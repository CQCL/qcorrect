import builtins
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, Self, TypeVar

from guppylang.decorator import custom_guppy_decorator, get_calling_frame, guppy
from guppylang.definition.common import DefId
from guppylang.definition.function import ParsedFunctionDef
from guppylang.engine import DEF_STORE, ENGINE
from guppylang.tracing.object import GuppyDefinition
from guppylang.tys.subst import Inst
from guppylang.tys.ty import FuncInput, FunctionType
from hugr import ext as he
from hugr import ops
from hugr import tys as ht
from hugr.ext import Extension, OpDef, OpDefSig
from hugr.package import ModulePointer
from pydantic_extra_types.semantic_version import SemanticVersion

from qcorrect.tys import InnerStructType, RawInnerStructDef

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@custom_guppy_decorator
def block(cls: builtins.type[T]) -> builtins.type[T]:
    defn = RawInnerStructDef(DefId.fresh(), cls.__name__, None, cls)
    DEF_STORE.register_def(defn, get_calling_frame())
    for val in cls.__dict__.values():
        if isinstance(val, GuppyDefinition):
            DEF_STORE.register_impl(defn.id, val.wrapped.name, val.id)
    # We're pretending to return the class unchanged, but in fact we return
    # a `GuppyDefinition` that handles the comptime logic
    return GuppyDefinition(defn)  # type: ignore[return-value]


@custom_guppy_decorator
def operation(defn: F) -> F:
    """Decorator to define code operations"""

    defn.__setattr__("__qct_op__", True)

    return defn


class CodeDefinition:
    guppy_module: ModuleType
    hugr_ext: Extension
    compiled_defs: ClassVar[dict[str, tuple[DefId, ModulePointer]]] = {}

    @custom_guppy_decorator
    def get_module(self: Self) -> ModuleType:
        """Get guppy module for code definition instance. Code definitions are compiled
        and new `outer` operations are defined. Returns a Module with `outer` definitions
        that can be used in guppy programs."""
        self.guppy_module = ModuleType(self.__class__.__name__)
        self.hugr_ext = Extension(self.__class__.__name__, SemanticVersion(0, 1, 0))

        # Compile all `inner` operations
        for name, defn in inspect.getmembers(self):
            if hasattr(defn, "__qct_op__"):
                guppy_defn = defn()

                compiled_hugr = guppy.compile(guppy_defn)

                # Update FuncDefn name
                for _, data in compiled_hugr.module.nodes():
                    if (
                        isinstance(data.op, ops.FuncDefn)
                        and data.op.f_name == guppy_defn.wrapped.name
                    ):
                        data.op.f_name = name

                self.compiled_defs[name] = guppy_defn.id, compiled_hugr

        # Define new `outer` operations
        for inner_def_name in self.compiled_defs:
            inner_id, inner_module_ptr = self.compiled_defs[inner_def_name]

            parsed_def = ENGINE.get_parsed(inner_id)

            assert isinstance(parsed_def, ParsedFunctionDef)
            ty = self.define_inner_type_sig(parsed_def.ty)

            op_def = OpDef(
                name=inner_def_name,
                description=defn.__doc__ or "",
                signature=OpDefSig(poly_func=ty.to_hugr_poly()),
                lower_funcs=[
                    he.FixedHugr(
                        ht.ExtensionSet([self.hugr_ext.name]),
                        inner_module_ptr.module,
                    )
                ],
            )

            self.hugr_ext.add_op_def(op_def)

            def empty_dec() -> None: ...

            def hugr_op(
                op_def: OpDef,
            ) -> Callable[[ht.FunctionType, Inst], ops.DataflowOp]:
                def op(ty: ht.FunctionType, inst: Inst) -> ops.DataflowOp:
                    return ops.ExtOp(op_def, ty)

                return op

            guppy_op = guppy.hugr_op(
                hugr_op(op_def),
                name=inner_def_name,
                signature=ty,
            )(empty_dec)

            self.guppy_module.__setattr__(inner_def_name, guppy_op)

        return self.guppy_module

    def define_inner_type_sig(self, ty: FunctionType) -> FunctionType:
        """Define a new FunctionType that will be the signature of the `outer`
        operations.

        The function loops through all inputs/output types and replaces any
        InnerStructTypes with outer type definitions. All other types are unchanged."""

        outer_inputs = []

        for f_input in ty.inputs:
            if isinstance(f_input.ty, InnerStructType):
                outer_type = f_input.ty.outer_type
                self.hugr_ext.add_type_def(f_input.ty.hugr_type_def)

                outer_inputs.append(FuncInput(ty=outer_type, flags=f_input.flags))
            else:
                outer_inputs.append(f_input)

        if isinstance(ty.output, InnerStructType):
            outer_type = ty.output.outer_type
            self.hugr_ext.add_type_def(ty.output.hugr_type_def)

            outer_output = outer_type
        else:
            outer_output = ty.output

        return FunctionType(
            inputs=outer_inputs,
            output=outer_output,
            input_names=ty.input_names,
            params=ty.params,
            comptime_args=ty.comptime_args,
        )

    def lower(self, package: ModulePointer) -> ModulePointer:
        """Function to lower from `outer` operations to `inner`.

        Any `outer` operations are replace with calls to function definitions defined
        in the hugr extension from the code.
        """

        # Find all nodes to replace
        nodes_to_replace = [
            (node, data, data.op.op_def().name)
            for node, data in package.module.nodes()
            if isinstance(data.op, ops.ExtOp)
            and data.op.op_def().name in self.hugr_ext.operations
        ]

        # Add all lowering functions
        func_defn_node = {}

        for f_name, op in self.hugr_ext.operations.items():
            for lower_funcs in op.lower_funcs:
                lower_hugr = lower_funcs.hugr

                # Delete the module root and change to function definition
                assert isinstance(lower_hugr[lower_hugr.module_root].op, ops.Module)
                assert lower_hugr[lower_hugr.module_root].metadata["name"] == "__main__"

                # Find node for the function definition
                try:
                    func_node = next(
                        node
                        for node, data in lower_hugr.nodes()
                        if isinstance(data.op, ops.FuncDefn)
                        and data.op.f_name == f_name
                    )
                except StopIteration as e:
                    raise NameError(
                        f"Function Definition ({f_name}) not found in hugr."
                    ) from e

                defn_nodes = package.module.insert_hugr(
                    lower_hugr, package.module.module_root
                )

                # Get new node locations
                func_defn_node[f_name] = defn_nodes[func_node]
                new_module_entry = defn_nodes[lower_hugr.module_root]

                # Update parent/children
                for node in list(package.module[new_module_entry].children):
                    package.module[node].parent = package.module.module_root
                    package.module[package.module.module_root].children.append(node)
                    package.module[new_module_entry].children.remove(node)

                # Delete original entrypoint
                package.module.delete_node(new_module_entry)

        # Replace all outer ops with calls to lowering functions
        for node, data, op_name in nodes_to_replace:
            op_sig = self.hugr_ext.get_op(op_name).signature.poly_func
            op_ports_in = [port for _, port in package.module.incoming_links(node)]
            op_ports_out = [port for _, port in package.module.outgoing_links(node)]

            assert isinstance(op_sig, ht.PolyFuncType)

            func_call = ops.Call(signature=op_sig)

            # Remove outer node
            package.module.delete_node(node)

            call_node = package.module.add_node(
                func_call, data.parent, len(op_ports_out)
            )

            package.module.add_link(
                func_defn_node[op_name].out(0),
                call_node.inp(func_call._function_port_offset()),
            )

            # Link nodes
            for i, ports in enumerate(op_ports_in):
                for p in ports:
                    package.module.add_link(p, call_node.inp(i))
            for i, ports in enumerate(op_ports_out):
                for p in ports:
                    package.module.add_link(call_node.out(i), p)

        return package
