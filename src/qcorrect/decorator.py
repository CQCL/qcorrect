import builtins
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
    cast,
    dataclass_transform,
)

from guppylang.decorator import custom_guppy_decorator, get_calling_frame
from guppylang.defs import GuppyDefinition
from guppylang_internals.compiler.core import CompilerContext
from guppylang_internals.decorator import hugr_op
from guppylang_internals.definition.common import DefId
from guppylang_internals.engine import DEF_STORE, ENGINE
from guppylang_internals.tys.subst import Inst
from guppylang_internals.tys.ty import FuncInput, FunctionType
from hugr import ext as he
from hugr import ops
from hugr import tys as ht
from hugr.ext import Extension, OpDef, OpDefSig
from hugr.package import Package
from pydantic_extra_types.semantic_version import SemanticVersion

from qcorrect.tys import InnerStructType, RawInnerStructDef

if TYPE_CHECKING:
    from guppylang_internals.definition.function import ParsedFunctionDef

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@custom_guppy_decorator
def block(cls: builtins.type[T]) -> builtins.type[T]:
    defn = RawInnerStructDef(DefId.fresh(), cls.__name__, None, cls)
    DEF_STORE.register_def(defn, get_calling_frame())
    for val in cls.__dict__.values():
        if isinstance(val, GuppyDefinition):
            DEF_STORE.register_impl(defn.id, val.wrapped.name, val.id)
    # Prior to Python 3.13, the `__firstlineno__` attribute on classes is not set.
    # However, we need this information to precisely look up the source for the
    # class later. If it's not there, we can set it from the calling frame:
    if not hasattr(cls, "__firstlineno__"):
        object.__setattr__(cls, "__firstlineno__", get_calling_frame().f_lineno)
    # We're pretending to return the class unchanged, but in fact we return
    # a `GuppyDefinition` that handles the comptime logic
    return GuppyDefinition(defn)  # type: ignore[return-value]


@custom_guppy_decorator
def operation(defn: Callable[..., F]) -> F:
    """Decorator to define code operations"""

    defn.__setattr__("__qct_op__", True)

    return defn  #  type: ignore[return-value]


def op_dec(
    op_name: str,
    compiled_defn: Package,
    hugr_ext: Extension,
) -> Callable[[ht.FunctionType, Inst, CompilerContext], ops.DataflowOp]:
    def op(ty: ht.FunctionType, op_inst: Inst, ctx: CompilerContext) -> ops.DataflowOp:
        # Define `outer` hugr operation
        op_def = OpDef(
            name=op_name,
            description="",
            signature=OpDefSig(poly_func=ty),
            lower_funcs=[
                he.FixedHugr(
                    ht.ExtensionSet([ext.name for ext in compiled_defn.extensions]),
                    hugr_module,
                )
                for hugr_module in compiled_defn.modules
            ],
        )

        # Add op_def to hugr extension
        hugr_ext.add_op_def(op_def)

        return ops.ExtOp(op_def, ty, [arg.to_hugr(ctx) for arg in op_inst])

    return op


@dataclass_transform()
class CodeDefinition(ModuleType):
    guppy_module: ModuleType
    hugr_ext: Extension
    compiled_defs: ClassVar[dict[str, tuple[DefId, Package]]] = {}

    def __new__(cls, *args, **kwargs):
        # Create new instance and initialise
        inst = super().__new__(dataclass(cls), *args, **kwargs)
        inst.__init__(*args, **kwargs)

        # Set name and docs for Module
        inst.__name__ = cls.__name__
        inst.__doc__ = cls.__doc__

        # Define new Hugr extension
        inst.hugr_ext = Extension(inst.__name__, SemanticVersion(0, 1, 0))

        # Add all definitions to Module
        for name, defn in inspect.getmembers(cls):
            if hasattr(defn, "__qct_op__"):
                inst.__setattr__(name, defn(inst))

        # Compile all definitions
        for name, defn in inspect.getmembers(inst):
            if isinstance(defn, GuppyDefinition):
                compiled_hugr = defn.compile()
                # Update FuncDefn name
                for module in compiled_hugr.modules:
                    for _, data in module.nodes():
                        if (
                            isinstance(data.op, ops.FuncDefn)
                            and data.op.f_name == defn.wrapped.name
                        ):
                            data.op.f_name = name
                inst.compiled_defs[name] = (defn.id, compiled_hugr)

        # Replace `inner` operations with `outer`
        for name, (
            defn_id,
            compiled_hugr,
        ) in inst.compiled_defs.items():
            if isinstance(defn, GuppyDefinition):

                def empty_dec() -> None: ...

                parsed_defn = cast("ParsedFunctionDef", ENGINE.get_parsed(defn_id))

                guppy_op = hugr_op(
                    op_dec(name, compiled_hugr, inst.hugr_ext),
                    name=name,
                    signature=inst.define_outer_type_sig(parsed_defn.ty),
                )(empty_dec)

                inst.__setattr__(name, guppy_op)

        return inst

    def define_outer_type_sig(self, ty: FunctionType) -> FunctionType:
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

    def lower(self, package: Package) -> Package:
        """Function to lower from `outer` operations to `inner`.

        Any `outer` operations are replaced with calls to function definitions defined
        in the hugr extension from the code.
        """

        # Find all nodes to replace
        nodes_to_replace = [
            (node, data, data.op.op_def().name)
            for node, data in package.modules[0].nodes()
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

                lower_hugr.entrypoint = lower_hugr.module_root

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

                defn_nodes = package.modules[0].insert_hugr(
                    lower_hugr, package.modules[0].module_root
                )

                # Get new node locations
                func_defn_node[f_name] = defn_nodes[func_node]
                new_module_entry = defn_nodes[lower_hugr.module_root]

                # Update parent/children
                for node in list(package.modules[0][new_module_entry].children):
                    package.modules[0][node].parent = package.modules[0].module_root
                    package.modules[0][package.modules[0].module_root].children.append(
                        node
                    )
                    package.modules[0][new_module_entry].children.remove(node)

                # Delete original entrypoint
                package.modules[0].delete_node(new_module_entry)

        # Replace all outer ops with calls to lowering functions
        for node, data, op_name in nodes_to_replace:
            op_sig = self.hugr_ext.get_op(op_name).signature.poly_func
            op_ports_in = [port for _, port in package.modules[0].incoming_links(node)]
            op_ports_out = [port for _, port in package.modules[0].outgoing_links(node)]

            assert isinstance(op_sig, ht.PolyFuncType)

            func_call = ops.Call(signature=op_sig)

            # Remove outer node
            parent = package.modules[0][node].parent
            if parent:
                package.modules[0][parent].children.remove(node)

            _, package.modules[0]._nodes[node.idx] = (
                package.modules[0]._nodes[node.idx],
                None,
            )

            # Free up the metadata dictionary
            node._metadata = {}

            package.modules[0]._free_nodes.append(node)

            call_node = package.modules[0].add_node(
                func_call, data.parent, len(op_ports_out)
            )

            package.modules[0].add_link(
                func_defn_node[op_name].out(0),
                call_node.inp(func_call._function_port_offset()),
            )

            # Link nodes
            for i, ports in enumerate(op_ports_in):
                for p in ports:
                    package.modules[0].delete_link(p, node.inp(i))
                    package.modules[0].add_link(p, call_node.inp(i))
            for i, ports in enumerate(op_ports_out):
                for p in ports:
                    package.modules[0].delete_link(node.out(i), p)
                    package.modules[0].add_link(call_node.out(i), p)

        return package
