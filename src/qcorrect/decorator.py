import builtins
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, TypeVar

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
    def get_module(self) -> ModuleType:
        self.guppy_module = ModuleType(self.__class__.__name__)
        self.hugr_ext = Extension(self.__class__.__name__, SemanticVersion(0, 1, 0))

        # Compile all `inner` operations
        for name, defn in inspect.getmembers(self):
            if hasattr(defn, "__qct_op__"):
                guppy_defn = defn()
                self.compiled_defs[name] = guppy_defn.id, guppy.compile(guppy_defn)

        # Define new `outer` operations
        for inner_def_name in self.compiled_defs:
            inner_id, inner_module_ptr = self.compiled_defs[inner_def_name]

            parsed_def = ENGINE.get_parsed(inner_id)

            assert isinstance(parsed_def, ParsedFunctionDef)
            ty = self.replace_inner_types(parsed_def.ty)

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

    def replace_inner_types(self, ty: FunctionType) -> FunctionType:
        "Replace all InnerStructTypes with new outer type definitions"
        assert isinstance(ty.inputs, list)
        for i, f_input in enumerate(ty.inputs):
            if isinstance(f_input.ty, InnerStructType):
                outer_type = f_input.ty.outer_type
                self.hugr_ext.add_type_def(f_input.ty.hugr_type_def)

                ty.inputs[i] = FuncInput(ty=outer_type, flags=f_input.flags)

        if isinstance(ty.output, InnerStructType):
            outer_type = ty.output.outer_type
            self.hugr_ext.add_type_def(ty.output.hugr_type_def)

            object.__setattr__(ty, "output", outer_type)

        ty_outer = FunctionType(
            inputs=ty.inputs,
            output=ty.output,
            input_names=ty.input_names,
            params=ty.params,
            comptime_args=ty.comptime_args,
        )

        return ty_outer

    def lower(self, package: ModulePointer) -> ModulePointer:
        # Find all nodes to replace
        nodes_to_replace = [
            (node, data)
            for node, data in package.module.nodes()
            if isinstance(data.op, ops.ExtOp)
        ]

        # Add all lowering functions
        func_defn_node = {}

        for f_name, op in self.hugr_ext.operations.items():
            lower_hugr = op.lower_funcs[0].hugr
            lower_hugr[ops.Node(1)].op.f_name = f_name # TODO: update name
            lower_hugr.delete_node(ops.Node(0))
            lower_hugr.module_root = ops.Node(1)
            lower_hugr.entrypoint = ops.Node(1)

            defn_nodes = package.module.insert_hugr(
                lower_hugr, package.module.module_root
            )

            func_defn_node[f_name] = defn_nodes[ops.Node(1)]

        # Replace all outer ops with calls to lowering functions
        for node, data in nodes_to_replace:
            if isinstance(data.op, ops.ExtOp):
                op_name = data.op.op_def().name

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
