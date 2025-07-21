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

            def hugr_op(op_def):
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
