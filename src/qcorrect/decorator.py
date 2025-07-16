import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from guppylang.checker.core import Globals
from guppylang.decorator import custom_guppy_decorator, get_calling_frame, guppy
from guppylang.definition.common import DefId
from guppylang.definition.function import ParsedFunctionDef
from guppylang.definition.struct import RawStructDef
from guppylang.definition.ty import OpaqueTypeDef
from guppylang.engine import DEF_STORE, ENGINE
from guppylang.tracing.object import GuppyDefinition
from guppylang.tys.subst import Inst
from hugr import ext as he
from hugr import ops
from hugr import tys as ht
from hugr.ext import ExplicitBound, Extension, OpDef, OpDefSig, TypeDef
from pydantic_extra_types.semantic_version import SemanticVersion

if TYPE_CHECKING:
    from hugr.package import ModulePointer


@dataclass(frozen=True)
class RawInnerStructDef(RawStructDef):
    pass


@custom_guppy_decorator
def type(cls):
    defn = RawInnerStructDef(DefId.fresh(), cls.__name__, None, cls)
    DEF_STORE.register_def(defn, get_calling_frame())
    for val in cls.__dict__.values():
        if isinstance(val, GuppyDefinition):
            DEF_STORE.register_impl(defn.id, val.wrapped.name, val.id)
    # We're pretending to return the class unchanged, but in fact we return
    # a `GuppyDefinition` that handles the comptime logic
    return GuppyDefinition(defn)  # type: ignore[return-value]


@custom_guppy_decorator
def operation(defn):
    """Decorator to define code operations"""

    defn.__setattr__("_qct_op", True)

    return defn


class CodeDefinition:
    guppy_module: ModuleType
    hugr_ext: Extension

    @custom_guppy_decorator
    def get_module(self) -> ModuleType:
        self.guppy_module = ModuleType(self.__class__.__name__)
        self.hugr_ext = Extension(self.__class__.__name__, SemanticVersion(0, 1, 0))
        self.compiled_defs: dict[str, tuple[DefId, ModulePointer]] = {}
        self.qct_types: dict[str, GuppyDefinition] = {}

        # Compile all `inner` operations
        for name, defn in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(defn, "_qct_op"):
                self.compiled_defs[name] = defn().id, guppy.compile(defn())

        # Define new `outer` operations
        for inner_def_name in self.compiled_defs:
            inner_id, inner_module_ptr = self.compiled_defs[inner_def_name]

            parsed_def = ENGINE.get_parsed(inner_id)

            assert isinstance(parsed_def, ParsedFunctionDef)

            ty = parsed_def.ty

            op_def = OpDef(
                name=name,
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
