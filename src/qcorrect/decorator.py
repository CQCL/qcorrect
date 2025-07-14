import inspect
from types import ModuleType

from guppylang.checker.core import Globals
from guppylang.decorator import custom_guppy_decorator, get_calling_frame, guppy
from guppylang.definition.common import DefId
from guppylang.definition.struct import RawStructDef
from guppylang.definition.value import CallableDef
from guppylang.engine import DEF_STORE, ENGINE
from guppylang.tys.subst import Inst
from hugr import ops
from hugr import tys as ht
from hugr.ext import ExplicitBound, Extension, OpDef, OpDefSig, TypeDef
from pydantic_extra_types.semantic_version import SemanticVersion

# This is required to get parsed function definitions
ENGINE.reset()

# Currently used to store hugr extensions for types
# Should be moved to the `code` decorator
hugr_ext = Extension("qcorrect", SemanticVersion(0, 1, 0))


@custom_guppy_decorator
def type(cls):
    defn = RawStructDef(DefId.fresh(), cls.__name__, None, cls)

    parsed_defn = defn.parse(Globals(get_calling_frame()), DEF_STORE.sources)

    type_def = TypeDef(
        name=cls.__name__,
        description=cls.__doc__ or "",
        params=[],
        bound=ExplicitBound(ht.TypeBound.Any),
    )

    extType = ht.ExtType(type_def=type_def, args=[])

    hugr_ext.add_type_def(type_def)

    return guppy.type(
        extType,
        copyable=False,
        droppable=False,
        params=parsed_defn.params,
    )(cls)


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
        self.inner_defs = {}

        # Get all `inner` operations
        for name, defn in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(defn, "_qct_op"):
                self.inner_defs[name] = defn()

                # Define `outer` operations
                guppy_def = self.inner_defs[name]
                parsed_def = ENGINE.get_parsed(guppy_def.id)

                assert isinstance(parsed_def, CallableDef)

                ty = parsed_def.ty

                op_def = OpDef(
                    name=name,
                    description=defn.__doc__ or "",
                    signature=OpDefSig(poly_func=ty.to_hugr_poly()),
                    lower_funcs=[
                        # FixedHugr(
                        #     extensions=ht.ExtensionSet(),
                        #     hugr=compiled_def.package.to_str(),
                        # )
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
                    name=name,
                    signature=ty,
                )(empty_dec)
                self.guppy_module.__setattr__(name, guppy_op)

        return self.guppy_module
