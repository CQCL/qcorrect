from guppylang.checker.core import Globals
from guppylang.checker.func_checker import check_signature
from guppylang.decorator import custom_guppy_decorator, get_calling_frame, guppy
from guppylang.definition.common import DefId
from guppylang.definition.function import parse_py_func
from guppylang.definition.struct import RawStructDef
from guppylang.engine import DEF_STORE, ENGINE
from guppylang.std._internal.util import quantum_op
from hugr import tys as ht
from hugr.ext import ExplicitBound, Extension, OpDef, OpDefSig, TypeDef
from pydantic_extra_types.semantic_version import SemanticVersion

# This is required to get parsed function definitions
ENGINE.reset()

hugr_ext = Extension("qcorrect", SemanticVersion(0, 1, 0))


@custom_guppy_decorator
def type(copyable: bool = True, droppable: bool = True):
    frame = get_calling_frame()

    def wrapper(cls):
        defn = RawStructDef(DefId.fresh(), cls.__name__, None, cls)

        parsed_defn = defn.parse(Globals(frame), DEF_STORE.sources)

        type_def = TypeDef(
            name=cls.__name__,
            description=cls.__doc__,
            params=[],
            bound=ExplicitBound(ht.TypeBound.Any),
        )

        extType = ht.ExtType(type_def=type_def, args=[])

        hugr_ext.add_type_def(type_def)

        return guppy.type(
            extType,
            copyable=copyable,
            droppable=droppable,
            params=parsed_defn.params,
        )(cls)

    return wrapper


@custom_guppy_decorator
def operation(defn):
    guppy_dec = guppy.declare(defn)
    func_ast, _ = parse_py_func(
        DEF_STORE.raw_defs[guppy_dec.id].python_func, DEF_STORE.sources
    )
    ty = check_signature(func_ast, Globals(DEF_STORE.frames[guppy_dec.id]))

    op_def = OpDef(
        name=defn.__name__,
        description="",
        signature=OpDefSig(poly_func=ty.to_hugr_poly()),
        lower_funcs=[],
    )

    hugr_ext.add_op_def(op_def)

    def empty_dec() -> None: ...

    return guppy.hugr_op(
        quantum_op(defn.__name__, ext=hugr_ext), name=defn.__name__, signature=ty
    )(empty_dec)
