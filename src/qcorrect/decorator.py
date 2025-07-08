import functools
import inspect
from types import ModuleType

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

# Currently used to store hugr extensions for types
# Should be moved to the `code` decorator
hugr_ext = Extension("qcorrect", SemanticVersion(0, 1, 0))


@custom_guppy_decorator
def type(copyable: bool = True, droppable: bool = True):
    """Decorator to define code types"""
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
    """Decorator to define code operations"""

    @functools.wraps(defn)
    def dec(self, *args, **kwargs):
        func_ast, _ = parse_py_func(defn, DEF_STORE.sources)
        ty = check_signature(func_ast, Globals(self.frame))

        op_def = OpDef(
            name=defn.__name__,
            description=defn.__doc__ or "",
            signature=OpDefSig(poly_func=ty.to_hugr_poly()),
            lower_funcs=[],
        )

        self.hugr_ext.add_op_def(op_def)

        def empty_dec() -> None: ...

        return guppy.hugr_op(
            quantum_op(defn.__name__, ext=self.hugr_ext),
            name=defn.__name__,
            signature=ty,
        )(empty_dec)

    dec.__setattr__("qct_op", True)

    return dec


@custom_guppy_decorator
def code(cls):
    """Decorator to define a code"""

    class CodeDefBase(cls):
        @custom_guppy_decorator
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.guppy_module = ModuleType(cls.__name__)
            self.hugr_ext = Extension(cls.__name__, SemanticVersion(0, 1, 0))

        def get_module(self):
            for name, defn in inspect.getmembers(self, predicate=inspect.ismethod):
                if hasattr(defn, "qct_op"):
                    self.guppy_module.__setattr__(name, defn())

            return self.guppy_module

    return CodeDefBase
