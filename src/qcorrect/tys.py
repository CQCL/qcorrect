from collections.abc import Callable, Sequence
from dataclasses import dataclass

from guppylang_internals.ast_util import AstNode
from guppylang_internals.checker.core import Globals
from guppylang_internals.definition.common import DefId
from guppylang_internals.definition.struct import (
    CheckedStructDef,
    ParsedStructDef,
    RawStructDef,
)
from guppylang_internals.definition.ty import OpaqueTypeDef
from guppylang_internals.engine import DEF_STORE
from guppylang_internals.span import SourceMap
from guppylang_internals.tys.arg import Argument
from guppylang_internals.tys.common import QuantifiedToHugrContext, ToHugrContext
from guppylang_internals.tys.ty import OpaqueType, StructType
from hugr import tys as ht
from hugr.ext import ExplicitBound, TypeDef


@dataclass(frozen=True)
class RawInnerStructDef(RawStructDef):
    def parse(self, globals: Globals, sources: SourceMap) -> "ParsedInnerStructDef":
        parsed_struct_def = super().parse(globals, sources)

        hugr_type_def = TypeDef(
            name=parsed_struct_def.name,
            description=parsed_struct_def.description,
            params=[
                param.to_hugr(QuantifiedToHugrContext(parsed_struct_def.params))
                for param in parsed_struct_def.params
            ],
            bound=ExplicitBound(ht.TypeBound.Linear),
        )

        def to_hugr_gen(
            type_def: TypeDef,
        ) -> Callable[[Sequence[Argument], ToHugrContext], ht.Type]:
            def to_hugr(args: Sequence[Argument], ctx: ToHugrContext) -> ht.Type:
                return ht.ExtType(
                    type_def=type_def, args=[arg.to_hugr(ctx) for arg in args]
                )

            return to_hugr

        outer_type_defn = OpaqueTypeDef(
            DefId.fresh(),
            parsed_struct_def.name,
            None,
            parsed_struct_def.params,
            True,
            True,
            to_hugr_gen(hugr_type_def),
            ht.TypeBound.Linear,
        )

        return ParsedInnerStructDef(
            self.id,
            self.name,
            parsed_struct_def.defined_at,
            parsed_struct_def.params,
            parsed_struct_def.fields,
            hugr_type_def,
            outer_type_defn,
        )


@dataclass(frozen=True)
class CheckedInnerStructDef(CheckedStructDef):
    pass


@dataclass(frozen=True)
class InnerStructType(StructType):
    defn: CheckedInnerStructDef
    hugr_type_def: TypeDef
    outer_type: OpaqueType


@dataclass(frozen=True)
class ParsedInnerStructDef(ParsedStructDef):
    hugr_type_def: TypeDef
    outer_type_defn: OpaqueTypeDef

    def check(self, globals: Globals) -> CheckedInnerStructDef:
        checked_struct_def = super().check(globals)

        return CheckedInnerStructDef(
            checked_struct_def.id,
            checked_struct_def.name,
            checked_struct_def.defined_at,
            checked_struct_def.params,
            checked_struct_def.fields,
        )

    def check_instantiate(
        self, args: Sequence[Argument], loc: AstNode | None = None
    ) -> InnerStructType:
        super().check_instantiate(args, loc)

        globals = Globals(DEF_STORE.frames[self.id])

        checked_def = self.check(globals)

        outer_type = self.outer_type_defn.check_instantiate(args, loc)

        return InnerStructType(args, checked_def, self.hugr_type_def, outer_type)
