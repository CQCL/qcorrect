from collections.abc import Sequence
from dataclasses import dataclass

from guppylang.ast_util import AstNode
from guppylang.checker.core import Globals
from guppylang.definition.common import DefId
from guppylang.definition.struct import CheckedStructDef, ParsedStructDef, RawStructDef
from guppylang.definition.ty import OpaqueTypeDef
from guppylang.engine import DEF_STORE
from guppylang.span import SourceMap
from guppylang.tys.arg import Argument
from guppylang.tys.ty import StructType
from hugr import tys as ht
from hugr.ext import ExplicitBound, Extension, TypeDef


@dataclass(frozen=True)
class RawInnerStructDef(RawStructDef):
    def parse(self, globals: Globals, sources: SourceMap) -> "ParsedInnerStructDef":
        parsed_struct_def = super().parse(globals, sources)
        return ParsedInnerStructDef(
            self.id,
            self.name,
            parsed_struct_def.defined_at,
            parsed_struct_def.params,
            parsed_struct_def.fields,
        )

    def get_outer_def(self, hugr_ext: Extension) -> OpaqueTypeDef:
        type_def = TypeDef(
            name=self.name,
            description=self.__doc__ or "",
            params=[],
            bound=ExplicitBound(ht.TypeBound.Any),
        )

        hugr_ext.add_type_def(type_def)

        return OpaqueTypeDef(
            DefId.fresh(),
            self.name,
            None,
            [],
            not False,  # copyable
            not False,  # droppable
            lambda _: ht.ExtType(type_def=type_def, args=[]),
            None,
        )


@dataclass(frozen=True)
class CheckedInnerStructDef(CheckedStructDef):
    pass


@dataclass(frozen=True)
class InnerStructType(StructType):
    defn: CheckedInnerStructDef


@dataclass(frozen=True)
class ParsedInnerStructDef(ParsedStructDef):
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
    ) -> "InnerStructType":
        super().check_instantiate(args, loc)

        globals = Globals(DEF_STORE.frames[self.id])

        checked_def = self.check(globals)

        return InnerStructType(args, checked_def)
