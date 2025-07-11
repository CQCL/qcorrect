from collections.abc import Callable
from typing import Generic

import pytest
from guppylang.decorator import get_calling_frame, guppy
from guppylang.error import GuppyError, GuppyTypeError
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned
from hugr.package import ModulePointer

import qcorrect as qct

N = guppy.nat_var("N")


@qct.type(copyable=False, droppable=False)
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


class CodeDef(qct.CodeDefinition):
    def __init__(self, n: nat):
        self.n: nat = n
        self.frame = get_calling_frame()

    @qct.operation
    def zero(self) -> Callable:
        @guppy
        def circuit() -> "CodeBlock[comptime(self.n)]":
            return CodeBlock(array(phys.qubit() for _ in range(comptime(self.n))))

        return circuit

    @qct.operation
    def measure(self) -> Callable:
        @guppy
        def circuit(
            q: "CodeBlock[comptime(self.n)] @ owned",
        ) -> "array[bool, comptime(self.n)]":
            return phys.measure_array(q.data_qs)

        return circuit


def test_code_usage():
    code = CodeDef(5).get_module()

    @guppy
    def main() -> None:
        q = code.zero()
        code.measure(q)

    hugr = main.compile()

    assert isinstance(hugr, ModulePointer)


def test_mismatched_codes():
    code4 = CodeDef(4).get_module()
    code5 = CodeDef(5).get_module()

    @guppy
    def main() -> None:
        q = code4.zero()
        code5.measure(q)

    with pytest.raises(GuppyTypeError):
        main.compile()


def test_block_dropped():
    code = CodeDef(5).get_module()

    @guppy
    def main() -> None:
        q = code.zero()

    with pytest.raises(GuppyError):
        main.compile()
