from typing import Generic

import pytest
from guppylang.decorator import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, owned
from guppylang_internals.error import GuppyError, GuppyTypeError
from hugr.package import Package

import qcorrect as qct

N = guppy.nat_var("N")


@qct.block
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


class CodeDef(qct.CodeDefinition):
    n: int

    @qct.operation
    def zero(self):
        @guppy
        def circuit() -> "CodeBlock[comptime(self.n)]":
            return CodeBlock(array(phys.qubit() for _ in range(comptime(self.n))))

        return circuit

    @qct.operation
    def measure(self):
        @guppy
        def circuit(
            q: "CodeBlock[comptime(self.n)] @ owned",
        ) -> "array[bool, comptime(self.n)]":
            return phys.measure_array(q.data_qs)

        return circuit


def test_code_usage():
    code = CodeDef(n=5)

    @guppy
    def main() -> None:
        q = code.zero()
        code.measure(q)

    hugr = main.compile()

    assert isinstance(hugr, Package)


def test_mismatched_codes():
    code4 = CodeDef(n=4)
    code5 = CodeDef(n=5)

    @guppy
    def main() -> None:
        q = code4.zero()
        code5.measure(q)

    with pytest.raises(GuppyTypeError):
        main.compile()


def test_block_dropped():
    code = CodeDef(n=5)

    @guppy
    def main() -> None:
        q = code.zero()

    with pytest.raises(GuppyError):
        main.compile()
