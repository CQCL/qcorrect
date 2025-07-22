from collections.abc import Callable
from typing import Generic

import pytest
from guppylang.decorator import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned

import qcorrect as qct

N = guppy.nat_var("N")


@qct.block
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


class CodeDef(qct.CodeDefinition):
    def __init__(self, n: nat):
        self.n: nat = n

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


phys_n = 5

@guppy
def zero() -> "CodeBlock[comptime(phys_n)]":
    return CodeBlock(array(phys.qubit() for _ in range(comptime(phys_n))))


@guppy
def measure(
    q: "CodeBlock[comptime(phys_n)] @ owned",
) -> "array[bool, comptime(phys_n)]":
    return phys.measure_array(q.data_qs)


def test_lowering():
    code = CodeDef(5).get_module()

    @guppy
    def main() -> None:
        q = code.zero()
        code.measure(q)

    qct_hugr = main.compile()

    @guppy
    def main() -> None:
        q = zero()
        measure(q)

    phys_hugr = main.compile()

    # TODO: Check qct_hugr matches phys_hugr
