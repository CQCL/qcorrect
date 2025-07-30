from collections.abc import Callable
from typing import Generic

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


def test_lowering():
    n = 5

    @guppy
    def phys_zero() -> "CodeBlock[comptime(n)]":
        return CodeBlock(array(phys.qubit() for _ in range(comptime(n))))

    @guppy
    def phys_measure(
        q: "CodeBlock[comptime(n)] @ owned",
    ) -> "array[bool, comptime(n)]":
        return phys.measure_array(q.data_qs)

    code_def = CodeDef(n)

    code = code_def.get_module()

    @guppy
    def main() -> None:
        q = code.zero()
        code.measure(q)

    qct_hugr = code_def.lower(main.compile())

    @guppy
    def phys_main() -> None:
        phys_measure(phys_zero())

    phys_hugr = phys_main.compile()

    # TODO: Check qct_hugr matches phys_hugr


def test_phys_and_code_operations():
    code_def = CodeDef(5)

    code = code_def.get_module()

    @guppy
    def main() -> None:
        q_block = code.zero()
        phys_qubit = phys.qubit()

        code.measure(q_block)
        phys.measure(phys_qubit)

    code_def.lower(main.compile())
