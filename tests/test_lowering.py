from collections.abc import Callable
from typing import Generic

from guppylang.decorator import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned
from hugr.package import Package

import qcorrect as qct

N = guppy.nat_var("N")


@qct.block
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


@qct.code
class CodeDef:
    n: nat

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
    def zero() -> "CodeBlock[comptime(n)]":
        return CodeBlock(array(phys.qubit() for _ in range(comptime(n))))

    @guppy
    def measure(
        q: "CodeBlock[comptime(n)] @ owned",
    ) -> "array[bool, comptime(n)]":
        return phys.measure_array(q.data_qs)

    code = CodeDef(n)

    @guppy
    def main() -> None:
        q = code.zero()
        code.measure(q)

    qct_hugr = qct.lower(code, main.compile())

    qct_node_names = {data.op.name() for _, data in qct_hugr.modules[0].nodes()}

    @guppy
    def main() -> None:
        q = zero()
        measure(q)

    phys_hugr = main.compile()

    phys_node_names = {data.op.name() for _, data in phys_hugr.modules[0].nodes()}

    # We are testing that the set of names matches between the hugr modules
    # Future tests should be more comprehensive but we are currently limited by
    # how `insert_hugr` and builtin functions being inserted twice. This should
    # be fixed in the future with hugr linking.
    assert qct_node_names == phys_node_names


def test_phys_and_code_operations():
    code = CodeDef(5)

    @guppy
    def main() -> None:
        q_block = code.zero()
        phys_qubit = phys.qubit()

        code.measure(q_block)
        phys.measure(phys_qubit)

    hugr = main.compile()

    hugr.extensions.append(code.hugr_ext)

    lowered_hugr = qct.lower(code, hugr)

    assert isinstance(lowered_hugr, Package)


if __name__ == "__main__":
    test_phys_and_code_operations()
