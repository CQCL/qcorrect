from collections import Counter
from typing import Generic

from guppylang import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, owned, result
from hugr.package import Package
from hugr.qsystem.result import QsysResult
from selene_hugr_qis_compiler import check_hugr
from selene_sim import Quest, build

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
    def x(self):
        @guppy
        def circuit(qb: "CodeBlock[comptime(self.n)]") -> None:
            for i in range(comptime(self.n)):
                phys.x(qb.data_qs[i])

        return circuit

    @qct.operation
    def measure(self):
        @guppy
        def circuit(
            q: "CodeBlock[comptime(self.n)] @ owned",
        ) -> "array[bool, comptime(self.n)]":
            return phys.measure_array(q.data_qs)

        return circuit


def test_lowering():
    n = 5

    # Define circuits at physical level
    @guppy
    def zero() -> "CodeBlock[comptime(n)]":
        return CodeBlock(array(phys.qubit() for _ in range(comptime(n))))

    @guppy
    def x(qb: "CodeBlock[comptime(n)]") -> None:
        for i in range(comptime(n)):
            phys.x(qb.data_qs[i])

    @guppy
    def measure(
        q: "CodeBlock[comptime(n)] @ owned",
    ) -> "array[bool, comptime(n)]":
        return phys.measure_array(q.data_qs)

    @guppy
    def main() -> None:
        q = zero()
        x(q)
        measure(q)

    phys_hugr = main.compile()

    phys_node_names = {data.op.name() for _, data in phys_hugr.modules[0].nodes()}

    # Create code instance
    code = CodeDef(n=n)

    @guppy
    def main() -> None:
        q = code.zero()
        code.x(q)
        code.measure(q)

    qct_hugr = code.lower(main.compile())

    qct_node_names = {data.op.name() for _, data in qct_hugr.modules[0].nodes()}

    # We are testing that the set of names matches between the hugr modules
    # Future tests should be more comprehensive but we are currently limited by
    # how `insert_hugr` and builtin functions being inserted twice. This should
    # be fixed in the future with hugr linking.
    assert qct_node_names == phys_node_names

    # Check hugr
    check_hugr(qct_hugr.to_bytes())


def test_phys_and_code_operations():
    code = CodeDef(n=5)

    @guppy
    def main() -> None:
        q_block = code.zero()
        phys_qubit = phys.qubit()

        code.measure(q_block)
        phys.measure(phys_qubit)

    hugr = main.compile()

    hugr.extensions.append(code.hugr_ext)

    lowered_hugr = code.lower(hugr)

    assert isinstance(lowered_hugr, Package)


def test_simulation():
    n_qubits = 5

    code = CodeDef(n=n_qubits)

    @guppy
    def main() -> None:
        q = code.zero()
        code.x(q)
        result("res", code.measure(q))

    lowered_hugr = code.lower(main.compile())

    res = QsysResult(
        build(lowered_hugr, "time_shot").run_shots(
            Quest(), n_qubits=n_qubits, n_shots=1
        )
    ).register_counts()["res"]

    assert Counter({f"{'1' * n_qubits}": 1}) == res
