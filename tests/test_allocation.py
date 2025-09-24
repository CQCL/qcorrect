from guppylang import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array
from selene_hugr_qis_compiler import check_hugr

import qcorrect as qct


def test_qubit_allocation():
    @guppy.comptime
    def main() -> None:
        q0 = phys.qubit()
        q1 = phys.qubit()
        phys.h(q0)
        phys.cx(q0, q1)
        phys.measure(q0)
        phys.measure(q1)

    hugr = main.compile()

    qct.allocate(hugr, qct.LinearAllocation(block_size=1))

    check_hugr(hugr.to_bytes())


def test_qubit_array_allocation():
    @guppy.comptime
    def main() -> None:
        qb = array(phys.qubit() for _ in range(5))
        for q in qb:
            phys.measure(q)

    hugr = main.compile()

    qct.allocate(hugr, qct.LinearAllocation(block_size=5))

    check_hugr(hugr.to_bytes())


def test_op_loop():
    @guppy.comptime
    def main() -> None:
        q0 = phys.qubit()
        for _ in range(2):
            phys.h(q0)
        phys.measure(q0)

    hugr = main.compile()

    qct.allocate(hugr, qct.LinearAllocation(block_size=5))

    check_hugr(hugr.to_bytes())


def test_mid_circuit_measurement():
    @guppy.comptime
    def main() -> None:
        q0 = phys.qubit()
        for _ in range(5):
            anc = phys.qubit()
            phys.cx(q0, anc)
            phys.measure(anc)
        phys.measure(q0)

    hugr = main.compile()

    qct.allocate(hugr, qct.LinearAllocation(block_size=2))

    check_hugr(hugr.to_bytes())
