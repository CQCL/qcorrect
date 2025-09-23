from guppylang import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array
from selene_hugr_qis_compiler import check_hugr

import qcorrect as qct


def test_qubit_allocation():
    @guppy
    def main() -> None:
        q = phys.qubit()
        phys.measure(q)

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
