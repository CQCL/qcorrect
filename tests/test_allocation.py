from dataclasses import dataclass

import pytest
from guppylang import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array
from hugr.hugr import Node
from selene_hugr_qis_compiler import check_hugr

import qcorrect as qct


# Define programs to test
@guppy.comptime
def main_entangle() -> None:
    q0 = phys.qubit()
    q1 = phys.qubit()
    phys.h(q0)
    phys.cx(q0, q1)
    phys.measure(q0)
    phys.measure(q1)


@guppy.comptime
def main_qb_array() -> None:
    qb = array(phys.qubit() for _ in range(5))
    for q in qb:
        phys.measure(q)


@guppy.comptime
def main_op_loop() -> None:
    q0 = phys.qubit()
    for _ in range(2):
        phys.h(q0)
    phys.measure(q0)


@guppy.comptime
def main_mid_circuit_measure() -> None:
    q0 = phys.qubit()
    for _ in range(5):
        anc = phys.qubit()
        phys.cx(q0, anc)
        phys.measure(anc)
    phys.measure(q0)


# Allocation strategies to test
linear_allocator = qct.LinearAllocation(block_size=2)
random_allocator = qct.RandomAllocation(block_size=2)


@pytest.mark.parametrize(
    "main", [main_entangle, main_qb_array, main_op_loop, main_mid_circuit_measure]
)
@pytest.mark.parametrize("allocator", [linear_allocator, random_allocator])
def test_qubit_allocation(main, allocator):
    hugr = main.compile()

    qct.allocate(hugr, allocator)

    check_hugr(hugr.to_bytes())

    qct.validate_allocation(hugr)


# To test errors
@dataclass(kw_only=True)
class ConstantAllocation(qct.AllocationStrategy):
    """Constant allocation strategy. Every qubit is assigned the same address."""

    block: int
    position: int

    def __next__(self) -> qct.QubitAddress:
        return qct.QubitAddress(self.block, self.position)


def test_qalloc_address():
    hugr = main_entangle.compile()

    with pytest.raises(
        KeyError, match="Qubit address dict not found for QAlloc node \\(Node\\(4\\)\\)"
    ):
        qct.validate_allocation(hugr, block_size=1, max_blocks=1)


def test_previously_allocated():
    hugr = main_entangle.compile()

    qct.allocate(hugr, ConstantAllocation(block=0, position=0, block_size=2))

    with pytest.raises(
        ValueError,
        match="Address QubitAddress\\(block\\=0, position\\=0\\) "
        "has previously been allocated\\.",
    ):
        qct.validate_allocation(hugr, block_size=2, max_blocks=1)


def test_qalloc_addresses():
    hugr = main_entangle.compile()

    qct.allocate(hugr, ConstantAllocation(block=0, position=0, block_size=2))

    hugr.modules[0][Node(4)].metadata["qubit_addresses"][1] = {
        "block_id": 0,
        "position": 1,
    }
    with pytest.raises(
        KeyError,
        match="QAlloc should have a single address\\. Keys found: \\[0, 1\\]",
    ):
        qct.validate_allocation(hugr, block_size=1, max_blocks=1)


def test_max_block_size():
    hugr = main_entangle.compile()

    qct.allocate(hugr, ConstantAllocation(block=0, position=2, block_size=1))

    with pytest.raises(
        ValueError,
        match="Assigned position exceeds block size\\. position\\=2, block size \\= 2",
    ):
        qct.validate_allocation(hugr, block_size=2, max_blocks=1)


def test_max_blocks():
    hugr = main_entangle.compile()

    qct.allocate(hugr, ConstantAllocation(block=1, position=0, block_size=1))

    with pytest.raises(
        ValueError,
        match="Assigned block id exceeds maximum blocks\\. block_id=1, max_blocks=1",
    ):
        qct.validate_allocation(hugr, block_size=1, max_blocks=1)


def test_trace_addresses_match():
    hugr = main_entangle.compile()

    qct.allocate(hugr, qct.LinearAllocation(block_size=2))

    hugr.modules[0][Node(6)].metadata["qubit_addresses"][0] = {
        "block_id": 0,
        "position": 1,
    }

    with pytest.raises(
        ValueError,
        match="Qubit address \\(QubitAddress\\(block\\=0, position\\=1\\)\\) "
        "does not match input \\(QubitAddress\\(block\\=0, position\\=0\\)\\) "
        "at Node\\(6\\) offset 0\\.",
    ):
        qct.validate_allocation(hugr, block_size=2, max_blocks=1)


def test_trace_address_missing():
    hugr = main_entangle.compile()

    qct.allocate(hugr, qct.LinearAllocation(block_size=2))

    hugr.modules[0][Node(6)].metadata["qubit_addresses"].pop(0, None)

    with pytest.raises(
        KeyError, match="Qubit address dict not found for node \\(Node\\(6\\)\\)"
    ):
        qct.validate_allocation(hugr, block_size=2, max_blocks=1)
