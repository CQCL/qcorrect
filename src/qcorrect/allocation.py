from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import choice
from typing import cast

import hugr.tys as ht
from hugr.hugr import Hugr, Node, NodeData
from hugr.ops import ExtOp, Op
from hugr.package import Package


@dataclass(frozen=True)
class QubitAddress:
    block: int
    position: int


@dataclass(kw_only=True)
class AllocationStrategy(ABC):
    """Abstract base class for qubit allocation strategies."""

    block_size: int
    allocated_blocks: list[int] = field(default_factory=list)
    allocated_positions: list[int] = field(default_factory=list)

    @abstractmethod
    def __next__(self) -> QubitAddress: ...


@dataclass(kw_only=True)
class LinearAllocation(AllocationStrategy):
    """Linear allocation strategy. Qubits are assigned to blocks in order of appearance
    in the hugr.
    """

    def __next__(self) -> QubitAddress:
        block_id = (
            self.allocated_blocks[-1] + 1 if len(self.allocated_blocks) > 0 else 0
        )
        block_position = (
            self.allocated_positions[-1] + 1 if len(self.allocated_positions) > 0 else 0
        )
        self.allocated_positions.append(block_position)

        if len(self.allocated_positions) == self.block_size:
            self.allocated_positions = []
            self.allocated_blocks.append(block_id)
        return QubitAddress(block_id, block_position)


@dataclass(kw_only=True)
class RandomAllocation(AllocationStrategy):
    """Random allocation strategy. Qubits are assigned addresses at random into each
    block. Blocks are still assigned sequentially.
    """

    def __next__(self) -> QubitAddress:
        block_id = (
            self.allocated_blocks[-1] + 1 if len(self.allocated_blocks) > 0 else 0
        )
        block_position = choice(  # noqa: S311
            list(set(range(self.block_size)) - set(self.allocated_positions))
        )
        self.allocated_positions.append(block_position)
        if len(self.allocated_positions) == self.block_size:
            self.allocated_positions = []
            self.allocated_blocks.append(block_id)
        return QubitAddress(block_id, block_position)


def get_qalloc_nodes(hugr: Hugr[Op]) -> list[Node]:
    """Helper method to get QAlloc ops from a hugr."""
    return [
        node
        for node, data in hugr.nodes()
        if isinstance(data.op, ExtOp) and data.op.name() == "tket.quantum.QAlloc"
    ]


def add_qubit_address_label(
    hugr: Hugr[Op], node: Node, port_offset: int, address: QubitAddress
) -> None:
    """Add qubit address (block_id and position) to a node metadata for a specific
    port_offset.

    Args:
        hugr: hugr which contains the node
        node: node to be labelled
        port_offset: port offset that corresponds to qubit
        address: block_id and position in logical block
    """
    if "qubit_addresses" not in hugr[node].metadata:
        hugr[node].metadata["qubit_addresses"] = {}

    hugr[node].metadata["qubit_addresses"][port_offset] = {
        "block_id": address.block,
        "position": address.position,
    }


def trace_qubit(
    hugr: Hugr[Op],
    start_node: Node,
    port_offset: int = 0,
    address: QubitAddress | None = None,
) -> list[tuple[Node, int]]:
    """Trace a qubit type through a HUGR. From `start_node` we find the next node from
    `post_offset` out link and recursively trace until we reach a port with no matching
    out link.

    If an `address` is provided then nodes will be labelled as the hugr is traversed.

    Args:
        hugr: HUGR to be searched.
        start_node: Starting node in the HUGR.
        port_offset: Offset for output port for qubit. Defaults to 0.
        address: Address to allocate for each qubit. Defaults to None.

    Returns:
        A list of tuples for each node and port offset.
    """
    if address is not None:
        add_qubit_address_label(hugr, start_node, port_offset, address)

    # Get link to next node and port kind
    out_link = list(hugr.outgoing_links(start_node))[port_offset]

    # Get out port type
    start_node_data = cast("NodeData", hugr.get(start_node))
    out_port_kind = start_node_data.op.port_kind(out_link[0])

    # If out port is qubit type then go to next node
    if isinstance(out_port_kind, ht.ValueKind) and isinstance(
        out_port_kind.ty, ht._QubitDef
    ):
        return [
            (start_node, port_offset),
            *trace_qubit(hugr, out_link[1][0].node, out_link[1][0].offset, address),
        ]
    # Else trace is finished
    else:
        return [(start_node, port_offset)]


def allocate(hugr: Package, strategy: AllocationStrategy) -> None:
    """Method to allocate qubits in a hugr to code blocks. This method loops through
    all nodes to find QAlloc nodes. The links are traced until we reach a measurement or
    discard. Each node is labelled to indicate which qubit should be allocated to which
    block address.

    Args:
        hugr: hugr to be allocated.
        strategy: strategy used to allocate addresses.
    """
    # Loop through all hugr modules
    for module in hugr.modules:
        # Loop through all qalloc nodes and trace qubit
        for node in get_qalloc_nodes(module):
            trace_qubit(module, node, port_offset=0, address=next(strategy))


# def validate_allocation(hugr: Package):

#     for module in hugr.modules[0]:
#         for node in get_qalloc_nodes(modules):
