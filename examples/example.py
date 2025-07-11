from collections.abc import Callable
from typing import Generic

from guppylang.decorator import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned

import qcorrect as qct

# Define logical code block
N = guppy.nat_var("N")


@qct.type(copyable=False, droppable=False)
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


# Define code operations
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


# Create code instance and get guppy module
code = CodeDef(5).get_module()


# Write logical guppy program
@guppy
def main() -> None:
    q = code.zero()
    code.measure(q)


hugr = main.compile()
