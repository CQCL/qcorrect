from collections.abc import Callable
from typing import Generic, no_type_check

from guppylang.decorator import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned

import qcorrect as qct

# Define logical code block
N = guppy.nat_var("N")


@qct.block
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


# Define code operations
class CodeDef(qct.CodeDefinition):
    def __init__(self, n: nat):
        self.n: nat = n

    @qct.operation
    def zero(self) -> Callable:
        @guppy
        @no_type_check
        def circuit() -> "CodeBlock[comptime(self.n)]":
            return CodeBlock(array(phys.qubit() for _ in range(comptime(self.n))))

        return circuit

    @qct.operation
    def cx(self) -> Callable:
        @guppy
        @no_type_check
        def circuit(
            ctl: "CodeBlock[comptime(self.n)]", tgt: "CodeBlock[comptime(self.n)]"
        ) -> None:
            for i in range(comptime(self.n)):
                phys.cx(ctl.data_qs[i], tgt.data_qs[i])

        return circuit

    @qct.operation
    def measure(self) -> Callable:
        @guppy
        @no_type_check
        def circuit(
            q: "CodeBlock[comptime(self.n)] @ owned",
        ) -> "array[bool, comptime(self.n)]":
            return phys.measure_array(q.data_qs)

        return circuit


# Create code instance and get guppy module
code_def = CodeDef(5)

code = code_def.get_module()


# Write logical guppy program
@guppy
def main() -> None:
    q0 = code.zero()
    q1 = code.zero()
    code.cx(q0, q1)
    code.measure(q0)
    code.measure(q1)


hugr = main.compile()

hugr = code_def.lower(hugr)
