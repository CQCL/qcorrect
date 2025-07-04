from types import ModuleType
from typing import Generic

from guppylang import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned

import qcorrect as qct

N = guppy.nat_var("N")


@qct.type(copyable=False, droppable=False)
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


def CodeDef(name: str, n: nat):
    module = ModuleType(name)

    @qct.operation
    def zero() -> CodeBlock[comptime(n)]:
        return CodeBlock(array(phys.qubit() for _ in range(comptime(n))))

    object.__setattr__(module, "zero", zero)

    @qct.operation
    def measure(q: CodeBlock[comptime(n)] @ owned) -> array[bool, comptime(n)]:
        return phys.measure_array(q.data_qs)

    object.__setattr__(module, "measure", measure)

    return module


# Create new code instance
code = CodeDef(name="code", n=5)


# Write logical guppy program
@guppy
def main() -> None:
    q = code.zero()
    code.measure(qs)


# Compile program to Hugr
hugr = main.compile()
