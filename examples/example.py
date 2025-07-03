from typing import Generic

from guppylang import guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned

import qcorrect as qct

# Define a new codeblock type with parameter `N`
N = guppy.nat_var("N")

class ExampleCode:

    @qct.type(copyable=False, droppable=False)
    class CodeBlock(Generic[N]):
        data_qs: array[phys.qubit, N]

    # Define logical operations
    @qct.operation
    def zero(n: nat @ comptime) -> "CodeBlock[n]":
        return CodeBlock(array(phys.qubit() for _ in range(n)))

    @qct.operation
    def measure(q: CodeBlock[N] @ owned) -> array[bool, N]:
        return phys.measure_array(q.data_qs)



# Define a new codeblock type with parameter `N`
N = guppy.nat_var("N")

@qct.type(copyable=False, droppable=False)
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]

class ExampleCode:

    # Define logical operations
    @qct.operation
    def zero(n: nat @ comptime) -> "CodeBlock[n]":
        return CodeBlock(array(phys.qubit() for _ in range(n)))

    @qct.operation
    def measure(q: CodeBlock[N] @ owned) -> array[bool, N]:
        return phys.measure_array(q.data_qs)
# Define a code instance
code = ExampleCode()

# Use new code to create a hugr module
@guppy
def main() -> None:
    q = code.zero(6)
    code.measure(q)

hugr = main.compile()
