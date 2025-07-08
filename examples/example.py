from typing import Generic

from guppylang.decorator import get_calling_frame, guppy
from guppylang.std import quantum as phys
from guppylang.std.builtins import array, comptime, nat, owned

import qcorrect as qct

N = guppy.nat_var("N")


@qct.type(copyable=False, droppable=False)
class CodeBlock(Generic[N]):
    data_qs: array[phys.qubit, N]


@qct.code
class CodeDef:
    def __init__(self, n: nat):
        self.n = n
        # Currently necessary to store calling frame to parse guppy definitions
        # Ideally this would move to the `qct.code` decorator
        self.frame = get_calling_frame()

    @qct.operation
    def zero() -> "CodeBlock[comptime(self.n)]":
        return CodeBlock(array(phys.qubit() for _ in range(comptime(self.n))))

    @qct.operation
    def measure(
        q: "CodeBlock[comptime(self.n)] @ owned",
    ) -> "array[bool, comptime(self.n)]":
        return phys.measure_array(q.data_qs)


code = CodeDef(5).get_module()


@guppy
def main() -> None:
    q = code.zero()
    code.measure(q)


hugr = main.compile()
