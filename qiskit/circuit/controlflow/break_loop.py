# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"Circuit operation representing a ``break`` from a loop."

from typing import Optional

from qiskit.circuit.instruction import Instruction


class BreakLoopOp(Instruction):
    """A circuit operation which, when encountered, jumps to the end of
    the nearest enclosing loop.

    .. note:

        Can be inserted only within the body of a loop op, and must span
        the full width of that block.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────────┐
        q_0: ┤0             ├
             │              │
        q_1: ┤1             ├
             │  break_loop  │
        q_2: ┤2             ├
             │              │
        c_0: ╡0             ╞
             └──────────────┘

    """

    def __init__(self, num_qubits: int, num_clbits: int, label: Optional[str] = None):

        super().__init__("break_loop", num_qubits, num_clbits, [], label=label)
