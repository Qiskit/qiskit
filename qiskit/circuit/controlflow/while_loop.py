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

"Circuit operation representing a ``while`` loop."

from typing import Optional, Tuple, Union

from qiskit.circuit import Clbit, ClassicalRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .control_flow import ControlFlowOp


class WhileLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit (``body``) until
    a condition (``condition``) evaluates as False.

    Parameters:
        condition: A condition to be checked prior to executing ``body``. Can be
            specified as either a tuple of a ``ClassicalRegister`` to be tested
            for equality with a given ``int``, or as a tuple of a ``Clbit`` to
            be compared to either a ``bool`` or an ``int``.
        body: The loop body to be repeatedly executed.
        label: An optional label for identifying the instruction.

    The classical bits used in ``condition`` must be a subset of those attached
    to ``body``.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────────┐
        q_0: ┤0            ├
             │             │
        q_1: ┤1            ├
             │  while_loop │
        q_2: ┤2            ├
             │             │
        c_0: ╡0            ╞
             └─────────────┘

    """

    def __init__(
        self,
        condition: Union[
            Tuple[ClassicalRegister, int],
            Tuple[Clbit, int],
            Tuple[Clbit, bool],
        ],
        body: QuantumCircuit,
        label: Optional[str] = None,
    ):
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits

        super().__init__("while_loop", num_qubits, num_clbits, [body], label=label)

        try:
            lhs, rhs = condition
        except TypeError as err:
            raise CircuitError(
                "WhileLoopOp expects a condition argument as either a "
                "Tuple[ClassicalRegister, int], a Tuple[Clbit, bool] or "
                f"a Tuple[Clbit, int], but received {condition} of type "
                f"{type(condition)}."
            ) from err

        if not (
            (isinstance(lhs, ClassicalRegister) and isinstance(rhs, int))
            or (isinstance(lhs, Clbit) and isinstance(rhs, (int, bool)))
        ):
            raise CircuitError(
                "WhileLoopOp expects a condition argument as either a "
                "Tuple[ClassicalRegister, int], a Tuple[Clbit, bool] or "
                f"a Tuple[Clbit, int], but receieved a {type(condition)}"
                f"[{type(lhs)}, {type(rhs)}]."
            )

        self.condition = condition

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        (body,) = parameters

        if not isinstance(body, QuantumCircuit):
            raise CircuitError(
                "WhileLoopOp expects a body parameter of type "
                f"QuantumCircuit, but received {type(body)}."
            )

        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the WhileLoopOp. "
                f"WhileLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}."
            )

        self._params = [body]

    @property
    def blocks(self):
        return (self._params[0],)

    def c_if(self, classical, val):
        raise NotImplementedError(
            "WhileLoopOp cannot be classically controlled through Instruction.c_if. "
            "Please use an IfElseOp instead."
        )
