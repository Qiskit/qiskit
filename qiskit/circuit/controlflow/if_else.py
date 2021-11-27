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

"Circuit operation representing an ``if/else`` statement."


from typing import Optional, Tuple, Union

from qiskit.circuit import Clbit, ClassicalRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .control_flow import ControlFlowOp


class IfElseOp(ControlFlowOp):
    """A circuit operation which executes a program (``true_body``) if a
    provided condition (``condition``) evaluates to true, and
    optionally evaluates another program (``false_body``) otherwise.

    Parameters:
        condition: A condition to be evaluated at circuit runtime which,
            if true, will trigger the evaluation of ``true_body``. Can be
            specified as either a tuple of a ``ClassicalRegister`` to be
            tested for equality with a given ``int``, or as a tuple of a
            ``Clbit`` to be compared to either a ``bool`` or an ``int``.
        true_body: A program to be executed if ``condition`` evaluates
            to true.
        false_body: A optional program to be executed if ``condition``
            evaluates to false.
        label: An optional label for identifying the instruction.

    If provided, ``false_body`` must be of the same ``num_qubits`` and
    ``num_clbits`` as ``true_body``.

    The classical bits used in ``condition`` must be a subset of those attached
    to the circuit on which this ``IfElseOp`` will be appended.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1          ├
             │  if_else  │
        q_2: ┤2          ├
             │           │
        c_0: ╡0          ╞
             └───────────┘

    """

    def __init__(
        self,
        condition: Union[
            Tuple[ClassicalRegister, int],
            Tuple[Clbit, int],
            Tuple[Clbit, bool],
        ],
        true_body: QuantumCircuit,
        false_body: Optional[QuantumCircuit] = None,
        label: Optional[str] = None,
    ):
        # Type checking generally left to @params.setter, but required here for
        # finding num_qubits and num_clbits.
        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(
                "IfElseOp expects a true_body parameter "
                f"of type QuantumCircuit, but received {type(true_body)}."
            )

        num_qubits = true_body.num_qubits
        num_clbits = true_body.num_clbits

        super().__init__("if_else", num_qubits, num_clbits, [true_body, false_body], label=label)

        try:
            lhs, rhs = condition
        except (TypeError, ValueError) as err:
            raise CircuitError(
                "IfElseOp expects a condition argument as either a "
                "Tuple[ClassicalRegister, int], a Tuple[Clbit, bool] or "
                f"a Tuple[Clbit, int], but received {condition} of type "
                f"{type(condition)}."
            ) from err

        if not (
            (isinstance(lhs, ClassicalRegister) and isinstance(rhs, int))
            or (isinstance(lhs, Clbit) and isinstance(rhs, (int, bool)))
        ):
            raise CircuitError(
                "IfElseOp expects a condition argument as either a "
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
        true_body, false_body = parameters

        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(
                "IfElseOp expects a true_body parameter of type "
                f"QuantumCircuit, but received {type(true_body)}."
            )

        if true_body.num_qubits != self.num_qubits or true_body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a true_body parameter with a num_qubits or "
                "num_clbits different than that of the IfElseOp. "
                f"IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {true_body.num_qubits}/{true_body.num_clbits}."
            )

        if false_body is not None:
            if not isinstance(false_body, QuantumCircuit):
                raise CircuitError(
                    "IfElseOp expects a false_body parameter of type "
                    f"QuantumCircuit, but received {type(false_body)}."
                )

            if false_body.num_qubits != self.num_qubits or false_body.num_clbits != self.num_clbits:
                raise CircuitError(
                    "Attempted to assign a false_body parameter with a num_qubits or "
                    "num_clbits different than that of the IfElseOp. "
                    f"IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                    f"Supplied body num_qubits/clbits: {false_body.num_qubits}/{false_body.num_clbits}."
                )

        self._params = [true_body, false_body]

    @property
    def blocks(self):
        if self.params[1] is None:
            return (self.params[0],)
        else:
            return (self.params[0], self.params[1])

    def c_if(self, classical, val):
        raise NotImplementedError(
            "WhileLoopOp cannot be classically controlled through Instruction.c_if. "
            "Please use an IfElseOp instead."
        )
