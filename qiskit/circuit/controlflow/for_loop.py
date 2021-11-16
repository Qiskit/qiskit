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

"Circuit operation representing a ``for`` loop."

import warnings
from typing import Iterable, Optional, Union

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .control_flow import ControlFlowOp


class ForLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit
    (``body``) parameterized by a parameter ``loop_parameter`` through
    the set of integer values provided in ``indexset``.

    Parameters:
        loop_parameter: The placeholder parameterizing ``body`` to which
            the values from ``indexset`` will be assigned.
        indexset: A collection of integers to loop over.
        body: The loop body to be repeatedly executed.
        label: An optional label for identifying the instruction.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1          ├
             │  for_loop │
        q_2: ┤2          ├
             │           │
        c_0: ╡0          ╞
             └───────────┘

    """

    def __init__(
        self,
        loop_parameter: Union[Parameter, None],
        indexset: Iterable[int],
        body: QuantumCircuit,
        label: Optional[str] = None,
    ):
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits

        super().__init__(
            "for_loop", num_qubits, num_clbits, [loop_parameter, indexset, body], label=label
        )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        loop_parameter, indexset, body = parameters

        if not isinstance(loop_parameter, (Parameter, type(None))):
            raise CircuitError(
                "ForLoopOp expects a loop_parameter parameter to "
                "be either of type Parameter or None, but received "
                f"{type(loop_parameter)}."
            )

        if not isinstance(body, QuantumCircuit):
            raise CircuitError(
                "ForLoopOp expects a body parameter to be of type "
                f"QuantumCircuit, but received {type(body)}."
            )

        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the ForLoopOp. "
                f"ForLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}."
            )

        if (
            loop_parameter is not None
            and loop_parameter not in body.parameters
            and loop_parameter.name in (p.name for p in body.parameters)
        ):
            warnings.warn(
                "The Parameter provided as a loop_parameter was not found "
                "on the loop body and so no binding of the indexset to loop "
                "parameter will occur. A different Parameter of the same name "
                f"({loop_parameter.name}) was found. If you intended to loop "
                "over that Parameter, please use that Parameter instance as "
                "the loop_parameter.",
                stacklevel=2,
            )

        # Consume indexset into a tuple unless it was provided as a range.
        # Preserve ranges so that they can be exported as OpenQASM3 ranges.
        indexset = indexset if isinstance(indexset, range) else tuple(indexset)

        self._params = [loop_parameter, indexset, body]

    @property
    def blocks(self):
        return (self._params[2],)
