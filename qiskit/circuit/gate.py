# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary gate."""

from typing import List, Optional, Union
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from .instruction import Instruction
from .operation import Operation
from .broadcast import Broadcaster


class Gate(Instruction, Operation):
    """Unitary gate."""

    broadcast_arguments = Broadcaster.gate

    def __init__(
        self, name: str, num_qubits: int, params: List, label: Optional[str] = None
    ) -> None:
        """Create a new gate.

        Args:
            name: The Qobj name of the gate.
            num_qubits: The number of qubits the gate acts on.
            params: A list of parameters.
            label: An optional label for the gate.
        """
        self.definition = None
        super().__init__(name, num_qubits, 0, params, label=label)

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the gate unitary matrix.

        Returns:
            np.ndarray: if the Gate subclass has a matrix definition.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        if hasattr(self, "__array__"):
            # pylint: disable=no-member
            return self.__array__(dtype=complex)
        raise CircuitError(f"to_matrix not defined for this {type(self)}")

    def power(self, exponent: float):
        """Creates a unitary gate as `gate^exponent`.

        Args:
            exponent (float): Gate^exponent

        Returns:
            qiskit.extensions.UnitaryGate: To which `to_matrix` is self.to_matrix^exponent.

        Raises:
            CircuitError: If Gate is not unitary
        """
        from qiskit.quantum_info.operators import Operator  # pylint: disable=cyclic-import
        from qiskit.extensions.unitary import UnitaryGate  # pylint: disable=cyclic-import
        from scipy.linalg import schur

        # Should be diagonalized because it's a unitary.
        decomposition, unitary = schur(Operator(self).data, output="complex")
        # Raise the diagonal entries to the specified power
        decomposition_power = []

        decomposition_diagonal = decomposition.diagonal()
        # assert off-diagonal are 0
        if not np.allclose(np.diag(decomposition_diagonal), decomposition):
            raise CircuitError("The matrix is not diagonal")

        for element in decomposition_diagonal:
            decomposition_power.append(pow(element, exponent))
        # Then reconstruct the resulting gate.
        unitary_power = unitary @ np.diag(decomposition_power) @ unitary.conj().T
        return UnitaryGate(unitary_power, label=f"{self.name}^{exponent}")

    def _return_repeat(self, exponent: float) -> "Gate":
        return Gate(name=f"{self.name}*{exponent}", num_qubits=self.num_qubits, params=self.params)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        """Return controlled version of gate. See :class:`.ControlledGate` for usage.

        Args:
            num_ctrl_qubits: number of controls to add to gate (default=1)
            label: optional gate label
            ctrl_state: The control state in decimal or as a bitstring
                (e.g. '111'). If None, use 2**num_ctrl_qubits-1.

        Returns:
            qiskit.circuit.ControlledGate: Controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.

        Raises:
            QiskitError: unrecognized mode or invalid ctrl_state
        """
        # pylint: disable=cyclic-import
        from .add_control import add_control

        return add_control(self, num_ctrl_qubits, label, ctrl_state)

    def validate_parameter(self, parameter):
        """Gate parameters should be int, float, or ParameterExpression"""
        if isinstance(parameter, ParameterExpression):
            if len(parameter.parameters) > 0:
                return parameter  # expression has free parameters, we cannot validate it
            if not parameter.is_real():
                msg = f"Bound parameter expression is complex in gate {self.name}"
                raise CircuitError(msg)
            return parameter  # per default assume parameters must be real when bound
        if isinstance(parameter, (int, float)):
            return parameter
        elif isinstance(parameter, (np.integer, np.floating)):
            return parameter.item()
        else:
            raise CircuitError(f"Invalid param type {type(parameter)} for gate {self.name}.")
