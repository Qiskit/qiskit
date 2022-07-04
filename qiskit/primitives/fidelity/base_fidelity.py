# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Base fidelity primitive
"""

from abc import abstractmethod
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from typing import List, Union


class BaseFidelity:
    """
    Implements the interface to calculate fidelities.
    """

    def __init__(
        self,
        left_circuit: QuantumCircuit,
        right_circuit: QuantumCircuit,
    ):
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            |<left_circuit(x), right_circuit(y)>|^2,
        where x and y are parametrizations of the circuits.
        Args:
            - left_circuit: (Parametrized) quantum circuit
            - right_circuit: (Parametrized) quantum circuit
        Raises:
            - ValueError: left_circuit and right_circuit don't have the same number of qubits
        """

        if left_circuit.num_qubits != right_circuit.num_qubits:
            raise ValueError(
                f"The number of qubits for the left circuit ({left_circuit.num_qubits}) and right circuit ({right_circuit.num_qubits}) do not coincide."
            )

        self._left_circuit = left_circuit
        self._right_circuit = right_circuit

        # Assigning parameter arrays to the two circuits
        self._left_parameters = ParameterVector("x", left_circuit.num_parameters)
        self._left_circuit = left_circuit.assign_parameters(self._left_parameters)

        self._right_parameters = ParameterVector("y", right_circuit.num_parameters)
        self._right_circuit = right_circuit.assign_parameters(self._right_parameters)

    @abstractmethod
    def compute(
        self,
        values_left: Union[np.ndarray, List[np.ndarray]],
        values_right: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[float, List[float]]:
        """Compute the overlap of two quantum states bound by the parametrizations values_left and values_right.

        Args:
            values_left: Numerical parameters to be bound to the left circuit
            values_right: Numerical parameters to be bound to the right circuit

        Returns:
            The overlap of two quantum states defined by two parametrized circuits.
        """
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass
