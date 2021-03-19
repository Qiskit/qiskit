# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The absolute value of the average of a linear system of equations solution."""

from typing import Union, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow import I, Z, TensoredOp
from qiskit.quantum_info import Statevector

from .linear_system_observable import LinearSystemObservable


class AbsoluteAverage(LinearSystemObservable):
    r"""An observable for the absolute average of a linear system of equations solution.

    For a vector :math:`x=(x_1,...,x_N)`, the absolute average is defined as
    :math:`\abs{\frac{1}{N}\sum_{i=1}^{N}x_i}`.
    """

    def observable(self, num_qubits: int) -> Union[TensoredOp, List[TensoredOp]]:
        """The observable operator.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a sum of Pauli strings.
        """
        zero_op = (I + Z) / 2
        return TensoredOp(num_qubits * [zero_op])

    def observable_circuit(self, num_qubits: int) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The circuit implementing the absolute average observable.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a QuantumCircuit.
        """
        qc = QuantumCircuit(num_qubits)
        qc.h(qc.qubits)
        return qc

    def post_processing(self, solution: Union[float, List[float]], num_qubits: int,
                        scaling: float = 1) -> float:
        """Evaluates the absolute average on the solution to the linear system.

        Args:
            solution: The probability calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            scaling: Scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            ValueError: If the input is not in the correct format.
        """
        if isinstance(solution, list):
            if len(solution) == 1:
                solution = solution[0]
            else:
                raise ValueError("Solution probability must be given as a single value.")

        return np.real(np.sqrt(solution / (2 ** num_qubits)) / scaling)

    def evaluate_classically(self, solution: Union[np.array, QuantumCircuit]) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array or the circuit that prepares it.

        Returns:
            The value of the observable.
        """
        # Check if it is QuantumCircuits and get the array from them
        if isinstance(solution, QuantumCircuit):
            solution = Statevector(solution).data
        return np.abs(np.mean(solution))
