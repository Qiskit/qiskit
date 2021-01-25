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

"""The absolute value of the average of the components of the vector solution to the
linear systems."""
from typing import Union, Optional, List
import numpy as np

from qiskit.aqua.operators import PauliSumOp
from .linear_system_observable import LinearSystemObservable
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import I, Z, Zero, One, TensoredOp


class AbsoluteAverage(LinearSystemObservable):
    r"""A class for the absolute value of the average of the components of the vector solution
    to the linear systems.
    For a vector :math:`x=(x_1,...,x_N)`, the average is defined as
    :math:`\abs{\frac{1}{N}\sum_{i-1}^{N}x_i}`."""
    def observable(self, num_qubits: int) -> Union[PauliSumOp, List[PauliSumOp]]:
        """The observable operator.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a sum of Pauli strings.

        Raises:
            TODO
        """
        ZeroOp = ((I + Z) / 2)
        return TensoredOp(num_qubits * [ZeroOp])

    def post_rotation(self, num_qubits: int) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The observable circuit.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a QuantumCircuit.

        Raises:
            TODO
        """
        qr = QuantumRegister(num_qubits)
        qc = QuantumCircuit(qr)
        qc.h(qr)
        return qc

    # TODO: can remove the list option or does it have to be as in the abstract method?.
    def post_processing(self, solution: Union[float, List[float]],
                        num_qubits: int,
                        constant: Optional[float] = 1) -> float:
        """Evaluates the absolute average on the solution to the linear system.

        Args:
            solution: The probability calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            constant: If known, scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            TODO
        """
        if num_qubits is None:
            raise ValueError("Number of qubits must be defined to calculate the absolute average.")
        if isinstance(solution, list):
            raise ValueError("Solution probability must be given as a single value.")

        return np.real(np.sqrt(solution / (2 ** num_qubits)) / constant)

    def evaluate_classically(self, solution: np.array) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array.

        Returns:
            The value of the observable.

        Raises:
            TODO
        """
        result = 0
        for xi in solution:
            result += xi
        return np.abs(result) / len(solution)
