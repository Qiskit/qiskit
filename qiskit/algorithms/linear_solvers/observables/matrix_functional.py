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

"""The matrix functional of the vector solution to the linear systems."""
from typing import Union, Optional, List
import numpy as np

from qiskit.aqua.operators import PauliSumOp
from .linear_system_observable import LinearSystemObservable
from qiskit import QuantumCircuit
from qiskit.opflow import I, Z, Zero, One, TensoredOp
from scipy.sparse import diags


class MatrixFunctional(LinearSystemObservable):
    """A class for the matrix functional of the vector solution to the linear systems."""

    def __init__(self, main_diag: float, off_diag: int) -> None:
        """
        Args:
            main_diag: The main diagonal of the tridiagonal Toeplitz symmetric matrix to compute
             the functional.
            off_diag: The off diagonal of the tridiagonal Toeplitz symmetric matrix to compute
             the functional.
        """
        self._main_diag = main_diag
        self._off_diag = off_diag

    def observable(self, num_qubits: int) -> Union[PauliSumOp, List[PauliSumOp]]:
        """The observable operators.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of sums of Pauli strings.

        Raises:
            TODO
        """
        ZeroOp = ((I + Z) / 2)
        OneOp = ((I - Z) / 2)
        observables = []
        # First we measure the norm of x
        observables.append(I ^ num_qubits)
        for i in range(0,num_qubits):
            j = num_qubits - i - 1
            observables.append([(I ^ j) ^ ZeroOp ^ TensoredOp(i * [OneOp]),
                                (I ^ j) ^ OneOp ^ TensoredOp(i * [OneOp])])
        return observables

    def post_rotation(self, num_qubits: int) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The observable circuits.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of QuantumCircuits.

        Raises:
            TODO
        """
        qcs = []
        # Again, the first value in the list will correspond to the norm of x
        qcs.append(QuantumCircuit(num_qubits))
        for i in range(0, num_qubits):
            qc = QuantumCircuit(num_qubits)
            for j in range(0, i):
                qc.cx(i, j)
            qc.h(i)
            qcs.append(qc)

        return qcs

    def post_processing(self, solution: Union[float, List[float]],
                        num_qubits: int,
                        constant: Optional[float] = 1) -> float:
        """Evaluates the matrix functional on the solution to the linear system.

        Args:
            solution: The list of probabilities calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            constant: If known, scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            TODO
        """
        if num_qubits is None:
            raise ValueError("Number of qubits must be defined to calculate the absolute average.")
        if not isinstance(solution, list):
            raise ValueError("Solution probabilities must be given in list form.")

        # Calculate the value from the off-diagonal elements
        off_val = 0
        for v in solution[1::]:
            off_val += (v[0]-v[1]) / (constant ** 2)
        main_val = solution[0] / (constant ** 2)
        return np.real(self._main_diag * main_val + self._off_diag * off_val)

    def evaluate_classically(self, solution: np.array) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array.

        Returns:
            The value of the observable.

        Raises:
            TODO
        """

        matrix = diags([self._off_diag, self._main_diag, self._off_diag], [-1, 0, 1],
                       shape=(len(solution), len(solution))).toarray()

        return np.dot(solution.transpose(), np.dot(matrix, solution))
