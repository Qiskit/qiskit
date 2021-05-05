# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The matrix functional of the vector solution to the linear systems."""

from typing import Union, List
import numpy as np
from scipy.sparse import diags

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import I, Z, TensoredOp

from .linear_system_observable import LinearSystemObservable


class MatrixFunctional(LinearSystemObservable):
    """A class for the matrix functional of the vector solution to the linear systems.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.algorithms.linear_solvers.observables.matrix_functional import
             MatrixFunctional
            from qiskit.transpiler.passes import RemoveResetInZeroState

            tpass = RemoveResetInZeroState()

            vector = [1.0, -2.1, 3.2, -4.3]
            observable = MatrixFunctional(1, -1 / 3)

            init_state = vector / np.linalg.norm(vector)
            num_qubits = int(np.log2(len(vector)))

            # Get observable circuits
            obs_circuits = observable.observable_circuit(num_qubits)
            qcs = []
            for obs_circ in obs_circuits:
                qc = QuantumCircuit(num_qubits)
                qc.isometry(init_state, list(range(num_qubits)), None)
                qc.append(obs_circ, list(range(num_qubits)))
                qcs.append(tpass(qc.decompose()))

            # Get observables
            observable_ops = observable.observable(num_qubits)
            state_vecs = []
            # First is the norm
            state_vecs.append((~StateFn(observable_ops[0]) @ StateFn(qcs[0])).eval())
            for i in range(1, len(observable_ops), 2):
                state_vecs += [(~StateFn(observable_ops[i]) @ StateFn(qcs[i])).eval(),
                               (~StateFn(observable_ops[i + 1]) @ StateFn(qcs[i + 1])).eval()]

            # Obtain result
            result = observable.post_processing(state_vecs, num_qubits)

            # Obtain analytical evaluation
            exact = observable.evaluate_classically(init_state)
    """

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

    def observable(self, num_qubits: int) -> Union[TensoredOp, List[TensoredOp]]:
        """The observable operators.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of sums of Pauli strings.
        """
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        observables = []
        # First we measure the norm of x
        observables.append(I ^ num_qubits)
        for i in range(num_qubits):
            j = num_qubits - i - 1

            # TODO this if can be removed once the bug in Opflow is fixed where
            # TensoredOp([X, TensoredOp([])]).eval() ends up in infinite recursion
            if i > 0:
                observables += [
                    (I ^ j) ^ zero_op ^ TensoredOp(i * [one_op]),
                    (I ^ j) ^ one_op ^ TensoredOp(i * [one_op]),
                ]
            else:
                observables += [(I ^ j) ^ zero_op, (I ^ j) ^ one_op]

        return observables

    def observable_circuit(self, num_qubits: int) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The circuits to implement the matrix functional observable.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of QuantumCircuits.
        """
        qcs = []
        # Again, the first value in the list will correspond to the norm of x
        qcs.append(QuantumCircuit(num_qubits))
        for i in range(0, num_qubits):
            qc = QuantumCircuit(num_qubits)
            for j in range(0, i):
                qc.cx(i, j)
            qc.h(i)
            qcs += [qc, qc]

        return qcs

    def post_processing(
        self, solution: Union[float, List[float]], num_qubits: int, scaling: float = 1
    ) -> float:
        """Evaluates the matrix functional on the solution to the linear system.

        Args:
            solution: The list of probabilities calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            scaling: Scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            ValueError: If the input is not in the correct format.
        """
        if not isinstance(solution, list):
            raise ValueError("Solution probabilities must be given in list form.")

        # Calculate the value from the off-diagonal elements
        off_val = 0
        for i in range(1, len(solution), 2):
            off_val += (solution[i] - solution[i + 1]) / (scaling ** 2)
        main_val = solution[0] / (scaling ** 2)
        return np.real(self._main_diag * main_val + self._off_diag * off_val)

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

        matrix = diags(
            [self._off_diag, self._main_diag, self._off_diag],
            [-1, 0, 1],
            shape=(len(solution), len(solution)),
        ).toarray()

        return np.dot(solution.transpose(), np.dot(matrix, solution))
