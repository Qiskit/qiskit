# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""The HHL algorithm."""

from typing import Optional, Union, List, Callable, Tuple
import numpy as np

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.opflow import Z, I, StateFn, TensoredOp
from .exact_inverse import ExactInverse
from .linear_solver import LinearSolver, LinearSolverResult
from .observables.linear_system_observable import LinearSystemObservable


# logger = logging.getLogger(__name__)

class HHL(LinearSolver):
    """The HHL algorithm to solve systems of linear equations"""
    def __init__(self, epsilon: float = 1e-2) -> None:
        """
        Args:
         epsilon : Error tolerance.

        .. note::

        References:

        [1]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020).
             Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
             `arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`_

        """
        super().__init__()

        self._epsilon = epsilon
        # Tolerance for the different parts of the algorithm as per [1]
        self._epsilon_r = epsilon / 3  # conditioned rotation
        self._epsilon_s = epsilon / 3  # state preparation
        self._epsilon_a = epsilon / 6  # hamiltonian simulation

        # Time of the evolution. Once the matrix is specified,
        # it can be updated 2 * np.pi / lambda_max
        self._evo_time = 2 * np.pi

        # Circuits for the different blocks of the algorithm
        self._post_rotation = None

        # For now the default inverse implementation is exact
        self._exact_inverse = True

    def get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float:
        """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.

        Args:
            n_l: The number of qubits to represent the eigenvalues.
            lambda_min: the smallest eigenvalue.
            lambda_max: the largest eigenvalue.

        Returns:
            The value of the scaling factor.
        """
        formatstr = "#0" + str(n_l + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2 ** n_l - 1) / lambda_max)
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep

    def calculate_norm(self, qc: QuantumCircuit, scaling: Optional[float] = 1) -> float:
        """Calculates the value of the euclidean norm of the solution.

        Args:
            qc: The quantum circuit preparing the solution x to the system.
            scaling: Factor scaling the solution vector.

        Returns:
            The value of the euclidean norm of the solution.
        """
        # Calculate the number of qubits
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        na = qc.num_ancillas

        # Create the Operators Zero and One
        zero_op = ((I + Z) / 2)
        one_op = ((I - Z) / 2)

        # Norm observable
        observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ (I ^ nb)
        norm_2 = (~StateFn(observable) @ StateFn(qc)).eval()

        return np.real(np.sqrt(norm_2) / scaling)

    def calculate_observable(self, qc: QuantumCircuit,
                             observable: Optional[Union[LinearSystemObservable, BaseOperator,
                                                        List[BaseOperator]]] = None,
                             post_rotation: Optional[Union[QuantumCircuit, List[QuantumCircuit]]]
                             = None,
                             post_processing: Optional[Callable[[Union[float, List[float]]],
                                                                Union[float, List[float]]]] = None,
                             scaling: Optional[float] = 1) -> Tuple[Union[float, List[float]],
                                                                    Union[float, List[float]]]:
        """Calculates the value of the observable(s) given.

        Args:
            qc: The quantum circuit preparing the solution x to the system.
            observable: Information to be extracted from the solution.
            post_rotation: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.
            scaling: Factor scaling the solution vector.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a
             tuple.
        """
        # Get the number of qubits
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        na = qc.num_ancillas

        if observable is not None and isinstance(observable, LinearSystemObservable):
            post_rotation = observable.post_rotation(nb)
            post_processing = observable.post_processing
            observable = observable.observable(nb)

        # Create the Operators Zero and One
        zero_op = ((I + Z) / 2)
        one_op = ((I - Z) / 2)
        # List of quantum circuits with post_rotation gates appended
        qcs = []
        # Observable gates
        if post_rotation is not None and isinstance(post_rotation, list):
            for circ in post_rotation:
                qc_temp = QuantumCircuit(qc.num_qubits)
                qc_temp.append(qc, list(range(qc.num_qubits)))
                qc_temp.append(circ, list(range(nb)))
                qcs.append(qc_temp)
        elif post_rotation:
            qc.append(post_rotation, list(range(nb)))

        # Update observable to include ancilla and rotation qubit
        result = []
        if isinstance(observable, list):
            for i, obs in enumerate(observable):
                if isinstance(obs, list):
                    result_temp = []
                    for ob in obs:
                        new_observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ ob
                        if qcs:
                            result_temp.append((~StateFn(new_observable) @ StateFn(qcs[i])).eval())
                        else:
                            result_temp.append((~StateFn(new_observable) @ StateFn(qc)).eval())
                    result.append(result_temp)
                else:
                    if obs is None:
                        obs = I ^ nb
                    new_observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ obs
                    if qcs:
                        result.append((~StateFn(new_observable) @ StateFn(qcs[i])).eval())
                    else:
                        result.append((~StateFn(new_observable) @ StateFn(qc)).eval())
        else:
            if observable is None:
                observable = I ^ nb

            new_observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ observable
            if qcs:
                for circ in qcs:
                    result.append((~StateFn(new_observable) @ StateFn(circ)).eval())
            else:
                result = (~StateFn(new_observable) @ StateFn(qc)).eval()

        if isinstance(result, list):
            circuit_results = result
        else:
            circuit_results = [result]
        if post_processing is not None:

            return post_processing(result, nb, scaling), circuit_results
        else:
            return result, circuit_results

    def construct_circuit(self, matrix: Union[np.ndarray, QuantumCircuit],
                          vector: Union[np.ndarray, QuantumCircuit]) -> QuantumCircuit:
        """Construct the HHL circuit.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        # State preparation circuit - default is qiskit
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, np.ndarray):
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb).initialize(vector / np.linalg.norm(vector),
                                                           list(range(nb)))
        else:
            raise ValueError("Input vector type must be either QuantumCircuit or numpy ndarray.")

        # If state preparation is probabilistic the number of qubit flags should increase
        nf = 1

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
            matrix_array = matrix.matrix()
        elif isinstance(matrix, np.ndarray):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2 ** vector_circuit.num_qubits:
                raise ValueError("Input vector dimension does not match input "
                                 "matrix dimension!")
            matrix_array = matrix
            # TODO add message not yet supported - check matrixOp & then call evolution
        else:
            raise ValueError("Input matrix type must be either QuantumCircuit or numpy ndarray.")

        lambda_max = max(np.abs(np.linalg.eigvals(matrix_array)))
        lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))
        kappa = np.linalg.cond(matrix_array)
        # Update the number of qubits required to represent the eigenvalues
        nl = max(nb + 1, int(np.log2(lambda_max / lambda_min)) + 1)

        # Constant from the representation of eigenvalues
        delta = self.get_delta(nl, lambda_min, lambda_max)

        # Update evolution time
        evo_time = 2 * np.pi * delta / lambda_min
        matrix_circuit.tolerance = self._epsilon_a
        matrix_circuit.time = evo_time

        if self._exact_inverse:
            inverse_circuit = ExactInverse(nl, delta)
            # Update number of ancilla qubits
            na = matrix_circuit.num_ancillas

        else:
            # Calculate breakpoints for the inverse approximation
            num_values = 2 ** nl
            constant = delta
            a = int(round(num_values ** (2 / 3)))  # pylint: disable=invalid-name

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(nb, int(np.log(1 + (16.23 * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2) *
                                             kappa * (2 * kappa - self._epsilon_r)) /
                                        self._epsilon_r)))
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            breakpoints = []
            for i in range(0, num_intervals):
                # Add the breakpoint to the list
                breakpoints.append(a * (5 ** i))

                # Define the right breakpoint of the interval
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)

            inverse_circuit = PiecewiseChebyshev(lambda x: np.arcsin(constant / x), degree,
                                                 breakpoints, nl)
            na = max(matrix_circuit.num_ancillas, inverse_circuit.num_ancillas)

        # Initialise the quantum registers
        qb = QuantumRegister(nb)  # right hand side and solution
        ql = QuantumRegister(nl)  # eigenvalue evaluation qubits
        if na > 0:
            qa = AncillaRegister(na)  # ancilla qubits
        qf = QuantumRegister(nf)  # flag qubits

        if na > 0:
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        qc.append(vector_circuit, qb[:])
        # QPE
        phase_estimation = PhaseEstimation(nl, matrix_circuit)
        if na > 0:
            qc.append(phase_estimation, ql[:] + qb[:] + qa[:matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        # Conditioned rotation
        if self._exact_inverse:
            qc.append(inverse_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(inverse_circuit.to_instruction(), ql[:] + [qf[0]] +
                      qa[:inverse_circuit.num_ancillas])
        # QPE inverse
        if na > 0:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:] +
                      qa[:matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        return qc

    def solve(self, matrix: Union[np.ndarray, QuantumCircuit],
              vector: Union[np.ndarray, QuantumCircuit],
              observable: Optional[Union[LinearSystemObservable, BaseOperator,
                                         List[BaseOperator]]] = None,
              post_rotation: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
              post_processing: Optional[Callable[[Union[float, List[float]]],
                                                 Union[float, List[float]]]] = None) \
            -> LinearSolverResult:
        """Tries to solve the given linear system.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Information to be extracted from the solution.
                Default is `EuclideanNorm`
            post_rotation: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.

        Returns:
            The result of the linear system.

        Raises:
            ValueError: If the input is not in the correct format.
        """
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
        elif isinstance(vector, np.ndarray):
            nb = int(np.log2(len(vector)))
        else:
            raise ValueError("Input vector type must be either QuantumCircuit or numpy ndarray.")

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            matrix_array = matrix.matrix()
        elif isinstance(matrix, np.ndarray):
            matrix_array = matrix
        else:
            raise ValueError("Input matrix type must be either QuantumCircuit or numpy ndarray.")

        lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))

        solution = LinearSolverResult()
        solution.state = self.construct_circuit(matrix, vector)
        solution.euclidean_norm = self.calculate_norm(solution.state, lambda_min)
        # The post-rotating gates have already been applied
        solution.observable, solution.circuit_results =\
            self.calculate_observable(solution.state, observable, post_rotation, post_processing,
                                      lambda_min)
        return solution
