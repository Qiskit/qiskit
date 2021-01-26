# -*- coding: utf-8 -*-

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

        # Number of qubits for the different registers
        self._nb = None  # number of qubits for the solution register
        self._nl = None  # number of qubits for the eigenvalue register
        self._na = None  # number of ancilla qubits
        self._nf = None  # number of flag qubits
        # Time of the evolution. Once the matrix is specified,
        # it can be updated 2 * np.pi / lambda_max
        self._evo_time = 2 * np.pi
        self._lambda_max = None
        self._lambda_min = None
        self._kappa = 1  # Set default condition number to 1

        # Circuits for the different blocks of the algorithm
        self._vector_circuit = None
        self._matrix_circuit = None
        self._inverse_circuit = None
        self._post_rotation = None

        # For now the default inverse implementation is exact
        self._exact_inverse = True

    @property
    def _num_qubits(self) -> int:
        """Return the number of qubits needed in the circuit.

        Returns:
            The total number of qubits.
        """
        num_qubits = self._na + self._nl + self._nl + self._nf

        return num_qubits

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

    def calculate_norm(self, qc: QuantumCircuit) -> float:
        """Calculates the value of the euclidean norm of the solution.

        Args:
            qc: The quantum circuit preparing the solution x to the system.

        Returns:
            The value of the euclidean norm of the solution.
        """
        # Create the Operators Zero and One
        zero_op = ((I + Z) / 2)
        one_op = ((I - Z) / 2)

        # Norm observable
        observable = one_op ^ TensoredOp((self._nl + self._na) * [zero_op]) ^ (I ^ self._nb)
        norm_2 = (~StateFn(observable) @ qc).eval()

        return np.real(np.sqrt(norm_2) / self._lambda_min)

    def calculate_observable(self, qc: QuantumCircuit,
                             observable: Optional[Union[BaseOperator, List[BaseOperator]]] = None,
                             post_processing: Optional[Callable[[Union[float, List[float]]],
                                                                Union[float, List[float]]]]
                             = None) -> Tuple[Union[float, List[float]], Union[float,
                                                                               List[float]]]:
        """Calculates the value of the observable(s) given.

        Args:
            qc: The quantum circuit preparing the solution x to the system.
            observable: The (list of) observable(s).
            post_processing: Function(s) to compute the value of the observable(s).

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a
             tuple.
        """
        # Create the Operators Zero and One
        zero_op = ((I + Z) / 2)
        one_op = ((I - Z) / 2)
        # List of quantum circuits with post_rotation gates appended
        qcs = []
        # Observable gates
        if self._post_rotation is not None and isinstance(self._post_rotation, list):
            for circ in self._post_rotation:
                qc_temp = QuantumCircuit(qc.num_qubits)
                qc_temp.append(qc, list(range(qc.num_qubits)))
                qc_temp.append(circ, list(range(self._nb)))
                qcs.append(qc_temp)
        elif self._post_rotation:
            qc.append(self._post_rotation, list(range(self._nb)))

        # Update observable to include ancilla and rotation qubit
        result = []
        if isinstance(observable, list):
            for i, obs in enumerate(observable):
                if isinstance(obs, list):
                    result_temp = []
                    for ob in obs:
                        new_observable = one_op ^ TensoredOp((self._nl + self._na) * [zero_op]) ^ ob
                        if qcs:
                            result_temp.append((~StateFn(new_observable) @ qcs[i]).eval())
                        else:
                            result_temp.append((~StateFn(new_observable) @ qc).eval())
                    result.append(result_temp)
                else:
                    if obs is None:
                        obs = I ^ self._nb
                    new_observable = one_op ^ TensoredOp((self._nl + self._na) * [zero_op]) ^ obs
                    if qcs:
                        result.append((~StateFn(new_observable) @ qcs[i]).eval())
                    else:
                        result.append((~StateFn(new_observable) @ qc).eval())
        else:
            if observable is None:
                observable = I ^ self._nb

            new_observable = one_op ^ TensoredOp((self._nl + self._na) * [zero_op]) ^ observable
            if qcs:
                for circ in qcs:
                    result.append((~StateFn(new_observable) @ circ).eval())
            else:
                result = (~StateFn(new_observable) @ qc).eval()

        if isinstance(result, list):
            circuit_results = result
        else:
            circuit_results = [result]
        if post_processing is not None:

            return post_processing(result, self._nb, self._lambda_min), circuit_results
        else:
            return result, circuit_results

    def construct_circuit(self) -> QuantumCircuit:
        """Construct the HHL circuit.

            Returns:
                QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        # Initialise the quantum registers
        qb = QuantumRegister(self._nb)  # right hand side and solution
        ql = QuantumRegister(self._nl)  # eigenvalue evaluation qubits
        if self._na > 0:
            qa = AncillaRegister(self._na)  # ancilla qubits
        qf = QuantumRegister(self._nf)  # flag qubits

        if self._na > 0:
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        qc.append(self._vector_circuit, qb[:])
        # QPE
        phase_estimation = PhaseEstimation(self._nl, self._matrix_circuit)
        if self._na > 0:
            qc.append(phase_estimation, ql[:] + qb[:] + qa[:self._matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        # Conditioned rotation
        if self._exact_inverse:
            qc.append(self._inverse_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(self._inverse_circuit.to_instruction(), ql[:] + [qf[0]] +
                      qa[:self._inverse_circuit.num_ancillas])
        # QPE inverse
        if self._na > 0:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:] +
                      qa[:self._matrix_circuit.num_ancillas])
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
        # State preparation circuit - default is qiskit
        if isinstance(vector, QuantumCircuit):
            self._nb = vector.num_qubits
            self._vector_circuit = vector
        elif isinstance(vector, np.ndarray):
            self._nb = int(np.log2(len(vector)))
            self._vector_circuit = QuantumCircuit(self._nb).initialize(vector /
                                                                       np.linalg.norm(vector),
                                                                       list(range(self._nb)))
        else:
            raise ValueError("Input vector type must be either QuantumCircuit or numpy ndarray.")

        # If state preparation is probabilistic the number of qubit flags should increase
        self._nf = 1

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            self._matrix_circuit = matrix
            matrix_array = matrix.matrix()
        elif isinstance(matrix, np.ndarray):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2 ** self._vector_circuit.num_qubits:
                raise ValueError("Input vector dimension does not match input "
                                 "matrix dimension!")
            matrix_array = matrix
        else:
            raise ValueError("Input matrix type must be either QuantumCircuit or numpy ndarray.")

        self._lambda_max = max(np.abs(np.linalg.eigvals(matrix_array)))
        self._lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))
        self._kappa = np.linalg.cond(matrix_array)
        # Update the number of qubits required to represent the eigenvalues
        self._nl = max(self._nb + 1, int(np.log2(self._lambda_max / self._lambda_min)) + 1)

        # Constant from the representation of eigenvalues
        delta = self.get_delta(self._nl, self._lambda_min, self._lambda_max)

        # Update evolution time
        self._evo_time = 2 * np.pi * delta / self._lambda_min
        self._matrix_circuit.set_simulation_params(self._evo_time, self._epsilon_a)

        if self._exact_inverse:
            self._inverse_circuit = ExactInverse(self._nl, delta)
            # Update number of ancilla qubits
            self._na = self._matrix_circuit.num_ancillas

        else:
            # Calculate breakpoints for the inverse approximation
            num_values = 2 ** self._nl
            constant = delta
            a = int(round(num_values ** (2 / 3)))  # pylint: disable=invalid-name

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(self._nb, int(np.log(1 + (16.23 * np.sqrt(np.log(r) ** 2 + (np.pi / 2)
                                                                   ** 2) * self._kappa *
                                                   (2 * self._kappa - self._epsilon_r)) /
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
                                                 breakpoints, self._nl)
            self._inverse_circuit = inverse_circuit.to_instruction()
            self._inverse_circuit._build()
            self._na = max(self._matrix_circuit.num_ancillas, inverse_circuit.num_ancillas)

        if observable is not None and isinstance(observable, LinearSystemObservable):
            self._post_rotation = observable.post_rotation(self._nb)
            post_processing = observable.post_processing
            observable = observable.observable(self._nb)
        else:
            self._post_rotation = post_rotation

        solution = LinearSolverResult()
        solution.state = self.construct_circuit()
        solution.euclidean_norm = self.calculate_norm(solution.state)
        # The post-rotating gates have already been applied
        (solution.observable, solution.circuit_results) =\
            self.calculate_observable(solution.state, observable, post_processing)
        return solution
