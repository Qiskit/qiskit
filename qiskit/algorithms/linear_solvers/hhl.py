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

"""The HHL algorithm."""

from typing import Optional, Union, List, Callable, Tuple
import numpy as np
from qiskit.aqua.operators import PauliExpectation

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.opflow import Z, I, StateFn, TensoredOp, ExpectationBase, CircuitSampler, ListOp
from qiskit.providers import Backend, BaseBackend
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import QuantumInstance

from .linear_solver import LinearSolver, LinearSolverResult
from .matrices.linear_system_matrix import LinearSystemMatrix
from .observables.linear_system_observable import LinearSystemObservable


class HHL(LinearSolver):
    """The HHL algorithm to solve systems of linear equations"""

    def __init__(self,
                 epsilon: float = 1e-2,
                 exp_val: Optional[ExpectationBase] = None,
                 quantum_instance: Optional[Union[Backend, QuantumInstance]] = None) -> None:
        r"""
        Args:
            epsilon: Error tolerance of the approximation to the solution, i.e. if x is the exact
                solution and ::math::`\tilde{x}` the one calculated by the algorithm, then
                :math:`||x - \tilde{x}|| < epsilon`.
            exp_val: The expectation converter applied to the expectation values before
                evaluation. If None then PauliExpectation is used.
            quantum_instance: Quantum Instance or Backend. If None, a Statevector calculation is
                done.


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

        if quantum_instance is not None:
            self._sampler = CircuitSampler(quantum_instance)
        else:
            self._sampler = None

        self._exp_val = exp_val if exp_val else PauliExpectation()
        self._quantum_instance = quantum_instance

        # For now the default reciprocal implementation is exact
        self._exact_reciprocal = True

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance,
                                                       BaseBackend, Backend]) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance
        if quantum_instance is not None:
            self._sampler = CircuitSampler(quantum_instance)
        else:
            self._sampler = None

    def _get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float:
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

    def _calculate_norm(self, qc: QuantumCircuit, scaling: Optional[float] = 1) -> float:
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
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2

        # Norm observable
        observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ (I ^ nb)
        norm_2 = (~StateFn(observable) @ StateFn(qc)).eval()

        return np.real(np.sqrt(norm_2) / scaling)

    def _calculate_observable(self, solution: QuantumCircuit,
                              observable: Optional[Union[LinearSystemObservable,
                                                         BaseOperator]] = None,
                              observable_circuit: Optional[QuantumCircuit] = None,
                              post_processing: Optional[Callable[[Union[float, List[float]]],
                                                                 Union[float, List[float]]]] = None,
                              scaling: float = 1) -> Tuple[Union[float, List[float]],
                                                           Union[float, List[float]]]:
        """Calculates the value of the observable(s) given.

        Args:
            solution: The quantum circuit preparing the solution x to the system.
            observable: Information to be extracted from the solution.
            observable_circuit: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.
            scaling: Factor scaling the solution vector.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a
             tuple.
        """
        # Get the number of qubits
        nb = solution.qregs[0].size
        nl = solution.qregs[1].size
        na = solution.num_ancillas

        # if the observable is given construct post_processing and observable_circuit
        if observable is not None:
            observable_circuit = observable.observable_circuit(nb)
            post_processing = observable.post_processing

            if isinstance(observable, LinearSystemObservable):
                observable = observable.observable(nb)

        # in the other case use the identity as observable
        else:
            observable = I ^ nb

        # Create the Operators Zero and One
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2

        is_list = True
        if not isinstance(observable_circuit, list):
            is_list = False
            observable_circuit = [observable_circuit]
            observable = [observable]

        expectations = []
        for circ, obs in zip(observable_circuit, observable):
            circuit = QuantumCircuit(solution.num_qubits)
            circuit.append(solution, circuit.qubits)
            circuit.append(circ, range(nb))

            ob = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ obs
            expectations.append(~StateFn(ob) @ StateFn(circuit))

        if is_list:
            # execute all in a list op to send circuits in batches
            expectations = ListOp(expectations)
        else:
            expectations = expectations[0]

        if self._exp_val is not None:
            expectations = self._exp_val.convert(expectations)

        if self._sampler is not None:
            expectations = self._sampler.convert(expectations)

        # evaluate
        expectation_results = expectations.eval()

        # apply post_processing
        result = post_processing(expectation_results, nb, scaling)

        return result, expectation_results

    def construct_circuit(self, matrix: Union[np.ndarray, QuantumCircuit],
                          vector: Union[np.ndarray, QuantumCircuit]) -> QuantumCircuit:
        """Construct the HHL circuit.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit

        Raises:
            ValueError: If the input is not in the correct format.
        """
        # State preparation circuit - default is qiskit
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, np.ndarray):
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb)
            vector_circuit.isometry(vector / np.linalg.norm(vector),
                                    list(range(nb)), None)

        # If state preparation is probabilistic the number of qubit flags should increase
        nf = 1

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
        elif isinstance(matrix, np.ndarray):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2 ** vector_circuit.num_qubits:
                raise ValueError("Input vector dimension does not match input "
                                 "matrix dimension! Vector dimension: " +
                                 str(vector_circuit.num_qubits) +
                                 ". Matrix dimension: " +
                                 str(matrix.shape[0]))

        # Set the tolerance for the matrix approximation
        matrix_circuit.tolerance = self._epsilon_a

        # check if the matrix can calculate the condition number and store the upper bound
        if hasattr(matrix, "condition_bounds") and matrix.condition_bounds is not None:
            kappa = matrix.condition_bounds[1]
        else:
            kappa = 1
        # Update the number of qubits required to represent the eigenvalues
        nl = max(nb + 1, int(np.log2(kappa)) + 1)

        # check if the matrix can calculate bounds for the eigenvalues
        if hasattr(matrix, "eigs_bounds") and matrix.eigs_bounds is not None:
            lambda_min, lambda_max = matrix.eigs_bounds
            # Constant so that the minimum eigenvalue is represented exactly, since it contributes
            # the most to the solution of the system
            delta = self._get_delta(nl, lambda_min, lambda_max)
            # Update evolution time
            matrix_circuit.evolution_time = 2 * np.pi * delta / lambda_min
        else:
            delta = 1

        if self._exact_reciprocal:
            reciprocal_circuit = ExactReciprocal(nl, delta)
            # Update number of ancilla qubits
            na = matrix_circuit.num_ancillas
        else:
            # Calculate breakpoints for the reciprocal approximation
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

            reciprocal_circuit = PiecewiseChebyshev(lambda x: np.arcsin(constant / x), degree,
                                                    breakpoints, nl)
            na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)

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
        if self._exact_reciprocal:
            qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(reciprocal_circuit.to_instruction(), ql[:] + [qf[0]] +
                      qa[:reciprocal_circuit.num_ancillas])
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
                                         List[LinearSystemObservable], List[BaseOperator]]] = None,
              observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
              post_processing: Optional[Callable[[Union[float, List[float]]],
                                                 Union[float, List[float]]]] = None) \
            -> LinearSolverResult:
        """Tries to solve the given linear system.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is `None`.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Raises:
            ValueError: If an invalid combination of observable, observable_circuit and
                post_processing is passed.

        Returns:
            The result of the linear system.
        """
        # verify input
        if observable is not None:
            if observable_circuit is not None or post_processing is not None:
                raise ValueError('If observable is passed, observable_circuit and post_processing '
                                 'cannot be set.')

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, LinearSystemMatrix):
            matrix_array = matrix.matrix
        elif isinstance(matrix, QuantumCircuit):
            matrix_array = Operator(matrix).data
        elif isinstance(matrix, np.ndarray):
            matrix_array = matrix

        lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))

        solution = LinearSolverResult()
        solution.state = self.construct_circuit(matrix, vector)
        solution.euclidean_norm = self._calculate_norm(solution.state, lambda_min)

        if observable is not None or observable_circuit is not None:
            solution.observable, solution.circuit_results = \
                self._calculate_observable(solution.state, observable, observable_circuit,
                                           post_processing, lambda_min)

        return solution
