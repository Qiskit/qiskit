# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An implementation of the AdaptVQE algorithm."""
from __future__ import annotations
from typing import Optional, List, Tuple, Union

import copy
import re
import logging

from enum import Enum
import numpy as np
from qiskit import QiskitError

from qiskit.algorithms import VQE
from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQEResult
from qiskit.algorithms import VariationalAlgorithm
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, PauliSumOp, ExpectationFactory
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.utils.validation import validate_min
from qiskit.algorithms.aux_ops_evaluator import eval_observables
from ..exceptions import AlgorithmError


logger = logging.getLogger(__name__)


class TerminationCriterion(Enum):
    """A class enumerating the various finishing criteria."""

    CONVERGED = "Threshold converged"
    CYCLICITY = "Aborted due to a cyclic selection of evolution operators"
    MAXIMUM = "Maximum number of iterations reached"


class AdaptVQE(VariationalAlgorithm):
    """The Adaptive Variational Quantum Eigensolver algorithm.

    `AdaptVQE <https://arxiv.org/abs/1812.11173>`__ is a quantum algorithm which creates a compact
    ansatz from a set of evolution operators. It iteratively extends the ansatz circuit, by
    selecting the building block that leads to the largest gradient from a set of candidates. In
    chemistry, this is usually a list of orbital excitations. This results in a wavefunction ansatz
    which is uniquely adapted to the operator whose minimum eigenvalue is being determined.
    This class relies on a supplied instance of :class:`~.VQE` to find the minimum eigenvalue.
    The performance of AdaptVQE significantly depends on the minimization routine.
    """

    def __init__(
        self,
        solver: VQE,
        excitation_pool: List[OperatorBase] = None,
        threshold: float = 1e-5,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Args:
            solver: a :class:`~.VQE` instance used internally to compute the minimum eigenvalues.
                It is a requirement that the `ansatz` of this solver is of type `EvolvedOperatorAnsatz`.
            excitation_pool: A list of quantum circuits out of which to build the ansatz.
            threshold: the eigenvalue convergence threshold. It has a minimum value of `1e-15`.
            max_iterations: the maximum number of iterations.
        """
        validate_min("threshold", threshold, 1e-15)

        self._threshold = threshold
        self._solver = solver
        self._tmp_ansatz = self._solver.ansatz
        self._max_iterations = max_iterations
        self._excitation_pool = excitation_pool
        self._excitation_list: List[OperatorBase] = []

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns the initial point."""
        return self._solver.initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Sets the initial point."""
        self._solver.initial_point = initial_point

    def _compute_gradients(
        self,
        theta: List[float],
        operator: OperatorBase,
        expectation: ExpectationBase,
    ) -> List[Tuple[complex, complex]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: List of (up to now) optimal parameters.
            operator: operator whose gradient needs to be computed.
            expectation: Expectation Base
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        commutators = []
        for exc in self._excitation_pool:
            # The excitations operators are applied later as exp(i*theta*excitation).
            # For this commutator, we need to explicitly pull in the imaginary phase.
            commutator = 1j * ((operator @ exc) - (exc @ operator)).reduce()
            commutators.append(commutator)

        wave_function = self._solver.ansatz.assign_parameters(theta)

        res = eval_observables(
            self._solver.quantum_instance, wave_function, commutators, expectation
        )
        return res

    @staticmethod
    def _check_cyclicity(indices: List[int]) -> bool:
        """
        Auxiliary function to check for cycles in the indices of the selected excitations.

        Args:
            indices: The list of chosen gradient indices.

        Returns:
            Whether repeating sequences of indices have been detected.
        """
        cycle_regex = re.compile(r"(\b.+ .+\b)( \b\1\b)+")
        # reg-ex explanation:
        # 1. (\b.+ .+\b) will match at least two numbers and try to match as many as possible. The
        #    word boundaries in the beginning and end ensure that now numbers are split into digits.
        # 2. the match of this part is placed into capture group 1
        # 3. ( \b\1\b)+ will match a space followed by the contents of capture group 1 (again
        #    delimited by word boundaries to avoid separation into digits).
        # -> this results in any sequence of at least two numbers being detected
        match = cycle_regex.search(" ".join(map(str, indices)))
        logger.debug("Cycle detected: %s", match)
        logger.info("Alternating sequence found. Finishing.")
        # Additionally we also need to check whether the last two numbers are identical, because the
        # reg-ex above will only find cycles of at least two consecutive numbers.
        # It is sufficient to assert that the last two numbers are different due to the iterative
        # nature of the algorithm.
        return match is not None or (len(indices) > 1 and indices[-2] == indices[-1])

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        aux_operators: Optional[ListOrDict[OperatorBase]] = None,
    ) -> AdaptVQEResult:
        """Computes the minimum eigenvalue.

        Args:
            operator: Operator whose minimum eigenvalue we want to find.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            QiskitError: If a solver other than :class:`~VQE` or an ansatz other than
                :class:`~.EvolvedOperatorAnsatz` is provided or if the algorithm finishes due
                to an unforeseen reason.
            AlgorithmError: If `quantum_instance` is not provided.
            QiskitError: If the chosen gradient method appears to result in all-zero gradients.

        Returns:
            An :class:`~.AdaptVQEResult` which is a :class:`~.VQEResult` but also but also
            includes runtime information about the AdaptVQE algorithm like the number of iterations,
            termination criterion, and the final maximum gradient.
        """
        if not isinstance(self._tmp_ansatz, EvolvedOperatorAnsatz):
            raise QiskitError("The AdaptVQE ansatz must be of the EvolvedOperatorAnsatz type.")

        if self._solver.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )

        # construct the expectation
        if self._solver.expectation is None:
            expectation = ExpectationFactory.build(
                operator=operator,
                backend=self._solver.quantum_instance,
                include_custom=self._solver.include_custom,
            )
        else:
            expectation = self._solver.expectation

        # Overwrite the solver's ansatz with the initial state
        solver_ansatz = copy.deepcopy(self._solver.ansatz)
        self._solver.ansatz = self._tmp_ansatz.initial_state

        prev_op_indices: List[int] = []
        theta: List[float] = []
        max_grad: Tuple[float, Optional[PauliSumOp]] = (0.0, None)
        self._excitation_list = []
        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))
            # compute gradients
            cur_grads = self._compute_gradients(theta, operator, expectation)
            # pick maximum gradient
            max_grad_index, max_grad = max(
                enumerate(cur_grads), key=lambda item: np.abs(item[1][0])
            )
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # log gradients
            if np.abs(max_grad[0]) < self._threshold:
                if iteration == 1:
                    raise QiskitError(
                        "Gradient choice is not suited as it leads to all zero gradients gradients. "
                        "Try a different gradient method."
                    )
                logger.info(
                    "AdaptVQE terminated successfully with a final maximum gradient: %s",
                    str(np.abs(max_grad[0])),
                )
                termination_criterion = TerminationCriterion.CONVERGED
                break
            # check indices of picked gradients for cycles
            if self._check_cyclicity(prev_op_indices):
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                termination_criterion = TerminationCriterion.CYCLICITY
                break
            # add new excitation to self._ansatz
            self._excitation_list.append(self._excitation_pool[max_grad_index])
            theta.append(0.0)
            # run VQE on current Ansatz
            self._tmp_ansatz.operators = self._excitation_list
            self._solver.ansatz = self._tmp_ansatz
            self._solver.initial_point = theta
            raw_vqe_result = self._solver.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
        else:
            # reached maximum number of iterations
            termination_criterion = TerminationCriterion.MAXIMUM
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        result = AdaptVQEResult()
        result.combine(raw_vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.termination_criterion = termination_criterion

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            bound_ansatz = self._solver.ansatz.bind_parameters(result.optimal_point)

            aux_values = eval_observables(
                self._solver.quantum_instance, bound_ansatz, aux_operators, expectation
            )
            result.aux_operator_eigenvalues = aux_values
        else:
            aux_values = None
        raw_vqe_result.aux_operator_eigenvalues = aux_values

        logger.info("The final energy is: %s", str(result.eigenvalue))
        self._solver.ansatz = solver_ansatz
        return result


class AdaptVQEResult(VQEResult):
    """AdaptVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._num_iterations: int = None
        self._final_max_gradient: float = None
        self._termination_criterion: str = ""

    @property
    def num_iterations(self) -> int:
        """Returns number of iterations"""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """Sets number of iterations"""
        self._num_iterations = value

    @property
    def final_max_gradient(self) -> float:
        """Returns final maximum gradient"""
        return self._final_max_gradient

    @final_max_gradient.setter
    def final_max_gradient(self, value: float) -> None:
        """Sets final maximum gradient"""
        self._final_max_gradient = value

    @property
    def termination_criterion(self) -> str:
        """Returns termination criterion"""
        return self._termination_criterion

    @termination_criterion.setter
    def termination_criterion(self, value: str) -> None:
        """Sets termination criterion"""
        self._termination_criterion = value
