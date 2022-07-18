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
from qiskit.opflow import OperatorBase, PauliSumOp, CircuitSampler
from qiskit.opflow.gradients import GradientBase, Gradient
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
    ansatz from a set of evolution operators. It iteratively appends excitations with the largest energy
    gradient to the circuit. This results in a wavefunction ansatz which is uniquely adapted to the
    operator whose minimum eigenvalue is being determined.
    This class relies on a supplied instance of `VQE` to find the minimum eigenvalue.
    The performance of AdaptVQE can significantly depend on the choice of gradient method, QFI
    solver (if applicable) and the epsilon value.
    """

    def __init__(
        self,
        solver: VQE,
        adapt_gradient: Optional[GradientBase] = None,
        excitation_pool: List[Union[OperatorBase, QuantumCircuit]] = None,
        threshold: float = 1e-5,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Args:
            solver: a `VQE` instance used internally to compute the minimum eigenvalues.
                It is a requirement that the `ansatz` of this solver is of type `EvolvedOperatorAnsatz`.
            adapt_gradient: a `Gradient` instance with which to compute the excitation operator
                gradients.
                If `None`, this will default to a parameter shift gradient.
            excitation_pool: A list of quantum circuits out of which to build the ansatz.
            threshold: the eigenvalue convergence threshold. It has a minimum value of `1e-15`.
            max_iterations: the maximum number of iterations.
        """
        validate_min("threshold", threshold, 1e-15)

        if adapt_gradient is None:
            adapt_gradient = Gradient(grad_method="param_shift")
        self._threshold = threshold
        self._solver = solver
        self._tmp_ansatz = self._solver.ansatz
        self._max_iterations = max_iterations
        self._adapt_gradient = adapt_gradient
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
    ) -> List[Tuple[float, PauliSumOp]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: List of (up to now) optimal parameters.
            operator: operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        sampler = CircuitSampler(self._solver.quantum_instance)
        for idx, exc in enumerate(self._excitation_pool):
            # add next excitation to ansatz
            self._tmp_ansatz.operators = self._excitation_list + [exc]
            # the ansatz needs to be decomposed for the gradient to work
            self._solver.ansatz = self._tmp_ansatz.decompose()
            param_sets = list(self._solver.ansatz.parameters)
            # zip will only iterate the length of the shorter list
            parameter = dict(zip(self._solver.ansatz.parameters, theta))
            op, expectation = self._solver.construct_expectation(
                parameter, operator, return_expectation=True
            )
            # compute gradient
            state_grad = self._adapt_gradient.convert(operator=op, params=param_sets)
            # Assign the parameters and evaluate the gradient
            state_grad_result = sampler.convert(
                state_grad, params={param_sets[-1]: self.initial_point[idx]}
            ).eval()
            logger.info("Gradient computed : %s", str(state_grad_result))
            res.append((np.abs(state_grad_result[-1]), exc))
        return res, expectation

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

    @staticmethod
    def _log_gradients(
        iteration: int,
        cur_grads: Tuple[float, PauliSumOp],
        max_grad: Tuple[float, PauliSumOp | None],
    ):
        if logger.isEnabledFor(logging.INFO):
            gradlog = []
            gradlog.append(
                f"\nGradients in iteration #{str(iteration)}\n"
                "ID: Excitation Operator: Gradient  <(*) maximum>"
            )
            for i, grad in enumerate(cur_grads):
                gradlog.append(f"\n{str(i)}: {str(grad[1])}: {str(grad[0])}")
                if grad[1] == max_grad[1]:
                    gradlog.append("\t(*)")
            logger.info(",".join(gradlog))

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        aux_operators: Optional[ListOrDict[PauliSumOp]] = None,
    ) -> AdaptVQEResult:
        """Computes the minimum eigenvalue.

        Args:
            operator: Operator to evaluate.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            QiskitError: If a solver other than VQE
                (`qiskit.algorithms.minimum_eigen_solvers.vqe.py`) or a ansatz other than
                EvolvedOperatorAnsatz (`qiskit.circuit.library.evolved_operator_ansatz.py`)
                is provided or if the algorithm finishes due to an unforeseen reason.
            AlgorithmError: If `quantum_instance` is not provided.
            QiskitError: If the user-provided `aux_operators` contain a name which clashes
                with an internally constructed auxiliary operator. Note: the names used for the
                internal auxiliary operators correspond to the `Property.name` attributes which
                generated the respective operators.
            QiskitError: If the chosen gradient method appears to result in all-zero gradients.

        Returns:
            An AdaptVQEResult (`qiskit.algorithms.minimum_eigen_solvers.adapt_vqe.AdaptVQEResult`)
            which is a VQEResult (`qiskit.algorithms.minimum_eigen_solvers.vqe.VQEResult`) but also
            includes runtime information about the AdaptVQE algorithm like the number of iterations,
            finishing criterion, and the final maximum gradient.
        """
        if not isinstance(self._tmp_ansatz, EvolvedOperatorAnsatz):
            raise QiskitError("The AdaptVQE ansatz must be of the EvolvedOperatorAnsatz type.")
        if self._solver.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )
        # We construct the ansatz once to be able to extract the full set of excitation operators.
        self._tmp_ansatz._build()

        prev_op_indices: List[int] = []
        theta: List[float] = []
        max_grad: Tuple[float, Optional[PauliSumOp]] = (0.0, None)
        self._excitation_list = []
        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))
            # compute gradients
            cur_grads, expectation = self._compute_gradients(theta, operator)
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
            self._excitation_list.append(max_grad[1])
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

        if termination_criterion is False:
            raise QiskitError("The algorithm finished due to an unforeseen reason!")

        result = AdaptVQEResult()
        result.combine(raw_vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.termination_criterion = termination_criterion

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            bound_ansatz = self._tmp_ansatz.bind_parameters(result.optimal_point)

            aux_values = eval_observables(
                self._solver.quantum_instance, bound_ansatz, aux_operators, expectation=expectation
            )
            result.aux_operator_eigenvalues = aux_values
        else:
            aux_values = None
        raw_vqe_result.aux_operator_eigenvalues = aux_values

        logger.info("The final energy is: %s", str(result.eigenvalue))
        return result


class AdaptVQEResult(VQEResult):
    """AdaptVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._num_iterations: int = 0
        self._final_max_gradient: float = 0.0
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
        """Returns finishing criterion"""
        return self._termination_criterion

    @termination_criterion.setter
    def termination_criterion(self, value: str) -> None:
        """Sets finishing criterion"""
        self._termination_criterion = value
