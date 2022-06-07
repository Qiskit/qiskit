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

"""A ground state calculation employing the AdaptVQE algorithm."""
from __future__ import annotations
from typing import Optional, List, Tuple, Union, Callable

import copy
import re
import logging

import numpy as np
from qiskit import QiskitError

from qiskit.algorithms import VQE
from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQEResult
from qiskit.algorithms import VariationalAlgorithm, VariationalResult
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, PauliSumOp, ExpectationBase, CircuitSampler,StateFn
from qiskit.opflow.gradients import GradientBase, Gradient
from qiskit.algorithms.minimum_eigen_solvers import MinimumEigensolver
from qiskit.circuit.library import EvolvedOperatorAnsatz, RealAmplitudes, PauliEvolutionGate
from qiskit.circuit.library import NLocal
from qiskit.utils.validation import validate_min


from qiskit.algorithms.optimizers import Optimizer
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend
from qiskit.algorithms.aux_ops_evaluator import eval_observables
#from ..aux_ops_evaluator import eval_observables
from enum import Enum

# from .minimum_eigensolver_factories import MinimumEigensolverFactory
# from .ground_state_eigensolver import GroundStateEigensolver
logger = logging.getLogger(__name__)


class Finishing_criterion(Enum):
    finishing_criterion = ""


class AdaptVQE(VariationalAlgorithm):
    """A ground state calculation employing the AdaptVQE algorithm.

    The performance of AdaptVQE can significantly depend on the choice of gradient method, QFI
    solver (if applicable) and the epsilon value.
    """

    def __init__(
        self,
        solver: VQE ,
        ansatz: Optional[QuantumCircuit] = None,
        threshold: float = 1e-5,
        max_iterations: Optional[int] = None,
        adapt_gradient: Optional[GradientBase] = None,
        #initial_point: Optional[np.ndarray] = None,
        expectation: Optional[ExpectationBase] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        excitation_pool: List[Union[OperatorBase, QuantumCircuit]] = None,
    ) -> None:
        """
        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            solver: a factory for the VQE solver employing a UCCSD ansatz.
            threshold: the energy convergence threshold. It has a minimum value of 1e-15.
            max_iterations: the maximum number of iterations of the AdaptVQE algorithm.
            gradient: a class that converts operator expression to the first-order gradient based
                on the method mentioned.
            initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            excitation_pool: An entire list of excitations.
        """
        validate_min("threshold", threshold, 1e-15)

        if adapt_gradient is None:
            adapt_gradient = Gradient(grad_method="param_shift")
        self._threshold = threshold
        self.solver = solver
        self._max_iterations = max_iterations
        self._adapt_gradient = adapt_gradient
        self._excitation_pool = excitation_pool
        self._tmp_ansatz = ansatz
        self.expectation = expectation
        self.quantum_instance = quantum_instance
        #self._initial_point = initial_point
        self._excitation_list: List[OperatorBase] = []

    def _compute_gradients(
        self,
        theta: List[float],
        operator: OperatorBase,
    ) -> List[Tuple[float, PauliSumOp]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: List of (up to now) optimal parameters
            vqe: The variational quantum eigensolver instance used for solving

        Returns:
            List of pairs consisting of gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        sampler = CircuitSampler(self.quantum_instance)
        for exc in self._excitation_pool:
            # add next excitation to ansatz
            self._tmp_ansatz.operators = self._excitation_list + [exc]
            # the ansatz needs to be decomposed for the gradient to work
            self.ansatz = self._tmp_ansatz.decompose()
            param_sets = list(self.ansatz.parameters)
            # zip will only iterate the length of the shorter list
            theta1 = dict(zip(self.ansatz.parameters, theta))
            """gradient1 = self._adapt_gradient.gradient_wrapper(
                ~StateFn(operator) @ StateFn(self.ansatz),
                bind_params=list(self.ansatz.parameters),
                backend=self.quantum_instance)"""
            op,expectation = self.solver.construct_expectation(theta1, operator, return_expectation=True)
            # compute gradient
            #print(op)
            state_grad = self._adapt_gradient.convert(operator=op, params=param_sets)
            # Assign the parameters and evaluate the gradient
            value_dict = {param_sets[-1]: 0.0}
            """for value in enumerate(value_dict):
                result = gradient1(value)
            print(result[-1])
            res.append((np.abs(result[-1]), exc))"""
            #print(state_grad)
            state_grad_result = sampler.convert(state_grad, params=value_dict).eval()
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
        match = cycle_regex.search(" ".join(map(str, indices)))
        logger.debug("Cycle detected: %s", match)
        logger.info("Alternating sequence found. Finishing.")
        return match is not None or (len(indices) > 1 and indices[-2] == indices[-1])

    @staticmethod
    def _log_gradients(
        iteration: int,
        cur_grads: Tuple[float, PauliSumOp],
        max_grad: Tuple[float, PauliSumOp | None],
    ):
        if logger.isEnabledFor(logging.INFO):
            gradlog = f"\nGradients in iteration #{str(iteration)}"
            gradlog += "\nID: Excitation Operator: Gradient  <(*) maximum>"
            for i, grad in enumerate(cur_grads):
                gradlog += f"\n{str(i)}: {str(grad[1])}: {str(grad[0])}"
                if grad[1] == max_grad[1]:
                    gradlog += "\t(*)"
            logger.info(gradlog)
        else:
            return

    def compute_minimum_eigensolver(
        self,
        operator: OperatorBase,
        aux_operators: Optional[ListOrDict[PauliSumOp]] = None,
    ) -> AdaptVQEResult:
        """Computes the ground state.

        Args:
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            QiskitError: if a solver other than VQE or a ansatz other than EvolvedOperatorAnsatz is provided
                or if the algorithm finishes due to an unforeseen reason.
            ValueError: If the grouped property object returned by the driver does not contain a
                main property as requested by the problem being solved (`problem.main_property_name`)
            QiskitError: If the user-provided `aux_operators` contain a name which clashes
                with an internally constructed auxiliary operator. Note: the names used for the
                internal auxiliary operators correspond to the `Property.name` attributes which
                generated the respective operators.
            QiskitError: If the chosen gradient method appears to result in all-zero gradients.

        Returns:
            An AdaptVQEResult which is an VQEResult but also includes runtime
            information about the AdaptVQE algorithm like the number of iterations, finishing
            criterion, and the final maximum gradient.
        """
        if not isinstance(self._tmp_ansatz, EvolvedOperatorAnsatz):
            raise QiskitError(
                "The AdaptVQE algorithm requires the use of the  evolved operator ansatz"
            )
        # We construct the ansatz once to be able to extract the full set of excitation operators.
        self._tmp_ansatz._build()

        prev_op_indices: List[int] = []
        theta: List[float] = []
        max_grad: Tuple[float, Optional[PauliSumOp]] = (0.0, None)
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
            self._log_gradients(iteration, cur_grads, max_grad)
            if np.abs(max_grad[0]) < self._threshold:
                if iteration == 1:
                    raise QiskitError(
                        "Gradient choice is not suited as it leads to all zero gradients gradients. "
                        "Try a different gradient method."
                    )
                logger.info(
                    "Adaptive VQE terminated successfully with a final maximum gradient: %s",
                    str(np.abs(max_grad[0])),
                )
                Finishing_criterion.finishing_criterion = "Threshold converged"
                break
            # check indices of picked gradients for cycles
            if self._check_cyclicity(prev_op_indices):
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                Finishing_criterion.finishing_criterion = "Aborted due to cyclicity"
                break
            # add new excitation to self._ansatz
            self._excitation_list.append(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            self._tmp_ansatz.operators = self._excitation_list
            self.ansatz = self._tmp_ansatz
            self.initial_point = theta
            raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
        else:
            # reached maximum number of iterations
            Finishing_criterion.finishing_criterion = "Maximum number of iterations reached"
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        if Finishing_criterion.finishing_criterion == False:
            raise QiskitError("The algorithm finished due to an unforeseen reason!")

        result = AdaptVQEResult()
        result.combine(raw_vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.finishing_criterion = Finishing_criterion.finishing_criterion

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            bound_ansatz = self.ansatz.bind_parameters(result.optimal_point)

            aux_values = eval_observables(
                self.quantum_instance, bound_ansatz, aux_operators, expectation=expectation
            )
            result.aux_operator_eigenvalues = aux_values
        else:
            aux_values = None
        raw_vqe_result.aux_operator_eigenvalues = aux_values

        logger.info("The final energy is: %s", str(result.computed_energies[0]))
        return result


class AdaptVQEResult(VariationalResult):
    """AdaptVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._num_iterations: int = 0
        self._final_max_gradient: float = 0.0
        self._finishing_criterion: str = ""

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
    def finishing_criterion(self) -> str:
        """Returns finishing criterion"""
        return self._finishing_criterion

    @finishing_criterion.setter
    def finishing_criterion(self, value: str) -> None:
        """Sets finishing criterion"""
        self._finishing_criterion = value
