# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Deflation Algorithm for computing higher energy states.

See https://arxiv.org/abs/1805.08138.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from time import time
from typing import Any

import numpy as np

from qiskit.algorithms.state_fidelities import BaseStateFidelity
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import SparsePauliOp

from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, Minimizer, OptimizerResult
from ..variational_algorithm import VariationalAlgorithm
from .eigensolver import Eigensolver, EigensolverResult
from ..utils import validate_bounds, validate_initial_point
from ..exceptions import AlgorithmError
from ..observables_evaluator import estimate_observables

# private function as we expect this to be updated in the next release
from ..utils.set_batching import _set_default_batchsize

logger = logging.getLogger(__name__)


class VQD(VariationalAlgorithm, Eigensolver):
    r"""The Variational Quantum Deflation algorithm. Implementation using primitives.

    `VQD <https://arxiv.org/abs/1805.08138>`__ is a quantum algorithm that uses a
    variational technique to find
    the k eigenvalues of the Hamiltonian :math:`H` of a given system.

    The algorithm computes excited state energies of generalised hamiltonians
    by optimising over a modified cost function where each succesive eigenvalue
    is calculated iteratively by introducing an overlap term with all
    the previously computed eigenstates that must be minimised, thus ensuring
    higher energy eigenstates are found.

    An instance of VQD requires defining three algorithmic sub-components:
    an integer k denoting the number of eigenstates to calculate, a trial
    state (a.k.a. ansatz) which is a :class:`QuantumCircuit`,
    and one of the classical :mod:`~qiskit.algorithms.optimizers`.
    The optimizer varies the circuit parameters
    The trial state :math:`|\psi(\vec\theta)\rangle` is varied by the optimizer,
    which modifies the set of ansatz parameters :math:`\vec\theta`
    such that the expectation value of the operator on the corresponding
    state approaches a minimum. The algorithm does this by iteratively refining
    each excited state to be orthogonal to all the previous excited states.

    An optional array of parameter values, via the *initial_point*, may be provided
    as the starting point for the search of the minimum eigenvalue. This feature is
    particularly useful when there are reasons to believe that the solution point
    is close to a particular point.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the ansatz. If the *initial_point* is left at the default
    of ``None``, then VQD will look to the ansatz for a preferred value, based on its
    given initial state. If the ansatz returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the ansatz provides ``None`` as the lower bound, then VQD
    will default it to :math:`-2\pi`; similarly, if the ansatz returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.

    The following attributes can be set via the initializer but can also be read and
    updated once the VQD object has been constructed.

    Attributes:
            estimator (BaseEstimator): The primitive instance used to perform the expectation
                estimation as indicated in the VQD paper.
            fidelity (BaseStateFidelity): The fidelity class instance used to compute the
                overlap estimation as indicated in the VQD paper.
            ansatz (QuantumCircuit): A parameterized circuit used as ansatz for the wave function.
            optimizer(Optimizer): A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            k (int): the number of eigenvalues to return. Returns the lowest k eigenvalues.
            betas (list[float]): Beta parameters in the VQD paper.
                Should have length k - 1, with k the number of excited states.
                These hyper-parameters balance the contribution of each overlap term to the cost
                function and have a default value computed as the mean square sum of the
                coefficients of the observable.
            initial point (list[float]): An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQD will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None):
                A callback that can access the intermediate data
                during the optimization. Four parameter values are passed to the callback as
                follows during each evaluation by the optimizer: the evaluation count,
                the optimizer parameters for the ansatz, the estimated value, the estimation
                metadata, and the current step.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        fidelity: BaseStateFidelity,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        *,
        k: int = 2,
        betas: Sequence[float] | None = None,
        initial_point: Sequence[float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ) -> None:
        """

        Args:
            estimator: The estimator primitive.
            fidelity: The fidelity class using primitives.
            ansatz: A parameterized circuit used as ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            k: The number of eigenvalues to return. Returns the lowest k eigenvalues.
            betas: Beta parameters in the VQD paper.
                Should have length k - 1, with k the number of excited states.
                These hyperparameters balance the contribution of each overlap term to the cost
                function and have a default value computed as the mean square sum of the
                coefficients of the observable.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQD will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            callback: A callback that can access the intermediate data
                during the optimization. Four parameter values are passed to the callback as
                follows during each evaluation by the optimizer: the evaluation count,
                the optimizer parameters for the ansatz, the estimated value,
                the estimation metadata, and the current step.
        """
        super().__init__()

        self.estimator = estimator
        self.fidelity = fidelity
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.k = k
        self.betas = betas
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback

        self._eval_count = 0

    @property
    def initial_point(self) -> Sequence[float] | None:
        """Returns initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Sequence[float]):
        """Sets initial point"""
        self._initial_point = initial_point

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self.ansatz is not None:
            if operator.num_qubits != self.ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.ansatz.num_qubits = operator.num_qubits
                except AttributeError as exc:
                    raise AlgorithmError(
                        "The number of qubits of the ansatz does not match the "
                        "operator, and the ansatz does not allow setting the "
                        "number of qubits using `num_qubits`."
                    ) from exc

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_eigenvalues(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> VQDResult:

        super().compute_eigenvalues(operator, aux_operators)

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_ansatz(operator)

        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)

        # We need to handle the array entries being zero or Optional i.e. having value None
        if aux_operators:
            zero_op = SparsePauliOp.from_list([("I" * self.ansatz.num_qubits, 0)])

            # Convert the None and zero values when aux_operators is a list.
            # Drop None and convert zero values when aux_operators is a dict.
            if isinstance(aux_operators, list):
                key_op_iterator = enumerate(aux_operators)
                converted = [zero_op] * len(aux_operators)
            else:
                key_op_iterator = aux_operators.items()
                converted = {}
            for key, op in key_op_iterator:
                if op is not None:
                    converted[key] = zero_op if op == 0 else op

            aux_operators = converted

        else:
            aux_operators = None

        if self.betas is None:

            if isinstance(operator, PauliSumOp):
                operator = operator.coeff * operator.primitive

            try:
                upper_bound = sum(np.abs(operator.coeffs))

            except Exception as exc:
                raise NotImplementedError(
                    r"Beta autoevaluation is not supported for operators"
                    f"of type {type(operator)}."
                ) from exc

            betas = [upper_bound * 10] * (self.k)
            logger.info("beta autoevaluated to %s", betas[0])
        else:
            betas = self.betas

        result = self._build_vqd_result()

        if aux_operators is not None:
            aux_values = []

        # We keep a list of the bound circuits with optimal parameters, to avoid re-binding
        # the same parameters to the ansatz if we do multiple steps
        prev_states = []

        for step in range(1, self.k + 1):

            # update list of optimal circuits
            if step > 1:
                prev_states.append(self.ansatz.bind_parameters(result.optimal_points[-1]))

            self._eval_count = 0
            energy_evaluation = self._get_evaluate_energy(
                step, operator, betas, prev_states=prev_states
            )

            start_time = time()

            # TODO: add gradient support after FidelityGradients are implemented
            if callable(self.optimizer):
                opt_result = self.optimizer(  # pylint: disable=not-callable
                    fun=energy_evaluation, x0=initial_point, bounds=bounds
                )
            else:
                # we always want to submit as many estimations per job as possible for minimal
                # overhead on the hardware
                was_updated = _set_default_batchsize(self.optimizer)

                opt_result = self.optimizer.minimize(
                    fun=energy_evaluation, x0=initial_point, bounds=bounds
                )

                # reset to original value
                if was_updated:
                    self.optimizer.set_max_evals_grouped(None)

            eval_time = time() - start_time

            self._update_vqd_result(result, opt_result, eval_time, self.ansatz.copy())

            if aux_operators is not None:
                aux_value = estimate_observables(
                    self.estimator, self.ansatz, aux_operators, result.optimal_points[-1]
                )
                aux_values.append(aux_value)

            if step == 1:
                logger.info(
                    "Ground state optimization complete in %s seconds.\n"
                    "Found opt_params %s in %s evals",
                    eval_time,
                    result.optimal_points,
                    self._eval_count,
                )
            else:
                logger.info(
                    (
                        "%s excited state optimization complete in %s s.\n"
                        "Found opt_params %s in %s evals"
                    ),
                    str(step - 1),
                    eval_time,
                    result.optimal_points,
                    self._eval_count,
                )

        # To match the signature of EigensolverResult
        result.eigenvalues = np.array(result.eigenvalues)
        result.optimal_points = np.array(result.optimal_points)
        result.optimal_values = np.array(result.optimal_values)
        result.cost_function_evals = np.array(result.cost_function_evals)
        result.optimizer_times = np.array(result.optimizer_times)

        if aux_operators is not None:
            result.aux_operators_evaluated = aux_values

        return result

    def _get_evaluate_energy(
        self,
        step: int,
        operator: BaseOperator | PauliSumOp,
        betas: Sequence[float],
        prev_states: list[QuantumCircuit] | None = None,
    ) -> Callable[[np.ndarray], float | list[float]]:
        """Returns a function handle to evaluate the ansatz's energy for any given parameters.
            This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            step: level of energy being calculated. 0 for ground, 1 for first excited state...
            operator: The operator whose energy to evaluate.
            betas: Beta parameters in the VQD paper.
            prev_states: List of optimal circuits from previous rounds of optimization.

        Returns:
            A callable that computes and returns the energy of the hamiltonian
            of each parameter.

        Raises:
            AlgorithmError: If the circuit is not parameterized (i.e. has 0 free parameters).
            AlgorithmError: If operator was not provided.
            RuntimeError: If the previous states array is of the wrong size.
        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

        if step > 1 and (len(prev_states) + 1) != step:
            raise RuntimeError(
                f"Passed previous states of the wrong size."
                f"Passed array has length {str(len(prev_states))}"
            )

        self._check_operator_ansatz(operator)

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:

            estimator_job = self.estimator.run(
                circuits=[self.ansatz], observables=[operator], parameter_values=[parameters]
            )

            total_cost = 0
            if step > 1:
                # compute overlap cost
                fidelity_job = self.fidelity.run(
                    [self.ansatz] * (step - 1),
                    prev_states,
                    [parameters] * (step - 1),
                )

                costs = fidelity_job.result().fidelities

                for state, cost in enumerate(costs):
                    total_cost += np.real(betas[state] * cost)

            try:
                estimator_result = estimator_job.result()

            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            values = estimator_result.values + total_cost

            if self.callback is not None:
                metadata = estimator_result.metadata
                for params, value, meta in zip([parameters], values, metadata):
                    self._eval_count += 1
                    self.callback(self._eval_count, params, value, meta, step)
            else:
                self._eval_count += len(values)

            return values if len(values) > 1 else values[0]

        return evaluate_energy

    @staticmethod
    def _build_vqd_result() -> VQDResult:

        result = VQDResult()
        result.optimal_points = []
        result.optimal_parameters = []
        result.optimal_values = []
        result.cost_function_evals = []
        result.optimizer_times = []
        result.eigenvalues = []
        result.optimizer_results = []
        result.optimal_circuits = []
        return result

    @staticmethod
    def _update_vqd_result(result, opt_result, eval_time, ansatz) -> VQDResult:

        result.optimal_points.append(opt_result.x)
        result.optimal_parameters.append(dict(zip(ansatz.parameters, opt_result.x)))
        result.optimal_values.append(opt_result.fun)
        result.cost_function_evals.append(opt_result.nfev)
        result.optimizer_times.append(eval_time)
        result.eigenvalues.append(opt_result.fun + 0j)
        result.optimizer_results.append(opt_result)
        result.optimal_circuits.append(ansatz)
        return result


class VQDResult(EigensolverResult):
    """VQD Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None
        self._optimizer_times = None
        self._optimal_values = None
        self._optimal_points = None
        self._optimal_parameters = None
        self._optimizer_results = None
        self._optimal_circuits = None

    @property
    def cost_function_evals(self) -> Sequence[int] | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: Sequence[int]) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def optimizer_times(self) -> Sequence[float] | None:
        """Returns time taken for optimization for each step"""
        return self._optimizer_times

    @optimizer_times.setter
    def optimizer_times(self, value: Sequence[float]) -> None:
        """Sets time taken for optimization for each step"""
        self._optimizer_times = value

    @property
    def optimal_values(self) -> Sequence[float] | None:
        """Returns optimal value for each step"""
        return self._optimal_values

    @optimal_values.setter
    def optimal_values(self, value: Sequence[float]) -> None:
        """Sets optimal values"""
        self._optimal_values = value

    @property
    def optimal_points(self) -> Sequence[np.ndarray] | None:
        """Returns optimal point for each step"""
        return self._optimal_points

    @optimal_points.setter
    def optimal_points(self, value: Sequence[np.ndarray]) -> None:
        """Sets optimal points"""
        self._optimal_points = value

    @property
    def optimal_parameters(self) -> Sequence[dict] | None:
        """Returns the optimal parameters for each step"""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: Sequence[dict]) -> None:
        """Sets optimal parameters"""
        self._optimal_parameters = value

    @property
    def optimizer_results(self) -> Sequence[OptimizerResult] | None:
        """Returns the optimizer results for each step"""
        return self._optimizer_results

    @optimizer_results.setter
    def optimizer_results(self, value: Sequence[OptimizerResult]) -> None:
        """Sets optimizer results"""
        self._optimizer_results = value

    @property
    def optimal_circuits(self) -> list[QuantumCircuit]:
        """The optimal circuits. Along with the optimal parameters,
        these can be used to retrieve the different eigenstates."""
        return self._optimal_circuits

    @optimal_circuits.setter
    def optimal_circuits(self, optimal_circuits: list[QuantumCircuit]) -> None:
        self._optimal_circuits = optimal_circuits
