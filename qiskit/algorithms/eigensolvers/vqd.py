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
from typing import Callable
import logging
from time import time
from warnings import warn
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.algorithms.state_fidelities import BaseStateFidelity

from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, SLSQP, Minimizer
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .eigensolver import Eigensolver, EigensolverResult
from ..minimum_eigen_solvers.vqe import _validate_bounds, _validate_initial_point
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)


class VQD(VariationalAlgorithm, Eigensolver):
    r"""The Variational Quantum Deflation algorithm. Implementation using primitives.

    `VQD <https://arxiv.org/abs/1805.08138>`__ is a quantum algorithm that uses a
    variational technique to find
    the k eigenvalues of the Hamiltonian :math:`H` of a given system.

    The algorithm computes excited state energies of generalised hamiltonians
    by optimising over a modified cost function where each succesive eigen value
    is calculated iteratively by introducing an overlap term with all
    the previously computed eigenstaes that must be minimised, thus ensuring
    higher energy eigen states are found.

    An instance of VQD requires defining three algorithmic sub-components:
    an integer k denoting the number of eigenstates to calculate, a trial
    state (a.k.a. ansatz)which is a :class:`QuantumCircuit`,
    and one of the classical :mod:`~qiskit.algorithms.optimizers`.
    The ansatz is varied, via its set of parameters, by the optimizer,
    such that it works towards a state, as determined by the parameters
    applied to the ansatz, that will result in the minimum expectation values
    being measured of the input operator (Hamiltonian). The algorithm does
    this by iteratively refining each excited state to be orthogonal to all
    the previous excited states.

    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the ansatz being used. If the *initial_point* is left at the default
    of ``None``, then VQD will look to the ansatz for a preferred value, based on its
    given initial state. If the ansatz returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the ansatz provides ``None`` as the lower bound, then VQD
    will default it to :math:`-2\pi`; similarly, if the ansatz returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        fidelity: BaseStateFidelity,
        ansatz: QuantumCircuit | None = None,
        *,
        k: int = 2,
        betas: list[float] | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        initial_point: np.ndarray | None = None,
        gradient: Callable | None = None,
        callback: Callable[[int, np.ndarray, float, float], None] | None = None,
    ) -> None:
        """

        Args:
            estimator: The estimator primitive.
            fidelity: The fidelity class using primitives.
            ansatz: A parameterized circuit used as ansatz for the wave function.
            k: the number of eigenvalues to return. Returns the lowest k eigenvalues.
            betas: beta parameters in the VQD paper.
                Should have length k - 1, with k the number of excited states.
                These hyperparameters balance the contribution of each overlap term to the cost
                function and have a default value computed as the mean square sum of the
                coefficients of the observable.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQD will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional fidelity gradient using primitives.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the ansatz, the
                evaluated mean, the evaluated standard deviation, and the current step.
        """
        super().__init__()

        self.estimator = estimator
        self.fidelity = fidelity
        self.k = k
        self.betas = betas
        self.callback = callback

        if ansatz is None:
            ansatz = RealAmplitudes()
        self.ansatz = ansatz

        if optimizer is None:
            optimizer = SLSQP()
        self.optimizer = optimizer

        if gradient is not None:
            warn(
                "The current VQD implementation does not offer gradient support."
                "Continuing without gradients."
            )

        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self._initial_point = initial_point

        self._eval_time = None
        self._eval_count = 0

    @property
    def initial_point(self) -> np.ndarray | None:
        """Returns initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray):
        """Sets initial point"""
        self._initial_point = initial_point

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self.ansatz is not None:
            if operator.num_qubits != self.ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.ansatz.num_qubits = operator.num_qubits
                except AttributeError as ex:
                    raise AlgorithmError(
                        "The number of qubits of the ansatz does not match the "
                        "operator, and the ansatz does not allow setting the "
                        "number of qubits using `num_qubits`."
                    ) from ex

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    # TODO: Replace with import
    def _eval_observables(
        self,
        estimator: BaseEstimator,
        quantum_state: QuantumCircuit,
        observables: ListOrDict[BaseOperator | PauliSumOp],
        threshold: float = 1e-12,
    ) -> ListOrDict[tuple[complex, complex]]:
        """
        Accepts a sequence of operators and calculates their expectation values - means
        and standard deviations. They are calculated with respect to a quantum state provided. A user
        can optionally provide a threshold value which filters mean values falling below the threshold.

        Args:
            estimator: An estimator primitive used for calculations.
            quantum_state: An unparametrized quantum circuit representing a quantum state that
                expectation values are computed against.
            observables: A list or a dictionary of operators whose expectation values are to be
                calculated.
            threshold: A threshold value that defines which mean values should be neglected (helpful for
                ignoring numerical instabilities close to 0).

        Returns:
            A list or a dictionary of tuples (mean, standard deviation).

        Raises:
            ValueError: If a ``quantum_state`` with free parameters is provided.
            AlgorithmError: If a primitive job is not successful.
        """

        if (
            isinstance(quantum_state, QuantumCircuit)  # State cannot be parametrized
            and len(quantum_state.parameters) > 0
        ):
            raise ValueError(
                "A parametrized representation of a quantum_state was provided. It is not "
                "allowed - it cannot have free parameters."
            )
        if isinstance(observables, dict):
            observables_list = list(observables.values())
        else:
            observables_list = observables
        quantum_state = [quantum_state] * len(observables)
        try:
            estimator_job = estimator.run(quantum_state, observables_list)
            expectation_values = estimator_job.result().values
        except Exception as exc:
            raise AlgorithmError("The primitive job failed!") from exc

        std_devs = self._compute_std_devs(estimator_job, len(expectation_values))

        # Discard values below threshold
        observables_means = expectation_values * (np.abs(expectation_values) > threshold)
        # zip means and standard deviations into tuples
        observables_results = list(zip(observables_means, std_devs))

        # Return None eigenvalues for None operators if observables is a list.
        return self._prepare_result(observables_results, observables)

    # TODO: Replace with import
    @staticmethod
    def _prepare_result(
        observables_results: list[tuple[complex, complex]],
        observables: ListOrDict[BaseOperator | PauliSumOp],
    ) -> ListOrDict[tuple[complex, complex]]:
        """
        Prepares a list of eigenvalues and standard deviations from ``observables_results`` and
        ``observables``.

        Args:
            observables_results: A list of of tuples (mean, standard deviation).
            observables: A list or a dictionary of operators whose expectation values are to be
                calculated.

        Returns:
            A list or a dictionary of tuples (mean, standard deviation).
        """

        if isinstance(observables, list):
            observables_eigenvalues = [None] * len(observables)
            key_value_iterator = enumerate(observables_results)
        else:
            observables_eigenvalues = {}
            key_value_iterator = zip(observables.keys(), observables_results)

        for key, value in key_value_iterator:
            if observables[key] is not None:
                observables_eigenvalues[key] = value
        return observables_eigenvalues

    # TODO: Replace with import
    @staticmethod
    def _compute_std_devs(
        estimator_result: EstimatorResult,
        results_length: int,
    ) -> list[complex | None]:
        """
        Calculates a list of standard deviations from expectation values of observables provided. If
        the choice of an underlying hardware is not shot-based and hence does not provide variance data,
        the standard deviation values will be set to ``None``.

        Args:
            estimator_result: An estimator result.
            results_length: Number of expectation values calculated.

        Returns:
            A list of standard deviations.
        """
        if not estimator_result.metadata:
            return [0] * results_length

        std_devs = []
        for metadata in estimator_result.metadata:
            if metadata and "variance" in metadata.keys() and "shots" in metadata.keys():
                variance = metadata["variance"]
                shots = metadata["shots"]
                std_devs.append(np.sqrt(variance / shots))
            else:
                std_devs.append(None)

        return std_devs

    def compute_eigenvalues(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> EigensolverResult:
        super().compute_eigenvalues(operator, aux_operators)

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_ansatz(operator)

        initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        bounds = _validate_bounds(self.ansatz)

        # We need to handle the array entries being zero or Optional i.e. having value None
        if aux_operators:
            zero_op = PauliSumOp.from_list([("I" * self.ansatz.num_qubits, 0)])

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
            upper_bound = (
                abs(operator.coeff)
                if isinstance(operator, PauliOp)
                else abs(operator.coeff) * sum(abs(operation.coeff) for operation in operator)
            )
            self.betas = [upper_bound * 10] * (self.k)
            logger.info("beta autoevaluated to %s", self.betas[0])

        result = VQDResult()
        result.optimal_point = []
        result.optimal_parameters = []
        result.optimal_value = []
        result.cost_function_evals = []
        result.optimizer_time = []
        result.eigenvalues = []

        if aux_operators is not None:
            aux_values = []

        for step in range(1, self.k + 1):

            self._eval_count = 0
            energy_evaluation = self.get_energy_evaluation(
                step, operator, prev_states=result.optimal_parameters
            )

            start_time = time()

            # TODO: add gradient support after FidelityGradients are implemented
            if callable(self.optimizer):
                opt_result = self.optimizer(  # pylint: disable=not-callable
                    fun=energy_evaluation, x0=initial_point, bounds=bounds
                )
            else:
                opt_result = self.optimizer.minimize(
                    fun=energy_evaluation, x0=initial_point, bounds=bounds
                )

            eval_time = time() - start_time

            result.optimal_point.append(opt_result.x)
            result.optimal_parameters.append(dict(zip(self.ansatz.parameters, opt_result.x)))
            result.optimal_value.append(opt_result.fun)
            result.cost_function_evals.append(opt_result.nfev)
            result.optimizer_time.append(eval_time)
            result.eigenvalues.append(opt_result.fun + 0j)

            if aux_operators is not None:
                bound_ansatz = self.ansatz.bind_parameters(result.optimal_point[-1])
                # TODO: replace this by the new eval_observables once merged
                aux_value = self._eval_observables(self.estimator, bound_ansatz, aux_operators)
                aux_values.append(aux_value)

            if step == 1:

                logger.info(
                    "Ground state optimization complete in %s seconds.\n"
                    "Found opt_params %s in %s evals",
                    eval_time,
                    result.optimal_point,
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
                    result.optimal_point,
                    self._eval_count,
                )

        # To match the signature of NumpyEigenSolver Result
        result.eigenvalues = np.array(result.eigenvalues)
        result.optimal_point = np.array(result.optimal_point)
        result.optimal_value = np.array(result.optimal_value)
        result.cost_function_evals = np.array(result.cost_function_evals)
        result.optimizer_time = np.array(result.optimizer_time)

        if aux_operators is not None:
            result.aux_operator_eigenvalues = aux_values

        return result

    def get_energy_evaluation(
        self,
        step: int,
        operator: BaseOperator | PauliSumOp,
        prev_states: list[np.ndarray] | None = None,
    ) -> Callable[[np.ndarray], float | list[float]]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.
        This return value is the objective function to be passed to the optimizer for evaluation.

        Args:
            step: level of energy being calculated. 0 for ground, 1 for first excited state...
            operator: The operator whose energy to evaluate.
            prev_states: List of parameters from previous rounds of optimization.

        Returns:
            A callable that computes and returns the energy of the hamiltonian
            of each parameter, and, optionally, the expectation

        Raises:
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).
            AlgorithmError: If operator was not provided.

        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        if operator is None:
            raise AlgorithmError("The operator was never provided.")

        if step > 1 and (len(prev_states) + 1) != step:
            raise RuntimeError(
                f"Passed previous states of the wrong size."
                f"Passed array has length {str(len(prev_states))}"
            )

        self._check_operator_ansatz(operator)

        prev_circs = []
        for state in range(step - 1):
            prev_circs.append(self.ansatz.bind_parameters(prev_states[state]))

        def energy_evaluation(parameters):
            estimator_job = self.estimator.run(
                circuits=[self.ansatz], observables=[operator], parameter_values=[parameters]
            )
            estimator_result = estimator_job.result()
            means = np.real(estimator_result.values)

            if step > 1:
                fidelity_job = self.fidelity.run(
                    [self.ansatz] * len(prev_circs),
                    prev_circs,
                    [parameters] * len(prev_circs),
                    [prev_states[:-1]],
                )
                costs = fidelity_job.result().fidelities

                for (state, cost) in zip(range(step - 1), costs):
                    means += np.real(self.betas[state] * cost)

            if self.callback is not None:
                variance = np.array([estimator_result.metadata[0].pop("variance", 0)])
                shots = np.array([estimator_result.metadata[0].pop("shots", 0)])
                estimator_error = np.sqrt(variance / shots)
                for i, param_set in enumerate([parameters]):
                    self._eval_count += 1
                    self.callback(self._eval_count, param_set, means[i], estimator_error[i], step)
            else:
                self._eval_count += len(means)
            return means if len(means) > 1 else means[0]

        return energy_evaluation


class VQDResult(VariationalResult, EigensolverResult):
    """VQD Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

    @property
    def cost_function_evals(self) -> int | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value
