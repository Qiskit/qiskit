# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
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

import logging
import warnings
from collections.abc import Callable
from time import time
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.providers import Backend
from qiskit.opflow import (
    OperatorBase,
    ExpectationBase,
    ExpectationFactory,
    StateFn,
    CircuitStateFn,
    ListOp,
    CircuitSampler,
    PauliSumOp,
)
from qiskit.opflow.gradients import GradientBase
from qiskit.utils.validation import validate_min
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.utils.deprecation import deprecate_func
from qiskit.utils import QuantumInstance
from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, SLSQP, Minimizer
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .eigen_solver import Eigensolver, EigensolverResult
from ..minimum_eigen_solvers.vqe import _validate_bounds, _validate_initial_point
from ..exceptions import AlgorithmError
from ..aux_ops_evaluator import eval_observables

logger = logging.getLogger(__name__)


class VQD(VariationalAlgorithm, Eigensolver):
    r"""Deprecated: Variational Quantum Deflation algorithm.

    The VQD class has been superseded by the
    :class:`qiskit.algorithms.eigensolvers.VQD` class.
    This class will be deprecated in a future release and subsequently
    removed after that.

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

    @deprecate_func(
        additional_msg=(
            "Instead, use the class ``qiskit.algorithms.eigensolvers.VQD``."
            "See https://qisk.it/algo_migration for a migration guide."
        ),
        since="0.24.0",
    )
    def __init__(
        self,
        ansatz: QuantumCircuit | None = None,
        k: int = 2,
        betas: list[float] | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        initial_point: np.ndarray | None = None,
        gradient: GradientBase | Callable | None = None,
        expectation: ExpectationBase | None = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Callable[[int, np.ndarray, float, float, int], None] | None = None,
        quantum_instance: QuantumInstance | Backend | None = None,
    ) -> None:
        """

        Args:
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
            gradient: An optional gradient function or operator for optimizer.
                Only used to compute the ground state at the moment.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When ``None`` (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQD performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                multiple points to compute the gradient can be passed and if computed in parallel
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the ansatz, the
                evaluated mean, the evaluated standard deviation, and the current step.
            quantum_instance: Quantum Instance or Backend

        """
        validate_min("max_evals_grouped", max_evals_grouped, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__()

        self._max_evals_grouped = max_evals_grouped
        self._circuit_sampler: CircuitSampler | None = None
        self._expectation = None
        self.expectation = expectation
        self._include_custom = include_custom

        # set ansatz -- still supporting pre 0.18.0 sorting

        self._ansatz: QuantumCircuit | None = None
        self.ansatz = ansatz

        self.k = k
        self.betas = betas

        self._optimizer: Optimizer | None = None
        self.optimizer = optimizer

        self._initial_point: np.ndarray | None = None
        self.initial_point = initial_point
        self._gradient: GradientBase | Callable | None = None
        self.gradient = gradient
        self._quantum_instance: QuantumInstance | None = None

        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        self._eval_time = None
        self._eval_count = 0
        self._callback: Callable[[int, np.ndarray, float, float, int], None] | None = None
        self.callback = callback

        logger.info(self.print_settings())

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit | None):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
                If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        self._ansatz = ansatz

    @property
    def gradient(self) -> GradientBase | Callable | None:
        """Returns the gradient."""
        return self._gradient

    @gradient.setter
    def gradient(self, gradient: GradientBase | Callable | None):
        """Sets the gradient."""
        self._gradient = gradient

    @property
    def quantum_instance(self) -> QuantumInstance | None:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance | Backend) -> None:
        """Sets a quantum_instance."""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance, param_qobj=is_aer_provider(quantum_instance.backend)
        )

    @property
    def initial_point(self) -> np.ndarray | None:
        """Returns initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray):
        """Sets initial point"""
        self._initial_point = initial_point

    @property
    def max_evals_grouped(self) -> int:
        """Returns max_evals_grouped"""
        return self._max_evals_grouped

    @max_evals_grouped.setter
    def max_evals_grouped(self, max_evals_grouped: int):
        """Sets max_evals_grouped"""
        self._max_evals_grouped = max_evals_grouped
        self.optimizer.set_max_evals_grouped(max_evals_grouped)

    @property
    def include_custom(self) -> bool:
        """Returns include_custom"""
        return self._include_custom

    @include_custom.setter
    def include_custom(self, include_custom: bool):
        """Sets include_custom. If set to another value than the one that was previsously set,
        the expectation attribute is reset to None.
        """
        if include_custom != self._include_custom:
            self._include_custom = include_custom
            self.expectation = None

    @property
    def callback(self) -> Callable[[int, np.ndarray, float, float, int], None] | None:
        """Returns callback"""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable[[int, np.ndarray, float, float, int], None] | None):
        """Sets callback"""
        self._callback = callback

    @property
    def expectation(self) -> ExpectationBase | None:
        """The expectation value algorithm used to construct the expectation measurement from
        the observable."""
        return self._expectation

    @expectation.setter
    def expectation(self, exp: ExpectationBase | None) -> None:
        self._expectation = exp

    def _check_operator_ansatz(self, operator: OperatorBase):
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

    @property
    def optimizer(self) -> Optimizer:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer | None):
        """Sets the optimizer attribute.

        Args:
            optimizer: The optimizer to be used. If None is passed, SLSQP is used by default.

        """
        if optimizer is None:
            optimizer = SLSQP()

        if isinstance(optimizer, Optimizer):
            optimizer.set_max_evals_grouped(self.max_evals_grouped)

        self._optimizer = optimizer

    @property
    def setting(self):
        """Prepare the setting of VQD as a string."""
        ret = f"Algorithm: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    def print_settings(self):
        """Preparing the setting of VQD into a string.

        Returns:
            str: the formatted setting of VQD.
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__
        )
        ret += f"{self.setting}"
        ret += "===============================================================\n"
        if self.ansatz is not None:
            ret += "{}".format(self.ansatz.draw(output="text"))
        else:
            ret += "ansatz has not been set"
        ret += "===============================================================\n"
        ret += f"{self._optimizer.setting}"
        ret += "===============================================================\n"
        return ret

    def construct_expectation(
        self,
        parameter: list[float] | list[Parameter] | np.ndarray,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> OperatorBase | tuple[OperatorBase, ExpectationBase]:
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.
            operator: Qubit operator of the Observable
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to compute the standard
                deviation of the expectation value.

        Returns:
            The Operator equalling the measurement of the ansatz :class:`StateFn` by the
            Observable's expectation :class:`StateFn`, and, optionally, the expectation converter.

        Raises:
            AlgorithmError: If no operator has been provided.
            AlgorithmError: If no expectation is passed and None could be inferred via the
                ExpectationFactory.
        """
        if operator is None:
            raise AlgorithmError("The operator was never provided.")

        self._check_operator_ansatz(operator)

        # if expectation was never created, try to create one
        if self.expectation is None:
            expectation = ExpectationFactory.build(
                operator=operator,
                backend=self.quantum_instance,
                include_custom=self._include_custom,
            )
        else:
            expectation = self.expectation

        wave_function = self.ansatz.assign_parameters(parameter)

        observable_meas = expectation.convert(StateFn(operator, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

        if return_expectation:
            return expect_op, expectation

        return expect_op

    def construct_circuit(
        self,
        parameter: list[float] | list[Parameter] | np.ndarray,
        operator: OperatorBase,
    ) -> list[QuantumCircuit]:
        """Return the circuits used to compute the expectation value.

        Args:
            parameter: Parameters for the ansatz circuit.
            operator: Qubit operator of the Observable

        Returns:
            A list of the circuits used to compute the expectation value.
        """
        expect_op = self.construct_expectation(parameter, operator).to_circuit_op()

        circuits = []

        # recursively extract circuits
        def extract_circuits(op):
            if isinstance(op, CircuitStateFn):
                circuits.append(op.primitive)
            elif isinstance(op, ListOp):
                for op_i in op.oplist:
                    extract_circuits(op_i)

        extract_circuits(expect_op)

        return circuits

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _eval_aux_ops(
        self,
        parameters: np.ndarray,
        aux_operators: ListOrDict[OperatorBase],
        expectation: ExpectationBase,
        threshold: float = 1e-12,
    ) -> ListOrDict[tuple[complex, complex]]:
        # Create new CircuitSampler to avoid breaking existing one's caches.
        sampler = CircuitSampler(self.quantum_instance)

        if isinstance(aux_operators, dict):
            list_op = ListOp(list(aux_operators.values()))
        else:
            list_op = ListOp(aux_operators)

        aux_op_meas = expectation.convert(StateFn(list_op, is_measurement=True))
        aux_op_expect = aux_op_meas.compose(CircuitStateFn(self.ansatz.bind_parameters(parameters)))
        aux_op_expect_sampled = sampler.convert(aux_op_expect)

        # compute means
        values = np.real(aux_op_expect_sampled.eval())

        # compute standard deviations
        variances = np.real(expectation.compute_variance(aux_op_expect_sampled))
        if not isinstance(variances, np.ndarray) and variances == 0.0:
            # when `variances` is a single value equal to 0., our expectation value is exact and we
            # manually ensure the variances to be a list of the correct length
            variances = np.zeros(len(aux_operators), dtype=float)
        std_devs = np.sqrt(variances / self.quantum_instance.run_config.shots)

        # Discard values below threshold
        aux_op_means = values * (np.abs(values) > threshold)
        # zip means and standard deviations into tuples
        aux_op_results = zip(aux_op_means, std_devs)

        # Return None eigenvalues for None operators if aux_operators is a list.
        # None operators are already dropped in compute_minimum_eigenvalue if aux_operators is a
        # dict.
        if isinstance(aux_operators, list):
            aux_operator_eigenvalues: ListOrDict[tuple[complex, complex]] = [None] * len(
                aux_operators
            )
            key_value_iterator = enumerate(aux_op_results)
        else:
            aux_operator_eigenvalues = {}
            key_value_iterator = zip(aux_operators.keys(), aux_op_results)

        for key, value in key_value_iterator:
            if aux_operators[key] is not None:
                aux_operator_eigenvalues[key] = value

        return aux_operator_eigenvalues

    def compute_eigenvalues(
        self, operator: OperatorBase, aux_operators: ListOrDict[OperatorBase] | None = None
    ) -> EigensolverResult:
        super().compute_eigenvalues(operator, aux_operators)

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )
        self.quantum_instance.circuit_summary = True

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_ansatz(operator)

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        bounds = _validate_bounds(self.ansatz)
        # We need to handle the array entries being zero or Optional i.e. having value None
        if aux_operators:
            zero_op = PauliSumOp.from_list([("I" * self.ansatz.num_qubits, 0)])

            # Convert the None and zero values when aux_operators is a list.
            # Drop None and convert zero values when aux_operators is a dict.
            if isinstance(aux_operators, list):
                key_op_iterator = enumerate(aux_operators)
                converted: ListOrDict[OperatorBase] = [zero_op] * len(aux_operators)
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
        result.eigenstates = []

        if aux_operators is not None:
            aux_values = []

        for step in range(1, self.k + 1):

            self._eval_count = 0
            energy_evaluation, expectation = self.get_energy_evaluation(
                step, operator, return_expectation=True, prev_states=result.optimal_parameters
            )

            # Convert the gradient operator into a callable function that is compatible with the
            # optimization routine. Only used for the ground state currently as Gradient() doesnt
            # support SumOps yet
            if isinstance(self._gradient, GradientBase):
                gradient = self._gradient.gradient_wrapper(
                    StateFn(operator, is_measurement=True) @ StateFn(self.ansatz),
                    bind_params=list(self.ansatz.parameters),
                    backend=self._quantum_instance,
                )
            else:
                gradient = self._gradient

            start_time = time()

            if callable(self.optimizer):
                opt_result = self.optimizer(  # pylint: disable=not-callable
                    fun=energy_evaluation, x0=initial_point, jac=gradient, bounds=bounds
                )
            else:
                opt_result = self.optimizer.minimize(
                    fun=energy_evaluation, x0=initial_point, jac=gradient, bounds=bounds
                )

            eval_time = time() - start_time

            result.optimal_point.append(opt_result.x)
            result.optimal_parameters.append(dict(zip(self.ansatz.parameters, opt_result.x)))
            result.optimal_value.append(opt_result.fun)
            result.cost_function_evals.append(opt_result.nfev)
            result.optimizer_time.append(eval_time)

            eigenvalue = (
                StateFn(operator, is_measurement=True)
                .compose(CircuitStateFn(self.ansatz.bind_parameters(result.optimal_parameters[-1])))
                .reduce()
                .eval()
            )

            result.eigenvalues.append(eigenvalue)
            result.eigenstates.append(self._get_eigenstate(result.optimal_parameters[-1]))

            if aux_operators is not None:
                bound_ansatz = self.ansatz.bind_parameters(result.optimal_point[-1])
                aux_value = eval_observables(
                    self.quantum_instance, bound_ansatz, aux_operators, expectation=expectation
                )
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
        result.eigenstates = ListOp([StateFn(vec) for vec in result.eigenstates])
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
        operator: OperatorBase,
        return_expectation: bool = False,
        prev_states: list[np.ndarray] | None = None,
    ) -> Callable[[np.ndarray], float | list[float]] | tuple[
        Callable[[np.ndarray], float | list[float]], ExpectationBase
    ]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.

        This return value is the objective function to be passed to the optimizer for evaluation.

        Args:
            step: level of energy being calculated. 0 for ground, 1 for first excited state...
            operator: The operator whose energy to evaluate.
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to evaluate other
                operators with the same expectation value converter.
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
        overlap_op = []

        ansatz_params = self.ansatz.parameters
        expect_op, expectation = self.construct_expectation(
            ansatz_params, operator, return_expectation=True
        )

        for state in range(step - 1):

            prev_circ = self.ansatz.bind_parameters(prev_states[state])
            overlap_op.append(~CircuitStateFn(prev_circ) @ CircuitStateFn(self.ansatz))

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            # Dict associating each parameter with the lists of parameterization values for it
            param_bindings = dict(zip(ansatz_params, parameter_sets.transpose().tolist()))

            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_bindings)
            means = np.real(sampled_expect_op.eval())

            for state in range(step - 1):
                sampled_final_op = self._circuit_sampler.convert(
                    overlap_op[state], params=param_bindings
                )
                cost = sampled_final_op.eval()
                means += np.real(self.betas[state] * np.conj(cost) * cost)

            if self._callback is not None:
                variance = np.real(expectation.compute_variance(sampled_expect_op))
                estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
                for i, param_set in enumerate(parameter_sets):
                    self._eval_count += 1
                    self._callback(self._eval_count, param_set, means[i], estimator_error[i], step)
            else:
                self._eval_count += len(means)

            return means if len(means) > 1 else means[0]

        if return_expectation:
            return energy_evaluation, expectation

        return energy_evaluation

    def _get_eigenstate(self, optimal_parameters) -> list[float] | dict[str, int]:
        """Get the simulation outcome of the ansatz, provided with parameters."""
        optimal_circuit = self.ansatz.bind_parameters(optimal_parameters)
        state_fn = self._circuit_sampler.convert(StateFn(optimal_circuit)).eval()
        if self.quantum_instance.is_statevector:
            state = state_fn.primitive.data  # VectorStateFn -> Statevector -> np.array
        else:
            state = state_fn.to_dict_fn().primitive  # SparseVectorStateFn -> DictStateFn -> dict

        return state


class VQDResult(VariationalResult, EigensolverResult):
    """Deprecated: VQD Result.

    The VQDResult class has been superseded by the
    :class:`qiskit.algorithms.eigensolvers.VQDResult` class.
    This class will be deprecated in a future release and subsequently
    removed after that.

    """

    @deprecate_func(
        additional_msg=(
            "Instead, use the class ``qiskit.algorithms.eigensolvers.VQDResult``."
            "See https://qisk.it/algo_migration for a migration guide."
        ),
        since="0.24.0",
    )
    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals: int | None = None

    @property
    def cost_function_evals(self) -> int | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def eigenstates(self) -> np.ndarray | None:
        """return eigen state"""
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """set eigen state"""
        self._eigenstates = value
