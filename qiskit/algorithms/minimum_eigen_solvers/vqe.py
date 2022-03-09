# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061
"""

from typing import Optional, List, Callable, Union, Dict, Tuple
import logging
import warnings
from time import time
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers import BaseBackend
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
from qiskit.utils import QuantumInstance, algorithm_globals
from ..optimizers import Optimizer, SLSQP
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult, ListOrDict
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)


class VQE(VariationalAlgorithm, MinimumEigensolver):
    r"""The Variational Quantum Eigensolver algorithm.

    `VQE <https://arxiv.org/abs/1304.3061>`__ is a quantum algorithm that uses a
    variational technique to find
    the minimum eigenvalue of the Hamiltonian :math:`H` of a given system.

    An instance of VQE requires defining two algorithmic sub-components:
    a trial state (a.k.a. ansatz) which is a :class:`QuantumCircuit`, and one of the classical
    :mod:`~qiskit.algorithms.optimizers`. The ansatz is varied, via its set of parameters, by the
    optimizer, such that it works towards a state, as determined by the parameters applied to the
    ansatz, that will result in the minimum expectation value being measured of the input operator
    (Hamiltonian).

    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  It provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the ansatz being used. If the *initial_point* is left at the default
    of ``None``, then VQE will look to the ansatz for a preferred value, based on its
    given initial state. If the ansatz returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the ansatz provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the ansatz returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.

    """

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """

        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When ``None`` (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.`
            quantum_instance: Quantum Instance or Backend

        """
        validate_min("max_evals_grouped", max_evals_grouped, 1)

        super().__init__()

        self._max_evals_grouped = max_evals_grouped
        self._circuit_sampler = None  # type: Optional[CircuitSampler]
        self._expectation = None
        self.expectation = expectation
        self._include_custom = include_custom

        self._ansatz = None
        self.ansatz = ansatz

        self._optimizer = None
        self.optimizer = optimizer

        self._initial_point = None
        self.initial_point = initial_point
        self._gradient = None
        self.gradient = gradient
        self._quantum_instance = None

        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        self._eval_time = None
        self._eval_count = 0
        self._callback = None
        self.callback = callback

        logger.info(self.print_settings())

        # TODO remove this once the stateful methods are deleted
        self._ret = None

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        self._ansatz = ansatz

    @property
    def gradient(self) -> Optional[Union[GradientBase, Callable]]:
        """Returns the gradient."""
        return self._gradient

    @gradient.setter
    def gradient(self, gradient: Optional[Union[GradientBase, Callable]]):
        """Sets the gradient."""
        self._gradient = gradient

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """Sets quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance, param_qobj=is_aer_provider(quantum_instance.backend)
        )

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns initial point"""
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
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns callback"""
        return self._callback

    @callback.setter
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]):
        """Sets callback"""
        self._callback = callback

    @property
    def expectation(self) -> Optional[ExpectationBase]:
        """The expectation value algorithm used to construct the expectation measurement from
        the observable."""
        return self._expectation

    @expectation.setter
    def expectation(self, exp: Optional[ExpectationBase]) -> None:
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
    def optimizer(self, optimizer: Optional[Optimizer]):
        """Sets the optimizer attribute.

        Args:
            optimizer: The optimizer to be used. If None is passed, SLSQP is used by default.

        """
        if optimizer is None:
            optimizer = SLSQP()

        optimizer.set_max_evals_grouped(self.max_evals_grouped)
        self._optimizer = optimizer

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
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
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
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
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Union[OperatorBase, Tuple[OperatorBase, ExpectationBase]]:
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
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
    ) -> List[QuantumCircuit]:
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
    ) -> ListOrDict[Tuple[complex, complex]]:
        # Create new CircuitSampler to avoid breaking existing one's caches.
        sampler = CircuitSampler(self.quantum_instance)

        if isinstance(aux_operators, dict):
            list_op = ListOp(list(aux_operators.values()))
        else:
            list_op = ListOp(aux_operators)

        aux_op_expect = expectation.convert(
            StateFn(list_op, is_measurement=True).compose(
                CircuitStateFn(self.ansatz.bind_parameters(parameters))
            )
        )
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
        # None operators are already dropped in compute_minimum_eigenvalue if aux_operators is a dict.
        if isinstance(aux_operators, list):
            aux_operator_eigenvalues = [None] * len(aux_operators)
            key_value_iterator = enumerate(aux_op_results)
        else:
            aux_operator_eigenvalues = {}
            key_value_iterator = zip(aux_operators.keys(), aux_op_results)

        for key, value in key_value_iterator:
            if aux_operators[key] is not None:
                aux_operator_eigenvalues[key] = value

        return aux_operator_eigenvalues

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)

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

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        if isinstance(self._gradient, GradientBase):
            gradient = self._gradient.gradient_wrapper(
                ~StateFn(operator) @ StateFn(self.ansatz),
                bind_params=list(self.ansatz.parameters),
                backend=self._quantum_instance,
            )
        else:
            gradient = self._gradient

        self._eval_count = 0
        energy_evaluation, expectation = self.get_energy_evaluation(
            operator, return_expectation=True
        )

        start_time = time()

        # keep this until Optimizer.optimize is removed
        try:
            opt_result = self.optimizer.minimize(
                fun=energy_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        except AttributeError:
            # self.optimizer is an optimizer with the deprecated interface that uses
            # ``optimize`` instead of ``minimize```
            warnings.warn(
                "Using an optimizer that is run with the ``optimize`` method is "
                "deprecated as of Qiskit Terra 0.19.0 and will be unsupported no "
                "sooner than 3 months after the release date. Instead use an optimizer "
                "providing ``minimize`` (see qiskit.algorithms.optimizers.Optimizer).",
                DeprecationWarning,
                stacklevel=2,
            )

            opt_result = self.optimizer.optimize(
                len(initial_point), energy_evaluation, gradient, bounds, initial_point
            )

        eval_time = time() - start_time

        result = VQEResult()
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        result.optimal_value = opt_result.fun
        result.cost_function_evals = opt_result.nfev
        result.optimizer_time = eval_time
        result.eigenvalue = opt_result.fun + 0j
        result.eigenstate = self._get_eigenstate(result.optimal_parameters)

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s in %s evals",
            eval_time,
            result.optimal_point,
            self._eval_count,
        )

        # TODO delete as soon as get_optimal_vector etc are removed
        self._ret = result

        if aux_operators is not None:
            aux_values = self._eval_aux_ops(opt_result.x, aux_operators, expectation=expectation)
            result.aux_operator_eigenvalues = aux_values

        return result

    def get_energy_evaluation(
        self,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            operator: The operator whose energy to evaluate.
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to evaluate other
                operators with the same expectation value converter.


        Returns:
            Energy of the hamiltonian of each parameter, and, optionally, the expectation
            converter.

        Raises:
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).

        """
        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        ansatz_params = self.ansatz.parameters
        expect_op, expectation = self.construct_expectation(
            ansatz_params, operator, return_expectation=True
        )

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            # Create dict associating each parameter with the lists of parameterization values for it
            param_bindings = dict(zip(ansatz_params, parameter_sets.transpose().tolist()))

            start_time = time()
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_bindings)
            means = np.real(sampled_expect_op.eval())

            if self._callback is not None:
                variance = np.real(expectation.compute_variance(sampled_expect_op))
                estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
                for i, param_set in enumerate(parameter_sets):
                    self._eval_count += 1
                    self._callback(self._eval_count, param_set, means[i], estimator_error[i])
            else:
                self._eval_count += len(means)

            end_time = time()
            logger.info(
                "Energy evaluation returned %s - %.5f (ms), eval count: %s",
                means,
                (end_time - start_time) * 1000,
                self._eval_count,
            )

            return means if len(means) > 1 else means[0]

        if return_expectation:
            return energy_evaluation, expectation

        return energy_evaluation

    def _get_eigenstate(self, optimal_parameters) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the ansatz, provided with parameters."""
        optimal_circuit = self.ansatz.bind_parameters(optimal_parameters)
        state_fn = self._circuit_sampler.convert(StateFn(optimal_circuit)).eval()
        if self.quantum_instance.is_statevector:
            state = state_fn.primitive.data  # VectorStateFn -> Statevector -> np.array
        else:
            state = state_fn.to_dict_fn().primitive  # SparseVectorStateFn -> DictStateFn -> dict

        return state


class VQEResult(VariationalResult, MinimumEigensolverResult):
    """VQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

    @property
    def cost_function_evals(self) -> Optional[int]:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def eigenstate(self) -> Optional[np.ndarray]:
        """return eigen state"""
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: np.ndarray) -> None:
        """set eigen state"""
        self._eigenstate = value


def _validate_initial_point(point, ansatz):
    expected_size = ansatz.num_parameters

    # try getting the initial point from the ansatz
    if point is None and hasattr(ansatz, "preferred_init_points"):
        point = ansatz.preferred_init_points
    # if the point is None choose a random initial point

    if point is None:
        # get bounds if ansatz has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(ansatz, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point


def _validate_bounds(ansatz):
    if hasattr(ansatz, "parameter_bounds") and ansatz.parameter_bounds is not None:
        bounds = ansatz.parameter_bounds
        if len(bounds) != ansatz.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({ansatz.num_parameters})."
            )
    else:
        bounds = [(None, None)] * ansatz.num_parameters

    return bounds
