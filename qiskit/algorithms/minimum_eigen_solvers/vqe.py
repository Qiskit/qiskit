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

from typing import Optional, List, Callable, Union, Dict
import logging
from time import time
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
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
    I,
    CircuitSampler,
)
from qiskit.opflow.gradients import GradientBase
from qiskit.utils.validation import validate_min
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.utils import QuantumInstance, algorithm_globals
from ..optimizers import Optimizer, SLSQP
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)

# disable check for ansatzes, optimizer setter because of pylint bug
# pylint: disable=no-member


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

    .. note::

        The VQE stores the parameters of ``ansatz`` sorted by name to map the values
        provided by the optimizer to the circuit. This is done to ensure reproducible results,
        for example such that running the optimization twice with same random seeds yields the
        same result. Also, the ``optimal_point`` of the result object can be used as initial
        point of another VQE run by passing it as ``initial_point`` to the initializer.

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
        if ansatz is None:
            ansatz = RealAmplitudes()

        if optimizer is None:
            optimizer = SLSQP()

        if quantum_instance is not None:
            if not isinstance(quantum_instance, QuantumInstance):
                quantum_instance = QuantumInstance(quantum_instance)

        super().__init__()

        self._max_evals_grouped = max_evals_grouped
        self._circuit_sampler = None  # type: Optional[CircuitSampler]
        self._expectation = expectation
        self._include_custom = include_custom

        self._ansatz = None
        self.ansatz = ansatz

        self._optimizer = optimizer
        self._initial_point = initial_point
        self._gradient = gradient
        self._quantum_instance = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        self._eval_time = None
        self._eval_count = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback

        logger.info(self.print_settings())

        # TODO remove this once the stateful methods are deleted
        self._ret = None

    def _try_set_expectation_value_from_factory(self, operator: OperatorBase) -> None:
        if operator is not None and self.quantum_instance is not None:
            self._expectation = ExpectationFactory.build(
                    operator=operator,
                    backend=self.quantum_instance,
                    include_custom=self._include_custom,
                )

    @property
    def ansatz(self) -> Optional[QuantumCircuit]:
        """Returns the ansatz"""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz"""
        if isinstance(ansatz, QuantumCircuit):
            # store the parameters
            self._ansatz_params = sorted(ansatz.parameters, key=lambda p: p.name)
            self._ansatz = ansatz
        elif ansatz is None:
            self._ansatz_params = None
            self._ansatz = ansatz
        else:
            raise ValueError('Unsupported type "{}" of ansatz'.format(type(ansatz)))

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """set quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance, param_qobj=is_aer_provider(quantum_instance.backend)
        )

    @property
    def expectation(self) -> ExpectationBase:
        """The expectation value algorithm used to construct the expectation measurement from
        the observable."""
        return self._expectation

    @expectation.setter
    def expectation(self, exp: ExpectationBase) -> None:
        self._expectation = exp

    def _check_operator_varform(self, operator: OperatorBase):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self.ansatz is not None:
            if operator.num_qubits != self.ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.ansatz.num_qubits = operator.num_qubits
                    self._ansatz_params = sorted(self.ansatz.parameters, key=lambda p: p.name)
                except AttributeError as ex:
                    raise AlgorithmError(
                        "The number of qubits of the ansatz does not match the "
                        "operator, and the ansatz does not allow setting the "
                        "number of qubits using `num_qubits`."
                    ) from ex

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        """Sets optimizer"""
        self._optimizer = optimizer
        if optimizer is not None:
            optimizer.set_max_evals_grouped(self._max_evals_grouped)

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
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
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        if hasattr(self._ansatz, "setting"):
            ret += "{}".format(self._ansatz.setting)
        elif hasattr(self._ansatz, "print_settings"):
            ret += "{}".format(self._ansatz.print_settings())
        elif isinstance(self._ansatz, QuantumCircuit):
            ret += "ansatz is a custom circuit"
        else:
            ret += "ansatz has not been set"
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def construct_expectation(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
    ) -> OperatorBase:
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.
            operator: Qubit operator of the Observable

        Returns:
            The Operator equalling the measurement of the ansatz :class:`StateFn` by the
            Observable's expectation :class:`StateFn`.

        Raises:
            AlgorithmError: If no operator has been provided.
        """
        if operator is None:
            raise AlgorithmError("The operator was never provided.")

        self._check_operator_varform(operator)

        param_dict = dict(zip(self._ansatz_params, parameter))  # type: Dict
        wave_function = self.ansatz.assign_parameters(param_dict)

        # Expectation was never created , try to create one
        if self.expectation is None:
            self._try_set_expectation_value_from_factory(operator)

        # If setting the expectation failed, raise an Error:
        if self.expectation is None:
            raise AlgorithmError(
                "No expectation set and could not automatically set one, please "
                "try explicitly setting an expectation or specify a backend so it "
                "can be chosen automatically."
            )

        observable_meas = self.expectation.convert(StateFn(operator, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        return observable_meas.compose(ansatz_circuit_op).reduce()

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

    def _eval_aux_ops(self, aux_operators: List[OperatorBase], threshold: float = 1e-12) -> np.ndarray:
        # Create new CircuitSampler to avoid breaking existing one's caches.
        sampler = CircuitSampler(self.quantum_instance)

        aux_op_meas = self.expectation.convert(StateFn(ListOp(aux_operators), is_measurement=True))
        aux_op_expect = aux_op_meas.compose(CircuitStateFn(self.get_optimal_circuit()))
        values = np.real(sampler.convert(aux_op_expect).eval())

        # Discard values below threshold
        aux_op_results = values * (np.abs(values) > threshold)
        # Deal with the aux_op behavior where there can be Nones or Zero qubit Paulis in the list
        _aux_op_nones = [op is None for op in aux_operators]
        aux_operator_eigenvalues = [
            None if is_none else [result]
            for (is_none, result) in zip(_aux_op_nones, aux_op_results)
        ]
        # As this has mixed types, since it can included None, it needs to explicitly pass object
        # data type to avoid numpy 1.19 warning message about implicit conversion being deprecated
        aux_operator_eigenvalues = np.array(
            [aux_operator_eigenvalues], dtype=object
        )
        return aux_operator_eigenvalues

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )
        self.quantum_instance.circuit_summary = True

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_varform(operator)

        initial_point = _validate_initial_point(self.initial_point,
                                                self.ansatz)

        # We need to handle the array entries being Optional i.e. having value None
        if aux_operators:
            zero_op = I.tensorpower(operator.num_qubits) * 0.0
            converted = []
            for op in aux_operators:
                if op is None:
                    converted.append(zero_op)
                else:
                    converted.append(op)

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]
        else:
            aux_operators = None

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        if isinstance(self._gradient, GradientBase):
            gradient = self._gradient.gradient_wrapper(
                ~StateFn(operator) @ StateFn(self._ansatz),
                bind_params=self._ansatz_params,
                backend=self._quantum_instance,
            )
        else:
            gradient = self._gradient

        self._eval_count = 0
        energy_evaluation = self.get_energy_evaluation(operator)

        start_time = time()
        opt_params, opt_value, nfev = self.optimizer.optimize(
            num_vars=len(initial_point),
            objective_function=energy_evaluation,
            gradient_function=gradient,
            initial_point=initial_point,
        )
        eval_time = time() - start_time

        result = VQEResult()
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self._ansatz_params, opt_params))
        result.cost_function_evals = nfev
        result.optimizer_time = eval_time
        result.eigenvalue = opt_value + 0j
        result.eigenstate = self._get_eigenstate(result.optimal_parameters)

        # TODO
        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s in %s evals",
            eval_time,
            result.optimal_point,
            self._eval_count,
        )

        # TODO
        # self._ret.eigenstate = self.get_optimal_vector()
        if aux_operators is not None:
            aux_values = self._eval_aux_ops(aux_operators)
            result.aux_operator_eigenvalues = aux_values[0]

        # TODO delete as soon as get_optimal_vector etc are removed
        self._ret = result
        return result

    def get_energy_evaluation(
        self, operator: OperatorBase
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Evaluate energy at given parameters for the ansatz.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            operator: The operator whose energy to evaluate.

        Returns:
            Energy of the hamiltonian of each parameter.

        Raises:
            RuntimeError: If the ansatz has no parameters.
        """
        num_parameters = self.ansatz.num_parameters
        if self._ansatz.num_parameters == 0:
            raise RuntimeError("The ansatz cannot have 0 parameters.")

        expect_op = self.construct_expectation(self._ansatz_params, operator)

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            # Create dict associating each parameter with the lists of parameterization values for it
            param_bindings = dict(
                zip(self._ansatz_params, parameter_sets.transpose().tolist())
            )

            start_time = time()
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_bindings)
            means = np.real(sampled_expect_op.eval())

            if self._callback is not None:
                variance = np.real(self.expectation.compute_variance(sampled_expect_op))
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

        return energy_evaluation

    def get_optimal_cost(self) -> float:
        """Get the minimal cost or energy found by the VQE."""
        if self._ret.optimal_point is None:
            raise AlgorithmError(
                "Cannot return optimal cost before running the " "algorithm to find optimal params."
            )
        return self._ret.optimal_value

    def get_optimal_circuit(self) -> QuantumCircuit:
        """Get the circuit with the optimal parameters."""
        if self._ret.optimal_point is None:
            raise AlgorithmError(
                "Cannot find optimal circuit before running the "
                "algorithm to find optimal params."
            )
        return self.ansatz.assign_parameters(self._ret.optimal_parameters)

    def get_optimal_vector(self) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the optimal circuit."""
        if self._ret.optimal_parameters is None:
            raise AlgorithmError(
                "Cannot find optimal circuit before running the "
                "algorithm to find optimal vector."
            )
        return self._get_eigenstate(self._ret.optimal_parameters)

    def _get_eigenstate(self, optimal_parameters) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the optimal circuit."""
        from qiskit.utils.run_circuits import find_regs_by_name

        optimal_circuit = self.ansatz.bind_parameters(optimal_parameters)
        min_vector = {}
        if self.quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(optimal_circuit)
            min_vector = ret.get_statevector(optimal_circuit)
        else:
            c = ClassicalRegister(optimal_circuit.width(), name="c")
            q = find_regs_by_name(optimal_circuit, "q")
            optimal_circuit.add_register(c)
            optimal_circuit.barrier(q)
            optimal_circuit.measure(q, c)
            ret = self.quantum_instance.execute(optimal_circuit)
            counts = ret.get_counts(optimal_circuit)
            # normalize, just as done in CircuitSampler.sample_circuits
            shots = self.quantum_instance._run_config.shots
            min_vector = {b: (v / shots) ** 0.5 for (b, v) in counts.items()}

        return min_vector

    @property
    def optimal_params(self) -> List[float]:
        """The optimal parameters for the ansatz."""
        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot find optimal params before running the algorithm.")
        return self._ret.optimal_point


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
        if self._eigenstate is None:
            self._eigenstate = self._get_optimal_vector()
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: np.ndarray) -> None:
        """set eigen state"""
        self._eigenstate = value


def _validate_initial_point(point, ansatz):
    expected_size = ansatz.num_parameters

    # if the point is None choose a random initial point
    if point is None:
        if hasattr(ansatz, "preferred_init_points"):
            point = ansatz.preferred_init_points
        else:
            # get bounds if ansatz has them set, otherwise use [-2pi, 2pi] for each parameter
            default_bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size
            bounds = getattr(ansatz, "parameter_bounds", default_bounds)

            # replace all Nones by [-2pi, 2pi]
            lower_bounds = []
            upper_bounds = []
            for lower, upper in bounds:
                lower_bounds.append(lower if lower is not None else -2 * np.pi)
                upper_bounds.append(upper if upper is not None else 2 * np.pi)

            # sample from within bounds
            point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(f'The dimension of the initial point ({len(point)}) does not match the '
                         f'number of parameters in the circuit ({expected_size}).')

    return point
