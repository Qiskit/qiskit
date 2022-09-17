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

"""The Subspace Search Variational Quantum Eigensolver algorithm.
See https://arxiv.org/abs/1810.09434
"""

from typing import Optional, List, Callable, Union, Dict, Tuple
import logging
import warnings
from time import time
import numpy as np
import scipy

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes
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
from qiskit.quantum_info import Statevector
from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, SLSQP, OptimizerResult
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .eigen_solver import Eigensolver, EigensolverResult
from ..exceptions import AlgorithmError
from ..aux_ops_evaluator import eval_observables

logger = logging.getLogger(__name__)


OBJECTIVE = Callable[[np.ndarray], float]
GRADIENT = Callable[[np.ndarray], np.ndarray]
RESULT = Union[scipy.optimize.OptimizeResult, OptimizerResult]
BOUNDS = List[Tuple[float, float]]

MINIMIZER = Callable[
    [
        OBJECTIVE,  # the objective function to minimize (the energy in the case of the VQE)
        np.ndarray,  # the initial point for the optimization
        Optional[GRADIENT],  # the gradient of the objective function
        Optional[BOUNDS],  # parameters bounds for the optimization
    ],
    RESULT,  # a result object (either SciPy's or Qiskit's)
]


class SSVQE(VariationalAlgorithm, Eigensolver):
    r"""The Subspace Search Variational Quantum Eigensolver algorithm.
    `VQE <https://arxiv.org/abs/1810.09434>`__ is a quantum algorithm that uses a
    variational technique to find
    the low-lying eigenvalues of the Hamiltonian :math:`H` of a given system.
    An instance of SSVQE requires defining two algorithmic sub-components:
    a trial state (a.k.a. ansatz) which is a :class:`QuantumCircuit`, and one of the classical
    :mod:`~qiskit.algorithms.optimizers`. The ansatz is varied, via its set of parameters, by the
    optimizer, such that it works towards a state, as determined by the parameters applied to the
    ansatz, that will result in the minimum expectation value being measured of the input operator
    (Hamiltonian).
    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the low-lying eigenvalues. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  It provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.
    The length of the *initial_point* list value must match the number of the parameters
    expected by the ansatz being used. If the *initial_point* is left at the default
    of ``None``, then SSVQE will look to the ansatz for a preferred value, based on its
    given initial state. If the ansatz returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the ansatz provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the ansatz returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.
    An optional list of initial states, via the *initial_states*, may also be provided. Choosing
    these states appropriately is a critical part of the algorithm. They must be mutually orthogonal
    as this is how the algorithm enforces the mutual orthogonality of the solution states. If
    the *initial_states* is left as ``None``, then SSVQE will automatically generate a list of
    computational basis states and use these as the initial states. For many physically-motivated
    problems, it is advised to not rely on these default values as doing so can easily result in
    an unphysical solution being returned. For example, if one wishes to find the low-lying
    excited states of a molecular Hamiltonian, then we expect the output states to belong to
    a particular particle-number subspace. If an ansatz that preserves particle number such as
    :class:`UCCSD` is used, then states belonging to the incorrect particle number subspace
    will be returned if the *initial_states* are not in the correct particle number subspace.
    A similar statement can often be made for the spin-magnetization quantum number.
    The optimizer can either be one of Qiskit's optimizers, such as
    :class:`~qiskit.algorithms.optimizers.SPSA` or a callable with the following signature:
    .. note::
        The callable _must_ have the argument names ``fun, x0, jac, bounds`` as indicated
        in the following code block.
    .. code-block::python
        from qiskit.algorithms.optimizers import OptimizerResult
        def my_minimizer(fun, x0, jac=None, bounds=None) -> OptimizerResult:
            # Note that the callable *must* have these argument names!
            # Args:
            #     fun (callable): the function to minimize
            #     x0 (np.ndarray): the initial point for the optimization
            #     jac (callable, optional): the gradient of the objective function
            #     bounds (list, optional): a list of tuples specifying the parameter bounds
            result = OptimizerResult()
            result.x = # optimal parameters
            result.fun = # optimal function value
            return result
    The above signature also allows to directly pass any SciPy minimizer, for instance as
    .. code-block::python
        from functools import partial
        from scipy.optimize import minimize
        optimizer = partial(minimize, method="L-BFGS-B")
    """

    def __init__(
        self,
        num_states: Optional[int] = 2,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Union[Optimizer, MINIMIZER]] = None,
        initial_point: Optional[np.ndarray] = None,
        initial_states: Optional[
            Union[List[QuantumCircuit], List[Statevector]]
        ] = None,  # Set of initial orthogonal states expressed as a list of QuantumCircuit objects.
        weight_vector: Optional[
            Union[np.ndarray, List]
        ] = None,  # set of weight factors to be used in the cost function
        gradient: Optional[Union[GradientBase, Callable]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
    ) -> None:
        """
        Args:
        num_states: The number of states which the algorithm will attempt to find.
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            initial_states: An optional list of mutually orthogonal initial states.
                If ``None``, then SSVQE will set these to be a list of mutually orthogonal
                computational basis states.
            weight_vector: An optional list or array of real positive numbers with length
                equal to the value of *num_states* to be used in the weighted energy summation
                objective function. This fixes the ordering of the returned eigenstate/eigenvalue
                pairs.
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

        self._num_states = num_states

        self._initial_states = None
        self.initial_states = initial_states
        self._weight_vector = None
        self.weight_vector = weight_vector
        self._initialized_ansatz_list = None

        super().__init__()

        self._max_evals_grouped = max_evals_grouped

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
    def weight_vector(self) -> Union[np.ndarray, List]:
        """Returns the weight vector used to order the eigenvalues."""
        return self._weight_vector

    @weight_vector.setter
    def weight_vector(self, vec: Union[np.ndarray, List]):
        """Sets the weight vector used to order the eigenvalues.
        Args:
            vec: The weight vector used in the objective function. If
            None is passed, then the weight vector is set to an array
            of the form [num_states, num_states - 1, .... ,1].
        """

        if vec is not None:
            self._weight_vector = vec
        else:
            self._weight_vector = np.asarray(
                [self._num_states - n for n in range(self._num_states)]
            )

    @property
    def num_states(self) -> int:
        """Returns the number of low-lying eigenstates which the algorithm will attempt to find."""
        return self._num_states

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
    def initial_states(self) -> List[Union[QuantumCircuit, Statevector]]:
        """Returns the initial states."""

        return self._initial_states

    @initial_states.setter
    def initial_states(self, states):
        """Sets the initial states.
        Args:
        states: The initial states to be used by the algorithm."""

        self._initial_states = states

    @property
    def initialized_ansatz_list(self) -> List[QuantumCircuit]:
        """Returns a list of ansatz circuits, where the nth element
        in the list is a QuantumCircuit consisting of the ansatz
        initialized with the nth element of the list
        of initial states."""

        return self._initialized_ansatz_list

    @initialized_ansatz_list.setter
    def initialized_ansatz_list(self, initial_states):
        """Sets the list of ansatz circuits initialized in the set of initial states.
        Args: initial_states: The list of orthogonal initial states."""

        self._initialized_ansatz_list = [
            initial_states[n].compose(self.ansatz) for n in range(self.num_states)
        ]

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
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, Backend]) -> None:
        """Sets quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler_list = [
            CircuitSampler(quantum_instance, param_qobj=is_aer_provider(quantum_instance.backend))
            for n in range(self.num_states)
        ]

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

        if isinstance(optimizer, Optimizer):
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
        if callable(self.optimizer):
            ret += "Optimizer is custom callable\n"
        else:
            ret += f"{self._optimizer.setting}"
        ret += "===============================================================\n"
        return ret

    def construct_expectation_list(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> List[Union[OperatorBase, Tuple[OperatorBase, ExpectationBase]]]:
        r"""
        Generate a list of the initialized ansatz circuits and
        expectation value measurements, and return their
        runnable compositions.
        Args:
            parameter: Parameters for the ansatz circuit.
            operator: Qubit operator of the Observable
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to compute the standard
                deviation of the expectation value.
        Returns:
            A list of the Operators equalling the measurement of the initialized ansatzes
            :class:`StateFn` (where the ansatzes are initialized in the initial orthogonal states)
            by the Observable's expectation :class:`StateFn`,
            and, optionally, the expectation converters.
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

        wave_function_list = [
            self.initialized_ansatz_list[n].assign_parameters(parameter)
            for n in range(self.num_states)
        ]

        observable_meas = expectation.convert(StateFn(operator, is_measurement=True))
        ansatz_circuit_op_list = [
            CircuitStateFn(wave_function_list[n]) for n in range(self.num_states)
        ]
        expect_op_list = [
            observable_meas.compose(ansatz_circuit_op_list[n]).reduce()
            for n in range(self.num_states)
        ]

        if return_expectation:
            return expect_op_list, expectation

        return expect_op_list

    def construct_circuits(  # check back on this later to make adjustments.
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
    ) -> List[QuantumCircuit]:
        """Return the circuits used to compute the expectation value.
        Args:
            parameter: Parameters for the ansatz circuit.
            operator: Qubit operator of the Observable
        Returns:
            A list of lists of the circuits used to compute the expectation value.
        """
        expect_op_list = self.construct_expectation_list(parameter, operator)

        for n in range(self.num_states):
            expect_op_list[n] = expect_op_list[n].to_circuit_op()

        circuits = [[]] * self.num_states
        # recursively extract circuits
        def extract_circuits(op_list):
            for n in range(self.num_states):

                if isinstance(op_list[n], CircuitStateFn):
                    circuits[n].append(op_list[n].primitive)
                elif isinstance(op_list[n], ListOp):
                    for op_i in op_list[n].oplist:
                        extract_circuits(op_i)

        extract_circuits(expect_op_list)

        return circuits

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_eigenvalues(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
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

        if self.initial_states is None:
            self.initial_states = [
                QuantumCircuit(self.ansatz.num_qubits) for n in range(self.num_states)
            ]
            for n in range(self.num_states):
                # if no initial states are provided, set them to be the first
                # n computational basis states.
                print(self.ansatz.num_qubits)
                self.initial_states[n].initialize(
                    Statevector.from_int(n, 2**self.ansatz.num_qubits)
                )

        for n in range(self.num_states):
            # if any element in the list of initial states is :class:`Statevector`,
            # convert it to :class:`QuantumCircuit`
            if isinstance(self.initial_states[n], Statevector):
                temporary_circ = QuantumCircuit(self.ansatz.num_qubits)
                temporary_circ.initialize(self.initial_states[n])
                self.initial_states[n] = temporary_circ

        if self.initialized_ansatz_list is None:
            self.initialized_ansatz_list = [
                self.initial_states[n].compose(self.ansatz) for n in range(self.num_states)
            ]

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

        if isinstance(self._gradient, GradientBase):

            gradient_op = 0
            for n in range(self.num_states):
                gradient_op += float(self.weight_vector[n]) * (
                    ~StateFn(operator) @ StateFn(self.initialized_ansatz_list[n])
                )

            gradient = self._gradient.gradient_wrapper(
                gradient_op,
                bind_params=list(self.ansatz.parameters),
                backend=self._quantum_instance,
            )
        else:
            gradient = self._gradient

        self._eval_count = 0
        weighted_energy_sum_evaluation, expectation = self.get_weighted_energy_sum_evaluation(
            operator, return_expectation=True
        )

        start_time = time()

        if callable(self.optimizer):
            opt_result = self.optimizer(  # pylint: disable=not-callable
                fun=weighted_energy_sum_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        else:
            # keep this until Optimizer.optimize is removed
            try:
                opt_result = self.optimizer.minimize(
                    fun=weighted_energy_sum_evaluation,
                    x0=initial_point,
                    jac=gradient,
                    bounds=bounds,
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
                    len(initial_point),
                    weighted_energy_sum_evaluation,
                    gradient,
                    bounds,
                    initial_point,
                )

        eval_time = time() - start_time

        result = SSVQEResult()
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        result.optimal_value = opt_result.fun
        result.cost_function_evals = opt_result.nfev
        result.optimizer_time = eval_time
        result.eigenvalues = self._get_eigenvalues(result.optimal_parameters, operator)
        result.eigenstates = self._get_eigenstates(result.optimal_parameters)

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s in %s evals",
            eval_time,
            result.optimal_point,
            self._eval_count,
        )

        # TODO delete as soon as get_optimal_vector etc are removed
        self._ret = result

        if aux_operators is not None:
            bound_ansatz_list = [
                self.initialized_ansatz_list[n].bind_parameters(result.optimal_point)
                for n in range(self.num_states)
            ]

            aux_values_list = [
                eval_observables(
                    self.quantum_instance,
                    bound_ansatz_list[n],
                    aux_operators,
                    expectation=expectation,
                )
                for n in range(self.num_states)
            ]
            result.aux_operator_eigenvalues = aux_values_list

        return result

    def get_weighted_energy_sum_evaluation(
        self,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Returns a function handle to evaluates the weighted energy sum at given parameters
        for the ansatz. This is the objective function to be passed to the optimizer
        that is used for evaluation.
        Args:
            operator: The operator whose energy levels to evaluate.
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to evaluate other
                operators with the same expectation value converter.
        Returns:
            Weighted energy sum of the hamiltonian of each parameter, and, optionally, the expectation
            converter.
        Raises:
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).
        """
        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        ansatz_params = self.ansatz.parameters
        expect_op_list, expectation = self.construct_expectation_list(
            ansatz_params, operator, return_expectation=True
        )

        def weighted_energy_sum_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            # Create dict associating each parameter with the lists of parameterization values for it
            param_bindings = dict(zip(ansatz_params, parameter_sets.transpose().tolist()))

            start_time = time()
            sampled_expect_op_list = [
                self._circuit_sampler_list[n].convert(expect_op_list[n], params=param_bindings)
                for n in range(self.num_states)
            ]
            list_of_means = np.asarray(
                [np.real(sampled_expect_op_list[n].eval()) for n in range(self.num_states)]
            )

            energy_values = np.copy(list_of_means)

            for n in range(self.num_states):
                list_of_means[n, :] *= self.weight_vector[n]

            weighted_sum_means = np.sum(list_of_means, axis=0)

            if self._callback is not None:
                variance_list = [
                    np.real(expectation.compute_variance(sampled_expect_op_list[n]))
                    for n in range(self.num_states)
                ]
                estimator_error_list = [
                    np.sqrt(variance_list[n] / self.quantum_instance.run_config.shots)
                    for n in range(self.num_states)
                ]
                for i, param_set in enumerate(parameter_sets):
                    self._eval_count += 1
                    self._callback(
                        self._eval_count,
                        param_set,
                        [energy_values[n][i] for n in range(self.num_states)],
                        weighted_sum_means[i],
                        [estimator_error_list[n][i] for n in range(self.num_states)],
                    )
            else:
                self._eval_count += len(weighted_sum_means)

            end_time = time()
            logger.info(
                "Energies evaluation returned %s - %.5f (ms), eval count: %s",
                list_of_means,
                (end_time - start_time) * 1000,
                self._eval_count,
            )

            return weighted_sum_means if len(weighted_sum_means) > 1 else weighted_sum_means[0]

        if return_expectation:
            return weighted_energy_sum_evaluation, expectation

        return weighted_energy_sum_evaluation

    def _get_eigenstates(self, optimal_parameters) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the ansatz, provided with parameters."""
        optimal_circuits_list = [
            self.initialized_ansatz_list[n].bind_parameters(optimal_parameters)
            for n in range(self.num_states)
        ]
        state_fns_list = [
            self._circuit_sampler_list[n].convert(StateFn(optimal_circuits_list[n])).eval()
            for n in range(self.num_states)
        ]
        if self.quantum_instance.is_statevector:
            list_of_states = [
                state_fns_list[n].primitive.data for n in range(self.num_states)
            ]  # VectorStateFn -> Statevector -> np.array
        else:
            list_of_states = [
                state_fns_list[n].to_dict_fn().primitive for n in range(self.num_states)
            ]  # SparseVectorStateFn -> DictStateFn -> dict

        return list_of_states

    def _get_eigenvalues(self, optimal_parameters, operator) -> List[float]:
        """Get the eigenvalue results at the optimal parameters."""

        if self.expectation is None:
            expectation = ExpectationFactory.build(
                operator=operator,
                backend=self.quantum_instance,
                include_custom=self._include_custom,
            )
        else:
            expectation = self.expectation

        optimal_circuits_list = [
            self.initialized_ansatz_list[n].assign_parameters(optimal_parameters)
            for n in range(self.num_states)
        ]

        observable_meas = expectation.convert(StateFn(operator, is_measurement=True))
        ansatz_circuit_op_list = [
            CircuitStateFn(optimal_circuits_list[n]) for n in range(self.num_states)
        ]
        expect_op_list = [
            observable_meas.compose(ansatz_circuit_op_list[n]).reduce()
            for n in range(self.num_states)
        ]
        sampled_expect_op_list = [
            self._circuit_sampler_list[n].convert(expect_op_list[n]) for n in range(self.num_states)
        ]
        list_of_means = np.asarray([np.real(sampled_expect_op_list[n].eval()) for n in range(self.num_states)])
        return list_of_means


class SSVQEResult(VariationalResult, EigensolverResult):
    """SSVQE Result."""

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
    def eigenstates(self) -> Optional[np.ndarray]:
        """return eigen states"""
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """set eigen state"""
        self._eigenstates = value


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
