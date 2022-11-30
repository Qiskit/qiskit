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

See https://arxiv.org/abs/1810.09434.
"""

from __future__ import annotations

import logging
import warnings
from time import time
from collections.abc import Callable, Sequence

import numpy as np

from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import Statevector

from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, Minimizer, OptimizerResult
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .eigensolver import Eigensolver, EigensolverResult

from ..observables_evaluator import estimate_observables

logger = logging.getLogger(__name__)


class SSVQE(VariationalAlgorithm, Eigensolver):
    r"""The Subspace Search Variational Quantum Eigensolver algorithm.
    `SSVQE <https://arxiv.org/abs/1810.09434>`__ is a quantum algorithm
    that uses a variational technique to find the low-lying eigenvalues
    of the Hamiltonian :math:`H` of a given system. SSVQE can be seen as
    a natural generalization of VQE. Whereas VQE minimizes the expectation
    value of :math:`H` with respect to one ansatz state, SSVQE takes a set
    of mutually orthogonal input states, applies the same parameterized ansatz
    to all of them, then minimizes a weighted sum of the expectation values of
    :math:`H` with respect to these states. An instance of SSVQE requires defining
    three algorithmic sub-components:

    An integer k denoting the number of eigenstates that the algorithm will attempt
    to find, a trial state (a.k.a. ansatz) which is a :class:`QuantumCircuit`, and
    one of the classical :mod:`~qiskit.algorithms.optimizers`. The ansatz is varied,
    via its set of parameters, by the optimizer, such that it works towards a set of
    mutually orthogonal states, as determined by the parameters applied to the ansatz,
    that will result in the minimum weighted sum of expectation values being measured
    of the input operator (Hamiltonian) with respect to these states. The weights given
    to this list of expectation values is given by ``weight_vector``. An optional array of
    parameter values, via the ``initial_point``, may be provided as the starting point for
    the search of the low-lying eigenvalues. This feature is particularly useful such as
    when there are reasons to believe that the solution point is close to a particular
    point. The length of the ``initial_point`` list value must match the number of the
    parameters expected by the ansatz being used. If the ``initial_point`` is left at the
    default of ``None``, then SSVQE will look to the ansatz for a preferred value, based
    on its given initial state. If the ansatz returns ``None``, then a random point will
    be generated within the parameter bounds set, as per above. If the ansatz provides
    ``None`` as the lower bound, then SSVQE will default it to :math:`-2\pi`; similarly,
    if the ansatz returns ``None`` as the upper bound, the default value will be :math:`2\pi`.

    An optional list of initial states, via the ``initial_states``, may also be provided.
    Choosing these states appropriately is a critical part of the algorithm. They must
    be mutually orthogonal because this is how the algorithm enforces the mutual
    orthogonality of the solution states. If the ``initial_states`` is left as ``None``,
    then SSVQE will automatically generate a list of computational basis states and use
    these as the initial states. For many physically-motivated problems, it is advised
    to not rely on these default values as doing so can easily result in an unphysical
    solution being returned. For example, if one wishes to find the low-lying excited
    states of a molecular Hamiltonian, then we expect the output states to belong to a
    particular particle-number subspace. If an ansatz that preserves particle number
    such as :class:`UCCSD` is used, then states belonging to the incorrect particle
    number subspace will be returned if the ``initial_states`` are not in the correct
    particle number subspace. A similar statement can often be made for the
    spin-magnetization quantum number.

    The following attributes can be set via the initializer but can also be read and
    updated once the SSVQE object has been constructed.

    Attributes:
            estimator (BaseEstimator): The primitive instance used to perform the expectation
                estimation of observables.
            k (int): The number of eigenstates that SSVQE will attempt to find.
            ansatz (QuantumCircuit): A parameterized circuit used as an ansatz for the
                wave function.
            optimizer (Optimizer): A classical optimizer, which can be either a Qiskit optimizer
                or a callable that takes an array as input and returns a Qiskit or SciPy optimization
                result.
            gradient (BaseEstimatorGradient | None): An optional estimator gradient to be used with the
                optimizer.
            initial_states (Sequence[QuantumCircuit]): An optional list of mutually orthogonal
                initial states. If ``None``, then SSVQE will set these to be a list of mutually
                orthogonal computational basis states.
            weight_vector (Sequence[float]): A 1D array of real positive-valued numbers to assign
                as weights to each of the expectation values. If ``None``, then SSVQE will default
                to [k, k-1, ..., 1].
            callback (Callable[[int, np.ndarray, Sequence[float], dict[str, Any]], None] | None): A
                function that can access the intermediate data at each optimization step. These data are
                the evaluation count, the optimizer parameters for the ansatz, the evaluated mean
                energies, and the metadata dictionary.
            check_input_states_orthogonality: A boolean that sets whether or not to check
                that the value of initial_states passed consists of a mutually orthogonal
                set of states. If ``True``, then SSVQE will check that these states are mutually
                orthogonal and return an error if they are not. This is set to ``True`` by default,
                but setting this to ``False`` may be desirable for larger numbers of qubits to avoid
                exponentially large computational overhead before the simulation even starts.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        k: int | None = 2,
        ansatz: QuantumCircuit | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        initial_point: Sequence[float] | None = None,
        initial_states: list[QuantumCircuit] | None = None,
        weight_vector: Sequence[float] | Sequence[int] | None = None,
        gradient: BaseEstimatorGradient | None = None,
        callback: Callable[[int, np.ndarray, Sequence[float], float], None] | None = None,
        check_input_states_orthogonality: bool = True,
    ) -> None:
        """
        Args:
            estimator: The estimator primitive.
            k: The number of eigenstates that the algorithm will attempt to find.
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
                equal to the value of ``num_states`` to be used in the weighted energy summation
                objective function. This fixes the ordering of the returned eigenstate/eigenvalue
                pairs. If ``None``, then SSVQE will default to [n, n-1, ..., 1] for `k` = n.
            gradient: An optional gradient function or operator for optimizer.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer ansatz parameters,
                the evaluated mean energies, and the metadata dictionary.
            check_input_states_orthogonality: A boolean that sets whether or not to check
                that the value of ``initial_states`` passed consists of a mutually orthogonal
                set of states. If ``True``, then SSVQE will check that these states are mutually
                orthogonal and return an error if they are not. This is set to ``True`` by default,
                but setting this to ``False`` may be desirable for larger numbers of qubits to avoid
                exponentially large computational overhead before the simulation even starts.
        """

        super().__init__()

        self.k = k
        self.initial_states = initial_states
        self.weight_vector = weight_vector
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_point = initial_point
        self.gradient = gradient
        self.callback = callback
        self.estimator = estimator
        self.check_initial_states_orthogonal = check_input_states_orthogonality

    @property
    def initial_point(self) -> Sequence[float] | None:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Sequence[float] | None):
        """Sets initial point"""
        self._initial_point = initial_point

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_eigenvalues(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> EigensolverResult:

        ansatz = self._check_operator_ansatz(operator)

        initial_point = _validate_initial_point(self.initial_point, ansatz)

        initial_states = self._check_operator_initial_states(self.initial_states, operator)

        bounds = _validate_bounds(ansatz)

        initialized_ansatz_list = [initial_states[n].compose(ansatz) for n in range(self.k)]

        self.weight_vector = self._check_weight_vector(self.weight_vector)

        evaluate_weighted_energy_sum = self._get_evaluate_weighted_energy_sum(
            initialized_ansatz_list, operator
        )

        if self.gradient is not None:  # need to implement _get_evaluate_gradient
            evaluate_gradient = self._get_evalute_gradient(initialized_ansatz_list, operator)
        else:
            evaluate_gradient = None

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

        start_time = time()

        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_weighted_energy_sum,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )
        else:
            optimizer_result = self.optimizer.minimize(
                fun=evaluate_weighted_energy_sum,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s",
            optimizer_time,
            optimizer_result.x,
        )

        if aux_operators is not None:
            bound_ansatz_list = [
                initialized_ansatz_list[n].bind_parameters(optimizer_result.x)
                for n in range(self.k)
            ]

            aux_values_list = [
                estimate_observables(
                    self.estimator,
                    bound_ansatz_list[n],
                    aux_operators,
                )
                for n in range(self.k)
            ]
        else:
            aux_values_list = None

        return self._build_ssvqe_result(
            optimizer_result, aux_values_list, optimizer_time, operator, initialized_ansatz_list
        )

    def _get_evaluate_weighted_energy_sum(
        self,
        initialized_ansatz_list: list[QuantumCircuit],
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], float | list[float]], dict]:
        """Returns a function handle to evaluate the weighted energy sum at given parameters
        for the ansatz. This is the objective function to be passed to the optimizer
        that is used for evaluation.
        Args:
            initialized_anastz_list: A list consisting of copies of the ansatz initialized
                in the initial states.
            operator: The operator whose expectation value with respect to each of the
                states in ``initialzed_ansatz_list`` is being measured.
        Returns:
            Weighted expectation value sum of the operator for each parameter.
        Raises:
            AlgorithmError: If the primitive job to evaluate the weighted energy
                sum fails.
        """
        num_parameters = initialized_ansatz_list[0].num_parameters

        eval_count = 0

        def evaluate_weighted_energy_sum(parameters):
            nonlocal eval_count
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batchsize = len(parameters)

            try:
                job = self.estimator.run(
                    [initialized_ansatz_list[m] for n in range(batchsize) for m in range(self.k)],
                    [operator] * self.k * batchsize,
                    [parameters[n] for n in range(batchsize) for m in range(self.k)],
                )
                result = job.result()
                values = result.values

                energies = np.reshape(values, (batchsize, self.k))
                weighted_energy_sums = np.dot(energies, self.weight_vector).tolist()
                energies = energies.tolist()

            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            if self.callback is not None:
                metadata = result.metadata
                for params, energies_value, metadata in zip(parameters, energies, metadata):
                    eval_count += 1
                    self.callback(eval_count, params, energies_value, metadata)

            return (
                weighted_energy_sums[0] if len(weighted_energy_sums) == 1 else weighted_energy_sums
            )

        return evaluate_weighted_energy_sum

    def _get_evalute_gradient(
        self,
        initialized_ansatz_list: list[QuantumCircuit],
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], np.ndarray]]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.
        Args:
            initialized_ansatz_list: The list of initialized ansatz preparing the quantum states.
            operator: The operator whose energy to evaluate.
        Returns:
            A function handle to evaluate the gradient at given parameters for the initialized
            ansatz list.
        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """

        def evaluate_gradient(parameters):
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run(
                    initialized_ansatz_list,
                    [operator] * self.k,
                    [parameters for n in range(self.k)],
                )
                energy_gradients = job.result().gradients
                weighted_energy_sum_gradient = sum(
                    [self.weight_vector[n] * energy_gradients[n] for n in range(self.k)]
                )
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            return weighted_energy_sum_gradient

        return evaluate_gradient

    def _check_circuit_num_qubits(
        self, operator: BaseOperator | PauliSumOp, circuit: QuantumCircuit, circuit_type: str
    ) -> QuantumCircuit:
        """Check that the number of qubits for the circuit passed matches
        the number of qubits  of the operator.
        """
        if operator.num_qubits != circuit.num_qubits:
            try:
                logger.info(
                    "Trying to resize %s to match operator on %s qubits.",
                    circuit_type,
                    operator.num_qubits,
                )
                circuit.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    f"The number of qubits of the {circuit_type} does not match the ",
                    f"operator, and the {circuit_type} does not allow setting the "
                    "number of qubits using `num_qubits`.",
                ) from error
        return circuit

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp) -> QuantumCircuit:
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        # set defaults
        if self.ansatz is None:
            ansatz = RealAmplitudes(num_qubits=operator.num_qubits, reps=6)
        else:
            ansatz = self.ansatz

        ansatz = self._check_circuit_num_qubits(
            operator=operator, circuit=ansatz, circuit_type="ansatz"
        )

        if ansatz.num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

        return ansatz

    def _check_operator_initial_states(
        self, list_of_states: list[QuantumCircuit] | None, operator: BaseOperator | PauliSumOp
    ) -> QuantumCircuit:

        """Check that the number of qubits of operator and all the initial states match."""

        if list_of_states is None:
            initial_states = [QuantumCircuit(operator.num_qubits) for n in range(self.k)]
            for n in range(self.k):
                initial_states[n].initialize(Statevector.from_int(n, 2**operator.num_qubits))

            warnings.warn(
                "No initial states have been provided to SSVQE, so they have been set to "
                "a subset of the computational basis states. This may result in unphysical "
                "results for some chemistry and physics problems."
            )

        else:
            initial_states = list_of_states
            if self.check_initial_states_orthogonal is True:
                stacked_states_array = np.hstack(
                    [np.asarray(Statevector(state)) for state in initial_states]
                )
                if not np.isclose(
                    stacked_states_array.transpose() @ stacked_states_array, np.eye(self.k)
                ):
                    raise AlgorithmError(
                        "The set of initial states provided is not mutually orthogonal."
                    )

        for initial_state in initial_states:
            initial_state = self._check_circuit_num_qubits(
                operator=operator, circuit=initial_state, circuit_type="initial state"
            )

        return initial_states

    def _check_weight_vector(self, weight_vector: Sequence[float]) -> Sequence[float]:
        """Check that the number of weights matches the number of states."""
        if weight_vector is None:
            weight_vector = [self.k - n for n in range(self.k)]
        elif len(weight_vector) != self.k:
            raise AlgorithmError(
                "The number of weights provided does not match the number of states."
            )

        return weight_vector

    def _eval_aux_ops(
        self,
        ansatz: QuantumCircuit,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp],
    ) -> ListOrDict[tuple(complex, complex)]:
        """Compute auxiliary operator eigenvalues."""

        if isinstance(aux_operators, dict):
            aux_ops = list(aux_operators.values())
        else:
            aux_ops = aux_operators

        num_aux_ops = len(aux_ops)

        try:
            aux_job = self.estimator.run([ansatz] * num_aux_ops, aux_ops)
            aux_values = aux_job.result().values
            aux_values = list(zip(aux_values, [0] * len(aux_values)))

            if isinstance(aux_operators, dict):
                aux_values = dict(zip(aux_operators.keys(), aux_values))

        except Exception as exc:
            raise AlgorithmError(
                "The primitive job to evaluate the aux operator values failed!"
            ) from exc

        return aux_values

    def _build_ssvqe_result(
        self,
        optimizer_result: OptimizerResult,
        aux_operators_evaluated: ListOrDict[tuple[complex, tuple[complex, int]]],
        optimizer_time: float,
        operator: BaseOperator | PauliSumOp,
        initialized_ansatz_list: list[QuantumCircuit],
    ) -> SSVQEResult:
        result = SSVQEResult()

        try:
            result.eigenvalues = (
                self.estimator.run(
                    initialized_ansatz_list, [operator] * self.k, [optimizer_result.x] * self.k
                )
                .result()
                .values
            )

        except Exception as exc:
            raise AlgorithmError("The primitive job to evaluate the eigenvalues failed!") from exc

        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated
        result.optimizer_result = optimizer_result

        return result


class SSVQEResult(VariationalResult, EigensolverResult):
    """SSVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

    @property
    def cost_function_evals(self) -> int:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value


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
