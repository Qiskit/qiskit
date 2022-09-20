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

"""The variational quantum eigensolver algorithm."""

from __future__ import annotations

import logging
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

from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, Minimizer
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult

from ..observables_evaluator import estimate_observables

logger = logging.getLogger(__name__)


class VQE(VariationalAlgorithm, MinimumEigensolver):
    r"""The variational quantum eigensolver (VQE) algorithm.

    VQE is a quantum algorithm that uses a variational technique to find the minimum eigenvalue of
    the Hamiltonian :math:`H` of a given system [1].

    The central sub-component of VQE is an Estimator primitive, which must be passed in as the
    first argument. This is used to estimate the expectation values of :math:`H`.

    An instance of VQE also requires defining two algorithmic sub-components: a trial state (a.k.a.
    ansatz) which is a :class:`QuantumCircuit`, and one of the classical
    :mod:`~qiskit.algorithms.optimizers`.

    The ansatz is varied, via its set of parameters, by the optimizer, such that it works towards a
    state, as determined by the parameters applied to the ansatz, that will result in the minimum
    expectation value being measured of the input operator (Hamiltonian).

    The optimizer can either be one of Qiskit's optimizers, such as
    :class:`~qiskit.algorithms.optimizers.SPSA` or a callable with the following signature:

    .. code-block:: python

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

    The above signature also allows one to directly pass any SciPy minimizer, for instance as

    .. code-block:: python

        from functools import partial
        from scipy.optimize import minimize

        optimizer = partial(minimize, method="L-BFGS-B")

    Attributes:
        estimator (BaseEstimator): The estimator primitive to compute the expectation value of the
            circuits.
        ansatz (Quantum Circuit): A parameterized circuit, preparing the ansatz for the wave
            function.
        optimizer (Optimizer | Minimizer): A classical optimizer to find the minimum energy. This
            can either be a Qiskit :class:`.Optimizer` or a callable implementing the
            :class:`.Minimizer` protocol.
        gradient (BaseEstimatorGradient | None): An optional estimator gradient to be used with the
            optimizer.
        initial_point (Sequence[float] | None): An optional initial point (i.e. initial parameter
            values) for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
            point and if not will simply compute a random one.
        callback (Callable[[int, np.ndarray, float, float], None] | None): A callback that can
            access the intermediate data during the optimization. Four parameter values are passed
            to the callback as follows during each evaluation by the optimizer for its current set
            of parameters as it works towards the minimum. These are: the evaluation count, the
            optimizer parameters for the ansatz, the evaluated mean and the evaluated standard
            deviation.

    References:
        [1] Peruzzo et al, "A variational eigenvalue solver on a quantum processor"
            `arXiv:1304.3061 https://arxiv.org/abs/1304.3061>`_
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        *,
        gradient: BaseEstimatorGradient | None = None,
        initial_point: Sequence[float] | None = None,
        callback: Callable[[int, np.ndarray, float, float], None] | None = None,
    ) -> None:
        """
        Args:
            estimator: The estimator primitive to compute the expectation value of the circuits.
            ansatz: A parameterized circuit, preparing the ansatz for the wave function.
            optimizer: A classical optimizer to find the minimum energy. This can either be a
                Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer`
                protocol.
            gradient: An optional estimator gradient to be used with the optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            callback: A callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the ansatz, the
                evaluated mean and the evaluated standard deviation.
        """
        super().__init__()

        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback

    @property
    def initial_point(self) -> Sequence[float] | None:
        """Return the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        """Set the initial point."""
        self._initial_point = value

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> VQEResult:
        self._check_operator_ansatz(operator)

        initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        bounds = _validate_bounds(self.ansatz)

        start_time = time()

        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)

        if self.gradient is not None:
            evaluate_gradient = self._get_evalute_gradient(self.ansatz, operator)
        else:
            evaluate_gradient = None

        # perform optimization
        if callable(self.optimizer):
            opt_result = self.optimizer(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )
        else:
            opt_result = self.optimizer.minimize(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )

        eval_time = time() - start_time

        result = VQEResult()
        result.eigenvalue = opt_result.fun
        result.cost_function_evals = opt_result.nfev
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        result.optimal_value = opt_result.fun
        result.optimizer_time = eval_time

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s",
            eval_time,
            result.optimal_point,
        )

        if aux_operators is not None:
            bound_ansatz = self.ansatz.bind_parameters(opt_result.x)
            aux_values = estimate_observables(self.estimator, bound_ansatz, aux_operators)
            result.aux_operator_eigenvalues = aux_values

        return result

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _get_evaluate_energy(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], float | list[float]], dict]:
        """Returns a function handle to evaluate the energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation.
        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.
        Returns:
            Energy of the Hamiltonian of each parameter.
        Raises:
            AlgorithmError: If the primitive job to evaluate the energy fails.
        """
        num_parameters = ansatz.num_parameters

        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        def evaluate_energy(parameters):
            nonlocal eval_count

            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batchsize = len(parameters)

            try:
                job = self.estimator.run(batchsize * [ansatz], batchsize * [operator], parameters)
                values = job.result().values
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            # TODO recover variance from estimator if has metadata has shots?

            if self.callback is not None:
                for params, value in zip(parameters, values):
                    eval_count += 1
                    self.callback(eval_count, params, value, 0.0)

            return values[0] if len(values) == 1 else values

        return evaluate_energy

    def _get_evalute_gradient(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], np.ndarray]]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A function handle to evaluate the gradient at given parameters for the ansatz.

        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """

        def evaluate_gradient(parameters):
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run([ansatz], [operator], [parameters])
                gradients = job.result().gradients
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            return gradients[0]

        return evaluate_gradient

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp):
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        if operator.num_qubits != self.ansatz.num_qubits:
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", operator.num_qubits
                )
                self.ansatz.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from error

        if self.ansatz.num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")


def _validate_initial_point(point: Sequence[float], ansatz: QuantumCircuit) -> Sequence[float]:
    expected_size = ansatz.num_parameters

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


def _validate_bounds(ansatz: QuantumCircuit) -> list[tuple(float | None, float | None)]:
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


class VQEResult(VariationalResult, MinimumEigensolverResult):
    """Variational quantum eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

    @property
    def cost_function_evals(self) -> int | None:
        """Returns number of cost optimizer evaluations."""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations."""
        self._cost_function_evals = value
