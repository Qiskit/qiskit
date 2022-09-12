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
from ..optimizers import Optimizer, Minimizer, SLSQP
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult

# from qiskit.algorithms.observables_evaluator import eval_observables

logger = logging.getLogger(__name__)


class VQE(VariationalAlgorithm, MinimumEigensolver):
    r"""The variational quantum eigensolver (VQE) algorithm.

    VQE is a quantum algorithm that uses a variational technique to find the minimum eigenvalue of
    the Hamiltonian :math:`H` of a given system [1].

    An instance of VQE requires defining two algorithmic sub-components: a trial state (a.k.a.
    ansatz) which is a :class:`QuantumCircuit`, and one of the classical
    :mod:`~qiskit.algorithms.optimizers`.

    The ansatz is varied, via its set of parameters, by the optimizer, such that it works towards a
    state, as determined by the parameters applied to the ansatz, that will result in the minimum
    expectation value being measured of the input operator (Hamiltonian).

    The optimizer can either be one of Qiskit's optimizers, such as
    :class:`~qiskit.algorithms.optimizers.SPSA` or a callable with the following signature:

    .. note::

        The callable _must_ have the argument names ``fun, x0, jac, bounds`` as indicated
        in the following code block.

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
        estimator: The estimator primitive to compute the expectation value of the circuits.
        ansatz: The parameterized circuit used as an ansatz for the wave function.
        optimizer: A classical optimizer to find the minimum energy. This can either be a
            Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
        gradient: An optional gradient function or operator for the optimizer.
        initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            If not provided, a random initial point with values in the interval :math:`[0, 2\pi]`
            is used.

    References:
        [1] Peruzzo et al, "A variational eigenvalue solver on a quantum processor"
            `arXiv:1304.3061 https://arxiv.org/abs/1304.3061>`_
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        gradient: BaseEstimatorGradient | None = None,
        initial_point: Sequence[float] | None = None,
    ) -> None:
        """
        Args:
            estimator: The estimator primitive to compute the expectation value of the circuits.
            ansatz: A parameterized circuit, preparing the ansatz for the wave function. If not
                provided, this defaults to a :class:`.RealAmplitudes` circuit.
            optimizer: A classical optimizer to find the minimum energy. This can either be a
                Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
                Defaults to :class:`.SLSQP`.
            gradient: An optional gradient function or operator for the optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
        """
        super().__init__()

        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient

        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point

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
        # set defaults
        if self.ansatz is None:
            ansatz = RealAmplitudes(num_qubits=operator.num_qubits)
        else:
            ansatz = self.ansatz

        ansatz = self._check_operator_ansatz(operator, ansatz)

        optimizer = SLSQP() if self.optimizer is None else self.optimizer

        initial_point = _validate_initial_point(self.initial_point, ansatz)

        start_time = time()

        energy_evaluation = self._get_energy_evaluation(ansatz, operator)

        if self.gradient is not None:
            gradient_evaluation = self._get_gradient_evaluation(ansatz, operator)
        else:
            gradient_evaluation = None

        # perform optimization
        if callable(optimizer):
            opt_result = optimizer(
                fun=energy_evaluation, x0=initial_point, jac=gradient_evaluation
            )
        else:
            opt_result = optimizer.minimize(
                fun=energy_evaluation, x0=initial_point, jac=gradient_evaluation
            )

        eval_time = time() - start_time

        result = VQEResult()
        result.eigenvalue = opt_result.fun + 0j
        result.cost_function_evals = opt_result.nfev
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(ansatz.parameters, opt_result.x))
        result.optimal_value = opt_result.fun
        result.optimizer_time = eval_time

        logger.info(
            f"Optimization complete in {eval_time} seconds.\nFound opt_params {result.optimal_point}."
        )

        if aux_operators:
            # not None and not empty list or dict
            bound_ansatz = ansatz.bind_parameters(opt_result.x)
            # aux_values = eval_observables(self.estimator, bound_ansatz, aux_operators)
            # TODO remove once eval_operators have been ported.
            aux_values = self._eval_aux_ops(bound_ansatz, aux_operators)
            result.aux_operator_eigenvalues = aux_values

        return result

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _get_energy_evaluation(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], float | list[float]], dict]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation.
        Args:
            operator: The operator whose energy to evaluate.
            ansatz: The ansatz preparing the quantum state.
        Returns:
            Energy of the hamiltonian of each parameter.
        """
        num_parameters = ansatz.num_parameters

        def energy_evaluation(parameters):
            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batchsize = len(parameters)

            job = self.estimator.run(batchsize * [ansatz], batchsize * [operator], parameters)
            values = job.result().values
            return values[0] if len(values) == 1 else values

        return energy_evaluation

    def _get_gradient_evaluation(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], np.ndarray]]:
        """Returns a function handle to evaluate the gradient at given parameters for the ansatz."""

        def gradient_evaluation(parameters):
            # broadcasting not required for the estimator gradients
            result = self.gradient.run([ansatz], [operator], [parameters]).result()
            return result.gradients[0]

        return gradient_evaluation

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp, ansatz: QuantumCircuit) -> QuantumCircuit:
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        if operator.num_qubits != ansatz.num_qubits:
            try:
                logger.info(
                    f"Trying to resize ansatz to match operator on {operator.num_qubits} qubits."
                )
                ansatz.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from error

        if ansatz.num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

        return ansatz

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
        aux_job = self.estimator.run([ansatz] * num_aux_ops, aux_ops)
        aux_values = aux_job.result().values
        aux_values = list(zip(aux_values, [0] * len(aux_values)))

        if isinstance(aux_operators, dict):
            aux_values = dict(zip(aux_operators.keys(), aux_values))

        return aux_values


def _validate_initial_point(point, ansatz):
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
