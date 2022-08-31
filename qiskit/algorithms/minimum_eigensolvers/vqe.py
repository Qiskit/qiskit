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
from dataclasses import dataclass
import logging
from time import time

import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import (
    OperatorBase,
)
from qiskit.primitives import BaseEstimator
from qiskit.utils.validation import validate_min

from ..exceptions import AlgorithmError
from ..optimizers import Optimizer, Minimizer, SLSQP
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult

logger = logging.getLogger(__name__)


class VQE(MinimumEigensolver):
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

    The above signature also allows to directly pass any SciPy minimizer, for instance as

    .. code-block:: python

        from functools import partial
        from scipy.optimize import minimize

        optimizer = partial(minimize, method="L-BFGS-B")

    Attributes:
        estimator: The estimator primitive to compute the expectation value of the circuits.
        ansatz: A parameterized circuit, preparing the ansatz for the wave function. If not
            provided, this defaults to a :class:`.RealAmplitudes` circuit.
        optimizer: A classical optimizer to find the minimum energy. This can either be a
            Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
            Defaults to :class:`.SLSQP`.
        initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            If not provided, a random initial point with values in the interval :math:`[0, 2\pi]`
            is used.
        max_evals_grouped: Specifies how many parameter sets can be evaluated simultaneously.
            This information is forwarded to the optimizer, which can use it for batch evaluation.

    References:
        [1] Peruzzo et al, "A variational eigenvalue solver on a quantum processor"
            `arXiv:1304.3061 https://arxiv.org/abs/1304.3061>`_
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        gradient=None,
        initial_point: np.ndarray | None = None,
        max_evals_grouped: int = 1,
        # TODO Attach callback to optimizer instead.
        callback=None,
    ) -> None:
        """
        Args:
            estimator: The estimator primitive to compute the expectation value of the circuits.
            ansatz: The parameterized circuit used as ansatz for the wave function.
            optimizer: The classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            gradient: An optional gradient function or operator for optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
        """
        super().__init__()

        validate_min("max_evals_grouped", max_evals_grouped, 1)

        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        self.initial_point = initial_point
        self.max_evals_grouped = max_evals_grouped
        self.callback = callback

        # TODO remove this
        self._eval_count = 0

    def compute_minimum_eigenvalue(self, operator: OperatorBase, aux_operators=None):
        # Set defaults
        if self.ansatz is None:
            ansatz = RealAmplitudes(num_qubits=operator.num_qubits)
        else:
            ansatz = self.ansatz.copy()
            _check_operator_ansatz(operator, ansatz)

        if self.optimizer is None:
            optimizer = SLSQP()
        else:
            optimizer = self.optimizer

        if isinstance(aux_operators, dict):
            aux_ops = list(aux_operators.values())
        elif aux_operators:
            # Not None and not empty list
            non_nones = [i for i, x in enumerate(aux_operators) if x is not None]
            aux_ops = [x for x in aux_operators if x is not None]
        else:
            aux_ops = None

        operators = [operator] + ([] if aux_ops is None else aux_ops)

        self._eval_count = 0

        def energy(point):
            job = self.estimator.run([ansatz], [operator], [point])
            return job.result().values[0]

        # def gradient(point):
        #     job = self.gradient.run([self.ansatz], [operator], [point])
        #     return job.result()

        def expectation(point):
            value = energy(point)
            self._eval_count += 1
            if self.callback is not None:
                self.callback(self._eval_count, point, value, 0)
            return value

        initial_point = self.initial_point
        if not initial_point:
            initial_point = np.random.random(ansatz.num_parameters)

        start_time = time()

        # Perform optimization
        if callable(optimizer):
            opt_result = optimizer(  # pylint: disable=not-callable
                fun=expectation, x0=initial_point  # , jac=gradient, bounds=bounds
            )
        else:
            opt_result = optimizer.minimize(
                fun=expectation, x0=initial_point  # , jac=gradient, bounds=bounds
            )

        eval_time = time() - start_time

        optimal_point = opt_result.x
        logger.info(
            f"Optimization complete in {eval_time} seconds.\nFound opt_params {optimal_point}."
        )

        # Compute auxiliary operator eigenvalues
        aux_values = None
        if aux_ops:
            num_aux_ops = len(aux_ops)
            aux_job = self.estimator.run(
                [ansatz] * num_aux_ops, operators[1:], [optimal_point] * num_aux_ops
            )
            aux_eigs = aux_job.result().values
            aux_eigs = list(zip(aux_eigs, [0] * len(aux_eigs)))
            if isinstance(aux_operators, dict):
                aux_values = dict(zip(aux_operators.keys(), aux_eigs))
            else:
                aux_values = [None] * len(aux_operators)
                for i, x in enumerate(non_nones):
                    aux_values[x] = aux_eigs[i]

        result = VQEResult(
            eigenvalue=opt_result.fun + 0j,
            cost_function_evals=opt_result.nfev,
            optimal_point=optimal_point,
            optimal_parameters=dict(zip(self.ansatz.parameters, optimal_point)),
            optimal_value=opt_result.fun,
            optimizer_time=eval_time,
            aux_operator_eigenvalues=aux_values,
            # TODO Add variances for the eigenvalues.
        )

        return result

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True


def _check_operator_ansatz(operator: OperatorBase, ansatz: QuantumCircuit) -> QuantumCircuit:
    """Check that the number of qubits of operator and ansatz match and that the ansatz is
    parameterized."""
    if operator.num_qubits != ansatz.num_qubits:
        # Try to set the number of qubits on the ansatz.
        try:
            logger.info(
                f"Trying to resize ansatz to match operator on {operator.num_qubits} qubits."
            )
            ansatz.num_qubits = operator.num_qubits
        except AttributeError as ex:
            raise AlgorithmError(
                "The number of qubits of the ansatz does not match the "
                "operator, and the ansatz does not allow setting the "
                "number of qubits using `num_qubits`."
            ) from ex

    if ansatz.num_parameters == 0:
        raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

    return ansatz


@dataclass(frozen=True)
class VQEResult(MinimumEigensolverResult):
    "Variational quantum eigensolver result."
    cost_function_evals: int
    optimal_point: np.ndarray
    optimal_parameters: dict[Parameter, float]
    optimal_value: float
    optimizer_time: float
