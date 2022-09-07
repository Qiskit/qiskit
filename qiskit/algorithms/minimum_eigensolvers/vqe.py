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
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator


from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, Minimizer
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult


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
        ansatz: The parameterized circuit used as an ansatz for the wave function.
        optimizer: A classical optimizer to find the minimum energy. This can either be a
            Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
        estimator: The estimator primitive to compute the expectation value of the circuits.
        gradient: An optional gradient function or operator for the optimizer.
        initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            If not provided, a random initial point with values in the interval :math:`[0, 2\pi]`
            is used.
        callback: An optional callback function to plot the energy at each evaluation.

    References:
        [1] Peruzzo et al, "A variational eigenvalue solver on a quantum processor"
            `arXiv:1304.3061 https://arxiv.org/abs/1304.3061>`_
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        estimator: BaseEstimator,
        *,
        gradient: BaseEstimatorGradient | None = None,
        initial_point: Sequence[float] | None = None,
        # TODO remove callback and attach to optimizer instead
        callback: Callable[[int, np.ndarray, float, float], None] | None = None,
    ) -> None:
        """
        Args:
            ansatz: The parameterized circuit used as ansatz for the wave function.
            optimizer: The classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            estimator: The estimator primitive to compute the expectation value of the circuits.
            gradient: An optional gradient function or operator for the optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            callback: An optional callback function to plot the energy at each evaluation.
        """
        super().__init__()

        self.ansatz = ansatz
        self.optimizer = optimizer
        self.estimator = estimator
        self.gradient = gradient

        # TODO change VariationalAlgorithm interface to use a public attribute
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self._initial_point = initial_point

        # TODO remove callback and attach to optimizer instead
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
        ansatz = self._check_operator_ansatz(operator)

        initial_point = self.initial_point
        if initial_point is None:
            initial_point = np.random.random(ansatz.num_parameters)
        elif len(initial_point) != ansatz.num_parameters:
            raise ValueError(
                f"The dimension of the initial point ({len(self.initial_point)}) does not match "
                f"the number of parameters in the circuit ({ansatz.num_parameters})."
            )

        start_time = time()

        eval_energy = self.get_eval_energy(ansatz, operator)

        if self.gradient is not None:
            eval_gradient = self.get_eval_gradient(ansatz, operator)
        else:
            eval_gradient = None

        # perform optimization
        if callable(self.optimizer):
            opt_result = self.optimizer(fun=eval_energy, x0=initial_point, jac=eval_gradient)
        else:
            opt_result = self.optimizer.minimize(
                fun=eval_energy, x0=initial_point, jac=eval_gradient
            )

        eval_time = time() - start_time

        optimal_point = opt_result.x
        logger.info(
            f"Optimization complete in {eval_time} seconds.\nFound opt_params {optimal_point}."
        )

        aux_values = None
        if aux_operators:
            # not None and not empty list
            # TODO this is going to be replaced by eval_observables (see PR #8683)
            aux_values = self._eval_aux_ops(ansatz, optimal_point, aux_operators)

        result = VQEResult()
        result.eigenvalue = opt_result.fun + 0j
        result.aux_operator_eigenvalues = aux_values
        result.cost_function_evals = opt_result.nfev
        result.optimal_point = optimal_point
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimal_point))
        result.optimal_value = opt_result.fun
        result.optimizer_time = eval_time
        return result

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def get_eval_energy(
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
        iter_count = 0
        num_parameters = ansatz.num_parameters

        def eval_energy(parameters):
            nonlocal iter_count

            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batchsize = len(parameters)

            job = self.estimator.run(batchsize * [ansatz], batchsize * [operator], parameters)
            values = job.result().values
            iter_count += 1
            if self.callback is not None:
                self.callback(iter_count, parameters, values[0], 0.0)
            return values[0] if len(values) == 1 else values

        return eval_energy

    def get_eval_gradient(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator | PauliSumOp,
    ) -> tuple[Callable[[np.ndarray], np.ndarray]]:
        """Returns a function handle to evaluate the gradient at given parameters for the ansatz."""

        def eval_gradient(parameters):
            # broadcasting not required for the estimator gradients
            result = self.gradient.run([ansatz], [operator], [parameters]).result()
            return result.gradients[0]

        return eval_gradient

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp) -> QuantumCircuit:
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        ansatz = self.ansatz.copy()
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
        parameters: np.ndarray,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp],
    ) -> ListOrDict[tuple(complex, complex)]:
        """Compute auxiliary operator eigenvalues."""

        # TODO this is going to be replaced by eval_observables (see PR #8683)

        if isinstance(aux_operators, dict):
            aux_ops = list(aux_operators.values())
        else:
            non_nones = [i for i, x in enumerate(aux_operators) if x is not None]
            aux_ops = [x for x in aux_operators if x is not None]

        aux_values = None
        if aux_ops:
            num_aux_ops = len(aux_ops)
            aux_job = self.estimator.run(
                [ansatz] * num_aux_ops, aux_ops, [parameters] * num_aux_ops
            )
            aux_eigs = aux_job.result().values
            aux_eigs = list(zip(aux_eigs, [0] * len(aux_eigs)))
            if isinstance(aux_operators, dict):
                aux_values = dict(zip(aux_operators.keys(), aux_eigs))
            else:
                aux_values = [None] * len(aux_operators)
                for i, x in enumerate(non_nones):
                    aux_values[x] = aux_eigs[i]

        return aux_values


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
