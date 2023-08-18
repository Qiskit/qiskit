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

"""The Variational Quantum Eigensolver algorithm, optimized for diagonal Hamiltonians."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from time import time
from typing import Any

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseSampler
from qiskit.result import QuasiDistribution
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import Minimizer, Optimizer, OptimizerResult
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .diagonal_estimator import _DiagonalEstimator
from .sampling_mes import (
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
)
from ..observables_evaluator import estimate_observables
from ..utils import validate_initial_point, validate_bounds

# private function as we expect this to be updated in the next released
from ..utils.set_batching import _set_default_batchsize


logger = logging.getLogger(__name__)


class SamplingVQE(VariationalAlgorithm, SamplingMinimumEigensolver):
    r"""The Variational Quantum Eigensolver algorithm, optimized for diagonal Hamiltonians.

    VQE is a hybrid quantum-classical algorithm that uses a variational technique to find the
    minimum eigenvalue of a given diagonal Hamiltonian operator :math:`H_{\text{diag}}`.

    In contrast to the :class:`~qiskit.algorithms.minimum_eigensolvers.VQE` class, the
    ``SamplingVQE`` algorithm is executed using a :attr:`sampler` primitive.

    An instance of ``SamplingVQE`` also requires an :attr:`ansatz`, a parameterized
    :class:`.QuantumCircuit`, to prepare the trial state :math:`|\psi(\vec\theta)\rangle`. It also
    needs a classical :attr:`optimizer` which varies the circuit parameters :math:`\vec\theta` to
    minimize the objective function, which depends on the chosen :attr:`aggregation`.

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

    The above signature also allows one to use any SciPy minimizer, for instance as

    .. code-block:: python

        from functools import partial
        from scipy.optimize import minimize

        optimizer = partial(minimize, method="L-BFGS-B")

    The following attributes can be set via the initializer but can also be read and updated once
    the ``SamplingVQE`` object has been constructed.

    Attributes:
        sampler (BaseSampler): The sampler primitive to sample the circuits.
        ansatz (QuantumCircuit): A parameterized quantum circuit to prepare the trial state.
        optimizer (Optimizer | Minimizer): A classical optimizer to find the minimum energy. This
            can either be a Qiskit :class:`.Optimizer` or a callable implementing the
            :class:`.Minimizer` protocol.
        aggregation (float | Callable[[list[tuple[float, complex]], float] | None):
            A float or callable to specify how the objective function evaluated on the basis states
            should be aggregated. If a float, this specifies the :math:`\alpha \in [0,1]` parameter
            for a CVaR expectation value [1]. If a callable, it takes a list of basis state
            measurements specified as  ``[(probability, objective_value)]`` and return an objective
            value as float. If None, all an ordinary expectation value is calculated.
        callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback that
            can access the intermediate data at each optimization step. These data are: the
            evaluation count, the optimizer parameters for the ansatz, the evaluated value, and the
            metadata dictionary.

    References:
        [1]: Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., and Woerner, S.,
            "Improving Variational Quantum Optimization using CVaR"
            `arXiv:1907.04769 <https://arxiv.org/abs/1907.04769>`_
    """

    def __init__(
        self,
        sampler: BaseSampler,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        *,
        initial_point: Sequence[float] | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ) -> None:
        r"""
        Args:
            sampler: The sampler primitive to sample the circuits.
            ansatz: A parameterized quantum circuit to prepare the trial state.
            optimizer: A classical optimizer to find the minimum energy. This can either be a Qiskit
                :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
            initial_point: An optional initial point (i.e. initial parameter values) for the
                optimizer. The length of the initial point must match the number of :attr:`ansatz`
                parameters. If ``None``, a random point will be generated within certain parameter
                bounds. ``SamplingVQE`` will look to the ansatz for these bounds. If the ansatz does
                not specify bounds, bounds of :math:`-2\pi`, :math:`2\pi` will be used.
            aggregation: A float or callable to specify how the objective function evaluated on the
                basis states should be aggregated.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer parameters for the ansatz, the
                estimated value, and the metadata dictionary.
        """
        super().__init__()

        self.sampler = sampler
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.aggregation = aggregation
        self.callback = callback

        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self._initial_point = initial_point

    @property
    def initial_point(self) -> Sequence[float] | None:
        """Return the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        """Set the initial point."""
        self._initial_point = value

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

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> SamplingMinimumEigensolverResult:
        # check that the number of qubits of operator and ansatz match, and resize if possible
        self._check_operator_ansatz(operator)

        if len(self.ansatz.clbits) > 0:
            self.ansatz.remove_final_measurements()
        self.ansatz.measure_all()

        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)

        evaluate_energy, best_measurement = self._get_evaluate_energy(
            operator, self.ansatz, return_best_measurement=True
        )

        start_time = time()

        if callable(self.optimizer):
            optimizer_result = self.optimizer(fun=evaluate_energy, x0=initial_point, bounds=bounds)
        else:
            # we always want to submit as many estimations per job as possible for minimal
            # overhead on the hardware
            was_updated = _set_default_batchsize(self.optimizer)

            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy, x0=initial_point, bounds=bounds
            )

            # reset to original value
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s.",
            optimizer_time,
            optimizer_result.x,
        )

        final_state = self.sampler.run([self.ansatz], [optimizer_result.x]).result().quasi_dists[0]

        if aux_operators is not None:
            aux_operators_evaluated = estimate_observables(
                _DiagonalEstimator(sampler=self.sampler),
                self.ansatz,
                aux_operators,
                optimizer_result.x,
            )
        else:
            aux_operators_evaluated = None

        return self._build_sampling_vqe_result(
            self.ansatz.copy(),
            optimizer_result,
            aux_operators_evaluated,
            best_measurement,
            final_state,
            optimizer_time,
        )

    def _get_evaluate_energy(
        self,
        operator: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        return_best_measurement: bool = False,
    ) -> Callable[[np.ndarray], np.ndarray | float] | tuple[
        Callable[[np.ndarray], np.ndarray | float], dict[str, Any]
    ]:
        """Returns a function handle to evaluate the energy at given parameters.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            operator: The operator whose energy to evaluate.
            ansatz: The ansatz preparing the quantum state.
            return_best_measurement: If True, a handle to a dictionary containing the best
                measurement evaluated with the cost function.

        Returns:
            A tuple of a callable evaluating the energy and (optionally) a dictionary containing the
            best measurement of the energy evaluation.

        Raises:
            AlgorithmError: If the circuit is not parameterized (i.e. has 0 free parameters).

        """
        num_parameters = ansatz.num_parameters
        if num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has 0 free parameters.")

        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        best_measurement = {"best": None}

        def store_best_measurement(best):
            for best_i in best:
                if best_measurement["best"] is None or _compare_measurements(
                    best_i, best_measurement["best"]
                ):
                    best_measurement["best"] = best_i

        estimator = _DiagonalEstimator(
            sampler=self.sampler, callback=store_best_measurement, aggregation=self.aggregation
        )

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count
            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batch_size = len(parameters)

            estimator_result = estimator.run(
                batch_size * [ansatz], batch_size * [operator], parameters
            ).result()
            values = estimator_result.values

            if self.callback is not None:
                metadata = estimator_result.metadata
                for params, value, meta in zip(parameters, values, metadata):
                    eval_count += 1
                    self.callback(eval_count, params, value, meta)

            result = values if len(values) > 1 else values[0]
            return np.real(result)

        if return_best_measurement:
            return evaluate_energy, best_measurement

        return evaluate_energy

    def _build_sampling_vqe_result(
        self,
        ansatz: QuantumCircuit,
        optimizer_result: OptimizerResult,
        aux_operators_evaluated: ListOrDict[tuple[complex, tuple[complex, int]]],
        best_measurement: dict[str, Any],
        final_state: QuasiDistribution,
        optimizer_time: float,
    ) -> SamplingVQEResult:
        result = SamplingVQEResult()
        result.eigenvalue = optimizer_result.fun
        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated
        result.optimizer_result = optimizer_result
        result.best_measurement = best_measurement["best"]
        result.eigenstate = final_state
        result.optimal_circuit = ansatz
        return result


class SamplingVQEResult(VariationalResult, SamplingMinimumEigensolverResult):
    """VQE Result."""

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


def _compare_measurements(candidate, current_best):
    """Compare two best measurements. Returns True if the candidate is better than current value.

    This compares the following two criteria, in this precedence:

        1. The smaller objective value is better
        2. The higher probability for the objective value is better

    """
    if candidate["value"] < current_best["value"]:
        return True
    elif candidate["value"] == current_best["value"]:
        return candidate["probability"] > current_best["probability"]
    return False
