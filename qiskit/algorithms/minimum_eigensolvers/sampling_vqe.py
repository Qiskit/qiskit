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
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import Minimizer, Optimizer
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .diagonal_estimator import _DiagonalEstimator
from .sampling_mes import (
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
)


logger = logging.getLogger(__name__)


class SamplingVQE(VariationalAlgorithm, SamplingMinimumEigensolver):
    r"""The Variational Quantum Eigensolver algorithm, optimized for diagonal Hamiltonians."

    Attributes:
        sampler: The sampler primitive to sample the circuits.
        ansatz: A parameterized circuit, preparing the ansatz for the wave function. If not
            provided, this defaults to a :class:`.RealAmplitudes` circuit.
        optimizer: A classical optimizer to find the minimum energy. This can either be a
            Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
            Defaults to :class:`.SLSQP`.
        aggregation: A float or callable to specify how the objective function evaluated on the
            basis states should be aggregated. If a float, this specifies the :math:`\alpha \in [0,1]`
            parameter for a CVaR expectation value (see also [1]).
        callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback that
            can access the intermediate data at each optimization step. These data are: the
            evaluation count, the optimizer parameters for the ansatz, the evaluated value, and the
            metadata dictionary.

    References:

        [1] Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., and Woerner, S.,
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
        callback: Callable[[int, np.ndarray, float, dict[str, Any], dict[str, Any]], None]
        | None = None,
    ) -> None:
        r"""
        Args:
            sampler: The sampler primitive to sample the circuits.
            ansatz: The parameterized circuit used as ansatz for the wave function.
            optimizer: The classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
           initial_point: An optional initial point (i.e. initial parameter values) for the
                optimizer. The length of the initial point must match the number of :attr:`ansatz`
                parameters. If ``None``, a random point will be generated within certain parameter
                bounds. ``VQE`` will look to the ansatz for these bounds. If the ansatz does not
                specify bounds, bounds of :math:`-2\pi`, :math:`2\pi` will be used.
            aggregation: A float or callable to specify how the objective function evaluated on the
                basis states should be aggregated.
            callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback
                that can access the intermediate data at each optimization step. These data are: the
                evaluation count, the optimizer parameters for the ansatz, the evaluated value, the
                the metadata dictionary, and the best measurement.
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

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp) -> QuantumCircuit:
        """Check that the number of qubits of operator and ansatz match."""
        ansatz = self.ansatz.copy()

        if operator.num_qubits != ansatz.num_qubits:
            # try to set the number of qubits on the ansatz, if possible
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", {operator.num_qubits}
                )
                ansatz.num_qubits = operator.num_qubits
            except AttributeError as ex:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from ex

        return ansatz

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> SamplingMinimumEigensolverResult:
        # check that the number of qubits of operator and ansatz match, and resize if possible
        ansatz = self._check_operator_ansatz(operator)
        ansatz.measure_all()

        # TODO once VQE is merged, replace with validate_initial_point and validate_bounds.
        if self.initial_point is None:
            initial_point = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
        elif len(self.initial_point) != ansatz.num_parameters:
            raise ValueError(
                f"The dimension of the initial point ({len(self.initial_point)}) does not match the "
                f"number of parameters in the circuit ({ansatz.num_parameters})."
            )
        else:
            initial_point = self.initial_point

        # set an expectation for this algorithm run (will be reset to None at the end)
        # initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        evaluate_energy, best_measurement = self._get_evaluate_energy(
            operator, ansatz, return_best_measurement=True
        )

        start_time = time()

        if callable(self.optimizer):
            # pylint: disable=not-callable
            opt_result = self.optimizer(fun=evaluate_energy, x0=initial_point)
        else:
            opt_result = self.optimizer.minimize(fun=evaluate_energy, x0=initial_point)

        eval_time = time() - start_time

        final_state = self.sampler.run([ansatz], [opt_result.x]).result().quasi_dists

        result = SamplingVQEResult()
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(ansatz.parameters, opt_result.x))
        result.optimal_value = opt_result.fun
        result.cost_function_evals = opt_result.nfev
        result.optimizer_time = eval_time
        result.best_measurement = best_measurement["best"]
        result.eigenvalue = opt_result.fun + 0j
        result.eigenstate = final_state

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s.",
            eval_time,
            result.optimal_point,
        )

        if aux_operators is not None:
            result.aux_operator_values = self._eval_aux_ops(ansatz, opt_result.x, aux_operators)

        return result

    def _get_evaluate_energy(
        self,
        operator: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        return_best_measurement: bool = False,
    ) -> tuple[Callable[[np.ndarray], np.ndarray | float], dict]:
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
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).

        """
        num_parameters = ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        best_measurement = {"best": None}

        def store_best_measurement(best):
            for best_i in best:
                if best_measurement["best"] is None or _compare_measurements(
                    best_i, best_measurement["best"]
                ):
                    best_measurement["best"] = best_i

        estimator = _DiagonalEstimator(sampler=self.sampler, callback=store_best_measurement)

        def evaluate_energy(parameters):
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
                    self.callback(eval_count, params, value, meta, best_measurement)

            result = values if len(values) > 1 else values[0]
            return np.real(result)

        if return_best_measurement:
            return evaluate_energy, best_measurement

        return evaluate_energy

    def _eval_aux_ops(self, ansatz, parameters, aux_operators):
        # convert to list if necessary and store the keys
        if isinstance(aux_operators, dict):
            is_dict = True
            keys = list(aux_operators.keys())
            aux_operators = list(aux_operators.values())
        else:
            is_dict = False

        # evaluate all aux operators
        num = len(aux_operators)
        estimator = _DiagonalEstimator(sampler=self.sampler)
        results = estimator.run(num * [ansatz], aux_operators, num * [parameters]).result()
        values = list(results.values)  # convert array to list

        # bring back into the right shape and return
        if is_dict:
            return dict(zip(keys, values))

        return values


class SamplingVQEResult(VariationalResult, SamplingMinimumEigensolverResult):
    """VQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

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
