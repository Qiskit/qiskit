# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
The Variational Quantum Algorithm Base Class.

This class can be used an interface for working with Variation Quantum Algorithms, such as VQE,
QAOA, or QSVM, and also provides helper utilities for implementing new variational algorithms.
Writing a new variational algorithm is a simple as extending this class, implementing a cost
function for the new algorithm to pass to the optimizer, and running :meth:`find_minimum` method
of this class to carry out the optimization. Alternatively, all of the functions below can be
overridden to opt-out of this infrastructure but still meet the interface requirements.
"""

from typing import Optional, Callable
import time
import logging
import warnings
from abc import abstractmethod
import numpy as np

from qiskit.aqua.algorithms import AlgorithmResult, QuantumAlgorithm
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class VQAlgorithm(QuantumAlgorithm):
    """
    The Variational Quantum Algorithm Base Class.
    """
    def __init__(self,
                 var_form: VariationalForm,
                 optimizer: Optimizer,
                 cost_fn: Optional[Callable] = None,
                 initial_point: Optional[np.ndarray] = None) -> None:
        """
        Args:
            var_form: An optional parameterized variational form (ansatz).
            optimizer: A classical optimizer.
            cost_fn: An optional cost function for optimizer. If not supplied here must be
                supplied on :meth:`find_minimum`.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer.
        Raises:
             ValueError: for invalid input
        """
        super().__init__()

        self._var_form = var_form
        self._optimizer = optimizer
        self._cost_fn = cost_fn
        self._initial_point = initial_point

        self._parameterized_circuits = None

    @property
    def var_form(self) -> Optional[VariationalForm]:
        """ Returns variational form """
        return self._var_form

    @var_form.setter
    def var_form(self, var_form: VariationalForm):
        """ Sets variational form """
        self._var_form = var_form

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """ Returns optimizer """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        """ Sets optimizer """
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """ Returns initial point """
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray):
        """ Sets initial point """
        self._initial_point = initial_point

    def find_minimum(self,
                     initial_point: Optional[np.ndarray] = None,
                     var_form: Optional[VariationalForm] = None,
                     cost_fn: Optional[Callable] = None,
                     optimizer: Optional[Optimizer] = None,
                     gradient_fn: Optional[Callable] = None) -> 'VQResult':
        """
        Optimize to find the minimum cost value.

        Args:
            initial_point: If not `None` will be used instead of any initial point supplied via
                constructor. If `None` and `None` was supplied to constructor then a random
                point will be used if the optimizer requires an initial point.
            var_form: If not `None` will be used instead of any variational form supplied via
                constructor.
            cost_fn: If not `None` will be used instead of any cost_fn supplied via
                constructor.
            optimizer: If not `None` will be used instead of any optimizer supplied via
                constructor.
            gradient_fn: Optional gradient function for optimizer

        Returns:
            dict: Optimized variational parameters, and corresponding minimum cost value.

        Raises:
            ValueError: invalid input
        """
        initial_point = initial_point if initial_point is not None else self.initial_point
        var_form = var_form if var_form is not None else self.var_form
        cost_fn = cost_fn if cost_fn is not None else self._cost_fn
        optimizer = optimizer if optimizer is not None else self.optimizer

        if var_form is None:
            raise ValueError('Variational form neither supplied to constructor nor find minimum.')
        if cost_fn is None:
            raise ValueError('Cost function neither supplied to constructor nor find minimum.')
        if optimizer is None:
            raise ValueError('Optimizer neither supplied to constructor nor find minimum.')

        nparms = var_form.num_parameters
        bounds = var_form.parameter_bounds

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError(
                'Initial point size {} and parameter size {} mismatch'.format(
                    len(initial_point), nparms))
        if len(bounds) != nparms:
            raise ValueError('Variational form bounds size does not match parameter size')
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if optimizer.is_initial_point_required:
                low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                initial_point = self.random.uniform(low, high)

        start = time.time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None

        logger.info('Starting optimizer.\nbounds=%s\ninitial point=%s', bounds, initial_point)
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(var_form.num_parameters,
                                                                      cost_fn,
                                                                      variable_bounds=bounds,
                                                                      initial_point=initial_point,
                                                                      gradient_function=gradient_fn)
        eval_time = time.time() - start

        result = VQResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = opt_params

        return result

    def get_prob_vector_for_params(self, construct_circuit_fn, params_s,
                                   quantum_instance, construct_circuit_args=None):
        """ Helper function to get probability vectors for a set of params """
        circuits = []
        for params in params_s:
            circuit = construct_circuit_fn(params, **construct_circuit_args)
            circuits.append(circuit)
        results = quantum_instance.execute(circuits)

        probs_s = []
        for circuit in circuits:
            if quantum_instance.is_statevector:
                sv = results.get_statevector(circuit)
                probs = np.real(sv * np.conj(sv))
                probs_s.append(probs)
            else:
                counts = results.get_counts(circuit)
                probs_s.append(self.get_probabilities_for_counts(counts))
        return np.array(probs_s)

    def get_probabilities_for_counts(self, counts):
        """ get probabilities for counts """
        shots = sum(counts.values())
        states = int(2 ** len(list(counts.keys())[0]))
        probs = np.zeros(states)
        for k, v in counts.items():
            probs[int(k, 2)] = v / shots
        return probs

    @abstractmethod
    def get_optimal_cost(self):
        """ get optimal cost """
        raise NotImplementedError()

    @abstractmethod
    def get_optimal_circuit(self):
        """ get optimal circuit """
        raise NotImplementedError()

    @abstractmethod
    def get_optimal_vector(self):
        """ get optimal vector """
        raise NotImplementedError()

    @property
    @abstractmethod
    def optimal_params(self):
        """ returns optimal parameters """
        raise NotImplementedError()

    def cleanup_parameterized_circuits(self):
        """ set parameterized circuits to None """
        self._parameterized_circuits = None


class VQResult(AlgorithmResult):
    """ Variation Quantum Algorithm Result."""

    @property
    def optimizer_evals(self) -> int:
        """ Returns number of optimizer evaluations """
        return self.get('optimizer_evals')

    @optimizer_evals.setter
    def optimizer_evals(self, value: int) -> None:
        """ Sets number of optimizer evaluations """
        self.data['optimizer_evals'] = value

    @property
    def optimizer_time(self) -> float:
        """ Returns time taken for optimization """
        return self.get('optimizer_time')

    @optimizer_time.setter
    def optimizer_time(self, value: float) -> None:
        """ Sets time taken for optimization  """
        self.data['optimizer_time'] = value

    @property
    def optimal_value(self) -> float:
        """ Returns optimal value """
        return self.get('optimal_value')

    @optimal_value.setter
    def optimal_value(self, value: int) -> float:
        """ Sets optimal value """
        self.data['optimal_value'] = value

    @property
    def optimal_point(self) -> np.ndarray:
        """ Returns optimal point """
        return self.get('optimal_point')

    @optimal_point.setter
    def optimal_point(self, value: np.ndarray) -> None:
        """ Sets optimal point """
        self.data['optimal_point'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'num_optimizer_evals':
            warnings.warn('num_optimizer_evals deprecated, use optimizer_evals property.',
                          DeprecationWarning)
            return super().__getitem__('optimizer_evals')
        elif key == 'min_val':
            warnings.warn('min_val deprecated, use optimal_value property.',
                          DeprecationWarning)
            return super().__getitem__('optimal_value')
        elif key == 'opt_params':
            warnings.warn('opt_params deprecated, use optimal_point property.',
                          DeprecationWarning)
            return super().__getitem__('optimal_point')
        elif key == 'eval_time':
            warnings.warn('eval_time deprecated, use optimizer_time property.',
                          DeprecationWarning)
            return super().__getitem__('optimizer_time')

        return super().__getitem__(key)
