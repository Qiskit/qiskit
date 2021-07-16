# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Algorithm Base Class.

This class can be used an interface for working with Variation Algorithms, such as VQE,
QAOA, or QSVM, and also provides helper utilities for implementing new variational algorithms.
Writing a new variational algorithm is a simple as extending this class, implementing a cost
function for the new algorithm to pass to the optimizer, and running :meth:`find_minimum` method
of this class to carry out the optimization. Alternatively, all of the functions below can be
overridden to opt-out of this infrastructure but still meet the interface requirements.

.. note::

    This component has some function that is normally random. If you want to reproduce behavior
    then you should set the random number generator seed in the algorithm_globals
    (``qiskit.utils.algorithm_globals.random_seed = seed``).
"""

import warnings
from typing import Optional, Callable, Union, Dict
import time
import logging
from abc import abstractmethod
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.opflow.gradients import GradientBase
from qiskit.utils import QuantumInstance, algorithm_globals, deprecate_function
from .algorithm_result import AlgorithmResult
from .optimizers import Optimizer, SLSQP

logger = logging.getLogger(__name__)


class VariationalAlgorithm:
    """The Variational Algorithm Base Class."""

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Optimizer] = None,
        cost_fn: Optional[Callable] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        initial_point: Optional[np.ndarray] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Args:
            ansatz: An optional parameterized ansatz (a.k.a. variational form).
            optimizer: A classical optimizer.
            cost_fn: An optional cost function for optimizer. If not supplied here must be
                supplied on :meth:`find_minimum`.
            gradient: An optional gradient operator or function for optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer.
            quantum_instance: Quantum Instance or Backend

        Raises:
             ValueError: for invalid input
        """
        if any(
            arg is not None
            for arg in [ansatz, optimizer, cost_fn, gradient, initial_point, quantum_instance]
        ):
            warnings.warn(
                "The VariationalAlgorithm class has been reduced to an abstract "
                "interface. Passing any arguments to the initializer is deprecated as of "
                "Qiskit Terra 0.18.0 and will be unsupported no sooner than 3 months "
                "after the release date.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._quantum_instance = None
        if quantum_instance:
            self.quantum_instance = quantum_instance

        if optimizer is None:
            logger.info("No optimizer provided, setting it to SLSPQ.")
            optimizer = SLSQP()

        self._optimizer = optimizer
        self._gradient = gradient
        self._cost_fn = cost_fn
        self._initial_point = initial_point
        self._ansatz = ansatz
        self._ansatz_params = None
        if ansatz is not None:
            self.ansatz = ansatz

        self._parameterized_circuits = None

    @property
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
quantum_instance property is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
quantum_instance property is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """Sets quantum instance."""
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    @property
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
ansatz property is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def ansatz(self) -> Optional[QuantumCircuit]:
        """Returns the ansatz"""
        return self._ansatz

    @ansatz.setter
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
ansatz property is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
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
            raise ValueError(f'Unsupported type "{type(ansatz)}" of ansatz')

    @property
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
optimizer property is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def optimizer(self) -> Optional[Optimizer]:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
optimizer property is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def optimizer(self, optimizer: Optimizer):
        """Sets optimizer"""
        self._optimizer = optimizer

    @property
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. The
initial_point property will be made abstract no sooner than
3 months after the release date of Qiskit Terra 0.18.0. You should
make a concrete implementation in any derived class and not rely on
the implementation here which will be removed."""
    )
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. The
initial_point property will be made abstract no sooner than
3 months after the release date of Qiskit Terra 0.18.0. You should
make a concrete implementation in any derived class and not rely on
the implementation here which will be removed."""
    )
    def initial_point(self, initial_point: np.ndarray):
        """Sets initial point"""
        self._initial_point = initial_point

    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
find_minimum method is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def find_minimum(
        self,
        initial_point: Optional[np.ndarray] = None,
        ansatz: Optional[QuantumCircuit] = None,
        cost_fn: Optional[Callable] = None,
        optimizer: Optional[Optimizer] = None,
        gradient_fn: Optional[Callable] = None,
    ) -> "VariationalResult":
        """Optimize to find the minimum cost value.

        Args:
            initial_point: If not `None` will be used instead of any initial point supplied via
                constructor. If `None` and `None` was supplied to constructor then a random
                point will be used if the optimizer requires an initial point.
            ansatz: If not `None` will be used instead of any ansatz supplied via constructor.
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
        ansatz = ansatz if ansatz is not None else self.ansatz
        cost_fn = cost_fn if cost_fn is not None else self._cost_fn
        optimizer = optimizer if optimizer is not None else self.optimizer

        if ansatz is None:
            raise ValueError("Ansatz neither supplied to constructor nor find minimum.")
        if cost_fn is None:
            raise ValueError("Cost function neither supplied to constructor nor find minimum.")
        if optimizer is None:
            raise ValueError("Optimizer neither supplied to constructor nor find minimum.")

        nparms = ansatz.num_parameters

        if hasattr(ansatz, "parameter_bounds") and ansatz.parameter_bounds is not None:
            bounds = ansatz.parameter_bounds
        else:
            bounds = [(None, None)] * nparms

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError(
                "Initial point size {} and parameter size {} mismatch".format(
                    len(initial_point), nparms
                )
            )
        if len(bounds) != nparms:
            raise ValueError("Ansatz bounds size does not match parameter size")
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError("Problem has bounds but optimizer does not support bounds")
        else:
            if optimizer.is_bounds_required:
                raise ValueError("Problem does not have bounds but optimizer requires bounds")
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError("Optimizer does not support initial point")
        else:
            if optimizer.is_initial_point_required:
                if hasattr(ansatz, "preferred_init_points"):
                    # Note: default implementation returns None, hence check again after below
                    initial_point = ansatz.preferred_init_points

                if initial_point is None:  # If still None use a random generated point
                    low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                    high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                    initial_point = algorithm_globals.random.uniform(low, high)

        start = time.time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None
        else:
            if not gradient_fn:
                gradient_fn = self._gradient

        logger.info("Starting optimizer.\nbounds=%s\ninitial point=%s", bounds, initial_point)
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(
            nparms,
            cost_fn,
            variable_bounds=bounds,
            initial_point=initial_point,
            gradient_function=gradient_fn,
        )
        eval_time = time.time() - start

        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self._ansatz_params, opt_params))

        return result

    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
get_prob_vector_for_params method is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def get_prob_vector_for_params(
        self, construct_circuit_fn, params_s, quantum_instance, construct_circuit_args=None
    ):
        """Helper function to get probability vectors for a set of params"""
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

    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
get_probabilities_for_counts method is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def get_probabilities_for_counts(self, counts):
        """get probabilities for counts"""
        shots = sum(counts.values())
        states = int(2 ** len(list(counts.keys())[0]))
        probs = np.zeros(states)
        for k, v in counts.items():
            probs[int(k, 2)] = v / shots
        return probs

    @abstractmethod
    def get_optimal_cost(self):
        """get optimal cost"""
        raise NotImplementedError()

    @abstractmethod
    def get_optimal_circuit(self):
        """get optimal circuit"""
        raise NotImplementedError()

    @abstractmethod
    def get_optimal_vector(self):
        """get optimal vector"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def optimal_params(self):
        """returns optimal parameters"""
        raise NotImplementedError()

    @deprecate_function(
        """
The VariationalAlgorithm is reduced to an interface. Thus, the
cleanup_paramterized_circuits method is deprecated as of Qiskit Terra 0.18.0
and will be removed no sooner than 3 months after the releasedate."""
    )
    def cleanup_parameterized_circuits(self):
        """set parameterized circuits to None"""
        self._parameterized_circuits = None


class VariationalResult(AlgorithmResult):
    """Variation Algorithm Result."""

    def __init__(self) -> None:
        super().__init__()
        self._optimizer_evals = None
        self._optimizer_time = None
        self._optimal_value = None
        self._optimal_point = None
        self._optimal_parameters = None

    @property
    def optimizer_evals(self) -> Optional[int]:
        """Returns number of optimizer evaluations"""
        return self._optimizer_evals

    @optimizer_evals.setter
    def optimizer_evals(self, value: int) -> None:
        """Sets number of optimizer evaluations"""
        self._optimizer_evals = value

    @property
    def optimizer_time(self) -> Optional[float]:
        """Returns time taken for optimization"""
        return self._optimizer_time

    @optimizer_time.setter
    def optimizer_time(self, value: float) -> None:
        """Sets time taken for optimization"""
        self._optimizer_time = value

    @property
    def optimal_value(self) -> Optional[float]:
        """Returns optimal value"""
        return self._optimal_value

    @optimal_value.setter
    def optimal_value(self, value: int) -> None:
        """Sets optimal value"""
        self._optimal_value = value

    @property
    def optimal_point(self) -> Optional[np.ndarray]:
        """Returns optimal point"""
        return self._optimal_point

    @optimal_point.setter
    def optimal_point(self, value: np.ndarray) -> None:
        """Sets optimal point"""
        self._optimal_point = value

    @property
    def optimal_parameters(self) -> Optional[Dict]:
        """Returns the optimal parameters in a dictionary"""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: Dict) -> None:
        """Sets optimal parameters"""
        self._optimal_parameters = value
