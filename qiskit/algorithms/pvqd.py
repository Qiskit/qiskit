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

"""The projected Variational Quantum Dynamics Algorithm."""

from typing import Optional, Union, List, Tuple, Callable

import numpy as np

from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers import Backend
from qiskit.opflow import (
    OperatorBase, CircuitSampler, ExpectationBase, ListOp, StateFn, GradientBase
)
from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.utils import QuantumInstance

from .algorithm_result import AlgorithmResult


class PVQDResult(AlgorithmResult):
    """The result object for the pVQD algorithm."""

    times = None
    parameters = None
    fidelities = None
    estimated_error = None
    observables = None


class PVQD:
    """The projected Variational Quantum Dynamics Algorithm."""

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: np.ndarray,
        optimizer: Optimizer,
        quantum_instance: Union[Backend, QuantumInstance],
        expectation: ExpectationBase,
        gradient: Optional[GradientBase] = None,
        evolution: Optional[EvolutionSynthesis] = None,
    ) -> None:
        """
        Args:
            ansatz: A parameterized circuit preparing the variational ansatz to model the
                time evolved quantum state.
            initial_parameters: The initial parameters for the ansatz.
            optimizer: The classical optimizers used to minimize the overlap between
                Trotterization and ansatz.
            quantum_instance: The backend of quantum instance used to evaluate the circuits.
            expectation: The expectation converter to evaluate expectation values.
            evolution: The evolution synthesis to use for the construction of the Trotter step.
                Defaults to first-order Lie-Trotter decomposition.
        """
        if evolution is None:
            evolution = LieTrotter()

        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.optimizer = optimizer
        self.gradient = gradient
        self.expectation = expectation
        self.evolution = evolution

        self._sampler = CircuitSampler(quantum_instance)

    def step(
        self, hamiltonian: OperatorBase, theta: np.ndarray, dt: float, initial_guess: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Perform a single time step.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            theta: The current parameters.
            dt: The time step.

        Returns:
            A tuple consisting of the next parameters and the fidelity of the optimization.
        """
        # construct cost function
        overlap, gradient = self.get_overlap(hamiltonian, dt, theta)

        # call optimizer
        optimizer_result = self.optimizer.minimize(lambda x: -overlap(x), initial_guess, gradient)

        return theta + optimizer_result.x, -optimizer_result.fun

    def get_overlap(
        self, hamiltonian: OperatorBase, dt: float, current_parameters: np.ndarray
    ) -> Callable[[np.ndarray], float]:
        """Get a function to evaluate the overlap between Trotter step and ansatz.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            dt: The time step.
            current_parameters: The current parameters.

        Returns:
            A callable to evaluate the overlap.
        """
        # use Trotterization to evolve the current state
        trotterized = self.ansatz.bind_parameters(current_parameters)
        trotterized.append(
            PauliEvolutionGate(hamiltonian, time=dt, synthesis=self.evolution), self.ansatz.qubits
        )

        # define the overlap of the Trotterized state and the ansatz
        x = ParameterVector("w", self.ansatz.num_parameters)
        shifted = self.ansatz.assign_parameters(current_parameters + x)
        overlap = StateFn(trotterized).adjoint() @ StateFn(shifted)

        # apply the expectation converter
        converted = self.expectation.convert(overlap)

        ansatz_parameters = self.ansatz.parameters

        def evaluate_overlap(displacement: np.ndarray) -> float:
            """Evaluate the overlap of the ansatz with the Trotterized evolution.

            Args:
                displacement: The parameters for the ansatz.

            Returns:
                The fidelity of the ansatz with parameters ``theta`` and the Trotterized evolution.
            """
            # evaluate with the circuit sampler
            value_dict = dict(zip(x, displacement))
            sampled = self._sampler.convert(converted, params=value_dict)
            return np.abs(sampled.eval()) ** 2

        if self.gradient is not None:
            gradient = self.gradient.convert(overlap)

            def evaluate_gradient(displacement: np.ndarray) -> np.ndarray:
                """Evaluate the gradient."""
                # evaluate with the circuit sampler
                value_dict = dict(zip(ansatz_parameters, current_parameters + displacement))
                sampled = self._sampler.convert(gradient, params=value_dict)
                return 2 * sampled.eval()

            return evaluate_overlap, evaluate_gradient

        return evaluate_overlap, None

    def _get_observable_evaluator(self, observables):
        if isinstance(observables, list):
            observables = ListOp(observables)

        expectation_value = StateFn(observables, is_measurement=True) @ StateFn(self.ansatz)
        converted = self.expectation.convert(expectation_value)

        ansatz_parameters = self.ansatz.parameters

        def evaluate_observables(theta: np.ndarray) -> Union[float, List[float]]:
            """Evaluate the observables for the ansatz parameters ``theta``.

            Args:
                theta: The ansatz parameters.

            Returns:
                The observables evaluated at the ansatz parameters.
            """
            value_dict = dict(zip(ansatz_parameters, theta))
            sampled = self._sampler.convert(converted, params=value_dict)
            return sampled.eval()

        return evaluate_observables

    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        dt: float,
        observables: Optional[Union[OperatorBase, List[OperatorBase]]] = None,
    ) -> PVQDResult:
        """
        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            time: The total evolution time.
            dt: The time step.
            observables: The observables to evaluate at each time step.

        Returns:
            A result object containing the evolution information and evaluated observables.

        Raises:
            ValueError: If the evolution time is not positive or the timestep is too small.
        """
        if time <= 0:
            raise ValueError("The evolution time must be larger than 0.")

        if not 0 < dt <= time:
            raise ValueError(
                "The time step must be larger than 0 and smaller equal the evolution time."
            )

        # get the function to evaluate the observables for a given set of ansatz parameters
        evaluate_observables = self._get_observable_evaluator(observables)

        observable_values = [evaluate_observables(self.initial_parameters)]
        fidelities = [1]
        times = [0]
        parameters = [self.initial_parameters]

        current_time = 0
        initial_guess = np.random.random(self.initial_parameters.size) * 0.01
        # initial_guess = np.zeros(self.initial_parameters.size)
        while current_time < time:
            # perform VQE to find the next parameters
            next_parameters, fidelity = self.step(hamiltonian, parameters[-1], dt, initial_guess)

            # set initial guess to last parameter update
            initial_guess = next_parameters - parameters[-1]

            # store parameters
            parameters.append(next_parameters)
            fidelities.append(fidelity)
            observable_values.append(evaluate_observables(next_parameters))

            # increase time
            current_time += dt
            times.append(current_time)

        result = PVQDResult()
        result.times = times
        result.parameters = parameters
        result.fidelities = fidelities
        result.estimated_error = np.prod(result.fidelities)
        result.observables = np.asarray(observable_values)

        return result
