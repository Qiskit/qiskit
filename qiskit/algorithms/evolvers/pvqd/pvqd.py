# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
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

import logging
import numpy as np

from qiskit import QiskitError
from qiskit.algorithms.optimizers import Optimizer, Minimizer
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers import Backend
from qiskit.opflow import (
    OperatorBase,
    CircuitSampler,
    ExpectationBase,
    StateFn,
)
from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.utils import QuantumInstance

from .result import PVQDResult
from .utils import _get_observable_evaluator, _is_gradient_supported

from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult
from ..real_evolver import RealEvolver

logger = logging.getLogger(__name__)


class PVQD(RealEvolver):
    """The projected Variational Quantum Dynamics Algorithm.

    In each timestep this algorithm computes the next state with a Trotter formula
    (specified by the ``evolution`` argument) and projects it onto a variational form (``ansatz``).
    The projection is determined by maximizing the fidelity of the Trotter-evolved state
    and the ansatz, using a classical optimization routine. See Ref. [1] for details.

    Example:

        .. code-block:: python

            import numpy as np

            from qiskit import BasicAer
            from qiskit.circuit.library import EfficientSU2
            from qiskit.opflow import X, Z, I, MatrixExpectation

            backend = BasicAer.get_backend("statevector_simulator")
            expectation = MatrixExpectation()
            hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
            observable = Z ^ Z
            ansatz = EfficientSU2(2, reps=1)
            initial_parameters = np.zeros(ansatz.num_parameters)

            time = 0.02
            optimizer = L_BFGS_B()

            # setup the algorithm
            pvqd = PVQD(
                ansatz,
                initial_parameters,
                timestep=0.01,
                optimizer=optimizer,
                quantum_instance=backend,
                expectation=expectation
            )

            # specify the evolution problem
            problem = EvolutionProblem(
                hamiltonian, time, aux_operators=[hamiltonian, observable]
            )

            # and evolve!
            result = pvqd.evolve(problem)

    References:

        [1] Stefano Barison, Filippo Vicentini, and Giuseppe Carleo (2021), An efficient
            quantum algorithm for the time evolution of parameterized circuits,
            `Quantum 5, 512 <https://quantum-journal.org/papers/q-2021-07-28-512/>`_.
    """

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        initial_parameters: Optional[np.ndarray] = None,
        timestep: Optional[float] = None,
        optimizer: Optional[Union[Optimizer, Minimizer]] = None,
        expectation: Optional[ExpectationBase] = None,
        initial_guess: Optional[np.ndarray] = None,
        evolution: Optional[EvolutionSynthesis] = None,
        gradients: bool = True,
        quantum_instance: Optional[Union[Backend, QuantumInstance]] = None,
    ) -> None:
        """
        Args:
            ansatz: A parameterized circuit preparing the variational ansatz to model the
                time evolved quantum state.
            initial_parameters: The initial parameters for the ansatz.
            timestep: The time step.
            optimizer: The classical optimizers used to minimize the overlap between
                Trotterization and ansatz. Can be either a :class:`.Optimizer` or a callable
                using the :class:`.Minimizer` protocol.
            expectation: The expectation converter to evaluate expectation values.
            initial_guess: The initial guess for the first VQE optimization. Afterwards the
                previous iteration result is used as initial guess. If None, this is set to
                a random vector with elements in the interval :math:`[-0.01, 0.01]`.
            evolution: The evolution synthesis to use for the construction of the Trotter step.
                Defaults to first-order Lie-Trotter decomposition.
            gradients: If True, use the parameter shift rule to compute gradients. If False,
                the optimizer will not be passed a gradient callable.
            quantum_instance: The backend of quantum instance used to evaluate the circuits.
        """
        if evolution is None:
            evolution = LieTrotter()

        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.timestep = timestep
        self.optimizer = optimizer
        self.initial_guess = initial_guess
        self.expectation = expectation
        self.evolution = evolution
        self.gradients = gradients

        self._sampler = None
        self.quantum_instance = quantum_instance

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Return the current quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Optional[Union[Backend, QuantumInstance]]) -> None:
        """Set the quantum instance and circuit sampler."""
        if quantum_instance is not None:
            if not isinstance(quantum_instance, QuantumInstance):
                quantum_instance = QuantumInstance(quantum_instance)
            self._sampler = CircuitSampler(quantum_instance)

        self._quantum_instance = quantum_instance

    def step(
        self, hamiltonian: OperatorBase, theta: np.ndarray, dt: float, initial_guess: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Perform a single time step.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            theta: The current parameters.
            dt: The time step.
            initial_guess: The initial guess for the classical optimization of the
                fidelity between the next variational state and the Trotter-evolved last state.
                If None, this is set to a random vector with elements in the interval
                :math:`[-0.01, 0.01]`.

        Returns:
            A tuple consisting of the next parameters and the fidelity of the optimization.
        """
        loss, gradient = self.get_loss(hamiltonian, dt, theta)

        if initial_guess is None:
            initial_guess = np.random.random(self.initial_parameters.size) * 0.01

        if isinstance(self.optimizer, Optimizer):
            optimizer_result = self.optimizer.minimize(loss, initial_guess, gradient)
        else:
            optimizer_result = self.optimizer(loss, initial_guess, gradient)

        # clip the fidelity to [0, 1]
        fidelity = np.clip(1 - optimizer_result.fun, 0, 1)

        return theta + optimizer_result.x, fidelity

    def get_loss(
        self,
        hamiltonian: OperatorBase,
        dt: float,
        current_parameters: np.ndarray,
    ) -> Tuple[Callable[[np.ndarray], float], Optional[Callable[[np.ndarray], np.ndarray]]]:

        """Get a function to evaluate the infidelity between Trotter step and ansatz.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            dt: The time step.
            current_parameters: The current parameters.

        Returns:
            A callable to evaluate the infidelity and, if gradients are supported and required,
                a second callable to evaluate the gradient of the infidelity.
        """
        self._validate_setup()

        # use Trotterization to evolve the current state
        trotterized = self.ansatz.bind_parameters(current_parameters)
        trotterized.append(
            PauliEvolutionGate(hamiltonian, time=dt, synthesis=self.evolution), self.ansatz.qubits
        )

        # define the overlap of the Trotterized state and the ansatz
        x = ParameterVector("w", self.ansatz.num_parameters)
        shifted = self.ansatz.assign_parameters(current_parameters + x)
        overlap = StateFn(trotterized).adjoint() @ StateFn(shifted)

        converted = self.expectation.convert(overlap)

        def evaluate_loss(
            displacement: Union[np.ndarray, List[np.ndarray]]
        ) -> Union[float, List[float]]:
            """Evaluate the overlap of the ansatz with the Trotterized evolution.

            Args:
                displacement: The parameters for the ansatz.

            Returns:
                The fidelity of the ansatz with parameters ``theta`` and the Trotterized evolution.
            """
            if isinstance(displacement, list):
                displacement = np.asarray(displacement)
                value_dict = {x_i: displacement[:, i].tolist() for i, x_i in enumerate(x)}
            else:
                value_dict = dict(zip(x, displacement))

            sampled = self._sampler.convert(converted, params=value_dict)
            return 1 - np.abs(sampled.eval()) ** 2

        if _is_gradient_supported(self.ansatz) and self.gradients:

            def evaluate_gradient(displacement: np.ndarray) -> np.ndarray:
                """Evaluate the gradient with the parameter-shift rule.

                This is hardcoded here since the gradient framework does not support computing
                gradients for overlaps.

                Args:
                    displacement: The parameters for the ansatz.

                Returns:
                    The gradient.
                """
                # construct lists where each element is shifted by plus (or minus) pi/2
                dim = displacement.size
                plus_shifts = (displacement + np.pi / 2 * np.identity(dim)).tolist()
                minus_shifts = (displacement - np.pi / 2 * np.identity(dim)).tolist()

                evaluated = evaluate_loss(plus_shifts + minus_shifts)

                gradient = (evaluated[:dim] - evaluated[dim:]) / 2

                return gradient

        else:
            evaluate_gradient = None

        return evaluate_loss, evaluate_gradient

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Args:
            evolution_problem: The evolution problem containing the hamiltonian, total evolution
                time and observables to evaluate. Note that :class:`~.PVQD` currently does not support
                hamiltonians of type :class:`~.MatrixOp`.

        Returns:
            A result object containing the evolution information and evaluated observables.

        Raises:
            ValueError: If the evolution time is not positive or the timestep is too small.
        """
        self._validate_setup()

        time = evolution_problem.time
        observables = evolution_problem.aux_operators
        hamiltonian = evolution_problem.hamiltonian

        if not 0 < self.timestep <= time:
            raise ValueError(
                f"The time step ({self.timestep}) must be larger than 0 and smaller equal "
                f"the evolution time ({time})."
            )

        # get the function to evaluate the observables for a given set of ansatz parameters
        if observables is not None:
            evaluate_observables = _get_observable_evaluator(
                self.ansatz, observables, self.expectation, self._sampler
            )
            observable_values = [evaluate_observables(self.initial_parameters)]

        fidelities = [1]
        times = [0]
        parameters = [self.initial_parameters]

        current_time = 0
        initial_guess = self.initial_guess

        while current_time < time:
            # perform VQE to find the next parameters
            next_parameters, fidelity = self.step(
                hamiltonian, parameters[-1], self.timestep, initial_guess
            )

            # set initial guess to last parameter update
            initial_guess = next_parameters - parameters[-1]

            parameters.append(next_parameters)
            fidelities.append(fidelity)
            if observables is not None:
                observable_values.append(evaluate_observables(next_parameters))

            current_time += self.timestep
            times.append(current_time)

        evolved_state = self.ansatz.bind_parameters(parameters[-1])

        result = PVQDResult(
            evolved_state=evolved_state,
            times=times,
            parameters=parameters,
            fidelities=fidelities,
            estimated_error=np.prod(fidelities),
        )
        if observables is not None:
            result.observables = observable_values
            result.aux_ops_evaluated = observable_values[-1]

        return result

    def _validate_setup(self):
        """Validate the current setup and raise an error if something misses to run."""

        required_args = {
            "ansatz",
            "initial_parameters",
            "timestep",
            "optimizer",
            "quantum_instance",
            "expectation",
        }
        for arg in required_args:
            if getattr(self, arg) is None:
                raise ValueError(f"The {arg} attribute cannot be None.")

        if len(self.initial_parameters) != self.ansatz.num_parameters:
            raise QiskitError(
                f"Mismatching number of parameters in the ansatz ({self.ansatz.num_parameters}) "
                f"and the initial parameters ({len(self.initial_parameters)})."
            )
