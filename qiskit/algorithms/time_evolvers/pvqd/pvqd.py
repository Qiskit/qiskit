# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The projected Variational Quantum Dynamics Algorithm."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.utils import algorithm_globals

from ...exceptions import AlgorithmError, QiskitError
from ...optimizers import Minimizer, Optimizer
from ...state_fidelities.base_state_fidelity import BaseStateFidelity
from ..real_time_evolver import RealTimeEvolver
from ..time_evolution_problem import TimeEvolutionProblem
from ..time_evolution_result import TimeEvolutionResult
from .pvqd_result import PVQDResult
from .utils import _get_observable_evaluator, _is_gradient_supported

logger = logging.getLogger(__name__)


class PVQD(RealTimeEvolver):
    """The projected Variational Quantum Dynamics (p-VQD) Algorithm.

    In each timestep, this algorithm computes the next state with a Trotter formula
    (specified by the ``evolution`` argument) and projects the timestep onto a variational form
    (``ansatz``). The projection is determined by maximizing the fidelity of the Trotter-evolved
    state and the ansatz, using a classical optimization routine. See Ref. [1] for details.

    The following attributes can be set via the initializer but can also be read and
    updated once the PVQD object has been constructed.

    Attributes:

        ansatz (QuantumCircuit): The parameterized circuit representing the time-evolved state.
        initial_parameters (np.ndarray): The parameters of the ansatz at time 0.
        optimizer (Optional[Union[Optimizer, Minimizer]]): The classical optimization routine
            used to maximize the fidelity of the Trotter step and ansatz.
        num_timesteps (Optional[int]): The number of timesteps to take. If None, it is automatically
            selected to achieve a timestep of approximately 0.01.
        evolution (Optional[EvolutionSynthesis]): The method to perform the Trotter step.
            Defaults to first-order Lie-Trotter evolution.
        use_parameter_shift (bool): If True, use the parameter shift rule for loss function
            gradients (if the ansatz supports).
        initial_guess (Optional[np.ndarray]): The starting point for the first classical optimization
            run, at time 0. Defaults to random values in :math:`[-0.01, 0.01]`.

    Example:

        This snippet computes the real time evolution of a quantum Ising model on two
        neighboring sites and keeps track of the magnetization.

        .. code-block:: python

            import numpy as np

            from qiskit.algorithms.state_fidelities import ComputeUncompute
            from qiskit.algorithms.time_evolvers import TimeEvolutionProblem, PVQD
            from qiskit.primitives import Estimator, Sampler
            from qiskit.circuit.library import EfficientSU2
            from qiskit.quantum_info import SparsePauliOp, Pauli
            from qiskit.algorithms.optimizers import L_BFGS_B

            sampler = Sampler()
            fidelity = ComputeUncompute(sampler)
            estimator = Estimator()
            hamiltonian = 0.1 * SparsePauliOp(["ZZ", "IX", "XI"])
            observable = Pauli("ZZ")
            ansatz = EfficientSU2(2, reps=1)
            initial_parameters = np.zeros(ansatz.num_parameters)

            time = 1
            optimizer = L_BFGS_B()

            # setup the algorithm
            pvqd = PVQD(
                fidelity,
                ansatz,
                initial_parameters,
                estimator,
                num_timesteps=100,
                optimizer=optimizer,
            )

            # specify the evolution problem
            problem = TimeEvolutionProblem(
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
        fidelity: BaseStateFidelity,
        ansatz: QuantumCircuit,
        initial_parameters: np.ndarray,
        estimator: BaseEstimator | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        num_timesteps: int | None = None,
        evolution: EvolutionSynthesis | None = None,
        use_parameter_shift: bool = True,
        initial_guess: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            fidelity: A fidelity primitive used by the algorithm.
            ansatz: A parameterized circuit preparing the variational ansatz to model the
                time evolved quantum state.
            initial_parameters: The initial parameters for the ansatz. Together with the ansatz,
                these define the initial state of the time evolution.
            estimator: An estimator primitive used for calculating expected values of auxiliary
                operators (if provided via the problem).
            optimizer: The classical optimizers used to minimize the overlap between
                Trotterization and ansatz. Can be either a :class:`.Optimizer` or a callable
                using the :class:`.Minimizer` protocol. This argument is optional since it is
                not required for :meth:`get_loss`, but it has to be set before :meth:`evolve`
                is called.
            num_timesteps: The number of time steps. If ``None`` it will be set such that the
                timestep is close to 0.01.
            evolution: The evolution synthesis to use for the construction of the Trotter step.
                Defaults to first-order Lie-Trotter decomposition, see also
                :mod:`~qiskit.synthesis.evolution` for different options.
            use_parameter_shift: If True, use the parameter shift rule to compute gradients.
                If False, the optimizer will not be passed a gradient callable. In that case,
                Qiskit optimizers will use a finite difference rule to approximate the gradients.
            initial_guess: The initial guess for the first VQE optimization. Afterwards the
                previous iteration result is used as initial guess. If None, this is set to
                a random vector with elements in the interval :math:`[-0.01, 0.01]`.
        """
        super().__init__()
        if evolution is None:
            evolution = LieTrotter()

        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.num_timesteps = num_timesteps
        self.optimizer = optimizer
        self.initial_guess = initial_guess
        self.estimator = estimator
        self.fidelity_primitive = fidelity
        self.evolution = evolution
        self.use_parameter_shift = use_parameter_shift

    def step(
        self,
        hamiltonian: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        theta: np.ndarray,
        dt: float,
        initial_guess: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Perform a single time step.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            ansatz: The parameterized quantum circuit which attempts to approximate the
                time-evolved state.
            theta: The current parameters.
            dt: The time step.
            initial_guess: The initial guess for the classical optimization of the
                fidelity between the next variational state and the Trotter-evolved last state.
                If None, this is set to a random vector with elements in the interval
                :math:`[-0.01, 0.01]`.

        Returns:
            A tuple consisting of the next parameters and the fidelity of the optimization.
        """
        self._validate_setup()

        loss, gradient = self.get_loss(hamiltonian, ansatz, dt, theta)

        if initial_guess is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                initial_guess = algorithm_globals.random.random(self.initial_parameters.size) * 0.01

        if isinstance(self.optimizer, Optimizer):
            optimizer_result = self.optimizer.minimize(loss, initial_guess, gradient)
        else:
            optimizer_result = self.optimizer(loss, initial_guess, gradient)

        # clip the fidelity to [0, 1]
        fidelity = np.clip(1 - optimizer_result.fun, 0, 1)

        return theta + optimizer_result.x, fidelity

    def get_loss(
        self,
        hamiltonian: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        dt: float,
        current_parameters: np.ndarray,
    ) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]] | None:
        """Get a function to evaluate the infidelity between Trotter step and ansatz.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            ansatz: The parameterized quantum circuit which attempts to approximate the
                time-evolved state.
            dt: The time step.
            current_parameters: The current parameters.

        Returns:
            A callable to evaluate the infidelity and, if gradients are supported and required,
                a second callable to evaluate the gradient of the infidelity.
        """
        self._validate_setup(skip={"optimizer"})

        # use Trotterization to evolve the current state
        trotterized = ansatz.assign_parameters(current_parameters)

        evolution_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=self.evolution)

        trotterized.append(evolution_gate, ansatz.qubits)

        # define the overlap of the Trotterized state and the ansatz
        x = ParameterVector("w", ansatz.num_parameters)
        shifted = ansatz.assign_parameters(current_parameters + x)

        def evaluate_loss(displacement: np.ndarray | list[np.ndarray]) -> float | np.ndarray:
            """Evaluate the overlap of the ansatz with the Trotterized evolution.

            Args:
                displacement: The parameters for the ansatz.

            Returns:
                The fidelity of the ansatz with parameters ``theta`` and the Trotterized evolution.

            Raises:
                AlgorithmError: If a primitive job fails.
            """
            if isinstance(displacement, list):
                displacement = np.asarray(displacement)
                value_dict = {x_i: displacement[:, i].tolist() for i, x_i in enumerate(x)}
            else:
                value_dict = dict(zip(x, displacement))

            param_dicts = self._transpose_param_dicts(value_dict)
            num_of_param_sets = len(param_dicts)
            states1 = [trotterized] * num_of_param_sets
            states2 = [shifted] * num_of_param_sets
            param_dicts2 = [list(param_dict.values()) for param_dict in param_dicts]
            # the first state does not have free parameters so values_1 will be None by default
            try:
                job = self.fidelity_primitive.run(states1, states2, values_2=param_dicts2)
                fidelities = np.array(job.result().fidelities)
            except Exception as exc:
                raise AlgorithmError("The primitive job failed!") from exc

            if len(fidelities) == 1:
                fidelities = fidelities[0]

            # in principle, we could add different loss functions here, but we're currently
            # not aware of a use-case for a different one than in the paper
            return 1 - fidelities

        if _is_gradient_supported(ansatz) and self.use_parameter_shift:

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

    def _transpose_param_dicts(self, params: dict) -> list[dict[Parameter, float]]:
        p_0 = list(params.values())[0]
        if isinstance(p_0, (list, np.ndarray)):
            num_parameterizations = len(p_0)
            param_bindings = [
                {param: value_list[i] for param, value_list in params.items()}  # type: ignore
                for i in range(num_parameterizations)
            ]
        else:
            param_bindings = [params]

        return param_bindings

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> TimeEvolutionResult:
        r"""Perform real time evolution :math:`\exp(-i t H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`t`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        Args:
            evolution_problem: The evolution problem containing the hamiltonian, total evolution
                time and observables to evaluate.

        Returns:
            A result object containing the evolution information and evaluated observables.

        Raises:
            ValueError: If ``aux_operators`` provided in the time evolution problem but no estimator
                provided to the algorithm.
            NotImplementedError: If the evolution problem contains an initial state.
        """
        self._validate_setup()

        time = evolution_problem.time
        observables = evolution_problem.aux_operators
        hamiltonian = evolution_problem.hamiltonian

        # determine the number of timesteps and set the timestep
        num_timesteps = (
            int(np.ceil(time / 0.01)) if self.num_timesteps is None else self.num_timesteps
        )
        timestep = time / num_timesteps

        if evolution_problem.initial_state is not None:
            raise NotImplementedError(
                "Setting an initial state for the evolution is not yet supported for PVQD."
            )

        # get the function to evaluate the observables for a given set of ansatz parameters
        if observables is not None:
            if self.estimator is None:
                raise ValueError(
                    "The evolution problem contained aux_operators but no estimator was provided. "
                )
            evaluate_observables = _get_observable_evaluator(
                self.ansatz, observables, self.estimator
            )
            observable_values = [evaluate_observables(self.initial_parameters)]

        fidelities = [1.0]
        parameters = [self.initial_parameters]
        times = np.linspace(0, time, num_timesteps + 1).tolist()  # +1 to include initial time 0

        initial_guess = self.initial_guess

        for _ in range(num_timesteps):
            # perform VQE to find the next parameters
            next_parameters, fidelity = self.step(
                hamiltonian, self.ansatz, parameters[-1], timestep, initial_guess
            )

            # set initial guess to last parameter update
            initial_guess = next_parameters - parameters[-1]

            parameters.append(next_parameters)
            fidelities.append(fidelity)
            if observables is not None:
                observable_values.append(evaluate_observables(next_parameters))

        evolved_state = self.ansatz.assign_parameters(parameters[-1])

        result = PVQDResult(
            evolved_state=evolved_state,
            times=times,
            parameters=parameters,
            fidelities=fidelities,
            estimated_error=1 - np.prod(fidelities),
        )
        if observables is not None:
            result.observables = observable_values
            result.aux_ops_evaluated = observable_values[-1]

        return result

    def _validate_setup(self, skip=None):
        """Validate the current setup and raise an error if something misses to run."""

        if skip is None:
            skip = {}

        required_attributes = {"optimizer"}.difference(skip)

        for attr in required_attributes:
            if getattr(self, attr, None) is None:
                raise ValueError(f"The {attr} cannot be None.")

        if self.num_timesteps is not None and self.num_timesteps <= 0:
            raise ValueError(
                f"The number of timesteps must be positive but is {self.num_timesteps}."
            )

        if self.ansatz.num_parameters == 0:
            raise QiskitError(
                "The ansatz cannot have 0 parameters, otherwise it cannot be trained."
            )

        if len(self.initial_parameters) != self.ansatz.num_parameters:
            raise QiskitError(
                f"Mismatching number of parameters in the ansatz ({self.ansatz.num_parameters}) "
                f"and the initial parameters ({len(self.initial_parameters)})."
            )
