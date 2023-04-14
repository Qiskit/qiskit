# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Quantum Approximate Optimization Algorithm."""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from qiskit.algorithms.optimizers import Minimizer, Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.providers import Backend
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.utils.validation import validate_min
from qiskit.utils.deprecation import deprecate_func
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQE


class QAOA(VQE):
    """
    Deprecated: Quantum Approximate Optimization Algorithm.

    The QAOA class has been superseded by the
    :class:`qiskit.algorithms.minimum_eigensolvers.QAOA` class.
    This class will be deprecated in a future release and subsequently
    removed after that.

    `QAOA <https://arxiv.org/abs/1411.4028>`__ is a well-known algorithm for finding approximate
    solutions to combinatorial-optimization problems.

    The QAOA implementation directly extends :class:`VQE` and inherits VQE's optimization structure.
    However, unlike VQE, which can be configured with arbitrary ansatzes,
    QAOA uses its own fine-tuned ansatz, which comprises :math:`p` parameterized global
    :math:`x` rotations and :math:`p` different parameterizations of the problem hamiltonian.
    QAOA is thus principally configured  by the single integer parameter, *p*,
    which dictates the depth of the ansatz, and thus affects the approximation quality.

    An optional array of :math:`2p` parameter values, as the *initial_point*, may be provided as the
    starting **beta** and **gamma** parameters (as identically named in the
    original `QAOA paper <https://arxiv.org/abs/1411.4028>`__) for the QAOA ansatz.

    An operator or a parameterized quantum circuit may optionally also be provided as a custom
    `mixer` Hamiltonian. This allows, as discussed in
    `this paper <https://doi.org/10.1103/PhysRevApplied.5.034007>`__ for quantum annealing,
    and in `this paper <https://arxiv.org/abs/1709.03489>`__ for QAOA,
    to run constrained optimization problems where the mixer constrains
    the evolution to a feasible subspace of the full Hilbert space.
    """

    @deprecate_func(
        additional_msg=(
            "Instead, use the class ``qiskit.algorithms.minimum_eigensolvers.QAOA``. "
            "See https://qisk.it/algo_migration for a migration guide."
        ),
        since="0.24.0",
    )
    def __init__(
        self,
        optimizer: Optimizer | Minimizer | None = None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer: QuantumCircuit | OperatorBase = None,
        initial_point: np.ndarray | None = None,
        gradient: GradientBase | Callable[[np.ndarray | list], list] | None = None,
        expectation: ExpectationBase | None = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Callable[[int, np.ndarray, float, float], None] | None = None,
        quantum_instance: QuantumInstance | Backend | None = None,
    ) -> None:
        """
        Args:
            optimizer: A classical optimizer, see also :class:`~qiskit.algorithms.VQE` for
                more details on the possible types.
            reps: the integer parameter :math:`p` as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
                as well as warm-starting the optimization as introduced
                in http://arxiv.org/abs/2009.10095.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then it will simply compute a random one.
            gradient: An optional gradient operator respectively a gradient function used for
                      optimization.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When None (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to True (defaults to False).
            include_custom: When `expectation` parameter here is None setting this to True will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Ignored if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
            quantum_instance: Quantum Instance or Backend
        """
        validate_min("reps", reps, 1)

        self._reps = reps
        self._mixer = mixer
        self._initial_state = initial_state
        self._cost_operator: OperatorBase | None = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(
                ansatz=None,
                optimizer=optimizer,
                initial_point=initial_point,
                gradient=gradient,
                expectation=expectation,
                include_custom=include_custom,
                max_evals_grouped=max_evals_grouped,
                callback=callback,
                quantum_instance=quantum_instance,
            )

    def _check_operator_ansatz(self, operator: OperatorBase) -> None:
        # Recreates a circuit based on operator parameter.
        if operator != self._cost_operator:
            self._cost_operator = operator
            self.ansatz = QAOAAnsatz(
                operator, self._reps, initial_state=self._initial_state, mixer_operator=self._mixer
            ).decompose()  # TODO remove decompose once #6674 is fixed

    @property
    def initial_state(self) -> QuantumCircuit | None:
        """
        Returns:
            Returns the initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """
        Args:
            initial_state: Initial state to set.
        """
        self._initial_state = initial_state

    @property
    def mixer(self) -> QuantumCircuit | OperatorBase:
        """
        Returns:
            Returns the mixer.
        """
        return self._mixer

    @mixer.setter
    def mixer(self, mixer: QuantumCircuit | OperatorBase) -> None:
        """
        Args:
            mixer: Mixer to set.
        """
        self._mixer = mixer
