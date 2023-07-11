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

"""The quantum approximate optimization algorithm."""

from __future__ import annotations

from typing import Callable, Any
import numpy as np

from qiskit.algorithms.optimizers import Minimizer, Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseSampler
from qiskit.utils.validation import validate_min

from .sampling_vqe import SamplingVQE


class QAOA(SamplingVQE):
    r"""
    The Quantum Approximate Optimization Algorithm (QAOA).

    QAOA is a well-known algorithm for finding approximate solutions to combinatorial-optimization
    problems [1].

    The QAOA implementation directly extends :class:`.SamplingVQE` and inherits its optimization
    structure. However, unlike VQE, which can be configured with arbitrary ansatzes, QAOA uses its
    own fine-tuned ansatz, which comprises :math:`p` parameterized global :math:`x` rotations and
    :math:`p` different parameterizations of the problem hamiltonian. QAOA is thus principally
    configured  by the single integer parameter, ``reps``, which dictates the depth of the ansatz,
    and thus affects the approximation quality.

    An optional array of :math:`2p` parameter values, as the :attr:`initial_point`, may be provided
    as the starting :math:`\beta` and :math:`\gamma` parameters for the QAOA ansatz [1].

    An operator or a parameterized quantum circuit may optionally also be provided as a custom
    :attr:`mixer` Hamiltonian. This allows in the case of quantum annealing [2] and QAOA [3], to run
    constrained optimization problems where the mixer constrains the evolution to a feasible
    subspace of the full Hilbert space.

    The following attributes can be set via the initializer but can also be read and updated once
    the QAOA object has been constructed.

    Attributes:
        sampler (BaseSampler): The sampler primitive to sample the circuits.
        optimizer (Optimizer | Minimizer): A classical optimizer to find the minimum energy. This
            can either be a Qiskit :class:`.Optimizer` or a callable implementing the
            :class:`.Minimizer` protocol.
        reps (int): The integer parameter :math:`p`. Has a minimum valid value of 1.
        initial_state: An optional initial state to prepend the QAOA circuit with.
        mixer (QuantumCircuit | BaseOperator | PauliSumOp): The mixer Hamiltonian to evolve with or
            a custom quantum circuit. Allows support of optimizations in constrained subspaces [2,
            3] as well as warm-starting the optimization [4].
        aggregation (float | Callable[[list[float]], float] | None): A float or callable to specify
            how the objective function evaluated on the basis states should be aggregated. If a
            float, this specifies the :math:`\alpha \in [0,1]` parameter for a CVaR expectation
            value.
        callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback
            that can access the intermediate data at each optimization step. These data are: the
            evaluation count, the optimizer parameters for the ansatz, the evaluated value, the
            the metadata dictionary, and the best measurement.

    References:
        [1]: Farhi, E., Goldstone, J., Gutmann, S., "A Quantum Approximate Optimization Algorithm"
            `arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`__
        [2]: Hen, I., Spedalieri, F. M.,  "Quantum Annealing for Constrained Optimization"
            `PhysRevApplied.5.034007 <https://doi.org/10.1103/PhysRevApplied.5.034007>`__
        [3]: Hadfield, S. et al, "From the Quantum Approximate Optimization Algorithm to a Quantum
            Alternating Operator Ansatz" `arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__
        [4]: Egger, D. J., Marecek, J., Woerner, S., "Warm-starting quantum optimization"
            `arXiv: 2009.10095 <https://arxiv.org/abs/2009.10095>`__
    """

    def __init__(
        self,
        sampler: BaseSampler,
        optimizer: Optimizer | Minimizer,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer: QuantumCircuit | BaseOperator | PauliSumOp = None,
        initial_point: np.ndarray | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ) -> None:
        r"""
        Args:
            sampler: The sampler primitive to sample the circuits.
            optimizer: A classical optimizer to find the minimum energy. This can either be a
                Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer`
                protocol.
            reps: The integer parameter :math:`p`. Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with.
            mixer: The mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces [2, 3] as well as warm-starting the
                optimization [4].
            initial_point: An optional initial point (i.e. initial parameter values) for the
                optimizer. The length of the initial point must match the number of :attr:`ansatz`
                parameters. If ``None``, a random point will be generated within certain parameter
                bounds. ``QAOA`` will look to the ansatz for these bounds. If the ansatz does not
                specify bounds, bounds of :math:`-2\pi`, :math:`2\pi` will be used.
            aggregation: A float or callable to specify how the objective function evaluated on the
                basis states should be aggregated. If a float, this specifies the :math:`\alpha \in
                [0,1]` parameter for a CVaR expectation value.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer parameters for the ansatz, the
                evaluated value, the metadata dictionary.
        """
        validate_min("reps", reps, 1)

        self.reps = reps
        self.mixer = mixer
        self.initial_state = initial_state
        self._cost_operator = None

        super().__init__(
            sampler=sampler,
            ansatz=None,
            optimizer=optimizer,
            initial_point=initial_point,
            aggregation=aggregation,
            callback=callback,
        )

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp):
        # Recreates a circuit based on operator parameter.
        self.ansatz = QAOAAnsatz(
            operator, self.reps, initial_state=self.initial_state, mixer_operator=self.mixer
        ).decompose()  # TODO remove decompose once #6674 is fixed
