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

"""The quantum approximate optimization algorithm. """

from __future__ import annotations

from typing import Callable
import numpy as np

from qiskit.algorithms.optimizers import Minimizer, Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseSampler
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils.validation import validate_min

from .sampling_vqe import SamplingVQE


class QAOA(SamplingVQE):
    r"""
    The Quantum Approximate Optimization Algorithm.

    `QAOA <https://arxiv.org/abs/1411.4028>`__ is a well-known algorithm for finding approximate
    solutions to combinatorial-optimization problems.

    The QAOA implementation directly extends :class:`.SamplingVQE` and inherits its optimization
    structure. However, unlike VQE, which can be configured with arbitrary ansatzes, QAOA uses its
    own fine-tuned ansatz, which comprises :math:`p` parameterized global :math:`x` rotations and
    :math:`p` different parameterizations of the problem hamiltonian. QAOA is thus principally
    configured  by the single integer parameter, *p*, which dictates the depth of the ansatz, and
    thus affects the approximation quality.

    An optional array of :math:`2p` parameter values, as the *initial_point*, may be provided as the
    starting **beta** and **gamma** parameters (as identically named in the original `QAOA paper
    <https://arxiv.org/abs/1411.4028>`__) for the QAOA ansatz.

    An operator or a parameterized quantum circuit may optionally also be provided as a custom
    `mixer` Hamiltonian. This allows, as discussed in `this paper
    <https://doi.org/10.1103/PhysRevApplied.5.034007>`__ for quantum annealing, and in `this paper
    <https://arxiv.org/abs/1709.03489>`__ for QAOA, to run constrained optimization problems where
    the mixer constrains the evolution to a feasible subspace of the full Hilbert space.

    Attributes:
        sampler: The sampler primitive to sample the circuits.
        optimizer: A classical optimizer to find the minimum energy. This can either be a
            Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
            Defaults to :class:`.SLSQP`.
        reps: the integer parameter :math:`p` as specified in https://arxiv.org/abs/1411.4028,
            Has a minimum valid value of 1.
        initial_state: An optional initial state to prepend the QAOA circuit with.
        mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
            of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
            as well as warm-starting the optimization as introduced
            in http://arxiv.org/abs/2009.10095.
        initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            If not provided, a random initial point with values in the interval :math:`[0, 2\pi]`
            is used.
        aggregation: A float or callable to specify how the objective function evaluated on the
            basis states should be aggregated. If a float, this specifies the :math:`\alpha \in [0,1]`
            parameter for a CVaR expectation value (see also [1]).
    """

    def __init__(
        self,
        sampler: BaseSampler,
        optimizer: Optimizer | Minimizer | None = None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer: QuantumCircuit | BaseOperator | PauliSumOp = None,
        initial_point: np.ndarray | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        # callback: Callable[[int, np.ndarray, float, float], None] | None = None,
    ) -> None:
        """
        Args:
            sampler: The sampler primitive to sample the circuits.
            optimizer: A classical optimizer, see also :class:`~qiskit.algorithms.VQE` for
                more details on the possible types.
            reps: the integer parameter :math:`p` as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with.
            mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
                as well as warm-starting the optimization as introduced
                in http://arxiv.org/abs/2009.10095.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then it will simply compute a random one.
            aggregation: A float or callable to specify how the objective function evaluated on the
                basis states should be aggregated.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
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
            # callback=callback,
        )

    def _check_operator_ansatz(
        self, operator: BaseOperator | PauliSumOp
    ) -> BaseOperator | PauliSumOp:
        # Recreates a circuit based on operator parameter.
        ansatz = QAOAAnsatz(
            operator, self.reps, initial_state=self.initial_state, mixer_operator=self.mixer
        ).decompose()  # TODO remove decompose once #6674 is fixed
        return ansatz
