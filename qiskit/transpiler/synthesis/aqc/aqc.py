# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Main entry point to Approximate Quantum Compiler.
"""
from typing import Optional

import numpy as np

from qiskit.algorithms.optimizers import L_BFGS_B, Optimizer
from qiskit.quantum_info import Operator
from .approximate import ApproximateCircuit, ApproximatingObjective


class AQC:
    # todo: update
    r"""
    Implementation of Approximate Quantum Compiler as described in the paper.

    We are interested in compiling a quantum circuit, which we formalize as finding the best
    circuit representation in terms of an ordered gate sequence of a target unitary matrix
    :math:`U\in U(d)`, with some additional hardware constraints. In particular, we look at
    representations that could be constrained in terms of hardware connectivity, as well
    as gate depth, and we choose a gate basis in terms of CNOT and rotation gates.
    We recall that the combination of CNOT and rotation gates is universal in :math:`SU(d)` and
    therefore it does not limit compilation.

    To properly define what we mean by best circuit representation, we define the metric
    as the Frobenius norm between the unitary matrix of the compiled circuit :math:`V` and
    the target unitary matrix :math:`U`, i.e., :math:`\|V - U\|_{\mathrm{F}}`. This choice
    is motivated by mathematical programming considerations, and it is related to other
    formulations that appear in the literature.

    References:

        [1]: Liam Madden, Andrea Simonetto, Best Approximate Quantum Compiling Problems.
            `arXiv:2106.05649 <https://arxiv.org/abs/2106.05649>`_
    """

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            optimizer: an optimizer to be used in the optimization procedure of the search for
                the best approximate circuit. By default ``L_BFGS_B`` is used with max iterations
                is set to 1000.
            seed: a seed value to be user by a random number generator.
        """
        super().__init__()
        self._optimizer = optimizer
        self._seed = seed or 12345

    def compile_unitary(
        self,
        target_matrix: np.ndarray,
        approximate_circuit: ApproximateCircuit,
        approximating_objective: ApproximatingObjective,
        initial_point: Optional[np.ndarray] = None,
    ) -> None:
        """
        Approximately compiles a circuit represented as a unitary matrix by solving an optimization
        problem defined by ``approximating_objective`` and using ``approximate_circuit`` as a
        template for the approximate circuit.

        Args:
            target_matrix: a unitary matrix to approximate.
            approximate_circuit: a template circuit that will be filled with the parameter values
                obtained in the optimization procedure.
            approximating_objective: a definition of the optimization problem.
            initial_point: initial values of angles/parameters to start optimization from.
        """
        matrix_dim = target_matrix.shape[0]
        # check if it is actually a special unitary matrix
        target_det = np.linalg.det(target_matrix)
        if not np.isclose(target_det, 1):
            su_matrix = target_matrix / np.power(target_det, (1 / matrix_dim))
            global_phase_required = True
        else:
            su_matrix = target_matrix
            global_phase_required = False

        # set the matrix to approximate in the algorithm
        approximating_objective.target_matrix = su_matrix

        optimizer = self._optimizer or L_BFGS_B(maxiter=1000)

        if initial_point is None:
            np.random.seed(self._seed)
            initial_point = np.random.uniform(0, 2 * np.pi, approximating_objective.num_thetas)

        opt_result = optimizer.minimize(
            fun=approximating_objective.objective,
            x0=initial_point,
            jac=approximating_objective.gradient,
        )

        approximate_circuit.thetas = opt_result.x
        approximate_circuit.build()

        approx_matrix = Operator(approximate_circuit).data

        if global_phase_required:
            alpha = np.angle(np.trace(np.dot(approx_matrix.conj().T, target_matrix)))
            approximate_circuit.global_phase = alpha
