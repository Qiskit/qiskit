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
"""A generic implementation of Approximate Quantum Compiler."""
from typing import Optional

import numpy as np

from qiskit.algorithms.optimizers import L_BFGS_B, Optimizer
from qiskit.quantum_info import Operator
from .approximate import ApproximateCircuit, ApproximatingObjective


class AQC:
    """
    A generic implementation of Approximate Quantum Compiler. This implementation is agnostic of
    the underlying implementation of the approximate circuit, objective, and optimizer. Users may
    pass corresponding implementations of the abstract classes:

    * Optimizer is an instance of :class:`~qiskit.algorithms.optimizers.Optimizer` and used to run
      the optimization process. A choice of optimizer may affect overall convergence, required time
      for the optimization process and achieved objective value.

    * Approximate circuit represents a template which parameters we want to optimize.  Currently,
      there's only one implementation based on 4-rotations CNOT unit blocks:
      :class:`.CNOTUnitCircuit`. See the paper for more details.

    * Approximate objective is tightly coupled with the approximate circuit implementation and
      provides two methods for computing objective function and gradient with respect to approximate
      circuit parameters. This objective is passed to the optimizer. Currently, there are two
      implementations based on 4-rotations CNOT unit blocks: :class:`.DefaultCNOTUnitObjective` and
      its accelerated version :class:`.FastCNOTUnitObjective`. Both implementations share the same
      idea of maximization the Hilbert-Schmidt product between the target matrix and its
      approximation. The former implementation approach should be considered as a baseline one. It
      may suffer from performance issues, and is mostly suitable for a small number of qubits
      (up to 5 or 6), whereas the latter, accelerated one, can be applied to larger problems.

    * One should take into consideration the exponential growth of matrix size with the number of
      qubits because the implementation not only creates a potentially large target matrix, but
      also allocates a number of temporary memory buffers comparable in size to the target matrix.
    """

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            optimizer: an optimizer to be used in the optimization procedure of the search for
                the best approximate circuit. By default, :obj:`.L_BFGS_B` is used with max
                iterations set to 1000.
            seed: a seed value to be user by a random number generator.
        """
        super().__init__()
        self._optimizer = optimizer
        self._seed = seed

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

        approximate_circuit.build(opt_result.x)

        approx_matrix = Operator(approximate_circuit).data

        if global_phase_required:
            alpha = np.angle(np.trace(np.dot(approx_matrix.conj().T, target_matrix)))
            approximate_circuit.global_phase = alpha
