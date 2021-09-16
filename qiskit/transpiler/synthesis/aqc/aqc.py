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
    formulations that appear in the literature. Let's take a look at the problem in more details.

    Let :math:`n` be the number of qubits and :math:`d=2^n`. Given a CNOT structure :math:`ct`
    and a vector of rotation angles :math:`\theta`, the parametric circuit forms a matrix
    :math:`Vct(\theta)\in SU(d)`. If we are given a target circuit forming a matrix
    :math:`U\in SU(d)`, then we would like to compute

    .. math::

        argmax_{\theta}\frac{1}{d}|\langle Vct(\theta),U\rangle|

    where the inner product is the Frobenius inner product. Note that
    :math:`|\langle V,U\rangle|\leq d` for all unitaries :math:`U` and :math:`V`, so the objective
    has range in :math:`[0,1]`.

    Our strategy is to maximize

    .. math::

        \frac{1}{d}\Re \langle Vct(\theta),U\rangle

    using its gradient. We will now discuss the specifics by going through an example.

    While the range of :math:`Vct` is a subset of :math:`SU(d)` by construction, the target
    circuit may form a general unitary matrix. However, for any :math:`U\in U(d)`,

    .. math::

        \frac{\exp(2\pi i k/d)}{\det(U)^{1/d}}U\in SU(d)\text{ for all }k\in\{0,\ldots,d-1\}.

    Thus, we should normalize the target circuit by its global phase and then approximately
    compile the normalized circuit. We can add the global phase back in afterwards.

    In the algorithm let :math:`U'` denote the un-normalized target matrix and :math:`U`
    the normalized  target matrix. Now that we have :math:`U`, we give the gradient function
    to the Nesterov's method optimizer and compute :math:`\theta`.

    To add the global phase back in, we can form the control circuit as

    .. math::

        \frac{\langle Vct(\theta),U'\rangle}{|\langle Vct(\theta),U'\rangle|}Vct(\theta).

    Note that while we optimized using Nesterov's method in the paper, this was for its convergence
    guarantees, not its speed in practice. It is much faster to use L-BFGS which is used as a
    default optimizer in this implementation.

    A basic usage of the algorithm should consist of the following steps:

        1. Choose a layout of the CNOT structure for the approximate circuit, e.g. ``spin`` for
            a linear layout.

        2. Choose a connectivity type, e.g. ``full`` for full connectivity between qubits.

        3. Construct a CNOT unit structure via
            :func:`qiskit.transpiler.synthesis.aqc.make_cnot_network` with a desirable
            configuration.

        4. Create an optimizer to be used by AQC.

        5. Create an approximate circuit based on the chosen CNOT structure. At this time the only
            one template of an approximate circuit that is supported is
            :class:`qiskit.transpiler.synthesis.aqc.CNOTUnitCircuit`.

        6. Create an objective for the optimization problem. At this time the only one objective
            that is supported is: :class:`qiskit.transpiler.synthesis.aqc.DefaultCNOTUnitObjective`.

        7. Compile a target unitary. This may take significant time of a number of qubits is large
            and if the target circuit is deep. As a result of the optimization procedure
            ``approximate_circuit`` contains an circuit that approximates the target circuit.

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
