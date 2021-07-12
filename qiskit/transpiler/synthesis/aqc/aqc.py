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

from .optimizers import GDOptimizer
from .parametric_circuit import ParametricCircuit


class AQC:
    r"""
    Implementation of Approximate Quantum Compiler as described in the paper.

    We are interested in compiling a quantum circuit, which we formalize as finding the best
    circuit representation in terms of an ordered gate sequence of a target unitary matrix
    :math:`U\in \U(d)`, with some additional hardware constraints. In particular, we look at
    representations that could be constrained in terms of hardware connectivity, as well
    as gate depth, and we choose a gate basis in terms of CNOT and rotation gates.
    We recall that the combination of CNOT and rotation gates is universal in :math:`\SU(d)` and
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
        method: str = "nesterov",
        maxiter: int = 100,
        eta: float = 0.1,
        tol: float = 1e-5,
        eps: float = 0,
    ):
        """
        Args:
            method: gradient descent method, either ``vanilla`` or ``nesterov``.
                Default value ``nesterov``.
            maxiter: a maximum number of iterations to run. Default ``100``.
            eta: learning rate/step size. Default value ``0.1``.
            tol: defines an error tolerance when to stop the optimizer. Default value ``1e-5``.
            eps: size of the noise to be added to escape local minima. Default value ``0``,
                so no noise is added.
        """
        super().__init__()
        self._method = method
        self._maxiter = maxiter
        self._eta = eta
        self._tol = tol
        self._eps = eps

    def compile_unitary(
        self,
        target_matrix: np.ndarray,
        cnots: np.ndarray,
        thetas: Optional[np.ndarray] = None,
    ) -> ParametricCircuit:
        """
        Approximately compiles a circuit represented as a unitary matrix using a passed cnot unit
        structure and starting from arbitrary specified values of angles/parameters.

        Args:
            target_matrix: a unitary matrix to approximate.
            cnots: a cnot structure to be used to construct an approximate circuit.
            thetas: initial values of angles/parameters to start optimization from.

        Returns:
            A parametric circuit that approximate target matrix.
        """
        assert isinstance(target_matrix, np.ndarray)
        assert isinstance(cnots, np.ndarray)

        num_qubits = int(round(np.log2(target_matrix.shape[0])))
        self._compute_optional_parameters(num_qubits)

        parametric_circuit = ParametricCircuit(num_qubits=num_qubits, cnots=cnots, thetas=thetas)

        optimizer = GDOptimizer(self._method, self._maxiter, self._eta, self._tol, self._eps)

        _, _ = optimizer.optimize(target_matrix, parametric_circuit)
        return parametric_circuit

    def _compute_optional_parameters(self, num_qubits: int) -> None:
        """
        Computes parameters that initially were set to ``None``.

        Args:
            num_qubits: a number of qubits in an optimization problem
        """
        if num_qubits <= 3:
            self._maxiter = self._maxiter or 200
            self._eta = self._eta or 0.1
        elif num_qubits == 4:
            self._maxiter = self._maxiter or 350
            self._eta = self._eta or 0.06
        elif num_qubits >= 5:
            self._maxiter = self._maxiter or 500
            self._eta = self._eta or 0.03
