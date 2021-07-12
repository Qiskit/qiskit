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

import numpy as np

from .optimizers import GDOptimizer
from .parametric_circuit import ParametricCircuit


class AQC:
    """
    Main entry point to Approximate Quantum Compiler.
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
        self, target_matrix: np.ndarray, cnots: np.ndarray, thetas0: np.ndarray
    ) -> ParametricCircuit:
        """
        Approximately compiles a circuit represented as a unitary matrix using a passed cnot unit
        structure and starting from arbitrary specified values of angles/parameters.

        Args:
            target_matrix: a unitary matrix to approximate.
            cnots: a cnot structure to be used to construct an approximate circuit.
            thetas0: initial values of angles/parameters to start optimization from.

        Returns:
            A parametric circuit that approximate target matrix.
        """
        assert isinstance(target_matrix, np.ndarray)
        assert isinstance(cnots, np.ndarray)
        assert isinstance(thetas0, np.ndarray)

        num_qubits = int(round(np.log2(target_matrix.shape[0])))
        self._compute_optional_parameters(num_qubits)

        parametric_circuit = ParametricCircuit(num_qubits=num_qubits, cnots=cnots, thetas=thetas0)

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
