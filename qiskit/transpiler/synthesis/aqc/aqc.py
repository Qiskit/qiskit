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

import logging
import numpy as np
from .optimizers import FISTAOptimizer, GDOptimizer
from .parametric_circuit import ParametricCircuit
from .compressor import EulerCompressor

logger = logging.getLogger(__name__)

# tolerance constant for equality checks
EPS = 100.0 * np.finfo(np.float64).eps


class AQC:
    """
    Main entry point to Approximate Quantum Compiler.
    """

    # TODO: rename "reg" to something more meaningful, lambda ?
    def __init__(
        self,
        method: str = "nesterov",
        maxiter: int = 100,
        eta: float = 0.1,
        tol: float = 1e-5,
        eps: float = 0,
        reg: float = 0.2,
        group=False,
        group_size=4,
    ):
        """

        Args:
            method:
            maxiter:
            eta:
            tol:
            eps:
            reg:
            group:
            group_size:
        """
        super().__init__()
        self._method = method
        self._maxiter = maxiter
        self._eta = eta
        self._tol = tol
        self._eps = eps
        self._reg = reg
        self._group = group
        self._group_size = group_size

    def compile_unitary(
        self, target_matrix: np.ndarray, cnots: np.ndarray, thetas0: np.ndarray
    ) -> ParametricCircuit:
        """

        Args:
            target_matrix:
            cnots:
            thetas0:

        Returns:
            A parametric circuit that approximate target matrix.
        """
        assert isinstance(target_matrix, np.ndarray)
        assert isinstance(cnots, np.ndarray)
        assert isinstance(thetas0, np.ndarray)

        gradient_backend = "default"
        num_qubits = int(round(np.log2(target_matrix.shape[0])))
        self._compute_optional_parameters(num_qubits)

        logger.debug("Optimizing via FISTA ...")
        circuit = ParametricCircuit(num_qubits=num_qubits, cnots=cnots, thetas=thetas0)
        circuit.init_gradient_backend(gradient_backend)
        circuit.set_thetas(thetas0)
        optimizer = FISTAOptimizer(
            method=self._method,
            maxiter=self._maxiter,
            eta=self._eta,
            tol=self._tol,
            eps=self._eps,
            reg=self._reg,
            group=True,
        )
        thetas, _, _, _ = optimizer.optimize(target_matrix, circuit)
        print(f"FISTA thetas:\n{thetas}")
        # TODO: remove
        assert np.allclose(thetas, circuit.thetas, atol=EPS, rtol=EPS)

        logger.debug("Compressing the circuit ...")
        compressed_circuit = EulerCompressor(synth=False).compress(circuit)
        print(f"compressed_circuit.thetas {compressed_circuit.thetas}")
        print(f"compressed_circuit.cnots {compressed_circuit.cnots}")

        logger.debug("Re-optimizing via gradient descent ...")
        # todo: why re-instantiate the gradient? is it stateful? just rest should be enough!
        compressed_circuit.init_gradient_backend(gradient_backend)
        optimizer = GDOptimizer(self._method, self._maxiter, self._eta, self._tol, self._eps)
        thetas, _, _, thetas_min = optimizer.optimize(target_matrix, compressed_circuit)
        # TODO: remove
        assert np.allclose(thetas_min, compressed_circuit.thetas, atol=EPS, rtol=EPS)
        return compressed_circuit

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
