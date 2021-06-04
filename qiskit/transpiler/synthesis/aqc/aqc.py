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
        self, target_matrix: np.ndarray, cnots: np.ndarray, thetas0: np.ndarray, verbose: int = 0
    ) -> ParametricCircuit:
        """
        TODO: description.
        """
        _EPS = 100.0 * np.finfo(np.float64).eps
        assert isinstance(target_matrix, np.ndarray)
        assert isinstance(cnots, np.ndarray)
        assert isinstance(thetas0, np.ndarray)
        assert isinstance(verbose, int)
        gradient_backend = "default"
        num_qubits = int(round(np.log2(target_matrix.shape[0])))
        self._adjust_optimization_parameters(num_qubits)

        self._message("Optimizing via FISTA ...", verbose)
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
        thetas, obj, gra, _ = optimizer.optimize(target_matrix, circuit)
        assert np.allclose(thetas, circuit.thetas, atol=_EPS, rtol=_EPS)

        self._message("Compressing the circuit ...", verbose)
        compressed_circuit = EulerCompressor(synth=False).compress(circuit)

        self._message("Re-optimizing via gradient descent ...", verbose)
        compressed_circuit.init_gradient_backend(gradient_backend)
        optimizer = GDOptimizer(self._method, self._maxiter, self._eta, self._tol, self._eps)
        thetas, obj, gra, thetas_min = optimizer.optimize(target_matrix, compressed_circuit)
        assert np.allclose(thetas_min, compressed_circuit.thetas, atol=_EPS, rtol=_EPS)
        return compressed_circuit

    def _adjust_optimization_parameters(self, num_qubits: int):
        """
        TODO: description.
        """
        if num_qubits <= 3:
            self._maxiter = 200
            self._eta = 0.1
        elif num_qubits == 4:
            self._maxiter = 350
            self._eta = 0.06
        elif num_qubits >= 5:
            self._maxiter = 500
            self._eta = 0.03

    @staticmethod
    def _message(msg: str, verbose: int = 0):
        if verbose >= 1:
            print(msg)
        logger.debug(msg)
