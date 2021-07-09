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
Gradient descent optimizers for AQC.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy import linalg as la

from .parametric_circuit import ParametricCircuit


class OptimizerBase(ABC):
    """Interface class to any circuit optimizer."""

    @abstractmethod
    def optimize(
        self, target_matrix: np.ndarray, circuit: ParametricCircuit
    ) -> Tuple[np.ndarray, float]:
        """
        Optimizes parameters of parametric circuit taking the current circuit
        parameters for initial point. N O T E: parameters of the circuit
        will be modified upon completion.
        Args:
            target_matrix: target unitary matrix we try to approximate by
                           tuning the parameters of parametric circuit.
            circuit: an instance of parametric circuit as a circuit
                     whose parameters will be optimized.
        Returns:
            a tuple of an array with best thetas and a minimal error achieved.
        """
        raise NotImplementedError("Abstract method is called!")


class GDOptimizer(OptimizerBase):
    """Implements the gradient descent optimization algorithm."""

    def __init__(
        self,
        method: str = "nesterov",
        maxiter: int = 100,
        eta: float = 0.1,
        tol: float = 1e-5,
        eps: float = 0,
    ) -> None:
        """
        Args:
            method: gradient descent method, either ``vanilla`` or ``nesterov``.
            maxiter: maximum number of iterations to run.
            eta: learning rate/step size.
            tol: defines an error tolerance when to stop the optimizer.
            eps: size of the noise to be added to escape local minima.
        """
        super().__init__()
        self._method = method
        self._maxiter = maxiter
        self._eta = eta
        self._tol = tol
        self._eps = eps

    def optimize(
        self, target_matrix: np.ndarray, circuit: ParametricCircuit
    ) -> Tuple[np.ndarray, float]:
        """
        Gradient descent algorithm. See the base class description.
        """
        # thetas0 = circuit.thetas.copy()

        aux = np.empty(0)
        if self._method == "nesterov":
            aux = circuit.thetas
        # obj = np.zeros(self._maxiter)
        # gra = np.zeros(self._maxiter)

        error, der = circuit.get_gradient(target_matrix)
        # todo: add callback and pass error and derivative to the callback
        # obj[0] = error
        # gra[0] = la.norm(der)
        gra = la.norm(der)  # grad_norm

        iter_count = 1
        thetas = circuit.thetas
        thetas_best = circuit.thetas
        max_error = np.inf

        while gra > self._tol and iter_count < self._maxiter:
            # add noise if required
            if self._eps != 0:
                noise = np.random.normal(0, 2 * np.pi, circuit.num_thetas)
                noise = noise / la.norm(noise) * self._eps / iter_count
                der += noise
            if self._method == "vanilla":
                thetas = thetas - self._eta * der
            elif self._method == "nesterov":
                newaux = thetas - self._eta * der
                thetas = newaux + iter_count / (iter_count + 3) * (newaux - aux)
                aux = newaux

            # update thetas with new values
            circuit.set_thetas(thetas)

            # prepare for the next iteration
            error, der = circuit.get_gradient(target_matrix)
            # todo: add callback and pass error and derivative to the callback
            # obj[i] = error
            # gra[i] = la.norm(der)
            gra = la.norm(der)
            if error < max_error:
                thetas_best = thetas
                max_error = error
            iter_count += 1
        # obj = obj[0:i]
        # gra = gra[0:i]

        # update thetas in the circuit
        circuit.set_thetas(thetas_best)

        # final thetas, objective, and gradient, min thetas
        # return thetas, obj, gra, thetas_min
        # return thetas, objective function value
        return thetas_best, max_error
