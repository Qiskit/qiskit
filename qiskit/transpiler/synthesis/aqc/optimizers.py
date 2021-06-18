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
This is the Optimizer class: anything to optimize the circuit.
"""
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from numpy import linalg as la
from scipy.optimize import fmin_l_bfgs_b

from .parametric_circuit import ParametricCircuit


logger = logging.getLogger(__name__)


class OptimizerBase(ABC):
    """Interface class to any circuit optimizer."""

    @abstractmethod
    def optimize(
        self, target_matrix: np.ndarray, circuit: ParametricCircuit
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None]]:
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
            final thetas, objective, and gradient, min thetas.
        """
        raise NotImplementedError("Abstract method is called!")


class GDOptimizer(OptimizerBase):
    """Class implements the gradient descent optimization algorithm."""

    def __init__(
        self,
        method: str = "nesterov",
        maxiter: int = 100,
        eta: float = 0.1,
        tol: float = 1e-5,
        eps: float = 0,
    ) -> None:
        super().__init__()
        self._method = method
        self._maxiter = maxiter
        self._eta = eta
        self._tol = tol
        self._eps = eps

    def optimize(
        self, target_matrix: np.ndarray, circuit: ParametricCircuit
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """
        Gradient descent algorithm. See the base class description.
        """
        thetas0 = circuit.thetas.copy()
        num_thetas = circuit.num_thetas
        aux = np.empty(0)
        if self._method == "nesterov":
            aux = thetas0
        obj = np.zeros(self._maxiter)
        gra = np.zeros(self._maxiter)
        err, der = circuit.get_gradient(target_matrix)
        obj[0] = err
        gra[0] = la.norm(der)
        i = 1
        thetas = thetas0
        thetas_min = thetas0
        errmax = 10
        while gra[i - 1] > self._tol and i < self._maxiter:
            if self._eps != 0:
                nois = np.random.normal(0, 2 * np.pi, num_thetas)
                nois = nois / la.norm(nois) * self._eps / i
                der += nois
            if self._method == "vanilla":
                thetas = thetas - self._eta * der
            elif self._method == "nesterov":
                newaux = thetas - self._eta * der
                thetas = newaux + i / (i + 3) * (newaux - aux)
                aux = newaux
            circuit.set_thetas(thetas)
            err, der = circuit.get_gradient(target_matrix)
            obj[i] = err
            gra[i] = la.norm(der)
            if err < errmax:
                thetas_min = thetas
                errmax = err
            i += 1
        obj = obj[0:i]
        gra = gra[0:i]

        # perform optimality checks:
        # self.perform_checks(thetas, tol)

        circuit.set_thetas(thetas_min)  # TODO: why thetas_min?
        return thetas, obj, gra, thetas_min


class FISTAOptimizer(OptimizerBase):
    """Class implements the FISTA optimization algorithm."""

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
    ) -> None:
        super().__init__()
        self._method = method
        self._maxiter = maxiter
        self._eta = eta
        self._tol = tol
        self._eps = eps
        self._reg = reg
        self._group = group
        self._group_size = group_size

    @staticmethod
    def _soth(x, thresh):
        """
        Soft-thresholding operator for l1 regularization (LASSO).
        """
        y = np.multiply(np.sign(x), np.maximum(np.abs(x) - thresh, 0))
        return y

    @staticmethod
    def _group_soth(x, thresh, group_size):
        """
        Group soft-thresholding operator for group l2 regularization (GROUP LASSO).
        """
        l = len(x)
        n = int(np.ceil(l / group_size))
        y = np.zeros(l)
        for i in range(n - 1):
            grp = x[group_size * i : group_size * (i + 1)]
            nrm = la.norm(grp)
            if nrm > thresh:
                y[group_size * i : group_size * (i + 1)] = (1 - thresh / nrm) * grp
        grp = x[group_size * (n - 1) : l]
        nrm = la.norm(grp)
        if nrm > thresh:
            y[group_size * (n - 1) : l] = (1 - thresh / nrm) * grp
        return y

    def optimize(
        self, target_matrix: np.ndarray, circuit: ParametricCircuit
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """
        FISTA algorithm. See the base class description.
        """
        logger.debug("FISTA/Lasso optimization ...")

        thetas0 = circuit.thetas.copy()
        num_cnots = circuit.num_cnots
        aux = np.empty(0)
        alpha = 0
        if self._method == "nesterov":
            aux = thetas0
            alpha = 1
        obj = np.zeros(self._maxiter)
        gra = np.zeros(self._maxiter)
        err, der = circuit.get_gradient(target_matrix=target_matrix)
        stop_crit = self._tol + 1
        i = 0
        thetas = thetas0
        while stop_crit > self._tol and i < self._maxiter:
            obj[i] = err
            if self._method == "vanilla":
                new_thetas = thetas - self._eta * der
                if self._group:
                    # l1 + l2
                    new_thetas[0 : 4 * (num_cnots - 1)] = self._group_soth(
                        new_thetas[0 : 4 * (num_cnots - 1)], self._eta * self._reg, self._group_size
                    )
                else:
                    # l1
                    new_thetas[0 : 4 * (num_cnots - 1)] = self._soth(
                        new_thetas[0 : 4 * (num_cnots - 1)], self._eta * self._reg
                    )

                stop_crit = la.norm((new_thetas - thetas) / self._eta)
                thetas = new_thetas
            if self._method == "nesterov":
                new_alpha = (1 + np.sqrt(1 + 4 * alpha ** 2)) / 2
                new_aux = thetas - self._eta * der
                if self._group:
                    # l1 + l2
                    new_aux[0 : 4 * (num_cnots - 1)] = self._group_soth(
                        new_aux[0 : 4 * (num_cnots - 1)], self._eta * self._reg, self._group_size
                    )
                else:
                    # l1
                    new_aux[0 : 4 * (num_cnots - 1)] = self._soth(
                        new_aux[0 : 4 * (num_cnots - 1)], self._eta * self._reg
                    )

                thetas = new_aux + (alpha - 1) / new_alpha * (new_aux - aux)
                stop_crit = la.norm((new_aux - aux) / self._eta)
                aux = new_aux
                alpha = new_alpha
            circuit.set_thetas(thetas)
            err, der = circuit.get_gradient(target_matrix=target_matrix)
            gra[i] = stop_crit
            i += 1
            if i == 1 or ((i - 1) % 20) == 0:
                logger.debug(
                    "iteration: %05d, fobj: %0.16f, |grad|: %0.16f", i - 1, err, np.linalg.norm(der)
                )

        obj = obj[0:i]
        gra = gra[0:i]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("FISTA status:")
            logger.debug("objective function: %0.16f", err)
            logger.debug("gradient norm: %0.16f", np.linalg.norm(err))
            logger.debug("number of iterations: %d", int(i))

        circuit.set_thetas(thetas)
        return thetas, obj, gra, None


class LBFGSOptimizer(OptimizerBase):
    """
    Class uses quasi-Newton L-BFGS-B optimizer as a backend.
    This can be a faster replacement for the gradient descent optimizer.
    """

    def __init__(
        self,
        nonzero_theta_mask: (np.ndarray, None) = None,
        maxiter: (int, None) = None,
    ):
        if nonzero_theta_mask is not None:
            assert isinstance(nonzero_theta_mask, np.ndarray)
            assert nonzero_theta_mask.dtype == bool
            assert np.count_nonzero(nonzero_theta_mask) > 0
        if maxiter is not None:
            assert isinstance(maxiter, int) and maxiter >= 100
        self._nonzero_mask = nonzero_theta_mask
        self._maxiter = maxiter

    def optimize(
        self, target_matrix: np.ndarray, circuit: ParametricCircuit
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """
        Optimizes parameters of parametric circuit using L-BFGS-B algorithm.
        For input/output specification see description in the base class.
        """
        logger.debug("L-BFGS optimization ...")

        # If nonzero mask was specified, we extract a subset of thetas.
        if self._nonzero_mask is not None:
            assert circuit.num_thetas == self._nonzero_mask.size
            thetas0 = circuit.thetas[self._nonzero_mask.ravel()]
        else:
            thetas0 = circuit.thetas.copy()

        maxiter = self._maxiter if self._maxiter is not None else 15000

        logger.debug("#thetas: %d, optimization problem size: %d", circuit.num_thetas, thetas0.size)
        logger.debug("max. number of iterations: %d", maxiter)

        # Optimize.
        # bounds = [-2 * np.pi, 2 * np.pi] * thetas0.size  # TODO: makes sense?
        bounds = None
        iter_counter = [0]
        tmp_grad = np.zeros_like(thetas0)
        thetas_min, fobj_min, info = fmin_l_bfgs_b(
            func=self._objective_func,
            x0=thetas0,
            fprime=None,
            args=(self._nonzero_mask, target_matrix, circuit, iter_counter, tmp_grad),
            bounds=bounds,
            m=10,
            iprint=0,
            maxiter=maxiter,
        )

        # If nonzero mask was specified, we finally update a subset of thetas.
        if self._nonzero_mask is not None:
            circuit.set_nonzero_thetas(thetas_min, self._nonzero_mask)
        else:
            circuit.set_thetas(thetas_min)

        # Print out the final status of the optimizer:
        if logger.isEnabledFor(logging.DEBUG):
            status = int(info["warnflag"])
            logger.debug("LBFGS status: %s converged", "" if status == 0 else "not ")
            if status == 0:
                logger.debug("")
            elif status == 1:
                logger.debug("Too many function evaluations or too many iterations")
            elif status == 2:
                logger.debug("Stopped for the reason: %s", info["task"])
            else:
                logger.debug("Unknown reason")
            logger.debug("Objective function: %0.16f", fobj_min)
            logger.debug("Gradient norm: %0.16f", np.linalg.norm(info["grad"]))
            logger.debug("Number of function calls made: %d", int(info["funcalls"]))
            logger.debug("Number of iterations: %d", int(info["nit"]))

        return circuit.thetas, np.array([fobj_min]), np.empty(0), None

    @staticmethod
    def _objective_func(thetas: np.ndarray, *args) -> Tuple[float, np.ndarray]:
        """
        Computes the value and gradient of objective function.
        """
        nonzero_mask, target_matrix, circuit, iter_counter, tmp_grad = args
        if nonzero_mask is not None:
            circuit.set_nonzero_thetas(thetas, nonzero_mask)
        else:
            circuit.set_thetas(thetas)

        # TODO: temporary code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("target_matrix.shape:", target_matrix.shape)
        # print("", flush=True)
        # start = time.time()

        objective, grad = circuit.get_gradient(target_matrix)

        # TODO: temporary code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("gradient computation time:", time.time() - start); print("", flush=True)

        assert grad.ndim == 1
        if nonzero_mask is not None:
            tmp_grad[:] = grad[nonzero_mask.ravel()]
        else:
            tmp_grad[:] = grad[:]

        num_iter = int(iter_counter[0])
        if num_iter >= 0:  # verbosity enabled?
            if num_iter == 0 or (num_iter % 20) == 0:
                logger.debug(
                    "iteration: %05d, fobj: %0.16f, |grad|: %0.16f",
                    num_iter,
                    objective,
                    np.linalg.norm(tmp_grad),
                )
            iter_counter[0] = num_iter + 1

        return objective, tmp_grad
