# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Adam and AMSGRAD optimizers."""

from typing import Any, Optional, Callable, Dict, Tuple, List
import os

import csv
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_arguments
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

# pylint: disable=invalid-name


class ADAM(Optimizer):
    """Adam and AMSGRAD optimizers.

    Adam [1] is a gradient-based optimization algorithm that is relies on adaptive estimates of
    lower-order moments. The algorithm requires little memory and is invariant to diagonal
    rescaling of the gradients. Furthermore, it is able to cope with non-stationary objective
    functions and noisy and/or sparse gradients.

    AMSGRAD [2] (a variant of Adam) uses a 'long-term memory' of past gradients and, thereby,
    improves convergence properties.

    References:

        [1]: Kingma, Diederik & Ba, Jimmy (2014), Adam: A Method for Stochastic Optimization.
             `arXiv:1412.6980 <https://arxiv.org/abs/1412.6980>`_

        [2]: Sashank J. Reddi and Satyen Kale and Sanjiv Kumar (2018),
             On the Convergence of Adam and Beyond.
             `arXiv:1904.09237 <https://arxiv.org/abs/1904.09237>`_

    .. note::

        This component has some function that is normally random. If you want to reproduce behavior
        then you should set the random number generator seed in the algorithm_globals
        (``qiskit.utils.algorithm_globals.random_seed = seed``).

    """

    _OPTIONS = [
        "maxiter",
        "tol",
        "lr",
        "beta_1",
        "beta_2",
        "noise_factor",
        "eps",
        "amsgrad",
        "snapshot_dir",
    ]

    def __init__(
        self,
        maxiter: int = 10000,
        tol: float = 1e-6,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
        snapshot_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations
            tol: Tolerance for termination
            lr: Value >= 0, Learning rate.
            beta_1: Value in range 0 to 1, Generally close to 1.
            beta_2: Value in range 0 to 1, Generally close to 1.
            noise_factor: Value >= 0, Noise factor
            eps : Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given.
            amsgrad: True to use AMSGRAD, False if not
            snapshot_dir: If not None save the optimizer's parameter
                after every step to the given directory
        """
        super().__init__()
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                self._options[k] = v
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad

        # runtime variables
        self._t = 0  # time steps
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir:

            with open(os.path.join(self._snapshot_dir, "adam_params.csv"), mode="w") as csv_file:
                if self._amsgrad:
                    fieldnames = ["v", "v_eff", "m", "t"]
                else:
                    fieldnames = ["v", "m", "t"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    @property
    def settings(self) -> Dict[str, Any]:
        return {
            "maxiter": self._maxiter,
            "tol": self._tol,
            "lr": self._lr,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "noise_factor": self._noise_factor,
            "eps": self._eps,
            "amsgrad": self._amsgrad,
            "snapshot_dir": self._snapshot_dir,
        }

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def save_params(self, snapshot_dir: str) -> None:
        """Save the current iteration parameters to a file called ``adam_params.csv``.

        Note:

            The current parameters are appended to the file, if it exists already.
            The file is not overwritten.

        Args:
            snapshot_dir: The directory to store the file in.
        """
        if self._amsgrad:
            with open(os.path.join(snapshot_dir, "adam_params.csv"), mode="a") as csv_file:
                fieldnames = ["v", "v_eff", "m", "t"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({"v": self._v, "v_eff": self._v_eff, "m": self._m, "t": self._t})
        else:
            with open(os.path.join(snapshot_dir, "adam_params.csv"), mode="a") as csv_file:
                fieldnames = ["v", "m", "t"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({"v": self._v, "m": self._m, "t": self._t})

    def load_params(self, load_dir: str) -> None:
        """Load iteration parameters for a file called ``adam_params.csv``.

        Args:
            load_dir: The directory containing ``adam_params.csv``.
        """
        with open(os.path.join(load_dir, "adam_params.csv")) as csv_file:
            if self._amsgrad:
                fieldnames = ["v", "v_eff", "m", "t"]
            else:
                fieldnames = ["v", "m", "t"]
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            for line in reader:
                v = line["v"]
                if self._amsgrad:
                    v_eff = line["v_eff"]
                m = line["m"]
                t = line["t"]

        v = v[1:-1]
        self._v = np.fromstring(v, dtype=float, sep=" ")
        if self._amsgrad:
            v_eff = v_eff[1:-1]
            self._v_eff = np.fromstring(v_eff, dtype=float, sep=" ")
        m = m[1:-1]
        self._m = np.fromstring(m, dtype=float, sep=" ")
        t = t[1:-1]
        self._t = np.fromstring(t, dtype=int, sep=" ")

    @deprecate_arguments(
        {
            "objective_function": "fun",
            "initial_point": "x0",
            "gradient_function": "jac",
        }
    )
    # pylint: disable=arguments-differ
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        # pylint:disable=unused-argument
        objective_function: Optional[Callable[[np.ndarray], float]] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient_function: Optional[Callable[[np.ndarray], float]] = None,
        # ) -> Tuple[np.ndarray, float, int]:
    ) -> OptimizerResult:  # TODO find proper way to deprecate return type
        """Minimize the scalar function.

        Args:
            fun: The scalar function to minimize.
            x0: The initial point for the minimization.
            jac: The gradient of the scalar function ``fun``.
            bounds: Bounds for the variables of ``fun``. This argument might be ignored if the
                optimizer does not support bounds.
            objective_function: DEPRECATED. A function handle to the objective function.
            initial_point: DEPRECATED. The initial iteration point.
            gradient_function: DEPRECATED. A function handle to the gradient of the objective
                function.

        Returns:
            The result of the optimization, containing e.g. the result as attribute ``x``.
        """
        if jac is None:
            jac = Optimizer.wrap_function(Optimizer.gradient_num_diff, (fun, self._eps))

        derivative = jac(x0)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        params = params_new = x0
        while self._t < self._maxiter:
            if self._t > 0:
                derivative = jac(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v.flatten()) + self._noise_factor
                )
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v_eff.flatten()) + self._noise_factor
                )

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)

            # check termination
            if np.linalg.norm(params - params_new) < self._tol:
                break

            params = params_new

        result = OptimizerResult()
        result.x = params_new
        result.fun = fun(params_new)
        result.nfev = self._t
        return result

    def optimize(
        self,
        num_vars: int,
        objective_function: Callable[[np.ndarray], float],
        gradient_function: Optional[Callable[[np.ndarray], float]] = None,
        variable_bounds: Optional[List[Tuple[float, float]]] = None,
        initial_point: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """Perform optimization.

        Args:
            num_vars: Number of parameters to be optimized.
            objective_function: Handle to a function that computes the objective function.
            gradient_function: Handle to a function that computes the gradient of the objective
                function.
            variable_bounds: deprecated
            initial_point: The initial point for the optimization.

        Returns:
            A tuple (point, value, nfev) where\n
                point: is a 1D numpy.ndarray[float] containing the solution\n
                value: is a float with the objective function value\n
                nfev: is the number of objective function calls

        """
        super().optimize(
            num_vars, objective_function, gradient_function, variable_bounds, initial_point
        )
        if initial_point is None:
            initial_point = algorithm_globals.random.random(num_vars)

        result = self.minimize(objective_function, initial_point, gradient_function)
        return result.x, result.fun, result.nfev
