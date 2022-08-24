import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from qiskit.algorithms.optimizers import NFT, QNSPSA, SPSA, GradientDescent

from .optimizer import POINT, Optimizer, OptimizerResult, OptimizerSupportLevel


class SuffixAveragingOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, n_params_suffix: int = 50) -> None:
        """
        Args:
            optimizer: The optimizer used for optimizing parameterized quantum circuits.
            n_params_suffix: The number of circuit parameters for taking the suffix average.

        References:
            [1] S. Tamiya and H. Yamasaki. 2021.
            Stochastic Gradient Line Bayesian Optimization for Efficient Noise-Robust Optimization of Parameterized Quantum Circuits.
            arXiv preprint arXiv:2111.07952.
        """

        self._n_params_suffix = n_params_suffix
        self._optimizer = optimizer

        self._circ_params = []

        if isinstance(self._optimizer, SPSA):

            def load_params(nfev, x_next, fx_next, update_step, is_accepted):
                self._circ_params.append(x_next)

        elif isinstance(self._optimizer, QNSPSA):

            def load_params(nfev, x_next, fx_next, update_step, is_accepted):
                self._circ_params.append(x_next)

        elif isinstance(self._optimizer, GradientDescent):

            def load_params(nfevs, x_next, fx_next, stepsize):
                self._circ_params.append(x_next)

        else:

            def load_params(x):
                self._circ_params.append(x)

        self._optimizer.callback = load_params

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def _return_suffix_average(self) -> List[float]:

        if isinstance(self._optimizer, NFT):
            n_params = int(len(self._circ_params[0]))
            n_iterates = int(len(self._circ_params))
            n_repitition = int(len(self._circ_params) / n_params)
            if n_repitition < self._n_params_suffix:
                warnings.warn(
                    "The total number of iterations is less than the number of parameters for taking suffix averaging."
                )
                averaged_param = np.zeros_like(self._circ_params[0])
                for j in range(n_repitition):
                    averaged_param += self._circ_params[n_iterates - n_params * j - 1]
                averaged_param /= self._n_params_suffix
                return averaged_param

            averaged_param = np.zeros_like(self._circ_params[0])
            for j in range(self._n_params_suffix):
                averaged_param += self._circ_params[n_iterates - n_params * j - 1]
            averaged_param /= self._n_params_suffix

        else:
            n_iterates = len(self._circ_params)

            if n_iterates < self._n_params_suffix:
                warnings.warn(
                    "The total number of iterations is less than the number of parameters for taking suffix averaging."
                )
                return np.average(self._circ_params, axis=0)

            averaged_param = np.zeros_like(self._circ_params[0])
            for j in range(self._n_params_suffix):
                averaged_param += self._circ_params[n_iterates - j - 1]
            averaged_param /= self._n_params_suffix

        return averaged_param

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:

        result = self._optimizer.minimize(fun, x0, jac=jac, bounds=bounds)
        result.x = self._return_suffix_average()
        result.fun = fun(np.copy(result.x))

        return result
