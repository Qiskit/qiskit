# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Stable Noisy Optimization by Branch and FIT algorithm (SNOBFIT) optimizer."""

from typing import Any, Dict, Optional, Callable, Tuple, List

import numpy as np
from qiskit.utils import optionals as _optionals
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT


@_optionals.HAS_SKQUANT.require_in_instance
@_optionals.HAS_SQSNOBFIT.require_in_instance
class SNOBFIT(Optimizer):
    """Stable Noisy Optimization by Branch and FIT algorithm.

    SnobFit is used for the optimization of derivative-free, noisy objective functions providing
    robust and fast solutions of problems with continuous variables varying within bound.

    Uses skquant.opt installed with pip install scikit-quant.
    For further detail, please refer to
    https://github.com/scikit-quant/scikit-quant and https://qat4chem.lbl.gov/software.
    """

    def __init__(
        self,
        maxiter: int = 1000,
        maxfail: int = 10,
        maxmp: int = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.
            maxmp: Maximum number of  model points requested for the local fit.
                 Default = 2 * number of parameters + 6 set to this value when None.
            maxfail: Maximum number of failures to improve the solution. Stops the algorithm
                    after maxfail is reached.
            verbose: Provide verbose (debugging) output.

        Raises:
            MissingOptionalLibraryError: scikit-quant or SQSnobFit not installed
        """
        super().__init__()
        self._maxiter = maxiter
        self._maxfail = maxfail
        self._maxmp = maxmp
        self._verbose = verbose

    def get_support_level(self):
        """Returns support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.required,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self) -> Dict[str, Any]:
        return {
            "maxiter": self._maxiter,
            "maxfail": self._maxfail,
            "maxmp": self._maxmp,
            "verbose": self._verbose,
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        import skquant.opt as skq
        from SQSnobFit import optset

        snobfit_settings = {
            "maxmp": self._maxmp,
            "maxfail": self._maxfail,
            "verbose": self._verbose,
        }
        options = optset(optin=snobfit_settings)
        # counters the error when initial point is outside the acceptable bounds
        x0 = np.asarray(x0)
        for idx, theta in enumerate(x0):
            if abs(theta) > bounds[idx][0]:
                x0[idx] = x0[idx] % bounds[idx][0]
            elif abs(theta) > bounds[idx][1]:
                x0[idx] = x0[idx] % bounds[idx][1]

        res, history = skq.minimize(
            fun,
            x0,
            bounds=bounds,
            budget=self._maxiter,
            method="snobfit",
            options=options,
        )

        optimizer_result = OptimizerResult()
        optimizer_result.x = res.optpar
        optimizer_result.fun = res.optval
        optimizer_result.nfev = len(history)
        return optimizer_result

    def optimize(
        self,
        num_vars,
        objective_function,
        gradient_function=None,
        variable_bounds=None,
        initial_point=None,
    ):
        """Runs the optimization."""
        super().optimize(
            num_vars, objective_function, gradient_function, variable_bounds, initial_point
        )
        result = self.minimize(
            objective_function, initial_point, gradient_function, variable_bounds
        )
        return result.x, result.fun, result.nfev
