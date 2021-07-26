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

from typing import Any, Dict

import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
from .optimizer import Optimizer, OptimizerSupportLevel


try:
    import skquant.opt as skq

    _HAS_SKQUANT = True
except ImportError:
    _HAS_SKQUANT = False

try:
    from SQSnobFit import optset

    _HAS_SKSNOBFIT = True
except ImportError:
    _HAS_SKSNOBFIT = False


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
        if not _HAS_SKQUANT:
            raise MissingOptionalLibraryError(
                libname="scikit-quant", name="SNOBFIT", pip_install="pip install scikit-quant"
            )
        if not _HAS_SKSNOBFIT:
            raise MissingOptionalLibraryError(
                libname="SQSnobFit", name="SNOBFIT", pip_install="pip install SQSnobFit"
            )
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
        snobfit_settings = {
            "maxmp": self._maxmp,
            "maxfail": self._maxfail,
            "verbose": self._verbose,
        }
        options = optset(optin=snobfit_settings)
        # counters the error when initial point is outside the acceptable bounds
        for idx, theta in enumerate(initial_point):
            if abs(theta) > variable_bounds[idx][0]:
                initial_point[idx] = initial_point[idx] % variable_bounds[idx][0]
            elif abs(theta) > variable_bounds[idx][1]:
                initial_point[idx] = initial_point[idx] % variable_bounds[idx][1]
        res, history = skq.minimize(
            objective_function,
            np.array(initial_point, dtype=float),
            bounds=variable_bounds,
            budget=self._maxiter,
            method="snobfit",
            options=options,
        )
        return res.optpar, res.optval, len(history)
