# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Wrapper class of scipy.optimize.minimize."""

from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from scipy.optimize import Bounds, minimize

from qiskit.utils.validation import validate_min

from .optimizer import Optimizer, OptimizerSupportLevel


class SciPyOptimizer(Optimizer):
    """A general Qiskit Optimizer wrapping scipy.optimize.minimize.

    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _bounds_support_methods = {"l-bfgs-b", "tnc", "slsqp", "powell", "trust-constr"}
    _gradient_support_methods = {
        "cg",
        "bfgs",
        "newton-cg",
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    def __init__(
        self,
        method: Union[str, Callable],
        options: Optional[Dict[str, Any]] = None,
        max_evals_grouped: int = 1,
        **kwargs,
    ):
        """
        Args:
            method: Type of solver.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
            max_evals_grouped: Max number of default gradient evaluations performed simultaneously.
        """
        # pylint: disable=super-init-not-called
        self._method = method.lower() if isinstance(method, str) else method
        # Set support level
        if self._method in self._bounds_support_methods:
            self._bounds_support_level = OptimizerSupportLevel.supported
        else:
            self._bounds_support_level = OptimizerSupportLevel.ignored
        if self._method in self._gradient_support_methods:
            self._gradient_support_level = OptimizerSupportLevel.supported
        else:
            self._gradient_support_level = OptimizerSupportLevel.ignored
        self._initial_point_support_level = OptimizerSupportLevel.required

        self._options = options if options is not None else {}
        validate_min("max_evals_grouped", max_evals_grouped, 1)
        self._max_evals_grouped = max_evals_grouped
        self._kwargs = kwargs

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": self._gradient_support_level,
            "bounds": self._bounds_support_level,
            "initial_point": self._initial_point_support_level,
        }

    @property
    def settings(self) -> Dict[str, Any]:
        settings = {
            "max_evals_grouped": self._max_evals_grouped,
            "options": self._options,
            **self._kwargs,
        }
        # the subclasses don't need the "method" key as the class type specifies the method
        if self.__class__ == SciPyOptimizer:
            settings["method"] = self._method

        return settings

    def optimize(
        self,
        num_vars,
        objective_function,
        gradient_function=None,
        variable_bounds: Optional[Union[Sequence, Bounds]] = None,
        initial_point=None,
    ):
        # Remove ignored parameters to supress the warning of scipy.optimize.minimize
        if self.is_bounds_ignored:
            variable_bounds = None
        if self.is_gradient_ignored:
            gradient_function = None

        if self.is_gradient_supported and gradient_function is None and self._max_evals_grouped > 1:
            if "eps" in self._options:
                epsilon = self._options["eps"]
            else:
                epsilon = (
                    1e-8 if self._method in {"l-bfgs-b", "tnc"} else np.sqrt(np.finfo(float).eps)
                )
            gradient_function = Optimizer.wrap_function(
                Optimizer.gradient_num_diff, (objective_function, epsilon, self._max_evals_grouped)
            )

        # Workaround for L_BFGS_B because it does not accept np.ndarray.
        # See https://github.com/Qiskit/qiskit-terra/pull/6373.
        if gradient_function is not None and self._method == "l-bfgs-b":
            gradient_function = self._wrap_gradient(gradient_function)

        # Validate the input
        super().optimize(
            num_vars,
            objective_function,
            gradient_function=gradient_function,
            variable_bounds=variable_bounds,
            initial_point=initial_point,
        )

        res = minimize(
            fun=objective_function,
            x0=initial_point,
            method=self._method,
            jac=gradient_function,
            bounds=variable_bounds,
            options=self._options,
            **self._kwargs,
        )
        return res.x, res.fun, res.nfev

    @staticmethod
    def _wrap_gradient(gradient_function):
        def wrapped_gradient(x):
            gradient = gradient_function(x)
            if isinstance(gradient, np.ndarray):
                return gradient.tolist()
            return gradient

        return wrapped_gradient
