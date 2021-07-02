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

"""IMplicit FILtering (IMFIL) optimizer."""

from typing import Any, Dict

from qiskit.exceptions import MissingOptionalLibraryError
from .optimizer import Optimizer, OptimizerSupportLevel

try:
    import skquant.opt as skq

    _HAS_SKQUANT = True
except ImportError:
    _HAS_SKQUANT = False


class IMFIL(Optimizer):
    """IMplicit FILtering algorithm.

    Implicit filtering is a way to solve bound-constrained optimization problems for
    which derivatives are not available. In comparison to methods that use interpolation to
    reconstruct the function and its higher derivatives, implicit filtering builds upon
    coordinate search followed by interpolation to get an approximate gradient.

    Uses skquant.opt installed with pip install scikit-quant.
    For further detail, please refer to
    https://github.com/scikit-quant/scikit-quant and https://qat4chem.lbl.gov/software.
    """

    def __init__(
        self,
        maxiter: int = 1000,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.

        Raises:
            MissingOptionalLibraryError: scikit-quant not installed
        """
        if not _HAS_SKQUANT:
            raise MissingOptionalLibraryError(
                libname="scikit-quant", name="IMFIL", pip_install="pip install scikit-quant"
            )
        super().__init__()
        self._maxiter = maxiter

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
        res, history = skq.minimize(
            func=objective_function,
            x0=initial_point,
            bounds=variable_bounds,
            budget=self._maxiter,
            method="imfil",
        )
        return res.optpar, res.optval, len(history)
