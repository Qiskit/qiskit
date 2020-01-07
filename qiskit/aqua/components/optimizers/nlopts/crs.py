# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Controlled Random Search (CRS) with local mutation."""

import logging
from qiskit.aqua.components.optimizers import Optimizer
from ._nloptimizer import minimize
from ._nloptimizer import check_nlopt_valid

logger = logging.getLogger(__name__)

try:
    import nlopt
except ImportError:
    logger.info('nlopt is not installed. Please install it if you want to use them.')


class CRS(Optimizer):
    # pylint: disable=line-too-long
    """
    Controlled Random Search (CRS) with local mutation.

    NLopt global optimizer, derivative-free. See `NLOpt CRS documentation
    <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#controlled-random-search-crs-with-local-mutation>`_
    for more information.
    """

    _OPTIONS = ['max_evals']

    def __init__(self, max_evals: int = 1000) -> None:  # pylint: disable=unused-argument
        """
        Args:
            max_evals: Maximum allowed number of function evaluations.
        """
        check_nlopt_valid('CRS')
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v

    def get_support_level(self):
        """ return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function,
                         gradient_function, variable_bounds, initial_point)

        return minimize(nlopt.GN_CRS2_LM, objective_function,
                        variable_bounds, initial_point, **self._options)
