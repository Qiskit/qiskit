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

"""Conjugate Gradient algorithm."""

import logging

from scipy.optimize import minimize

from qiskit.aqua.components.optimizers import Optimizer


logger = logging.getLogger(__name__)


class CG(Optimizer):
    """Conjugate Gradient algorithm.

    Uses scipy.optimize.minimize CG
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    CONFIGURATION = {
        'name': 'CG',
        'description': 'CG Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'cg_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 20
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'gtol': {
                    'type': 'number',
                    'default': 1e-05
                },
                'tol': {
                    'type': ['number', 'null'],
                    'default': None
                },
                'eps': {
                    'type': 'number',
                    'default': 1.4901161193847656e-08
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp', 'gtol', 'eps'],
        'optimizer': ['local']
    }

    # pylint: disable=unused-argument
    def __init__(self, maxiter=20, disp=False, gtol=1e-5, tol=None, eps=1.4901161193847656e-08):
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter (int): Maximum number of iterations to perform.
            disp (bool): Set to True to print convergence messages.
            gtol (float): Gradient norm must be less than gtol before successful termination.
            tol (float or None): Tolerance for termination.
            eps (float): If jac is approximated, use this value for the step size.
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v
        self._tol = tol

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        if gradient_function is None and self._max_evals_grouped > 1:
            epsilon = self._options['eps']
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                        (objective_function, epsilon,
                                                         self._max_evals_grouped))

        res = minimize(objective_function, initial_point, jac=gradient_function,
                       tol=self._tol, method="CG", options=self._options)
        return res.x, res.fun, res.nfev
