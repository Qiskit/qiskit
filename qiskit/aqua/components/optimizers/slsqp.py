# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import logging

from scipy.optimize import minimize

from qiskit.aqua.components.optimizers import Optimizer

logger = logging.getLogger(__name__)


class SLSQP(Optimizer):
    """Sequential Least SQuares Programming algorithm

    Uses scipy.optimize.minimize SLSQP
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    CONFIGURATION = {
        'name': 'SLSQP',
        'description': 'SLSQP Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'cobyla_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 100
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'ftol': {
                    'type': 'number',
                    'default': 1e-06
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
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp', 'ftol', 'eps'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=100, disp=False, ftol=1e-06, tol=None, eps=1.4901161193847656e-08):
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter (int): Maximum number of iterations.
            disp (bool): Set to True to print convergence messages.
            ftol (float): Precision goal for the value of f in the stopping criterion.
            tol (float or None): Tolerance for termination.
            eps (float): Step size used for numerical approximation of the Jacobian.
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v
        self._tol = tol

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        if gradient_function is None and self._max_evals_grouped > 1:
            epsilon = self._options['eps']
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff, (objective_function, epsilon, self._max_evals_grouped))

        res = minimize(objective_function, initial_point, jac=gradient_function, tol=self._tol, bounds=variable_bounds, method="SLSQP",
                       options=self._options)
        return res.x, res.fun, res.nfev
