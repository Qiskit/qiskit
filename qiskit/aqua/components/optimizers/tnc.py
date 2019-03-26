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


class TNC(Optimizer):
    """Truncated Newton (TNC) algorithm.

    Uses scipy.optimize.minimize TNC
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    CONFIGURATION = {
        'name': 'TNC',
        'description': 'TNC Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'tnc_schema',
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
                'accuracy': {
                    'type': 'number',
                    'default': 0
                },
                'ftol': {
                    'type': 'number',
                    'default': -1
                },
                'xtol': {
                    'type': 'number',
                    'default': -1
                },
                'gtol': {
                    'type': 'number',
                    'default': -1
                },
                'tol': {
                    'type': ['number', 'null'],
                    'default': None
                },
                'eps': {
                    'type': 'number',
                    'default': 1e-08
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp', 'accuracy', 'ftol', 'xtol', 'gtol', 'eps'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=100, disp=False, accuracy=0, ftol=-1, xtol=-1,
                 gtol=-1, tol=None, eps=1e-08):
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter (int): Maximum number of function evaluation.
            disp (bool): Set to True to print convergence messages.
            accuracy (float): Relative precision for finite difference calculations.
                              If <= machine_precision, set to sqrt(machine_precision).
                              Defaults to 0.
            ftol (float): Precision goal for the value of f in the stopping criterion.
                          If ftol < 0.0, ftol is set to 0.0 defaults to -1.
            xtol (float): Precision goal for the value of x in the stopping criterion
                          (after applying x scaling factors).
                          If xtol < 0.0, xtol is set to sqrt(machine_precision).
                          Defaults to -1.
            gtol (float): Precision goal for the value of the projected gradient in
                          the stopping criterion (after applying x scaling factors).
                          If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
                          Setting it to 0.0 is not recommended. Defaults to -1.
            tol (float or None): Tolerance for termination.
            eps (float): Step size used for numerical approximation of the jacobian.
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

        res = minimize(objective_function, initial_point, jac=gradient_function, tol=self._tol,
                       bounds=variable_bounds, method="TNC", options=self._options)
        # Note: nfev here seems to be iterations not function evaluations
        return res.x, res.fun, res.nfev
