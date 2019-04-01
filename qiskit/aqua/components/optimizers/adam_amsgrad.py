
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

import csv
import numpy as np

from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua import aqua_globals
logger = logging.getLogger(__name__)


class ADAM(Optimizer):

    """
    Adam
    Kingma, Diederik & Ba, Jimmy. (2014).
    Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.

    AMSGRAD
    Sashank J. Reddi and Satyen Kale and Sanjiv Kumar. (2018).
    On the Convergence of Adam and Beyond. International Conference on Learning Representations.
    """
    CONFIGURATION = {
        'name': 'ADAM',
        'description': 'ADAM Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'adam_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 10000
                },
                'tol': {
                    'type': 'number',
                    'default': 1e-06
                },
                'lr': {
                    'type': 'number',
                    'default': 1e-03
                },
                'beta_1': {
                    'type': 'number',
                    'default': 0.9
                },
                'beta_2': {
                    'type': 'number',
                    'default': 0.99
                },
                'noise_factor': {
                    'type': 'number',
                    'default': 1e-08
                },
                'eps': {
                    'type': 'number',
                    'default': 1e-10
                },
                'amsgrad': {
                    'type': 'boolean',
                    'default': False
                },
                'save': {
                    'type': 'boolean',
                    'default': False
                },
                'path': {
                    'type': 'string',
                    'default': ''
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.supported
        },
        'options': ['maxiter', 'tol', 'lr', 'beta_1', 'beta_2', 'noise_factor', 'eps', 'amsgrad', 'save', 'path'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=10000, tol=1e-6, lr=1e-3, beta_1=0.9, beta_2=0.99, noise_factor=1e-8,
                 eps=1e-10, amsgrad=False, save=False, path=''):
        """
        Constructor.

        maxiter: int, Maximum number of iterations
        tol: float, Tolerance for termination
        lr: float >= 0, Learning rate.
        beta_1: float, 0 < beta < 1, Generally close to 1.
        beta_2: float, 0 < beta < 1, Generally close to 1.
        noise_factor: float >= 0, Noise factor
        eps: float >=0, Epsilon to be used for finite differences if no analytic gradient method is given.
        amsgrad: Boolean, use AMSGRAD or not
        save: Boolean, if True - save the optimizer's parameter after every step
        path: str, path where to save optimizer's parameters if save==True
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v
        self._maxiter = maxiter
        self._save = save
        self._path = path
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad
        self._t = 0 #time steps


    def minimize(self, objective_function, initial_point, gradient_function):
        derivative = gradient_function(initial_point)
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        if self._save:
            if self._amsgrad:
                with open(self._path + 'adam_params.csv', mode='w') as csv_file:
                    fieldnames = ['v', 'v_eff', 'm', 't']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                with open(self._path + 'adam_params.csv', mode='w') as csv_file:
                    fieldnames = ['v', 'm', 't']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        params = initial_point
        while self._t < self._maxiter:
            derivative = gradient_function(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = (params - lr_eff * self._m.flatten() / (np.sqrt(self._v.flatten()) + self._noise_factor))
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = (params - lr_eff * self._m.flatten() / (np.sqrt(self._v_eff.flatten()) + self._noise_factor))

            if self._save:
                if self._amsgrad:
                    with open(self._path + 'adam_params.csv', mode='a') as csv_file:
                        fieldnames = ['v', 'v_eff', 'm', 't']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'v': self._v, 'v_eff': self._v_eff,
                                         'm': self._m, 't': self._t})
                else:
                    with open(self._path + 'adam_params.csv', mode='a') as csv_file:
                        fieldnames = ['v', 'm', 't']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'v': self._v, 'm': self._m, 't': self._t})
            if np.linalg.norm(params - params_new) < self._tol:
                return params_new, objective_function(params_new), self._t
            else:
                params = params_new

        return params_new, objective_function(params_new), self._t

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None,
                 initial_point=None):
        """
        Perform optimization.
        Args:
            num_vars (int) : number of parameters to be optimized.
            objective_function (callable) : handle to a function that
                computes the objective function.
            gradient_function (callable) : handle to a function that
                computes the gradient of the objective function, or
                None if not available.
            variable_bounds (list[(float, float)]) : deprecated
            initial_point (numpy.ndarray[float]) : initial point.
        Returns:
            point, value, nfev
               point: is a 1D numpy.ndarray[float] containing the solution
               value: is a float with the objective function value
               nfev: number of objective function calls made if available or None
        """
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)
        if initial_point is None:
            initial_point = aqua_globals.random.rand(num_vars)
        if gradient_function is None:
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff, (objective_function, self._eps))

        point, value, nfev = self.minimize(objective_function, initial_point, gradient_function)
        return point, value, nfev

