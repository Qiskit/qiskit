
# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Adam
Kingma, Diederik & Ba, Jimmy. (2014).
Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.

AMSGRAD
Sashank J. Reddi and Satyen Kale and Sanjiv Kumar. (2018).
On the Convergence of Adam and Beyond. International Conference on Learning Representations.
"""

import logging
import os

import csv
import numpy as np

from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua import aqua_globals
logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class ADAM(Optimizer):

    """
    Adam
    Kingma, Diederik & Ba, Jimmy. (2014).
    Adam: A Method for Stochastic Optimization.
          International Conference on Learning Representations.

    AMSGRAD
    Sashank J. Reddi and Satyen Kale and Sanjiv Kumar. (2018).
    On the Convergence of Adam and Beyond. International Conference on Learning Representations.
    """
    CONFIGURATION = {
        'name': 'ADAM',
        'description': 'ADAM Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
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
                'snapshot_dir': {
                    'type': ['string', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.supported
        },
        'options': ['maxiter', 'tol', 'lr', 'beta_1', 'beta_2', 'noise_factor', 'eps',
                    'amsgrad', 'snapshot_dir'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=10000, tol=1e-6, lr=1e-3, beta_1=0.9, beta_2=0.99, noise_factor=1e-8,
                 eps=1e-10, amsgrad=False, snapshot_dir=None):
        """
        Constructor.

        maxiter: int, Maximum number of iterations
        tol: float, Tolerance for termination
        lr: float >= 0, Learning rate.
        beta_1: float, 0 < beta < 1, Generally close to 1.
        beta_2: float, 0 < beta < 1, Generally close to 1.
        noise_factor: float >= 0, Noise factor
        eps: float >=0, Epsilon to be used for finite differences if no analytic
                        gradient method is given.
        amsgrad: Boolean, use AMSGRAD or not
        snapshot_dir: str or None, if not None save the optimizer's parameter
                        after every step to the given directory
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad
        self._t = 0  # time steps
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir:

            with open(os.path.join(self._snapshot_dir, 'adam_params.csv'), mode='w') as csv_file:
                if self._amsgrad:
                    fieldnames = ['v', 'v_eff', 'm', 't']
                else:
                    fieldnames = ['v', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    def save_params(self, snapshot_dir):
        """ save params """
        if self._amsgrad:
            with open(os.path.join(snapshot_dir, 'adam_params.csv'), mode='a') as csv_file:
                fieldnames = ['v', 'v_eff', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'v': self._v, 'v_eff': self._v_eff,
                                 'm': self._m, 't': self._t})
        else:
            with open(os.path.join(snapshot_dir, 'adam_params.csv'), mode='a') as csv_file:
                fieldnames = ['v', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'v': self._v, 'm': self._m, 't': self._t})

    def load_params(self, load_dir):
        """ load params """
        with open(os.path.join(load_dir, 'adam_params.csv'), mode='r') as csv_file:
            if self._amsgrad:
                fieldnames = ['v', 'v_eff', 'm', 't']
            else:
                fieldnames = ['v', 'm', 't']
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            for line in reader:
                v = line['v']
                if self._amsgrad:
                    v_eff = line['v_eff']
                m = line['m']
                t = line['t']

        v = v[1:-1]
        self._v = np.fromstring(v, dtype=float, sep=' ')
        if self._amsgrad:
            v_eff = v_eff[1:-1]
            self._v_eff = np.fromstring(v_eff, dtype=float, sep=' ')
        m = m[1:-1]
        self._m = np.fromstring(m, dtype=float, sep=' ')
        t = t[1:-1]
        self._t = np.fromstring(t, dtype=int, sep=' ')

    def minimize(self, objective_function, initial_point, gradient_function):
        """ minimize """
        derivative = gradient_function(initial_point)
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        params = initial_point
        while self._t < self._maxiter:
            derivative = gradient_function(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = (params - lr_eff * self._m.flatten() /
                              (np.sqrt(self._v.flatten()) + self._noise_factor))
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = (params - lr_eff * self._m.flatten() /
                              (np.sqrt(self._v_eff.flatten()) + self._noise_factor))

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)
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
            tuple(numpy.ndarray, float, int):
               point: is a 1D numpy.ndarray[float] containing the solution
               value: is a float with the objective function value
               nfev: number of objective function calls made if available or None
        """
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)
        if initial_point is None:
            initial_point = aqua_globals.random.rand(num_vars)
        if gradient_function is None:
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                        (objective_function, self._eps))

        point, value, nfev = self.minimize(objective_function, initial_point, gradient_function)
        return point, value, nfev
