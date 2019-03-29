import logging

import csv
import numpy as np

from qiskit.aqua.components.optimizers import Optimizer


logger = logging.getLogger(__name__)


class Adam(Optimizer):

    """
    Adam
    Kingma, Diederik & Ba, Jimmy. (2014).
    Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.

    AMSGRAD
    Sashank J. Reddi and Satyen Kale and Sanjiv Kumar. (2018).
    On the Convergence of Adam and Beyond. International Conference on Learning Representations.

    Default parameters follow those provided in the original paper.
    """
    CONFIGURATION = {
        'name': 'Adam',
        'description': 'Adam Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'cg_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 10000
                },
                'save': {
                    'type': 'boolean',
                    'default': False
                },
                'tol': {
                    'type': 'number',
                    'default': 1e-04
                },
                'lr': {
                    'type': 'number',
                    'default': 1e-03
                },
                'beta_1': {
                    'type': 'number',
                    'default': 0.7
                },
                'beta_1': {
                    'type': 'number',
                    'default': 0.999
                },
                'eps': {
                    'type': 'number',
                    'default': 1e-06
                },
                'eps_fin_diff': {
                   'type': 'number',
                           'default': 1e-03
                },
                'amsgrad': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.supported
        },
        'options': ['maxiter', 'save', 'tol', 'lr', 'beta_1', 'beta_2', 'eps', 'eps_fin_diff', 'amsgrad'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=10000, save=False, path='', tol=1e-4, lr=1e-3, beta_1=0.7, beta_2=0.999, eps=1e-6,
                 eps_fin_diff = 1e-2, amsgrad=False):
        """
        Constructor.

        maxiter: int, Maximum number of iterations
        save: Boolean, if True - save the optimizer's parameter after every step
        path: str, path where to save optimizer's parameters if save==True
        tol: float, Tolerance for termination
        lr: float >= 0, Learning rate.
        beta_1: float, 0 < beta < 1, Generally close to 1.
        beta_2: float, 0 < beta < 1, Generally close to 1.
        eps: float >= 0, Noise factor
        eps_fin_diff: float >=0, Epsilon to be used for finite differences if no analytic gradient method is given.
        amsgrad: Boolean, use AMSGRAD or not
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
        self._eps = eps
        self._eps_fin_diff = eps_fin_diff
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
                with open(self._path + 'optim_params.csv', mode='w') as csv_file:
                    fieldnames = ['v', 'v_eff', 'm', 't']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                with open(self._path + 'optim_params.csv', mode='w') as csv_file:
                    fieldnames = ['v', 'm', 't']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        params = initial_point
        while self._t < self._maxiter:
            derivative = gradient_function(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * np.power(derivative, 2 * np.ones(np.shape(derivative)))
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = (params - lr_eff * self._m.flatten() / (np.sqrt(self._v.flatten()) + self._eps))
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = (params - lr_eff * self._m.flatten() / (np.sqrt(self._v_eff.flatten()) + self._eps))

            if self._save:
                if self._amsgrad:
                    with open(self._path + 'optim_params.csv', mode='a') as csv_file:
                        fieldnames = ['v', 'v_eff', 'm', 't']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'v': self._v, 'v_eff': self._v_eff,
                                         'm': self._m, 't': self._t})
                else:
                    with open(self._path + 'optim_params.csv', mode='a') as csv_file:
                        fieldnames = ['v', 'm', 't']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'v': self._v, 'm': self._m, 't': self._t})
            if np.linalg.norm(params - params_new) < self._tol:
                return params_new, objective_function(params_new), self._t
            else:
                params = params_new



        return params_new, objective_function(params_new), self._t


    def optimize(self, num_vars, objective_function, gradient_function = None, variable_bounds = None,
                 initial_point= None):
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
            initial_point = np.random.rand(num_vars)
        if gradient_function is None:
            """
            take fin diff of Aqua
            derivative = 
            """
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff, (objective_function, self._eps_fin_diff))

        point, value, nfev = self.minimize(objective_function, initial_point, gradient_function)
        return point, value, nfev

