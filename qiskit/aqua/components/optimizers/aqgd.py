# -*- coding: utf-8 -*-

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

"""Analytic Quantum Gradient Descent (AQGD) optimizer """

import logging
from copy import deepcopy
from numpy import pi, absolute, array, zeros
from qiskit.aqua.utils.validation import validate_range_exclusive_max
from .optimizer import Optimizer

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class AQGD(Optimizer):
    """Analytic Quantum Gradient Descent (AQGD) optimizer.

    Performs gradient descent optimization with a momentum term and analytic gradients
    for parametrized quantum gates, i.e. Pauli Rotations. See, for example:

    * K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. (2018).
      Quantum circuit learning. Phys. Rev. A 98, 032309.
      https://arxiv.org/abs/1803.00745

    * Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, Nathan Killoran. (2019).
      Evaluating analytic gradients on quantum hardware. Phys. Rev. A 99, 032331.
      https://arxiv.org/abs/1811.11184

    for further details on analytic gradients of parametrized quantum gates.

    Gradients are computed "analytically" using the quantum circuit when evaluating
    the objective function.
    """

    _OPTIONS = ['maxiter', 'eta', 'tol', 'disp']

    def __init__(self,
                 maxiter: int = 1000,
                 eta: float = 3.0,
                 tol: float = 1e-6,
                 disp: bool = False,
                 momentum: float = 0.25) -> None:
        """
        Args:
            maxiter: Maximum number of iterations, each iteration evaluation gradient.
            eta: The coefficient of the gradient update. Increasing this value
                 results in larger step sizes: param = previous_param - eta * deriv
            tol: The convergence criteria that must be reached before stopping.
                 Optimization stops when: absolute(loss - previous_loss) < tol
            disp: Set to True to display convergence messages.
            momentum: Bias towards the previous gradient momentum in current update.
                      Must be within the bounds: [0,1)

        """
        validate_range_exclusive_max('momentum', momentum, 0, 1)
        super().__init__()

        self._eta = eta
        self._maxiter = maxiter
        self._tol = tol if tol is not None else 1e-6
        self._disp = disp
        self._momentum_coeff = momentum
        self._previous_loss = None

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        }

    def deriv(self, j, params, obj):
        """
        Obtains the analytical quantum derivative of the objective function with
        respect to the jth parameter.

        Args:
            j (int): Index of the parameter to compute the derivative of.
            params (array): Current value of the parameters to evaluate
                            the objective function at.
            obj (callable): Objective function.
        Returns:
            float: The derivative of the objective function w.r.t. j
        """
        # create a copy of the parameters with the positive shift
        plus_params = deepcopy(params)
        plus_params[j] += pi / 2

        # create a copy of the parameters with the negative shift
        minus_params = deepcopy(params)
        minus_params[j] -= pi / 2

        # return the derivative value
        return 0.5 * (obj(plus_params) - obj(minus_params))

    def update(self, j, params, deriv, mprev):
        """
        Updates the jth parameter based on the derivative and previous momentum

        Args:
            j (int): Index of the parameter to compute the derivative of.
            params (array): Current value of the parameters to evaluate
                            the objective function at.
            deriv (float): Value of the derivative w.r.t. the jth parameter
            mprev (array): Array containing all of the parameter momentums
        Returns:
            tuple: params, new momentums
        """
        mnew = self._eta * (deriv * (1 - self._momentum_coeff) + mprev[j] * self._momentum_coeff)
        params[j] -= mnew
        return params, mnew

    def converged(self, objval, n=2):
        """
        Determines if the objective function has converged by finding the difference between
        the current value and the previous n values.

        Args:
            objval (float): Current value of the objective function.
            n (int): Number of previous steps which must be within the convergence criteria
                     in order to be considered converged. Using a larger number will prevent
                     the optimizer from stopping early.

        Returns:
            bool: Whether or not the optimization has converged.
        """
        if self._previous_loss is None:
            self._previous_loss = [objval + 2 * self._tol] * n

        if all(absolute(objval - prev) < self._tol for prev in self._previous_loss):
            # converged
            return True

        # store previous function evaluations
        for i in range(n):
            if i < n - 1:
                self._previous_loss[i] = self._previous_loss[i + 1]
            else:
                self._previous_loss[i] = objval

        return False

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        params = array(initial_point)
        it = 0
        momentum = zeros(shape=(num_vars,))
        objval = objective_function(params)

        if self._disp:
            print("Iteration: " + str(it) + " \t| Energy: " + str(objval))

        minobj = objval
        minparams = params

        while it < self._maxiter and not self.converged(objval):
            for j in range(num_vars):
                # update parameters in order based on quantum gradient
                derivative = self.deriv(j, params, objective_function)
                params, momentum[j] = self.update(j, params, derivative, momentum)

            # check the value of the objective function
            objval = objective_function(params)

            # keep the best parameters
            if objval < minobj:
                minobj = objval
                minparams = params

            # update the iteration count
            it += 1
            if self._disp:
                print("Iteration: " + str(it) + " \t| Energy: " + str(objval))

        return minparams, minobj, it
