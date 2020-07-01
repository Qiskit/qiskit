# -*- coding: utf-8 -*-

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

"""Simultaneous Perturbation Stochastic Approximation optimizer."""

import warnings
from typing import Optional, List, Callable
import logging

import numpy as np

from qiskit.aqua import aqua_globals
from qiskit.aqua.utils.validation import validate_min
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class SPSA(Optimizer):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA is an algorithmic method for optimizing systems with multiple unknown parameters.
    As an optimization method, it is appropriately suited to large-scale population models,
    adaptive modeling, and simulation optimization.

    .. seealso::
        Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.

    SPSA is a descent method capable of finding global minima,
    sharing this property with other methods as simulated annealing.
    Its main feature is the gradient approximation, which requires only two
    measurements of the objective function, regardless of the dimension of the optimization
    problem.

    .. note::

        SPSA can be used in the presence of noise, and it is therefore indicated in situations
        involving measurement uncertainty on a quantum computation when finding a minimum.
        If you are executing a variational algorithm using a Quantum ASseMbly Language (QASM)
        simulator or a real device, SPSA would be the most recommended choice among the optimizers
        provided here.

    The optimization process includes a calibration phase, which requires additional
    functional evaluations.

    For further details, please refer to https://arxiv.org/pdf/1704.05018v2.pdf#section*.11
    (Supplementary information Section IV.)
    """

    _C0 = 2 * np.pi * 0.1
    _OPTIONS = ['save_steps', 'last_avg']

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: int = 1000,
                 save_steps: int = 1,
                 last_avg: int = 1,
                 c0: float = _C0,
                 c1: float = 0.1,
                 c2: float = 0.602,
                 c3: float = 0.101,
                 c4: float = 0,
                 skip_calibration: float = False,
                 max_trials: Optional[int] = None) -> None:
        """
        Args:
            maxiter: Maximum number of iterations to perform.
            save_steps: Save intermediate info every save_steps step. It has a min. value of 1.
            last_avg: Averaged parameters over the last_avg iterations.
                If last_avg = 1, only the last iteration is considered. It has a min. value of 1.
            c0: The initial a. Step size to update parameters.
            c1: The initial c. The step size used to approximate gradient.
            c2: The alpha in the paper, and it is used to adjust a (c0) at each iteration.
            c3: The gamma in the paper, and it is used to adjust c (c1) at each iteration.
            c4: The parameter used to control a as well.
            skip_calibration: Skip calibration and use provided c(s) as is.
            max_trials: Deprecated, use maxiter.
        """
        validate_min('save_steps', save_steps, 1)
        validate_min('last_avg', last_avg, 1)
        super().__init__()
        if max_trials is not None:
            warnings.warn('The max_trials parameter is deprecated as of '
                          '0.8.0 and will be removed no sooner than 3 months after the release. '
                          'You should use maxiter instead.',
                          DeprecationWarning)
            maxiter = max_trials
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v
        self._maxiter = maxiter
        self._parameters = np.array([c0, c1, c2, c3, c4])
        self._skip_calibration = skip_calibration

    def get_support_level(self):
        """ return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        if not isinstance(initial_point, np.ndarray):
            initial_point = np.asarray(initial_point)

        logger.debug('Parameters: %s', self._parameters)
        if not self._skip_calibration:
            # at least one calibration, at most 25 calibrations
            num_steps_calibration = min(25, max(1, self._maxiter // 5))
            self._calibration(objective_function, initial_point, num_steps_calibration)
        else:
            logger.debug('Skipping calibration, parameters used as provided.')

        opt, sol, _, _, _, _ = self._optimization(objective_function,
                                                  initial_point,
                                                  maxiter=self._maxiter,
                                                  **self._options)
        return sol, opt, None

    def _optimization(self,
                      obj_fun: Callable,
                      initial_theta: np.ndarray,
                      maxiter: int,
                      save_steps: int = 1,
                      last_avg: int = 1) -> List:
        """Minimizes obj_fun(theta) with a simultaneous perturbation stochastic
        approximation algorithm.

        Args:
            obj_fun: the function to minimize
            initial_theta: initial value for the variables of obj_fun
            maxiter: the maximum number of trial steps ( = function
                calls/2) in the optimization
            save_steps: stores optimization outcomes each 'save_steps'
                trial steps
            last_avg: number of last updates of the variables to average
                on for the final obj_fun
        Returns:
            a list with the following elements:
                cost_final : final optimized value for obj_fun
                theta_best : final values of the variables corresponding to
                    cost_final
                cost_plus_save : array of stored values for obj_fun along the
                    optimization in the + direction
                cost_minus_save : array of stored values for obj_fun along the
                    optimization in the - direction
                theta_plus_save : array of stored variables of obj_fun along the
                    optimization in the + direction
                theta_minus_save : array of stored variables of obj_fun along the
                    optimization in the - direction
        """

        theta_plus_save = []
        theta_minus_save = []
        cost_plus_save = []
        cost_minus_save = []
        theta = initial_theta
        theta_best = np.zeros(initial_theta.shape)
        for k in range(maxiter):
            # SPSA Parameters
            a_spsa = float(self._parameters[0]) / np.power(k + 1 + self._parameters[4],
                                                           self._parameters[2])
            c_spsa = float(self._parameters[1]) / np.power(k + 1, self._parameters[3])
            delta = 2 * aqua_globals.random.integers(2, size=np.shape(initial_theta)[0]) - 1
            # plus and minus directions
            theta_plus = theta + c_spsa * delta
            theta_minus = theta - c_spsa * delta
            # cost function for the two directions
            if self._max_evals_grouped > 1:
                cost_plus, cost_minus = obj_fun(np.concatenate((theta_plus, theta_minus)))
            else:
                cost_plus = obj_fun(theta_plus)
                cost_minus = obj_fun(theta_minus)
            # derivative estimate
            g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
            # updated theta
            theta = theta - a_spsa * g_spsa
            # saving
            if k % save_steps == 0:
                logger.debug('Objective function at theta+ for step # %s: %1.7f', k, cost_plus)
                logger.debug('Objective function at theta- for step # %s: %1.7f', k, cost_minus)
                theta_plus_save.append(theta_plus)
                theta_minus_save.append(theta_minus)
                cost_plus_save.append(cost_plus)
                cost_minus_save.append(cost_minus)

            if k >= maxiter - last_avg:
                theta_best += theta / last_avg
        # final cost update
        cost_final = obj_fun(theta_best)
        logger.debug('Final objective function is: %.7f', cost_final)

        return [cost_final, theta_best, cost_plus_save, cost_minus_save,
                theta_plus_save, theta_minus_save]

    def _calibration(self,
                     obj_fun: Callable,
                     initial_theta: np.ndarray,
                     stat: int):
        """Calibrates and stores the SPSA parameters back.

        SPSA parameters are c0 through c5 stored in parameters array

        c0 on input is target_update and is the aimed update of variables on the first trial step.
        Following calibration c0 will be updated.

        c1 is initial_c and is first perturbation of initial_theta.

        Args:
            obj_fun: the function to minimize.
            initial_theta: initial value for the variables of obj_fun.
            stat: number of random gradient directions to average on in the calibration.
        """

        target_update = self._parameters[0]
        initial_c = self._parameters[1]
        delta_obj = 0
        logger.debug("Calibration...")
        for i in range(stat):
            if i % 5 == 0:
                logger.debug('calibration step # %s of %s', str(i), str(stat))
            delta = 2 * aqua_globals.random.integers(2, size=np.shape(initial_theta)[0]) - 1
            theta_plus = initial_theta + initial_c * delta
            theta_minus = initial_theta - initial_c * delta
            if self._max_evals_grouped > 1:
                obj_plus, obj_minus = obj_fun(np.concatenate((theta_plus, theta_minus)))
            else:
                obj_plus = obj_fun(theta_plus)
                obj_minus = obj_fun(theta_minus)
            delta_obj += np.absolute(obj_plus - obj_minus) / stat

        self._parameters[0] = target_update * 2 / delta_obj \
            * self._parameters[1] * (self._parameters[4] + 1)

        logger.debug('Calibrated SPSA parameter c0 is %.7f', self._parameters[0])
