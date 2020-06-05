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

""" Minimize using objective function """

from typing import List, Optional, Tuple, Callable
from enum import Enum
from abc import abstractmethod
import logging
import numpy as np
from qiskit.aqua.components.optimizers import Optimizer

logger = logging.getLogger(__name__)

_HAS_NLOPT = False
try:
    import nlopt
    logger.info('NLopt version: %s.%s.%s', nlopt.version_major(),
                nlopt.version_minor(), nlopt.version_bugfix())
    _HAS_NLOPT = True
except ImportError:
    logger.info('NLopt is not installed. Please install it to use these global optimizers.')


class NLoptOptimizerType(Enum):
    """ NLopt Valid Optimizer """
    GN_CRS2_LM = 1
    GN_DIRECT_L_RAND = 2
    GN_DIRECT_L = 3
    GN_ESCH = 4
    GN_ISRES = 5


class NLoptOptimizer(Optimizer):
    """
    NLopt global optimizer base class
    """

    _OPTIONS = ['max_evals']

    def __init__(self, max_evals: int = 1000) -> None:  # pylint: disable=unused-argument
        """
        Args:
            max_evals: Maximum allowed number of function evaluations.

        Raises:
            NameError: NLopt library not installed.
        """
        if not _HAS_NLOPT:
            raise NameError("Unable to instantiate '{}', nlopt is not installed. "
                            "Please install it if you want to use them.".format(
                                self.__class__.__name__))
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v

        self._optimizer_names = {
            NLoptOptimizerType.GN_CRS2_LM: nlopt.GN_CRS2_LM,
            NLoptOptimizerType.GN_DIRECT_L_RAND: nlopt.GN_DIRECT_L_RAND,
            NLoptOptimizerType.GN_DIRECT_L: nlopt.GN_DIRECT_L,
            NLoptOptimizerType.GN_ESCH: nlopt.GN_ESCH,
            NLoptOptimizerType.GN_ISRES: nlopt.GN_ISRES,
        }

    @abstractmethod
    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """ return NLopt optimizer enum type """
        raise NotImplementedError

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
        return self._minimize(self._optimizer_names[self.get_nlopt_optimizer()],
                              objective_function,
                              variable_bounds,
                              initial_point, **self._options)

    def _minimize(self,
                  name: str,
                  objective_function: Callable,
                  variable_bounds: Optional[List[Tuple[float, float]]] = None,
                  initial_point: Optional[np.ndarray] = None,
                  max_evals: int = 1000) -> Tuple[float, float, int]:
        """Minimize using objective function

        Args:
            name: NLopt optimizer name
            objective_function: handle to a function that
                                            computes the objective function.
            variable_bounds: list of variable
                                bounds, given as pairs (lower, upper). None means
                                unbounded.
            initial_point: initial point.
            max_evals: Maximum evaluations

        Returns:
            tuple(float, float, int): Solution at minimum found,
                    value at minimum found, num evaluations performed
        """
        threshold = 3 * np.pi
        low = [(l if l is not None else -threshold) for (l, u) in variable_bounds]
        high = [(u if u is not None else threshold) for (l, u) in variable_bounds]

        opt = nlopt.opt(name, len(low))
        logger.debug(opt.get_algorithm_name())

        opt.set_lower_bounds(low)
        opt.set_upper_bounds(high)

        eval_count = 0

        def wrap_objfunc_global(x, _grad):
            nonlocal eval_count
            eval_count += 1
            return objective_function(x)

        opt.set_min_objective(wrap_objfunc_global)
        opt.set_maxeval(max_evals)

        xopt = opt.optimize(initial_point)
        minf = opt.last_optimum_value()

        logger.debug('Global minimize found %s eval count %s', minf, eval_count)
        return xopt, minf, eval_count
