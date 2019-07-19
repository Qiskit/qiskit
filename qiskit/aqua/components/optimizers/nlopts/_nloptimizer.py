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

import numpy as np
import importlib
import logging
from qiskit.aqua import AquaError
logger = logging.getLogger(__name__)

try:
    import nlopt
except ImportError:
    logger.info('nlopt is not installed. Please install it if you want to use them.')


def check_pluggable_valid(name):
    err_msg = "Unable to instantiate '{}', nlopt is not installed. Please install it if you want to use them.".format(name)
    try:
        spec = importlib.util.find_spec('nlopt')
        if spec is not None:
            return
    except Exception as e:
        logger.debug('{} {}'.format(err_msg, str(e)))
        raise AquaError(err_msg) from e

    raise AquaError(err_msg)


def minimize(name, objective_function, variable_bounds=None, initial_point=None, max_evals=1000):
    """Minimize using objective function

    Args:
        name: NLopt optimizer name
        objective_function: Objective function to evaluate
        variable_bounds: Bounds
        initial_point: Initial point for optimizer
        max_evals: Maximum evaluations

    Returns:
        Solution at minimum found, value at minimum found, num evaluations performed
    """
    threshold = 3*np.pi
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

    logger.debug('Global minimize found {} eval count {}'.format(minf, eval_count))
    return xopt, minf, eval_count
