# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Differential Evolution optimizer"""

from typing import Callable, Tuple, List, Dict, Union

import numpy as np
from scipy.optimize import differential_evolution
from .optimizer import Optimizer, OptimizerSupportLevel


class DifferentialEvolution(Optimizer):
    """
    Differential Evolution optimizer

    The Differential Evolution Optimizer is a global and derivative-free
    optimizer that seeks to find the optimal value of an objective function,
    f(x) (where x represents the vector of possible parameters) through
    stochastically searching the optimization space of possible solutions.
    It belongs to the family of evolutionary optimization techniques
    such as genetic and particle swarm algorithms. It is best suited
    for combinatorial optimization problems and operates on real-parameter
    and real-valued objective functions.

    Specifically, the algorithm begins with an initial population
    of candidate solutions (can be random or user-specified) and new candidates
    are iteratively created through a combination of mutation, recombination
    and stochastic selection. The main parameters that govern the optimization
    include the evolutionary strategy, population size, mutation constant
    (differential weight) and recombination factor (crossover probability).
    The algorithm terminates if either the maximum number of iterations/generations
    have been reached or the standard deviation of the generation of candidates
    is below a user-specified threshold. Specificially, the total number of
    function evaluations equals (maxiter + 1)* popsize * len(x), without polishing
    (see below).

    Uses scipy.optimize.differential_evolution
    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    """

    _OPTIONS = ['maxiter', 'strategy', 'popsize', 'mutation',
                'recombination', 'init', 'tol', 'atol',
                'polish', 'disp', 'workers']

    def __init__(self,
                 maxiter: int = 10,
                 strategy: str = 'currenttobest1bin',
                 popsize: int = 10,
                 mutation: Union[float, tuple] = (0.5, 1),
                 recombination: float = 0.7,
                 init: str = 'latinhypercube',
                 tol: float = 0.01,
                 atol: float = 0,
                 polish: bool = False,
                 disp: bool = False,
                 workers: Union[int, Callable] = 1) -> None:
        """
        Args:
            maxiter: Maximum number of generations to evolve population
            strategy: Differential evolution strategy to employ for generating new trial candidates
                      in subsequent generations/iterations
            popsize: Number of individuals/candidates in each generation
            mutation: Mutation constant (differential weight) specified as a float in the range of
                      [0,2] or as a tuple(min, max). If specified as a tuple, the mutation constant
                      will be 'dithered' or randomly changed within the range specified in each
                      generation.
            recombination: Recombination constant (crossover probability) specified as a float
                           in the range [0,1]
            init: String/array to specify whether initial population is generately via ``random``,
                  ``latinhypercube`` sampling or via a specified array of individuals. If an array
                  is specified, its shape must match (popsize, params) where popsize and params
                  refer to population size and number of parameters in the objective function
            tol: Relative tolerance for convergence
            atol: Absolute tolerance for convergence
            polish: Boolean to specify whether or not best population in last generation is
                    optimized using scipy.optimize.minimize with the L-BFGS-B method. If
                    this is set to true, the maximum number of function evaluations will
                    be greater than (maxiter + 1)* popsize * len(x)
            disp: Boolean to specify if the objective function is printed at each iteration
            workers: Integer or map-like callable (multiprocessing.Pool.map) that specifies
                     how many individuals in the population to be evaluated in parallel
        """

        super().__init__()

        self._strategy = strategy
        self._maxiter = maxiter
        self._popsize = popsize
        self._mutation = mutation
        self._recombination = recombination
        self._tol = tol
        self._atol = atol
        self._init = init
        self._polish = polish
        self._disp = disp
        self._workers = workers

    def get_support_level(self) -> Dict[str, OptimizerSupportLevel]:
        """ Returns support level dictionary. """
        return {
            'gradient': OptimizerSupportLevel.ignored,
            'bounds': OptimizerSupportLevel.required,
            'initial_point': OptimizerSupportLevel.ignored
        }

    def optimize(self,
                 num_vars: int,
                 objective_function: Callable,
                 gradient_function: Callable = None,
                 variable_bounds: List[Tuple[float, float]] = None,
                 initial_point: np.ndarray = None) -> Tuple[np.ndarray, float, int]:
        """ Runs the optimization. """
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        res = differential_evolution(func=objective_function, bounds=variable_bounds, args=(),
                                     strategy=self._strategy, maxiter=self._maxiter,
                                     popsize=self._popsize, mutation=self._mutation,
                                     recombination=self._recombination, tol=self._tol,
                                     atol=self._atol, init=self._init, disp=self._disp,
                                     polish=self._polish, workers=self._workers)

        return res.x, res.fun, res.nfev
