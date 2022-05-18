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

"""Univariate Marginal Distribution Algorithm (Estimation-of-Distribution-Algorithm)."""


import pandas as pd
import numpy as np
from scipy.stats import norm

from .scipy_optimizer import SciPyOptimizer


class UMDA(SciPyOptimizer):
    """Continuous univariate marginal Estimation of Distribution algorithm.
    New individuals are sampled from a vector of univariate normal distributions.

    Implementations were obtained from EDAspy. For further detail, please refer to
    https://github.com/VicentePerezSoloviev/EDAspy
    """

    best_mae_global = 9999999
    best_ind_global = 9999999
    history = []

    elite_factor = 0.4
    std_bound = 0.3

    setting = "---"

    def __init__(self,
                 maxiter: int,
                 size_gen: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 disp: bool = False
                 ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.
            size_gen: Population size of each generation.
            n_variables: Number of variables to be optimized. For example in QAOA the number of variables is 2p, where p
            is the number of layers of the circuit.
            alpha: Percentage [0, 1] of the population to be selected as elite selection.
            disp: Set to True to print convergence messages.
        """

        self.disp = disp
        self.SIZE_GEN = size_gen
        self.MAX_ITER = maxiter
        self.alpha = alpha
        self.n_variables = n_variables
        self.vector = self.initialization()
        self.dead_iter = self.SIZE_GEN/5

        self.variables = list(self.vector.columns)

        self.best_mae_global = 999999999999
        self.truncation_length = int(size_gen * alpha)

        # initialization of generation
        mus = self.vector[self.variables].loc['mu'].to_list()
        stds = self.vector[self.variables].loc['std'].to_list()
        self.generation = pd.DataFrame(np.random.normal(mus, stds, [self.SIZE_GEN, len(self.variables)]),
                                       columns=self.variables, dtype='float_')

        super().__init__(method="UMDA")

    def initialization(self):
        vector = pd.DataFrame(columns=list(range(0, self.n_variables)))
        vector['data'] = ['mu', 'std', 'min', 'max']
        vector = vector.set_index('data')
        vector.loc['mu'] = np.pi
        vector.loc['std'] = 0.5

        vector.loc['min'] = 0  # optional
        vector.loc['max'] = np.pi * 2  # optional

        return vector

    def set_max_evals_grouped(self, max_evals_grouped):
        pass

    # build a generation of size SIZE_GEN from prob vector
    def new_generation(self):
        """Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """

        mus = self.vector[self.variables].loc['mu'].to_list()
        stds = self.vector[self.variables].loc['std'].to_list()
        gen = pd.DataFrame(np.random.normal(mus, stds, [self.SIZE_GEN, len(self.variables)]),
                           columns=self.variables, dtype='float_')

        self.generation = self.generation.nsmallest(int(self.elite_factor*len(self.generation)), 'cost')
        # self.generation = self.generation.append(gen).reset_index(drop=True)
        self.generation = pd.concat([self.generation, gen]).reset_index(drop=True)

    # truncate the generation at alpha percent
    def truncation(self):
        """Selection of the best individuals of the actual generation.
        Updates the generation by selecting the best individuals.
        """
        self.generation = self.generation.nsmallest(self.truncation_length, 'cost')

    # check each individual of the generation
    def check_generation(self, objective_function):
        """Check the cost of each individual in the cost function implemented by the user.
        """
        self.generation['cost'] = self.generation.apply(lambda row: objective_function(row[self.variables].to_list()),
                                                        axis=1)

    # update the probability vector
    def update_vector(self):
        """From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions
        """

        for var in self.variables:
            self.vector.at['mu', var], self.vector.at['std', var] = norm.fit(self.generation[var].values)
            if self.vector.at['std', var] < self.std_bound:
                self.vector.at['std', var] = self.std_bound

    def minimize(self, fun, x0, jac, bounds):
        self.history = []
        not_better = 0
        for i in range(self.MAX_ITER):
            self.check_generation(fun)
            self.truncation()
            self.update_vector()

            best_mae_local = self.generation['cost'].min()

            self.history.append(best_mae_local)
            best_ind_local = self.generation[self.generation['cost'] == best_mae_local]

            # update the best values ever
            if best_mae_local < self.best_mae_global:
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0

            else:
                not_better += 1
                if not_better == self.dead_iter:
                    return EdaResult(self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1],
                                     self.best_mae_global,
                                     len(self.history),
                                     self.disp)

            self.new_generation()

        return EdaResult(self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1],
                         self.best_mae_global,
                         len(self.history),
                         self.disp)


class EdaResult:
    def __init__(self, optimal_point, optimal_value, cost_function_evals, disp):
        self.x = optimal_point
        self.fun = optimal_value
        self.nfev = cost_function_evals

        if disp:
            print('\tNFVALS = ' + str(cost_function_evals) + ' F = ' + str(optimal_value))
            print('\tX = ' + str(optimal_point))
