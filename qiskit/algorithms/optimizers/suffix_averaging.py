from json import load
import os
import csv
import numpy as np
from qiskit.algorithms.optimizers import QNSPSA, SPSA, NFT, GradientDescent
from typing import Callable, List, Optional, Tuple

from scipy_optimizer import SciPyOptimizer
from optimizer import Optimizer, OptimizerResult, POINT, OptimizerSupportLevel

class SuffixAveragingOptimizer(Optimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        alpha: float = 0.3,
        suffix_dir: str = None,
        suffix_filename: str = None,
        save_params: bool = False,
        save_averaged_params: bool = False
    ) -> None:
        """
        Args:
            optimizer: The optimizer used for optimizing parameterized quantum circuits.
            alpha: The hyperparameter to determine the ratio of ansatz parameters to take the average.
            suffix_dir: The directory for storing ansatz parameters.
            suffix_filename: The filename for storing ansatz parameters.
            save_params: If True, intermediate ansatz parameters are stored.
            save_averaged_params: If True, intermediate ansatz parameters over which the suffix average is taken are stored.

        References:
            [1] S. Tamiya and H. Yamasaki. 2021.
            Stochastic Gradient Line Bayesian Optimization: Reducing Measurement Shots in Optimizing Parameterized Quantum Circuits.
            arXiv preprint arXiv:2111.07952.
        """

        self._alpha = alpha
        self._suffix_dir = suffix_dir
        self._suffix_filename = suffix_filename
        self._save_params = save_params
        self._save_averaged_params = save_averaged_params
        self._optimizer = optimizer

        self._circ_params = []
        
    
        if isinstance(self._optimizer, SPSA):
            def load_params(nfev, x_next, fx_next, update_step, is_accepted):
                self._circ_params.append(x_next)
        elif isinstance(self._optimizer, QNSPSA):
            def load_params(nfev, x_next, fx_next, update_step, is_accepted):
                self._circ_params.append(x_next)
        elif isinstance(self._optimizer, GradientDescent):
            def load_params(nfevs, x_next, fx_next, stepsize):
                self._circ_params.append(x_next)
        else:
            def load_params(x):
                self._circ_params.append(x)

        self._optimizer.callback = load_params

    def _save_circ_params(self, circ_params: List[float], csv_dir: str, csv_filename: str) -> None:
        directory = csv_dir+"/results"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, csv_filename+".csv"), mode="w") as csv_file:
            writer = csv.writer(csv_file, lineterminator='\n')
            writer.writerows(circ_params)

    @staticmethod
    def read_circ_params(csv_dir: str, csv_filename: str) -> List[float]:
        with open(os.path.join(csv_dir, "results", csv_filename+".csv")) as csv_file:
            reader = csv.reader(csv_file)
            circ_params = [list(map(float, row)) for row in reader]
        return circ_params
    
    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def return_suffix_average_from_log(self, fun: Callable[[POINT], float], alpha: float, total_iterates: int, csv_dir: str, csv_filename: str) -> float:
        params_list_temp = self.read_circ_params(csv_dir, csv_filename)
        params_list = params_list_temp[:total_iterates]
        n_iterates = len(params_list)
        averaged_param = np.zeros_like(params_list[0])
        for j in range(int(np.ceil(n_iterates*alpha))):
            averaged_param += params_list[n_iterates-j-1]
        averaged_param /= np.ceil(n_iterates*alpha)

        cost_func = fun(np.copy(averaged_param))

        return cost_func
    
    def _return_suffix_average(self) -> List[float]:
        if self._save_params:
            self._save_circ_params(self._circ_params, self._suffix_dir, self._suffix_filename)
        
        if self._save_averaged_params:
            n_iterates = len(self._circ_params)
            averaged_params = np.zeros_like(self._circ_params)
            for i in range(n_iterates):
                averaged_param = np.zeros_like(self._circ_params[0])
                for j in range(int(np.ceil(i*self._alpha))):
                    averaged_param += self._circ_params[i-j]
                averaged_param /= np.ceil(i*self._alpha)
                averaged_params[i] = averaged_param
            self._save_circ_params(averaged_params, self._suffix_dir, self._suffix_filename+"_suffix")
            return averaged_params[-1]
                
        else:
            if isinstance(self._optimizer, NFT):
                n_params = int(len(self._circ_params[0]))
                n_iterates = int(len(self._circ_params))
                n_repitition = int(len(self._circ_params)/n_params)
                averaged_param = np.zeros_like(self._circ_params[0])
                for j in range(int(np.ceil(n_repitition*self._alpha))):
                    averaged_param += self._circ_params[n_iterates-n_params*j-1]
                averaged_param /= np.ceil(n_repitition*self._alpha)
            else:
                n_iterates = len(self._circ_params)
                averaged_param = np.zeros_like(self._circ_params[0])
                for j in range(int(np.ceil(n_iterates*self._alpha))):
                    averaged_param += self._circ_params[n_iterates-j-1]
                averaged_param /= np.ceil(n_iterates*self._alpha)

            return averaged_param
        
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:

        result = self._optimizer.minimize(fun, x0, jac=jac, bounds=bounds)
        result.x_suff = self._return_suffix_average()
        result.fun_suff = fun(np.copy(result.x_suff))

        return result