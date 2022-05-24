
from typing import Optional, List, Callable, Union, Dict, Tuple

from qiskit import Aer
from qiskit.algorithms import VQE, VQEResult
from qiskit.algorithms.optimizers import SPSA, ADAM
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Y, Z, I, StateFn
from qiskit import Aer, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.opflow import *
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend
import numpy as np
import copy
from time import time


##----icans
class ICANS(VQE):
    # Init is the same as standard VQE
    # New parameters will appear at compute_minimum_eigenvalue function
    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        initial_point: Optional[np.ndarray] = None,
        # gradient: Optional[Union[GradientBase, Callable]] = None,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,        
        minShots: int =100,
        alpha: float = 0.01,
        max_iterations: Optional[int] = 500,
        L: Optional[float] = None,
        mu: Optional[float]= 0.99,
        b: Optional[float] = 0.000001,
        limit_learning_rate: Optional[bool] = False
    ) -> None:
        super().__init__(
            ansatz=ansatz,
            optimizer=None,  # iCANS is the optimizer
            initial_point=initial_point,
            gradient=None,  # not supported right now
            expectation=PauliExpectation(),  # must be Pauli expectation to adjust the shots per term
            callback=callback,
            quantum_instance=quantum_instance,
        )
        self.minShots = minShots
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.L = L
        self.mu = mu
        self.b = b
        self.limit_learning_rate = limit_learning_rate

    def single_shots(
        self, expectation: OperatorBase, values: np.ndarray, num_shots: int
    ) -> np.ndarray:
        # set the number of shots to the current value
        self._circuit_sampler.quantum_instance.run_config.shots = num_shots
        param_dict = dict(zip(self.ansatz.parameters, values))
        
        sampled = self._circuit_sampler.convert(expectation, params=param_dict)
        if not isinstance(sampled, SummedOp):
            sampled = SummedOp([sampled])

        n_components = len(sampled.oplist)
        out_Result = []
        for comp in range(n_components):
            read_outs = []
            composed = sampled.oplist[comp]
            meas2 = composed[0]
            samples2 = composed[1]
            
            for state, amplitude in samples2.primitive.items():
                occurence = int(np.round(num_shots * np.abs(amplitude) ** 2))
                value = np.real(meas2.eval(state))
                read_outs += occurence * [value]
                
            out_Result.append(read_outs)
            
        # Sum the components and return
        out = np.array(out_Result)
        
        return np.array(out.sum(axis=0))
    
    def compute_minimum_eigenvalue(
        self,
        operator
    ):
        # Execute optimization by iCans method
        # limit_learning_rate = Falsefor iCans1, True for iCans2
        start_time = time()

        epsilon = np.pi / 2  # Parameter shift rule
        
        n_param = self.ansatz.num_parameters
        
        if self.initial_point is None:
            lst_param = 2 * np.pi * np.random.rand(n_param)
        else:
            lst_param = self.initial_point

        # Define Lipschitz constant
        if self.L is None:
            # H = a*X + b*Z,
            # where X and Z are Pauli operators. The eigenvalues of X and Z are both {1, -1}, so the eigenvalues of a*X are {a, -a} and of b*Z are {b, -b}. Then we compute the upper bound for the Lipschitz constant from Eq. (5) as
            # L <= |a| + |b|.
            # :: self.L = np.sum(np.abs(self.operator.coeffs))
            self.L = np.sum(np.abs(operator.coeffs))
        
        expectation = self.construct_expectation(self.ansatz.parameters, operator)
        
        shots = np.zeros(n_param) + self.minShots  # Initial vector with number of shots
        self.hist_shots = np.zeros(self.max_iterations)  # historic total shots
        self.hist_results = np.zeros(self.max_iterations)  # historic partial results        
        xi = np.zeros(n_param)  # Variances
        chi = np.zeros(n_param)  # gradients
        gamma = np.zeros(n_param)  # Expected gain
        
        for k in range(self.max_iterations):
            grad = np.zeros(n_param)
            variances = np.zeros(n_param)
            
            # Calculation of gradients
            for param in range(n_param):
                parameters_plus = copy.deepcopy(lst_param)
                parameters_plus[param] += epsilon
                
                results_plus = self.single_shots(
                    expectation, parameters_plus, shots[param]
                )
                
                parameters_minus = copy.deepcopy(lst_param)
                parameters_minus[param] -= epsilon
                
                results_minus = self.single_shots(
                    expectation, parameters_minus, shots[param]
                )
                
                grad[param] = (
                    np.average(results_plus) - np.average(results_minus)
                ) / 2
                variances[param] = np.var(results_plus + results_minus)

            # Update gradient
            if not self.limit_learning_rate:
                lst_param = lst_param - self.alpha * grad
            else:
                alpha_bound = grad[param] ** 2 / (
                        self.L
                        * (
                            grad[param] ** 2
                            + variances[param] / shots[param]
                            + self.b * self.mu**k
                        )
                    )
                if self.alpha <=alpha_bound:
                    lst_param = lst_param - self.alpha * grad
                else:
                    lst_param = lst_param - alpha_bound * grad                    
            
            # Saves the historic of results
            self.hist_results[k] = np.average(results_plus)
            # Saves the accumulated shots
            if k >= 1:
                self.hist_shots[k] = self.hist_shots[k - 1] + 2 * shots.sum()
            else:
                self.hist_shots[k] = 2 * shots.sum()
            # Update number of shots
            for param, s in enumerate(shots):
                xi_aux = self.mu * xi[param] + (1 - self.mu) * variances[param]
                chi_aux = self.mu * chi[param] + (1 - self.mu) * grad[param]
                
                xi[param] = xi_aux / (1 - self.mu ** (k + 1))
                chi[param] = chi_aux / (1 - self.mu ** (k + 1))
                
                shots[param] = np.ceil(
                    2
                    * self.L
                    * self.alpha
                    / (2 - self.L * self.alpha)
                    * xi[param]
                    / (chi[param] ** 2 + self.b * self.mu**k)
                )
                if shots[param] < self.minShots:
                    shots[param] = self.minShots
                    
            # Expected return
            for i in range(n_param):
                gamma[i] = (
                    (self.alpha - self.L * self.alpha**2 / 2) * chi[i] ** 2
                    - self.L * self.alpha**2 * xi[i] / (2 * shots[i])
                ) / shots[i]
            idx = gamma.argmax(axis=0)
            smax = shots[idx]
            
            # Clip shots
            for param in range(n_param):
                if shots[param] > smax:
                    shots[param] = smax
        # print("Historic number shots: ", hist_shots)
        # print("Total shots: ", hist_shots.sum())
        
        eval_time = time() - start_time

        result = VQEResult()
        result.optimal_point = lst_param
        result.optimal_parameters = dict(zip(self.ansatz.parameters, lst_param))
        result.optimal_value = self.hist_results[self.max_iterations-1]
        result.cost_function_evals = 2*n_param*self.max_iterations
        result.optimizer_time = eval_time
        result.eigenvalue =self.hist_results[self.max_iterations-1] + 0j
        result.eigenstate = self._get_eigenstate(result.optimal_parameters)
        result.total_shots = self.hist_shots[self.max_iterations-1]

        return result