""" Implementation of icans
"""
from typing import Optional, Callable, Union
from copy import deepcopy
from time import time
import numpy as np
from qiskit.algorithms import VQE, VQEResult
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, PauliExpectation, SummedOp
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend

##----icans
class ICANS(VQE):
    """ Init is the same as standard VQE
    New parameters will appear at compute_minimum_eigenvalue function
    """
    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        initial_point: Optional[np.ndarray] = None,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        min_shots: int = 100,
        alpha: float = 0.01,
        max_iterations: Optional[int] = 500,
        lipschitz: Optional[float] = None,
        mu: Optional[float] = 0.99,
        b: Optional[float] = 0.000001,
        limit_learning_rate: Optional[bool] = False,
    ) -> None:
        super().__init__(
            ansatz=ansatz,
            optimizer=None,
            initial_point=initial_point,
            gradient=None,
            expectation=PauliExpectation(),
            callback=callback,
            quantum_instance=quantum_instance,
        )
        self.min_shots = min_shots
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.lipschitz = lipschitz
        self.mu = mu
        self.b = b
        self.limit_learning_rate = limit_learning_rate
        self.hist_shots =[]
        self.hist_results = []

    def single_shots(
        self, expectation: OperatorBase, values: np.ndarray, num_shots: int
    ) -> np.ndarray:
        """Do single shots
        """

        self._circuit_sampler.quantum_instance.run_config.shots = num_shots
        param_dict = dict(zip(self.ansatz.parameters, values))
        sampled = self._circuit_sampler.convert(expectation, params=param_dict)
        if not isinstance(sampled, SummedOp):
            sampled = SummedOp([sampled])

        n_components = len(sampled.oplist)
        out_result = []
        for comp in range(n_components):
            read_outs = []
            composed = sampled.oplist[comp]
            meas2 = composed[0]
            samples2 = composed[1]

            for state, amplitude in samples2.primitive.items():
                occurence = int(np.round(num_shots * np.abs(amplitude) ** 2))
                value = np.real(meas2.eval(state))
                read_outs += occurence * [value]

            out_result.append(read_outs)

        out = np.array(out_result)

        return np.array(out.sum(axis=0))

    def compute_minimum_eigenvalue(self, operator):
        """ Execute optimization by iCans method
         limit_learning_rate = Falsefor iCans1, True for iCans2
        """
        start_time = time()

        epsilon = np.pi / 2  # Parameter shift rule

        n_param = self.ansatz.num_parameters

        if self.initial_point is None:
            lst_param = 2 * np.pi * np.random.rand(n_param)
        else:
            lst_param = self.initial_point
        if self.lipschitz is None:
            self.lipschitz = np.sum(np.abs(operator.coeffs))

        expectation = self.construct_expectation(self.ansatz.parameters, operator)

        shots = np.zeros(n_param) + self.min_shots
        self.hist_shots = np.zeros(self.max_iterations)
        self.hist_results = np.zeros(self.max_iterations)
        xi = np.zeros(n_param)
        chi = np.zeros(n_param)
        gamma = np.zeros(n_param)

        for k in range(self.max_iterations):
            grad = np.zeros(n_param)
            variances = np.zeros(n_param)

            for param in range(n_param):
                parameters_plus = deepcopy(lst_param)
                parameters_plus[param] += epsilon
                results_plus = self.single_shots(expectation, parameters_plus, shots[param])

                parameters_minus = deepcopy(lst_param)
                parameters_minus[param] -= epsilon

                results_minus = self.single_shots(expectation, parameters_minus, shots[param])

                grad[param] = (np.average(results_plus) - np.average(results_minus)) / 2
                variances[param] = np.var(results_plus + results_minus)

            # Update gradient
            if not self.limit_learning_rate:
                lst_param = lst_param - self.alpha * grad
            else:
                alpha_bound = grad[param] ** 2 / (
                    self.lipschitz
                    * (grad[param] ** 2 + variances[param] / shots[param] + self.b * self.mu**k)
                )
                if self.alpha <= alpha_bound:
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
            for param, _ in enumerate(shots):
                xi_aux = self.mu * xi[param] + (1 - self.mu) * variances[param]
                chi_aux = self.mu * chi[param] + (1 - self.mu) * grad[param]

                xi[param] = xi_aux / (1 - self.mu ** (k + 1))
                chi[param] = chi_aux / (1 - self.mu ** (k + 1))

                shots[param] = np.ceil(
                    2
                    * self.lipschitz
                    * self.alpha
                    / (2 - self.lipschitz * self.alpha)
                    * xi[param]
                    / (chi[param] ** 2 + self.b * self.mu**k)
                )
                if shots[param] < self.min_shots:
                    shots[param] = self.min_shots

            # Expected return
            for i in range(n_param):
                gamma[i] = (
                    (self.alpha - self.lipschitz * self.alpha**2 / 2) * chi[i] ** 2
                    - self.lipschitz * self.alpha**2 * xi[i] / (2 * shots[i])
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
        result.optimal_value = self.hist_results[self.max_iterations - 1]
        result.cost_function_evals = 2 * n_param * self.max_iterations
        result.optimizer_time = eval_time
        result.eigenvalue = self.hist_results[self.max_iterations - 1] + 0j
        result.eigenstate = self._get_eigenstate(result.optimal_parameters)
        result.total_shots = self.hist_shots[self.max_iterations - 1]

        return result
