# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""VQE optimization with dynamic shot scheduling – ICANS optimizer."""

from typing import Optional, Callable, Union
import copy
from time import time
import numpy as np

from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, PauliExpectation, SummedOp
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend

from .vqe import VQE, VQEResult
from .minimum_eigen_solver import MinimumEigensolverResult


class ICANS(VQE):
    """
    The iCANS algorithm (individual coupled adaptive number of shots) [1], adaptively selects the
    number of shots in each iteration to reach the target accuracy with minimal number of quantum
    computational resources, both for a given iteration and for a given partial derivative in a
    stochastic gradient descent. As an optimization method, it is appropriately suited to noisy
    situations, when the cost of each shot is expensive.

    ICANS is implemented as a subclass of VQE. VQE [2] is a quantum algorithm that uses a
    variational technique to find the minimum eigenvalue of the Hamiltonian of a given system.
    According to Ref. [1], iCANS may perform comparably or better than other state-of-the-art
    optimizers, and especially well in the presence of realistic hardware noise.
    There are two variations, iCANS1 and iCANS2. The first is the more aggressive version,
    in relation to the learning rate, while iCANS2 is more cautious and limits the learning rate
    so that the expected gain is always guaranteed to be positive.

    Examples::

    This example runs ICANS with some specified parameters and limited learning rate.

    .. code-block::python

        import numpy as np

        from qiskit.algorithms import ICANS
        from qiskit.circuit.library import EfficientSU2
        from qiskit import Aer

        backend = Aer.get_backend("qasm_simulator")

        hamiltonian = PauliSumOp.from_list(
            [
                ("XXI", 1),
                ("XIX", 1),
                ("IXX", 1),
                ("YYI", 1),
                ("YIY", 1),
                ("IYY", 1),
                ("ZZI", 1),
                ("ZIZ", 1),
                ("IZZ", 1),
                ("IIZ", 3),
                ("IZI", 3),
                ("ZII", 3),
            ]
        )

        # parameters for iCANS
        maxiter = 500
        min_shots = 100
        alpha = 0.05

        icans = ICANS(ansatz, quantum_instance=backend, min_shots=min_shots, alpha=alpha,
                      maxiter=maxiter, limit_learning_rate=True)

        result_icans = icans1.compute_minimum_eigenvalue(hamiltonian)
        print(result_icans)



    References::

        [1] Jonas M. Kübler, Andrew Arrasmith, Lukasz Cincio, Patrick J. Coles (2019).
            An Adaptive Optimizer for Measurement-Frugal Variational Algorithms.
            `arXiv:1909.09082 <https://arxiv.org/abs/1909.09083>`_
        [2] Alberto Peruzzo, Et al. (2013). A variational eigenvalue solver on a quantum processor.
            `arXiv:1304.3061 <https://arxiv.org/abs/1304.3061>`_

    """

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        initial_point: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        min_shots: int = 100,
        alpha: float = 0.01,
        maxiter: int = 500,
        lipschitz: Optional[float] = None,
        mu: float = 0.99,
        b: float = 1e-6,
        limit_learning_rate: bool = False,
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
        self.maxiter = maxiter
        self.lipschitz = lipschitz
        self.mu = mu
        self.b = b
        self.limit_learning_rate = limit_learning_rate
        self.hist_shots = []
        self.hist_results = []

    def single_shots(
        self, expectation: OperatorBase, values: np.ndarray, num_shots: int
    ) -> np.ndarray:
        """Do single shots"""
        # set the number of shots in the quantum instance
        self._circuit_sampler.quantum_instance.run_config.shots = num_shots

        # evaluate the circuits
        param_dict = dict(zip(self.ansatz.parameters, values))
        sampled = self._circuit_sampler.convert(expectation, params=param_dict)

        if not isinstance(sampled, SummedOp):
            sampled = SummedOp([sampled])

        # evaluate each Pauli group
        out = []
        for component in sampled.oplist:
            read_outs = []
            measurement, samples = component

            for state, amplitude in samples.primitive.items():
                occurence = int(np.round(num_shots * np.abs(amplitude) ** 2))
                value = np.real(measurement.eval(state))
                read_outs += occurence * [value]

            out.append(read_outs)

        return np.array(out).sum(axis=0)

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> MinimumEigensolverResult:
        start_time = time()

        epsilon = np.pi / 2  # Parameter shift rule shifts the parameter values by pi/2
        n_param = self.ansatz.num_parameters

        if self.initial_point is None:
            lst_param = 2 * np.pi * np.random.rand(n_param)
        else:
            lst_param = self.initial_point
        if self.lipschitz is None:
            self.lipschitz = np.sum(np.abs(operator.coeffs))

        if self.lipschitz is None:
            lipschitz = np.sum(np.abs(operator.coeffs))
        else:
            lipschitz = self.lipschitz

        expectation = self.construct_expectation(self.ansatz.parameters, operator)
        energy_evaluation = self.get_energy_evaluation(operator)

        # algorithm variables (number of shots, estimated average, variance, ...)
        shots = np.zeros(n_param) + self.min_shots
        xi = np.zeros(n_param)
        chi = np.zeros(n_param)
        gamma = np.zeros(n_param)

        # keep track of the total number of shots taken
        shots_history = np.empty(self.maxiter)

        for k in range(self.maxiter):
            grad = np.zeros(n_param)
            variances = np.zeros(n_param)

            # Calculation of gradients
            for param in range(n_param):
                parameters_plus = deepcopy(lst_param)
                parameters_plus[param] += epsilon

                results_plus = self.single_shots(expectation, parameters_plus, shots[param])

                parameters_minus = copy.deepcopy(lst_param)
                parameters_minus[param] -= epsilon

                results_minus = self.single_shots(expectation, parameters_minus, shots[param])

                grad[param] = (np.average(results_plus) - np.average(results_minus)) / 2
                variances[param] = np.var(results_plus + results_minus)

            # Update gradient
            if not self.limit_learning_rate:
                lst_param = lst_param - self.alpha * grad
            else:
                alpha_bound = grad[param] ** 2 / (
                    lipschitz
                    * (grad[param] ** 2 + variances[param] / shots[param] + self.b * self.mu**k)
                )
                lst_param = lst_param - min(self.alpha, alpha_bound) * grad

            if self.callback is not None:
                # the energy evaluation function forwards the information to the callback
                _ = energy_evaluation(lst_param)

            # Saves the accumulated shots
            if k >= 1:
                shots_history[k] = shots_history[k - 1] + 2 * shots.sum()
            else:
                shots_history[k] = 2 * shots.sum()

            # Update number of shots
            for param, _ in enumerate(shots):
                xi_aux = self.mu * xi[param] + (1 - self.mu) * variances[param]
                chi_aux = self.mu * chi[param] + (1 - self.mu) * grad[param]

                xi[param] = xi_aux / (1 - self.mu ** (k + 1))
                chi[param] = chi_aux / (1 - self.mu ** (k + 1))

                shots[param] = np.ceil(
                    2
                    * lipschitz
                    * self.alpha
                    / (2 - lipschitz * self.alpha)
                    * xi[param]
                    / (chi[param] ** 2 + self.b * self.mu**k)
                )
                if shots[param] < self.min_shots:
                    shots[param] = self.min_shots

            # Expected return
            for i in range(n_param):
                gamma[i] = (
                    (self.alpha - lipschitz * self.alpha**2 / 2) * chi[i] ** 2
                    - lipschitz * self.alpha**2 * xi[i] / (2 * shots[i])
                ) / shots[i]
            idx = gamma.argmax(axis=0)
            smax = shots[idx]

            # Clip shots
            for param in range(n_param):
                if shots[param] > smax:
                    shots[param] = smax

        eval_time = time() - start_time

        result = VQEResult()
        result.optimal_point = lst_param
        result.optimal_parameters = dict(zip(self.ansatz.parameters, lst_param))
        result.optimal_value = energy_evaluation(lst_param)
        result.cost_function_evals = 2 * n_param * self.maxiter
        result.optimizer_time = eval_time
        result.eigenvalue = result.optimal_value + 0j
        result.eigenstate = self._get_eigenstate(result.optimal_parameters)
        result.total_shots = shots_history[-1]

        # calculate the expectation values of the aux operators if they are provided
        if aux_operators is not None:
            bound = self.ansatz.assign_parameters(result.optimal_point)
            aux_values = eval_observables(self.quantum_instance, bound, aux_operators, expectation)
            result.aux_operator_eigenvalues = aux_values

        return result
