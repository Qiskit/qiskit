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

"""Suffix Averaging optimier"""

import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from qiskit.algorithms.optimizers import NFT, QNSPSA, SPSA, GradientDescent

from .optimizer import POINT, Optimizer, OptimizerResult


class SuffixAveragingOptimizer(Optimizer):
    r"""The suffix averaging optimizer.

    Given a sequence of circuit parameters of parameterized quantum circuit
    obtained from :math:`T` iterations, :math:`\{\vec{\theta}^{(t)}\}_{t=1}^{t=T}`,
    the SuffixAveragingOptimizer returns the averaged point of the last n_params_suffix
    points,

    .. math::

        \overline{\vec{\theta}} = \frac{1}{n_params_suffix} \sum_{t=T-n_params_suffix-1}^{T}
        \vec{\theta}^{(t)}.

    Examples:
        .. code-block::python
            from qiskit import Aer
            from qiskit.utils import QuantumInstance
            from qiskit.algorithms import VQE
            from qiskit.opflow.gradients import Gradient
            from qiskit.circuit.library import TwoLocal
            from qiskit.algorithms.optimizers import ADAM
            from qiskit.algorithms.optimizers.suffix_averaging import SuffixAveragingOptimizer

            backend = Aer.get_backend("qasm_simulator")
            quantum_instance = QuantumInstance(backend=backend, shots=1000)
            ansatz = TwoLocal(4, ["rx", "rz"], "cx", "linear")
            optimizer = ADAM(maxiter=1000)
            suffix_optimizer = SuffixAveragingOptimizer(optimizer, n_params_suffix=50)
            gradient = Gradient(grad_method="param_shift")
            vqe = VQE(
                ansatz = ansatz,
                optimizer = suffix_optimizer,
                gradient = gradient,
                quantum_instance = quantum_instance
            )

    References:
        [1] S. Tamiya and H. Yamasaki. 2021.
        Stochastic Gradient Line Bayesian Optimization
        for Efficient Noise-Robust Optimization of Parameterized Quantum Circuits.
        arXiv preprint arXiv:2111.07952.
    """

    def __init__(self, optimizer: Optimizer, n_params_suffix: int = 50) -> None:
        """
        Args:
            optimizer: The optimizer used for optimizing parameterized quantum circuits.
            n_params_suffix: The number of circuit parameters for taking the suffix average.
        """

        self._n_params_suffix = n_params_suffix
        self._optimizer = optimizer

        self._circ_params = []

        if isinstance(self._optimizer, SPSA):

            # pylint: disable=unused-argument
            def load_params(nfev, x_next, fx_next, update_step, is_accepted):
                self._circ_params.append(x_next)

        elif isinstance(self._optimizer, QNSPSA):

            # pylint: disable=unused-argument
            def load_params(nfev, x_next, fx_next, update_step, is_accepted):
                self._circ_params.append(x_next)

        elif isinstance(self._optimizer, GradientDescent):

            # pylint: disable=unused-argument
            def load_params(nfevs, x_next, fx_next, stepsize):
                self._circ_params.append(x_next)

        else:

            def load_params(x):
                self._circ_params.append(x)

        self._optimizer.callback = load_params
        super().__init__()

    def get_support_level(self):
        """Return support level dictionary"""
        return self._optimizer.get_support_level()

    def _return_suffix_average(self) -> List[float]:

        if isinstance(self._optimizer, NFT):
            n_params = int(len(self._circ_params[0]))
            n_iterates = int(len(self._circ_params))
            n_repitition = int(len(self._circ_params) / n_params)
            if n_repitition < self._n_params_suffix:
                warnings.warn("The total number of iterations is less than n_params_suffix.")
                averaged_param = np.zeros_like(self._circ_params[0])
                for j in range(n_repitition):
                    averaged_param += self._circ_params[n_iterates - n_params * j - 1]
                averaged_param /= self._n_params_suffix
                return averaged_param

            averaged_param = np.zeros_like(self._circ_params[0])
            for j in range(self._n_params_suffix):
                averaged_param += self._circ_params[n_iterates - n_params * j - 1]
            averaged_param /= self._n_params_suffix

        else:
            n_iterates = len(self._circ_params)

            if n_iterates < self._n_params_suffix:
                warnings.warn("The total number of iterations is less than n_params_suffix.")
                return np.average(self._circ_params, axis=0)

            averaged_param = np.zeros_like(self._circ_params[0])
            for j in range(self._n_params_suffix):
                averaged_param += self._circ_params[n_iterates - j - 1]
            averaged_param /= self._n_params_suffix

        return averaged_param

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:

        result = self._optimizer.minimize(fun, x0, jac=jac, bounds=bounds)
        result.x = self._return_suffix_average()
        result.fun = fun(np.copy(result.x))

        return result
