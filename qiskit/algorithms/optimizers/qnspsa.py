# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QN-SPSA optimizer."""

from typing import Any, Iterator, Optional, Union, Callable, Dict

import numpy as np
from qiskit.providers import Backend
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.opflow import StateFn, CircuitSampler, ExpectationBase
from qiskit.utils import QuantumInstance

from .spsa import SPSA, _batch_evaluate

# the function to compute the fidelity
FIDELITY = Callable[[np.ndarray, np.ndarray], float]

# parameters, loss, stepsize, number of function evaluations, accepted
CALLBACK = Callable[[np.ndarray, float, float, int, bool], None]


class QNSPSA(SPSA):
    r"""The Quantum Natural SPSA (QN-SPSA) optimizer.

    The QN-SPSA optimizer [1] is a stochastic optimizer that belongs to the family of gradient
    descent methods. This optimizer is based on SPSA but attempts to improve the convergence by
    sampling the **natural gradient** instead of the vanilla, first-order gradient. It achieves
    this by approximating Hessian of the ``fidelity`` of the ansatz circuit.

    Compared to natural gradients, which require :math:`\mathcal{O}(d^2)` expectation value
    evaluations for a circuit with :math:`d` parameters, QN-SPSA only requires
    :math:`\mathcal{O}(1)` and can therefore significantly speed up the natural gradient calculation
    by sacrificing some accuracy. Compared to SPSA, QN-SPSA requires 4 additional function
    evaluations of the fidelity.

    The stochastic approximation of the natural gradient can be systematically improved by
    increasing the number of ``resamplings``. This leads to a Monte Carlo-style convergence to
    the exact, analytic value.

    Examples:

        This short example runs QN-SPSA for the ground state calculation of the ``Z ^ Z``
        observable where the ansatz is a ``PauliTwoDesign`` circuit.

        .. code-block:: python

            import numpy as np
            from qiskit.algorithms.optimizers import QNSPSA
            from qiskit.circuit.library import PauliTwoDesign
            from qiskit.opflow import Z, StateFn

            ansatz = PauliTwoDesign(2, reps=1, seed=2)
            observable = Z ^ Z
            initial_point = np.random.random(ansatz.num_parameters)

            def loss(x):
                bound = ansatz.bind_parameters(x)
                return np.real((StateFn(observable, is_measurement=True) @ StateFn(bound)).eval())

            fidelity = QNSPSA.get_fidelity(ansatz)
            qnspsa = QNSPSA(fidelity, maxiter=300)
            result = qnspsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)


    References:

        [1] J. Gacon et al, "Simultaneous Perturbation Stochastic Approximation of the Quantum
        Fisher Information", `arXiv:2103.09232 <https://arxiv.org/abs/2103.09232>`_

    """

    def __init__(
        self,
        fidelity: FIDELITY,
        maxiter: int = 100,
        blocking: bool = True,
        allowed_increase: Optional[float] = None,
        learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
        perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
        last_avg: int = 1,
        resamplings: Union[int, Dict[int, int]] = 1,
        perturbation_dims: Optional[int] = None,
        regularization: Optional[float] = None,
        hessian_delay: int = 0,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        initial_hessian: Optional[np.ndarray] = None,
        callback: Optional[CALLBACK] = None,
    ) -> None:
        r"""
        Args:
            fidelity: A function to compute the fidelity of the ansatz state with itself for
                two different sets of parameters.
            maxiter: The maximum number of iterations. Note that this is not the maximal number
                of function evaluations.
            blocking: If True, only accepts updates that improve the loss (up to some allowed
                increase, see next argument).
            allowed_increase: If ``blocking`` is ``True``, this argument determines by how much
                the loss can increase with the proposed parameters and still be accepted.
                If ``None``, the allowed increases is calibrated automatically to be twice the
                approximated standard deviation of the loss function.
            learning_rate: The update step is the learning rate is multiplied with the gradient.
                If the learning rate is a float, it remains constant over the course of the
                optimization. It can also be a callable returning an iterator which yields the
                learning rates for each optimization step.
                If ``learning_rate`` is set ``perturbation`` must also be provided.
            perturbation: Specifies the magnitude of the perturbation for the finite difference
                approximation of the gradients. Can be either a float or a generator yielding
                the perturbation magnitudes per step.
                If ``perturbation`` is set ``learning_rate`` must also be provided.
            last_avg: Return the average of the ``last_avg`` parameters instead of just the
                last parameter values.
            resamplings: The number of times the gradient (and Hessian) is sampled using a random
                direction to construct a gradient estimate. Per default the gradient is estimated
                using only one random direction. If an integer, all iterations use the same number
                of resamplings. If a dictionary, this is interpreted as
                ``{iteration: number of resamplings per iteration}``.
            perturbation_dims: The number of perturbed dimensions. Per default, all dimensions
                are perturbed, but a smaller, fixed number can be perturbed. If set, the perturbed
                dimensions are chosen uniformly at random.
            regularization: To ensure the preconditioner is symmetric and positive definite, the
                identity times a small coefficient is added to it. This generator yields that
                coefficient.
            hessian_delay: Start multiplying the gradient with the inverse Hessian only after a
                certain number of iterations. The Hessian is still evaluated and therefore this
                argument can be useful to first get a stable average over the last iterations before
                using it as preconditioner.
            lse_solver: The method to solve for the inverse of the Hessian. Per default an
                exact LSE solver is used, but can e.g. be overwritten by a minimization routine.
            initial_hessian: The initial guess for the Hessian. By default the identity matrix
                is used.
            callback: A callback function passed information in each iteration step. The
                information is, in this order: the parameters, the function value, the number
                of function evaluations, the stepsize, whether the step was accepted.
        """
        super().__init__(
            maxiter,
            blocking,
            allowed_increase,
            # trust region *must* be false for natural gradients to work
            trust_region=False,
            learning_rate=learning_rate,
            perturbation=perturbation,
            resamplings=resamplings,
            callback=callback,
            second_order=True,
            hessian_delay=hessian_delay,
            lse_solver=lse_solver,
            regularization=regularization,
            perturbation_dims=perturbation_dims,
            initial_hessian=initial_hessian,
        )

        self.fidelity = fidelity

    def _point_sample(self, loss, x, eps, delta1, delta2):
        loss_points = [x + eps * delta1, x - eps * delta1]
        fidelity_points = [
            (x, x + eps * delta1),
            (x, x - eps * delta1),
            (x, x + eps * (delta1 + delta2)),
            (x, x + eps * (-delta1 + delta2)),
        ]
        self._nfev += 6

        loss_values = _batch_evaluate(loss, loss_points, self._max_evals_grouped)
        fidelity_values = _batch_evaluate(self.fidelity, fidelity_points, self._max_evals_grouped)

        # compute the gradient approximation and additionally return the loss function evaluations
        gradient_estimate = (loss_values[0] - loss_values[1]) / (2 * eps) * delta1

        # compute the preconditioner point estimate
        diff = fidelity_values[2] - fidelity_values[0]
        diff -= fidelity_values[3] - fidelity_values[1]
        diff /= 2 * eps ** 2

        rank_one = np.outer(delta1, delta2)
        # -0.5 factor comes from the fact that we need -0.5 * fidelity
        hessian_estimate = -0.5 * diff * (rank_one + rank_one.T) / 2

        return gradient_estimate, hessian_estimate

    @property
    def settings(self) -> Dict[str, Any]:
        """The optimizer settings in a dictionary format.

        .. note::

            The ``fidelity`` property cannot be serialized and will not be contained
            in the dictionary. To construct a ``QNSPSA`` object from a dictionary you
            have to add it manually with the key ``"fidelity"``.

        """
        # re-use serialization from SPSA
        settings = super().settings

        # remove SPSA-specific arguments not in QNSPSA
        settings.pop("trust_region")
        settings.pop("second_order")

        return settings

    @staticmethod
    def get_fidelity(
        circuit: QuantumCircuit,
        backend: Optional[Union[Backend, QuantumInstance]] = None,
        expectation: Optional[ExpectationBase] = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        r"""Get a function to compute the fidelity of ``circuit`` with itself.

        Let ``circuit`` be a parameterized quantum circuit performing the operation
        :math:`U(\theta)` given a set of parameters :math:`\theta`. Then this method returns
        a function to evaluate

        .. math::

            F(\theta, \phi) = \big|\langle 0 | U^\dagger(\theta) U(\phi) |0\rangle  \big|^2.

        The output of this function can be used as input for the ``fidelity`` to the
        :class:~`qiskit.algorithms.optimizers.QNSPSA` optimizer.

        Args:
            circuit: The circuit preparing the parameterized ansatz.
            backend: A backend of quantum instance to evaluate the circuits. If None, plain
                matrix multiplication will be used.
            expectation: An expectation converter to specify how the expected value is computed.
                If a shot-based readout is used this should be set to ``PauliExpectation``.

        Returns:
            A handle to the function :math:`F`.

        """
        params_x = ParameterVector("x", circuit.num_parameters)
        params_y = ParameterVector("y", circuit.num_parameters)

        expression = ~StateFn(circuit.assign_parameters(params_x)) @ StateFn(
            circuit.assign_parameters(params_y)
        )

        if expectation is not None:
            expression = expectation.convert(expression)

        if backend is None:

            def fidelity(values_x, values_y):
                value_dict = dict(
                    zip(params_x[:] + params_y[:], values_x.tolist() + values_y.tolist())
                )
                return np.abs(expression.bind_parameters(value_dict).eval()) ** 2

        else:
            sampler = CircuitSampler(backend)

            def fidelity(values_x, values_y):
                value_dict = dict(
                    zip(params_x[:] + params_y[:], values_x.tolist() + values_y.tolist())
                )
                return np.abs(sampler.convert(expression, params=value_dict).eval()) ** 2

        return fidelity
