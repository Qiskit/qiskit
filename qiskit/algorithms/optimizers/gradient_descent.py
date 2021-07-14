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

"""A standard gradient descent optimizer."""

from typing import Iterator, Optional, Union, Callable, Dict, Any
from functools import partial

import numpy as np

from .optimizer import Optimizer, OptimizerSupportLevel

CALLBACK = Callable[[int, np.ndarray, float, float], None]


class GradientDescent(Optimizer):
    r"""The gradient descent minimization routine.

    For a function :math:`f` and an initial point :math:`\vec\theta_0`, the standard (or "vanilla")
    gradient descent method is an iterative scheme to find the minimum :math:`\vec\theta^*` of
    :math:`f` by updating the parameters in the direction of the negative gradient of :math:`f`

    .. math::

        \vec\theta_{n+1} = \vec\theta_{n} - \vec\eta\nabla f(\vec\theta_{n}),

    for a small learning rate :math:`\eta > 0`.

    You can either provide the analytic gradient :math:`\vec\nabla f` as ``gradient_function``
    in the ``optimize`` method, or, if you do not provide it, use a finite difference approximation
    of the gradient. To adapt the size of the perturbation in the finite difference gradients,
    set the ``perturbation`` property in the initializer.

    This optimizer supports a callback function. If provided in the initializer, the optimizer
    will call the callback in each iteration with the following information in this order:
    current number of function values, current parameters, current function value, norm of current
    gradient.

    Examples:

        A minimum example that will use finite difference gradients with a default perturbation
        of 0.01 and a default learning rate of 0.01.

        .. code-block::python

            from qiskit.algorithms.optimizers import GradientDescent

            def f(x):
                return (np.linalg.norm(x) - 1) ** 2

            initial_point = np.array([1, 0.5, -0.2])

            optimizer = GradientDescent(maxiter=100)
            x_opt, fx_opt, nfevs = optimizer.optimize(initial_point.size,
                                                      f,
                                                      initial_point=initial_point)

            print(f"Found minimum {x_opt} at a value of {fx_opt} using {nfevs} evaluations.")

        An example where the learning rate is an iterator and we supply the analytic gradient.
        Note how much faster this convergences (i.e. less ``nfevs``) compared to the previous
        example.

        .. code-block::python

            from qiskit.algorithms.optimizers import GradientDescent

            def learning_rate():
                power = 0.6
                constant_coeff = 0.1

                def powerlaw():
                    n = 0
                    while True:
                        yield constant_coeff * (n ** power)
                        n += 1

                return powerlaw()

            def f(x):
                return (np.linalg.norm(x) - 1) ** 2

            def grad_f(x):
                return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

            initial_point = np.array([1, 0.5, -0.2])

            optimizer = GradientDescent(maxiter=100, learning_rate=learning_rate)
            x_opt, fx_opt, nfevs = optimizer.optimize(initial_point.size,
                                                      f,
                                                      gradient_function=grad_f,
                                                      initial_point=initial_point)

            print(f"Found minimum {x_opt} at a value of {fx_opt} using {nfevs} evaluations.")

    """

    def __init__(
        self,
        maxiter: int = 100,
        learning_rate: Union[float, Callable[[], Iterator]] = 0.01,
        tol: float = 1e-7,
        callback: Optional[CALLBACK] = None,
        perturbation: Optional[float] = None,
    ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            learning_rate: A constant or generator yielding learning rates for the parameter
                updates. See the docstring for an example.
            tol: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
            perturbation: If no gradient is passed to ``GradientDescent.optimize`` the gradient is
                approximated with a symmetric finite difference scheme with ``perturbation``
                perturbation in both directions (defaults to 1e-2 if required).
                Ignored if a gradient callable is passed to ``GradientDescent.optimize``.
        """
        super().__init__()

        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.tol = tol
        self.callback = callback

    @property
    def settings(self) -> Dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        return {
            "maxiter": self.maxiter,
            "tol": self.tol,
            "learning_rate": learning_rate,
            "perturbation": self.perturbation,
            "callback": self.callback,
        }

    def _minimize(self, loss, grad, initial_point):
        # set learning rate
        if isinstance(self.learning_rate, float):
            eta = constant(self.learning_rate)
        else:
            eta = self.learning_rate()

        if grad is None:
            eps = 0.01 if self.perturbation is None else self.perturbation
            grad = partial(
                Optimizer.gradient_num_diff,
                f=loss,
                epsilon=eps,
                max_evals_grouped=self._max_evals_grouped,
            )

        # prepare some initials
        x = np.asarray(initial_point)
        nfevs = 0

        for _ in range(1, self.maxiter + 1):
            # compute update -- gradient evaluation counts as one function evaluation
            update = grad(x)
            nfevs += 1

            # compute next parameter value
            x_next = x - next(eta) * update

            # send information to callback
            stepsize = np.linalg.norm(update)
            if self.callback is not None:
                self.callback(nfevs, x_next, loss(x_next), stepsize)

            # update parameters
            x = x_next

            # check termination
            if stepsize < self.tol:
                break

        return x, loss(x), nfevs

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    # pylint: disable=unused-argument
    def optimize(
        self,
        num_vars,
        objective_function,
        gradient_function=None,
        variable_bounds=None,
        initial_point=None,
    ):
        return self._minimize(objective_function, gradient_function, initial_point)


def constant(eta=0.01):
    """Yield a constant."""

    while True:
        yield eta
