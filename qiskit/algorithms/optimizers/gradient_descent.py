"""A generalized SPSA optimizer including support for Hessians."""

from typing import Iterator, Optional, Union, Callable
from functools import partial

import numpy as np

from .optimizer import Optimizer, OptimizerSupportLevel

CALLBACK = Callable[[int, np.ndarray, float], None]


class GradientDescent(Optimizer):
    """The gradient descent minimization routine."""

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
                updates.
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
            update = update * next(eta)
            x_next = x - update

            # send information to callback
            if self.callback is not None:
                self.callback(nfevs, x_next, loss(x_next))

            # update parameters
            x = x_next

            # check termination
            if np.linalg.norm(update) < self.tol:
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
