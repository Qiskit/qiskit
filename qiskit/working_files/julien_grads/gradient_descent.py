"""A generalized SPSA optimizer including support for Hessians."""

from typing import Iterator, Optional, Union, Callable

import numpy as np

from qiskit.aqua.components.optimizers import Optimizer, OptimizerSupportLevel

# a preconditioner can either be a function (e.g. loss function to obtain the Hessian)
# or a metric (e.g. Fubini-Study metric to obtain the quantum Fisher information)
PRECONDITIONER = Union[Callable[[float], float], Callable[[float, float], float]]


class GradientDescent(Optimizer):
    """A generalized SPSA optimizer including support for Hessians."""

    def __init__(self, maxiter: int = 100,
                 blocking: bool = False,
                 trust_region: bool = False,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 tolerance: float = 1e-7
                 ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            blocking: If True, only accepts updates that improve the loss.
            trust_region: If True, restricts norm of the random direction to be <= 1.
            learning_rate: A generator yielding learning rates for the parameter updates,
                :math:`a_k`.
            tolerance: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
        """
        super().__init__()

        self.maxiter = maxiter
        self.blocking = blocking
        self.trust_region = trust_region
        self.learning_rate = learning_rate
        self.tolerance = tolerance

        self.history = None  # data of the last optimization run

    def _minimize(self, loss, grad, initial_point):
        # ensure learning rate is set
        if self.learning_rate is None:
            eps = stepseries()
        elif isinstance(self.learning_rate, float):
            eps = constant(self.learning_rate)
        else:
            eps = self.learning_rate()

        # prepare some initials
        x = np.asarray(initial_point)
        self.history = {'nfev': 0,  # number of function evaluations
                        'nfevs': [0],  # number of function evaluations per iteration
                        'fx': [loss(x)],  # function values
                        'x': [x],  # the parameter values
                        'accepted': [True],  # if the update step was accepted
                        'converged': False,  # whether the algorithm converged
                        }

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss(x)
            self.history['nfev'] += 1

        for _ in range(1, self.maxiter + 1):
            # compute update
            update = grad(x)

            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:
                    update = update / norm

            # compute next parameter value
            update = update * next(eps)
            x_next = x - update
            self.history['x'].append(x_next)
            self.history['fx'].append(loss(x_next))

            # blocking
            if self.blocking:
                fx_next = loss(x_next)
                self.history['nfev'] += 1
                if fx <= fx_next:  # discard update if it didn't improve the loss
                    self.history['accepted'].append(False)
                    self.history['nfevs'].append(self.history['nfev'] - self.history['nfevs'][-1])
                    continue
                fx = fx_next

            self.history['nfevs'].append(self.history['nfev'] - self.history['nfevs'][-1])
            self.history['accepted'].append(True)

            # update parameters
            x = x_next

            # check termination
            if np.linalg.norm(update) < self.tolerance:
                self.history['converged'] = True
                break

        self.history['nfev'] += 1
        return x, loss(x), self.history['nfev']

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            'gradient': OptimizerSupportLevel.ignored,  # could be supported though
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        return self._minimize(objective_function, gradient_function, initial_point)


def bernoulli_perturbation(dim):
    """Get a Bernoulli random perturbation."""
    return np.array([1 - 2 * np.random.binomial(1, 0.5) for _ in range(dim)])


def hseries(eta=0.1, divisor=2, cutoff=1e-3):
    """Yield a decreasing series cutoff to 0 at ``cutoff``."""
    while True:
        if eta > cutoff:
            yield eta
            eta /= divisor
        else:
            yield 0


def stepseries(eta=0.1, batchsize=20, divisor=2):
    """Yield a stepwise decreasing sequence."""

    count = 0
    while True:
        yield eta
        count += 1
        if count >= batchsize:
            eta /= divisor
            count = 0


def constant(eta=0.01):
    """Yield a constant."""

    while True:
        yield eta


def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a powerlaw."""

    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1
