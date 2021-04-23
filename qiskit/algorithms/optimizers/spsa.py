# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer."""

from typing import Iterator, Optional, Union, Callable, Tuple, Dict
import logging
import warnings
from time import time

from collections import deque
import numpy as np

from qiskit.utils import algorithm_globals

from .optimizer import Optimizer, OptimizerSupportLevel

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, float, bool], None]

logger = logging.getLogger(__name__)


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA [1] is an algorithmic method for optimizing systems with multiple unknown parameters.
    As an optimization method, it is appropriately suited to large-scale population models,
    adaptive modeling, and simulation optimization.

    .. seealso::

        Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.

    SPSA is a descent method capable of finding global minima,
    sharing this property with other methods as simulated annealing.
    Its main feature is the gradient approximation, which requires only two
    measurements of the objective function, regardless of the dimension of the optimization
    problem.

    .. note::

        SPSA can be used in the presence of noise, and it is therefore indicated in situations
        involving measurement uncertainty on a quantum computation when finding a minimum.
        If you are executing a variational algorithm using a Quantum ASseMbly Language (QASM)
        simulator or a real device, SPSA would be the most recommended choice among the optimizers
        provided here.

    The optimization process can includes a calibration phase if neither the ``learning_rate`` nor
    ``perturbation`` is provided, which requires additional functional evaluations.
    (Note that either both or none must be set.) For further details on the automatic calibration,
    please refer to the supplementary information section IV. of [2].

    References:

        [1]: J. C. Spall (1998). An Overview of the Simultaneous Perturbation Method for Efficient
        Optimization, Johns Hopkins APL Technical Digest, 19(4), 482–492.
        `Online. <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

        [2]: A. Kandala et al. (2017). Hardware-efficient Variational Quantum Eigensolver for
        Small Molecules and Quantum Magnets. Nature 549, pages242–246(2017).
        `arXiv:1704.05018v2 <https://arxiv.org/pdf/1704.05018v2.pdf#section*.11>`_

    """

    def __init__(self,
                 maxiter: int = 100,
                 blocking: bool = False,
                 allowed_increase: Optional[float] = None,
                 trust_region: bool = False,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
                 last_avg: int = 1,
                 resamplings: Union[int, Dict[int, int]] = 1,
                 perturbation_dims: Optional[int] = None,
                 callback: Optional[CALLBACK] = None,
                 ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            blocking: If True, only accepts updates that improve the loss (minus some allowed
                increase, see next argument).
            allowed_increase: If blocking is True, this sets by how much the loss can increase
                and still be accepted. If None, calibrated automatically to be twice the
                standard deviation of the loss function.
            trust_region: If True, restricts norm of the random direction to be :math:`\leq 1`.
            learning_rate: A generator yielding learning rates for the parameter updates,
                :math:`a_k`. If set, also ``perturbation`` must be provided.
            perturbation: A generator yielding the perturbation magnitudes :math:`c_k`. If set,
                also ``learning_rate`` must be provided.
            last_avg: Return the average of the ``last_avg`` parameters instead of just the
                last parameter values.
            resamplings: The number of times the gradient is sampled using a random direction to
                construct a gradient estimate. Per default the gradient is estimated using only
                one random direction. If an integer, all iterations use the same number of
                resamplings. If a dictionary, this is interpreted as
                ``{iteration: number of resamplings per iteration}``.
            perturbation_dims: The number of perturbed dimensions. Per default, all dimensions
                are perturbed, but a smaller, fixed number can be perturbed. If set, the perturbed
                dimensions are chosen uniformly at random.
            callback: A callback function passed information in each iteration step. The
                information is, in this order: the number of function evaluations, the parameters,
                the function value, the stepsize, whether the step was accepted.
        """
        super().__init__()

        if isinstance(learning_rate, float):
            self.learning_rate = lambda: constant(learning_rate)
        else:
            self.learning_rate = learning_rate

        if isinstance(perturbation, float):
            self.perturbation = lambda: constant(perturbation)
        else:
            self.perturbation = perturbation

        self.maxiter = maxiter
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.trust_region = trust_region
        self.callback = callback
        self.last_avg = last_avg
        self.resamplings = resamplings
        self.perturbation_dims = perturbation_dims

        # runtime arguments
        self._nfev = None

    @staticmethod
    def calibrate(loss: Callable[[np.ndarray], float],
                  initial_point: np.ndarray,
                  c: float = 0.2,
                  stability_constant: float = 0,
                  target_magnitude: Optional[float] = None,  # 2 pi / 10
                  alpha: float = 0.602,
                  gamma: float = 0.101,
                  modelspace: bool = False) -> Tuple[Iterator[float], Iterator[float]]:
        r"""Calibrate SPSA parameters with a powerseries as learning rate and perturbation coeffs.

        The powerseries are:

        .. math::

            a_k = \frac{a}{(A + k + 1)^\alpha}, c_k = \frac{c}{(k + 1)^\gamma}

        Args:
            loss: The loss function.
            initial_point: The initial guess of the iteration.
            c: The initial perturbation magnitude.
            stability_constant: The value of `A`.
            target_magnitude: The target magnitude for the first update step, defaults to
                :math:`2\pi / 10`.
            alpha: The exponent of the learning rate powerseries.
            gamma: The exponent of the perturbation powerseries.
            modelspace: Whether the target magnitude is the difference of parameter values
                or function values (= model space).

        Returns:
            tuple(generator, generator): A tuple of powerseries generators, the first one for the
                learning rate and the second one for the perturbation.
        """
        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)

        # compute the average magnitude of the first step
        steps = 25
        avg_magnitudes = 0
        for _ in range(steps):
            # compute the random directon
            pert = bernoulli_perturbation(dim)
            delta = loss(initial_point + c * pert) - loss(initial_point - c * pert)

            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes ** 2)
        else:
            a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            warnings.warn(f'Calibration failed, using {target_magnitude} for `a`')
            a = target_magnitude

        # set up the powerseries
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation

    @staticmethod
    def estimate_stddev(loss: Callable[[np.ndarray], float],
                        initial_point: np.ndarray,
                        avg: int = 25) -> float:
        """Estimate the standard deviation of the loss function."""
        losses = [loss(initial_point) for _ in range(avg)]
        return np.std(losses)

    def _point_sample(self, loss, x, eps, delta):
        """A single sample of the gradient at position ``x`` in direction ``delta``."""
        if self._max_evals_grouped > 1:
            plus, minus = loss(np.concatenate((x + eps * delta, x - eps * delta)))
        else:
            plus, minus = loss(x + eps * delta), loss(x - eps * delta)

        gradient_sample = (plus - minus) / (2 * eps) * delta
        self._nfev += 2

        return gradient_sample

    def _point_estimate(self, loss, x, eps, deltas):
        """The gradient estimate at point ``x`` consisting as average of all directions ``delta``.
        """
        # number of samples
        resamplings = len(deltas)

        # set up variables to store averages
        gradient_estimate = np.zeros(x.size)

        # iterate over the directions
        for delta in deltas:
            gradient_sample = self._point_sample(loss, x, eps, delta)
            gradient_estimate += gradient_sample

        return gradient_estimate / resamplings

    def _compute_update(self, loss, x, k, eps):
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            avg = self.resamplings.get(k, 1)
        else:
            avg = self.resamplings

        gradient = np.zeros(x.size)

        # accumulate the number of samples
        deltas = [bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(avg)]

        gradient = self._point_estimate(loss, x, eps, deltas)

        return gradient

    def _minimize(self, loss, initial_point):
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_learning_rate, get_perturbation = self.calibrate(loss, initial_point)
            # get iterator
            eta = get_learning_rate()
            eps = get_perturbation()
        elif self.learning_rate is None or self.perturbation is None:
            raise ValueError('If one of learning rate or perturbation is set, both must be set.')
        else:
            # get iterator
            eta = self.learning_rate()
            eps = self.perturbation()

        # prepare some initials
        x = np.asarray(initial_point)

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss(x)

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(loss, x)

        logger.info('=' * 30)
        logger.info('Starting SPSA optimization')
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        for k in range(1, self.maxiter + 1):
            iteration_start = time()
            # compute update
            update = self._compute_update(loss, x, k, next(eps))

            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update

            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = loss(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(self._nfev,  # number of function evals
                                      x_next,  # next parameters
                                      fx_next,  # loss at next parameters
                                      np.linalg.norm(update),  # size of the update step
                                      False)  # not accepted

                    logger.info('Iteration %s/%s rejected in %s.',
                                k, self.maxiter + 1, time() - iteration_start)
                    continue
                fx = fx_next

            logger.info('Iteration %s/%s done in %s.',
                        k, self.maxiter + 1, time() - iteration_start)

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = loss(x_next)

                self.callback(self._nfev,  # number of function evals
                              x_next,  # next parameters
                              fx_next,  # loss at next parameters
                              np.linalg.norm(update),  # size of the update step
                              True)  # accepted

            # update parameters
            x = x_next

            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

        logger.info('SPSA finished in %s', time() - start)
        logger.info('=' * 30)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        return x, loss(x), self._nfev

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            'gradient': OptimizerSupportLevel.ignored,
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        return self._minimize(objective_function, initial_point)


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(list(range(dim)), size=perturbation_dims,
                                              replace=False)
    result = np.zeros(dim)
    result[indices] = pert

    return result


def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a powerlaw."""

    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1


def constant(eta=0.01):
    """Yield a constant series."""

    while True:
        yield eta
