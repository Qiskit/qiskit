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

"""Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This implementation allows both, standard first-order as well as second-order SPSA.
"""

from typing import Iterator, Optional, Union, Callable, Tuple, Dict
import logging
import warnings
from time import time

from collections import deque
import scipy
import numpy as np

from qiskit.utils import algorithm_globals

from .optimizer import Optimizer, OptimizerSupportLevel

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, float, bool], None]

logger = logging.getLogger(__name__)


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA [1] is an gradient descent method for optimizing systems with multiple unknown parameters.
    As an optimization method, it is appropriately suited to large-scale population models,
    adaptive modeling, and simulation optimization.

    .. seealso::

        Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.

    The main feature of SPSA is the stochastic gradient approximation, which requires only two
    measurements of the objective function, regardless of the dimension of the optimization
    problem.

    Additionally to standard, first-order SPSA, where only gradient information is used, this
    implementation also allows second-order SPSA (2-SPSA) [2]. In 2-SPSA we also estimate the
    Hessian of the loss with a stochastic approximation and multiply the gradient with the
    inverse Hessian to take local curvature into account and improve convergence.
    Notably this Hessian estimate requires only a constant number of function evaluations
    unlike an exact evaluation of the Hessian, which scales quadratically in the number of
    function evaluations.

    .. note::

        SPSA can be used in the presence of noise, and it is therefore indicated in situations
        involving measurement uncertainty on a quantum computation when finding a minimum.
        If you are executing a variational algorithm using a Quantum ASseMbly Language (QASM)
        simulator or a real device, SPSA would be the most recommended choice among the optimizers
        provided here.

    The optimization process can includes a calibration phase if neither the ``learning_rate`` nor
    ``perturbation`` is provided, which requires additional functional evaluations.
    (Note that either both or none must be set.) For further details on the automatic calibration,
    please refer to the supplementary information section IV. of [3].

    .. note::

        This component has some function that is normally random. If you want to reproduce behavior
        then you should set the random number generator seed in the algorithm_globals
        (``qiskit.utils.algorithm_globals.random_seed = seed``).


    Examples:

        This short example runs SPSA for the ground state calculation of the ``Z ^ Z``
        observable where the ansatz is a ``PauliTwoDesign`` circuit.

        .. code-block:: python

            import numpy as np
            from qiskit.algorithms.optimizers import SPSA
            from qiskit.circuit.library import PauliTwoDesign
            from qiskit.opflow import Z, StateFn

            ansatz = PauliTwoDesign(2, reps=1, seed=2)
            observable = Z ^ Z
            initial_point = np.random.random(ansatz.num_parameters)

            def loss(x):
                bound = ansatz.bind_parameters(x)
                return np.real((StateFn(observable, is_measurement=True) @ StateFn(bound)).eval())

            spsa = SPSA(maxiter=300)
            result = spsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)

        To use the Hessian information, i.e. 2-SPSA, you can add `second_order=True` to the
        initializer of the `SPSA` class, the rest of the code remains the same.

        .. code-block:: python

            two_spsa = SPSA(maxiter=300, second_order=True)
            result = two_spsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)


    References:

        [1]: J. C. Spall (1998). An Overview of the Simultaneous Perturbation Method for Efficient
        Optimization, Johns Hopkins APL Technical Digest, 19(4), 482–492.
        `Online at jhuapl.edu. <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

        [2]: J. C. Spall (1997). Accelerated second-order stochastic optimization using only
        function measurements, Proceedings of the 36th IEEE Conference on Decision and Control,
        1417-1424 vol.2. `Online at IEEE.org. <https://ieeexplore.ieee.org/document/657661>`_

        [3]: A. Kandala et al. (2017). Hardware-efficient Variational Quantum Eigensolver for
        Small Molecules and Quantum Magnets. Nature 549, pages242–246(2017).
        `arXiv:1704.05018v2 <https://arxiv.org/pdf/1704.05018v2.pdf#section*.11>`_

    """

    def __init__(
        self,
        maxiter: int = 100,
        blocking: bool = False,
        allowed_increase: Optional[float] = None,
        trust_region: bool = False,
        learning_rate: Optional[Union[float, np.array, Callable[[], Iterator]]] = None,
        perturbation: Optional[Union[float, np.array, Callable[[], Iterator]]] = None,
        last_avg: int = 1,
        resamplings: Union[int, Dict[int, int]] = 1,
        perturbation_dims: Optional[int] = None,
        second_order: bool = False,
        regularization: Optional[float] = None,
        hessian_delay: int = 0,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        initial_hessian: Optional[np.ndarray] = None,
        callback: Optional[CALLBACK] = None,
    ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations. Note that this is not the maximal number
                of function evaluations.
            blocking: If True, only accepts updates that improve the loss (up to some allowed
                increase, see next argument).
            allowed_increase: If ``blocking`` is ``True``, this argument determines by how much
                the loss can increase with the proposed parameters and still be accepted.
                If ``None``, the allowed increases is calibrated automatically to be twice the
                approximated standard deviation of the loss function.
            trust_region: If ``True``, restricts the norm of the update step to be :math:`\leq 1`.
            learning_rate: The update step is the learning rate is multiplied with the gradient.
                If the learning rate is a float, it remains constant over the course of the
                optimization. If a NumPy array, the :math:`i`-th element is the learning rate for
                the :math:`i`-th iteration. It can also be a callable returning an iterator which
                yields the learning rates for each optimization step.
                If ``learning_rate`` is set ``perturbation`` must also be provided.
            perturbation: Specifies the magnitude of the perturbation for the finite difference
                approximation of the gradients. See ``learning_rate`` for the supported types.
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
            second_order: If True, use 2-SPSA instead of SPSA. In 2-SPSA, the Hessian is estimated
                additionally to the gradient, and the gradient is preconditioned with the inverse
                of the Hessian to improve convergence.
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
                information is, in this order: the number of function evaluations, the parameters,
                the function value, the stepsize, whether the step was accepted.

        Raises:
            ValueError: If ``learning_rate`` or ``perturbation`` is an array with less elements
                than the number of iterations.
        """
        super().__init__()

        # general optimizer arguments
        self.maxiter = maxiter
        self.trust_region = trust_region
        self.callback = callback

        # if learning rate and perturbation are arrays, check they are sufficiently long
        for attr, name in zip([learning_rate, perturbation], ["learning_rate", "perturbation"]):
            if isinstance(attr, (list, np.ndarray)):
                if len(attr) < maxiter:
                    raise ValueError(f"Length of {name} is smaller than maxiter ({maxiter}).")

        self.learning_rate = learning_rate
        self.perturbation = perturbation

        # SPSA specific arguments
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.last_avg = last_avg
        self.resamplings = resamplings
        self.perturbation_dims = perturbation_dims

        # 2-SPSA specific arguments
        if regularization is None:
            regularization = 0.01

        self.second_order = second_order
        self.hessian_delay = hessian_delay
        self.lse_solver = lse_solver
        self.regularization = regularization
        self.initial_hessian = initial_hessian

        # runtime arguments
        self._nfev = None  # the number of function evaluations
        self._smoothed_hessian = None  # smoothed average of the Hessians

    @staticmethod
    def calibrate(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        c: float = 0.2,
        stability_constant: float = 0,
        target_magnitude: Optional[float] = None,  # 2 pi / 10
        alpha: float = 0.602,
        gamma: float = 0.101,
        modelspace: bool = False,
    ) -> Tuple[Iterator[float], Iterator[float]]:
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
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        # set up the powerseries
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation

    @staticmethod
    def estimate_stddev(
        loss: Callable[[np.ndarray], float], initial_point: np.ndarray, avg: int = 25
    ) -> float:
        """Estimate the standard deviation of the loss function."""
        losses = [loss(initial_point) for _ in range(avg)]
        return np.std(losses)

    @property
    def settings(self):
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        if callable(self.perturbation):
            iterator = self.perturbation()
            perturbation = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            perturbation = self.perturbation

        return {
            "maxiter": self.maxiter,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "trust_region": self.trust_region,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "resamplings": self.resamplings,
            "perturbation_dims": self.perturbation_dims,
            "second_order": self.second_order,
            "hessian_delay": self.hessian_delay,
            "regularization": self.regularization,
            "lse_solver": self.lse_solver,
            "initial_hessian": self.initial_hessian,
            "callback": self.callback,
        }

    def _point_sample(self, loss, x, eps, delta1, delta2):
        """A single sample of the gradient at position ``x`` in direction ``delta``."""
        # points to evaluate
        points = [x + eps * delta1, x - eps * delta1]
        self._nfev += 2

        if self.second_order:
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            self._nfev += 2

        # batch evaluate the points (if possible)
        values = _batch_evaluate(loss, points, self._max_evals_grouped)

        plus = values[0]
        minus = values[1]
        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps ** 2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return gradient_sample, hessian_sample

    def _point_estimate(self, loss, x, eps, num_samples):
        """The gradient estimate at point x."""
        # set up variables to store averages
        gradient_estimate = np.zeros(x.size)
        hessian_estimate = np.zeros((x.size, x.size))

        # iterate over the directions
        deltas1 = [
            bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
        ]

        if self.second_order:
            deltas2 = [
                bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
            ]
        else:
            deltas2 = None

        for i in range(num_samples):
            delta1 = deltas1[i]
            delta2 = deltas2[i] if self.second_order else None

            gradient_sample, hessian_sample = self._point_sample(loss, x, eps, delta1, delta2)
            gradient_estimate += gradient_sample

            if self.second_order:
                hessian_estimate += hessian_sample

        return gradient_estimate / num_samples, hessian_estimate / num_samples

    def _compute_update(self, loss, x, k, eps, lse_solver):
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(k, 1)
        else:
            num_samples = self.resamplings

        # accumulate the number of samples
        gradient, hessian = self._point_estimate(loss, x, eps, num_samples)

        # precondition gradient with inverse Hessian, if specified
        if self.second_order:
            smoothed = k / (k + 1) * self._smoothed_hessian + 1 / (k + 1) * hessian
            self._smoothed_hessian = smoothed

            if k > self.hessian_delay:
                spd_hessian = _make_spd(smoothed, self.regularization)

                # solve for the gradient update
                gradient = np.real(lse_solver(spd_hessian, gradient))

        return gradient

    def _minimize(self, loss, initial_point):
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_eta, get_eps = self.calibrate(loss, initial_point)
        else:
            get_eta, get_eps = _validate_pert_and_learningrate(
                self.perturbation, self.learning_rate
            )
        eta, eps = get_eta(), get_eps()

        if self.lse_solver is None:
            lse_solver = np.linalg.solve
        else:
            lse_solver = self.lse_solver

        # prepare some initials
        x = np.asarray(initial_point)
        if self.initial_hessian is None:
            self._smoothed_hessian = np.identity(x.size)
        else:
            self._smoothed_hessian = self.initial_hessian

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss(x)

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(loss, x)

        logger.info("=" * 30)
        logger.info("Starting SPSA optimization")
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        for k in range(1, self.maxiter + 1):
            iteration_start = time()
            # compute update
            update = self._compute_update(loss, x, k, next(eps), lse_solver)

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
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
                            False,
                        )  # not accepted

                    logger.info(
                        "Iteration %s/%s rejected in %s.",
                        k,
                        self.maxiter + 1,
                        time() - iteration_start,
                    )
                    continue
                fx = fx_next

            logger.info(
                "Iteration %s/%s done in %s.", k, self.maxiter + 1, time() - iteration_start
            )

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = loss(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
                    True,
                )  # accepted

            # update parameters
            x = x_next

            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

        logger.info("SPSA finished in %s", time() - start)
        logger.info("=" * 30)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        return x, loss(x), self._nfev

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    def optimize(
        self,
        num_vars,
        objective_function,
        gradient_function=None,
        variable_bounds=None,
        initial_point=None,
    ):
        return self._minimize(objective_function, initial_point)


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(
        list(range(dim)), size=perturbation_dims, replace=False
    )
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


def _batch_evaluate(function, points, max_evals_grouped):
    # if the function cannot handle lists of points as input, cover this case immediately
    if max_evals_grouped == 1:
        # support functions with multiple arguments where the points are given in a tuple
        return [
            function(*point) if isinstance(point, tuple) else function(point) for point in points
        ]

    num_points = len(points)

    # get the number of batches
    num_batches = num_points // max_evals_grouped
    if num_points % max_evals_grouped != 0:
        num_batches += 1

    # split the points
    batched_points = np.split(np.asarray(points), num_batches)

    results = []
    for batch in batched_points:
        results += function(batch).tolist()

    return results


def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return psd + bias * identity


def _validate_pert_and_learningrate(perturbation, learning_rate):
    if learning_rate is None or perturbation is None:
        raise ValueError("If one of learning rate or perturbation is set, both must be set.")

    if isinstance(perturbation, float):

        def get_eps():
            return constant(perturbation)

    elif isinstance(perturbation, (list, np.ndarray)):

        def get_eps():
            return iter(perturbation)

    else:
        get_eps = perturbation

    if isinstance(learning_rate, float):

        def get_eta():
            return constant(learning_rate)

    elif isinstance(learning_rate, (list, np.ndarray)):

        def get_eta():
            return iter(learning_rate)

    else:
        get_eta = learning_rate

    return get_eta, get_eps
