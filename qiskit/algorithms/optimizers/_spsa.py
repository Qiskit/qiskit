"""A generalized SPSA optimizer including support for Hessians."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union, Callable, Tuple
import logging
import warnings
from time import time

from collections import deque
import scipy
import numpy as np

from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import Optimizer, OptimizerSupportLevel
from qiskit.opflow import ListOp, CircuitSampler
from qiskit.providers import BackendV1 as Backend
from qiskit.utils import QuantumInstance

# a preconditioner can either be a function (e.g. loss function to obtain the Hessian)
# or a metric (e.g. Fubini-Study metric to obtain the quantum Fisher information)
PRECONDITIONER = Union[Callable[[float], float],
                       Callable[[float, float], float]]

# parameters, loss, stepsize, number of function evaluations, accepted
CALLBACK = Callable[[np.ndarray, float, float, int, bool], None]

logger = logging.getLogger(__name__)


class It(ABC):
    @abstractmethod
    def serialize(self):
        raise NotImplementedError

    @abstractmethod
    def get_iterator(self):
        raise NotImplementedError

    @staticmethod
    def deserialize(serialized):
        name, inputs = serialized
        classes = {'Constant': Constant,
                   'Powerlaw': Powerlaw,
                   'Concatenated': Concatenated}
        return classes[name](**inputs)


class Constant(It):
    def __init__(self, value):
        self.value = value

    def get_iterator(self):
        def const():
            while True:
                yield self.value

        return const

    def serialize(self):
        return ('Constant', {'value': self.value})


class Powerlaw(It):
    def __init__(self, coeff, power, offset, skip=0):
        self.coeff = coeff
        self.power = power
        self.offset = offset
        self.skip = skip

    def serialize(self):
        return ('Powerlaw', {'coeff': self.coeff,
                             'power': self.power,
                             'offset': self.offset,
                             'skip': self.skip})

    def get_iterator(self):
        def powerlaw():
            n = 1
            while True:
                if n > self.skip:
                    yield self.coeff / ((n + self.offset) ** self.power)
                n += 1
        return powerlaw


class Concatenated(It):
    def __init__(self, iterators, breakpoints):
        self.iterators = []
        # deserialize if necessary
        for it in iterators:
            if isinstance(it, tuple):
                self.iterators.append(self.deserialize(it))
            else:
                self.iterators.append(it)

        self.breakpoints = breakpoints

    def serialize(self):
        return ('Concatenated', {'iterators':  [it.serialize() for it in self.iterators],
                                 'breakpoints': self.breakpoints})

    def get_iterator(self):
        iterators = [it.get_iterator()() for it in self.iterators]
        breakpoints = self.breakpoints

        def concat():
            i, n = 0, 0  # n counts always up, i is at which iterator/breakpoint pair we are
            while True:
                if i < len(breakpoints) and n >= breakpoints[i]:
                    i += 1
                yield next(iterators[i])
                n += 1

        return concat


class SPSA(Optimizer):
    """A generalized SPSA optimizer including support for Hessians."""

    def __init__(self,
                 maxiter: int = 100,
                 blocking: bool = False,
                 allowed_increase: Optional[float] = None,
                 trust_region: bool = False,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
                 resamplings: int = 1,
                 last_avg: int = 1,
                 callback: Optional[CALLBACK] = None,
                 # 2-SPSA arguments
                 second_order: bool = False,  # skip_calibration: bool = False) -> None:
                 hessian_delay: int = 0,
                 lse_solver: Optional[Union[str,
                                            Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 regularization: Optional[float] = None,
                 perturbation_dims: Optional[int] = None,
                 initial_hessian: Optional[np.ndarray] = None,
                 backend: Optional[Union[Backend, QuantumInstance]] = None,
                 ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            blocking: If True, only accepts updates that improve the loss.
            allowed_increase: If blocking is True, this sets by how much the loss can increase
                and still be accepted. If None, calibrated automatically to be twice the
                standard deviation of the loss function.
            trust_region: If True, restricts norm of the random direction to be <= 1.
            learning_rate: A generator yielding learning rates for the parameter updates,
                :math:`a_k`.
            perturbation: A generator yielding the perturbation magnitudes :math:`c_k`.
            tolerance: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
            last_avg: Return the average of the ``last_avg`` parameters instead of just the
                last parameter values.
            callback: A callback function passed information in each iteration step. The
                information is, in this order: the parameters, the function value, the number
                of function evaluations, the stepsize, whether the step was accepted.
            second_order: If True, use 2-SPSA instead of SPSA. In 2-SPSA, the Hessian is estimated
                additionally to the gradient, and the gradient is preconditioned with the inverse
                of the Hessian to improve convergence.
            hessian_delay: Start preconditioning only after a certain number of iterations.
                Can be useful to first get a stable average over the last iterations before using
                the preconditioner.
            hessian_resamplings: In each step, sample the preconditioner this many times. Default
                is 1.
            lse_solver: The method to solve for the inverse of the preconditioner. Per default an
                exact LSE solver is used, but can e.g. be overwritten by a minimization routine.
            regularization: To ensure the preconditioner is symmetric and positive definite, the
                identity times a small coefficient is added to it. This generator yields that
                coefficient.
            perturbation_dims: The number of dimensions to perturb at once. Per default all
                dimensions are perturbed simulatneously.
            initial_hessian: The initial guess for the Hessian. By default the identity matrix
                is used.
            backend: A backend to evaluate the circuits, if the overlap function is provided as
                a circuit and the objective function as operator expression.
        """
        super().__init__()

        if regularization is None:
            regularization = 0.01

        if lse_solver is None:
            lse_solver = np.linalg.solve

        self.maxiter = maxiter
        self.perturbation = perturbation
        self.learning_rate = learning_rate
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.trust_region = trust_region
        self.callback = callback
        self.resamplings = resamplings
        self.last_avg = last_avg
        self.second_order = second_order
        self.hessian_delay = hessian_delay
        self.lse_solver = lse_solver
        self.regularization = regularization
        self.perturbation_dims = perturbation_dims
        self.initial_hessian = initial_hessian
        self.trust_region = trust_region

        # runtime arguments
        self.grad_params = None
        self.grad_expr = None
        self.hessian_params = None
        self.hessian_expr = None
        self.gradient_expressions = None

        if backend is not None:
            self._sampler = CircuitSampler(backend, caching='all')
        else:
            self._sampler = None

        self._nfev = None
        self._moving_avg = None  # moving average of the preconditioner

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
            target_magnitude: The target magnitude for the first update step.
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
            pert = np.array([1 - 2 * np.random.binomial(1, 0.5)
                             for _ in range(dim)])
            delta = loss(initial_point + c * pert) - \
                loss(initial_point - c * pert)
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

    @property
    def name(self):
        return 'SPSA'

    def to_dict(self):
        for obj in [self.perturbation, self.learning_rate]:
            if not (obj is None or isinstance(obj, (It, float))):
                raise AttributeError('Learning rate and perturbation must be None or float.')

        if isinstance(self.learning_rate, It):
            learning_rate = self.learning_rate.serialize()
        else:
            learning_rate = self.learning_rate

        if isinstance(self.perturbation, It):
            perturbation = self.perturbation.serialize()
        else:
            perturbation = self.perturbation

        if self.callback is not None:
            raise AttributeError('Callback not serializable.')

        return {'maxiter': self.maxiter,
                'learning_rate': learning_rate,
                'perturbation': perturbation,
                'blocking': self.blocking,
                'allowed_increase': self.allowed_increase,
                'resamplings': self.resamplings,
                # 'trust_region': self.trust_region,
                # 'last_avg': self.last_avg,
                # 'second_order': self.second_order,
                'hessian_delay': self.hessian_delay,
                'regularization': self.regularization,
                # 'perturbation_dims': self.perturbation_dims,
                'initial_hessian': self.initial_hessian
                }

    def _point_sample_blackbox(self, loss, x, eps, delta1, delta2):
        pert1, pert2 = eps * delta1, eps * delta2

        # compute the gradient approximation and additionally return the loss function evaluations
        plus, minus = loss(x + eps * delta1), loss(x - eps * delta1)
        gradient_sample = (plus - minus) / (2 * eps) * delta1
        fx_sample = (plus + minus) / 2
        self._nfev += 2

        hessian_sample = None
        if self.second_order:
            # compute the preconditioner point estimate
            diff = loss(x + pert1 + pert2) - plus
            diff -= loss(x - pert1 + pert2) - minus
            diff /= 2 * eps ** 2

            self._nfev += 2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return gradient_sample, hessian_sample, fx_sample

    def _point_samples_blackbox(self, loss, x, eps, deltas1, deltas2):
        # number of samples
        resamplings = len(deltas1)

        # set up variables to store averages
        gradient_estimate, hessian_estimate = np.zeros(x.size), np.zeros((x.size, x.size))
        fx_estimate = 0

        # iterate over the directions
        for delta1, delta2 in zip(deltas1, deltas2):
            gradient_sample, hessian_sample, fx_sample = self._point_sample_blackbox(loss, x, eps,
                                                                                     delta1, delta2)
            gradient_estimate += gradient_sample
            fx_estimate += fx_sample

            if self.second_order:
                hessian_estimate += hessian_sample

        return (gradient_estimate / resamplings,
                hessian_estimate / resamplings,
                fx_estimate / resamplings)

    def _point_samples_circuits(self, loss, x, eps, deltas1, deltas2):
        # cache gradient epxressions
        if self.gradient_expressions is None:
            # sorted loss parameters
            sorted_params = sorted(loss.parameters, key=lambda p: p.name)

            # SPSA estimates
            theta_p = ParameterVector('th+', len(loss.parameters))
            theta_m = ParameterVector('th-', len(loss.parameters))

            # 2-SPSA estimates
            x_pp = ParameterVector('x++', len(loss.parameters))
            x_pm = ParameterVector('x+-', len(loss.parameters))
            x_mp = ParameterVector('x-+', len(loss.parameters))
            x_mm = ParameterVector('x--', len(loss.parameters))

            self.grad_expr = [
                loss.assign_parameters(dict(zip(sorted_params, theta_p))),
                loss.assign_parameters(dict(zip(sorted_params, theta_m)))
            ]
            self.grad_params = [theta_p, theta_m]

            # catch QNSPSA case. Could be put in a method to make it a bit nicer
            if self.hessian_expr is None:
                self.hessian_expr = [
                    loss.assign_parameters(dict(zip(sorted_params, x_pp))),
                    loss.assign_parameters(dict(zip(sorted_params, x_pm))),
                    loss.assign_parameters(dict(zip(sorted_params, x_mp))),
                    loss.assign_parameters(dict(zip(sorted_params, x_mm))),
                ]
                self.hessian_params = [x_pp, x_pm, x_mp, x_mm]

            self.gradient_expressions = ListOp(self.grad_expr + self.hessian_expr)

        num_parameters = x.size
        resamplings = len(deltas1)

        # SPSA parameters
        theta_p_ = np.array([x + eps * delta1 for delta1 in deltas1])
        theta_m_ = np.array([x - eps * delta1 for delta1 in deltas1])

        # 2-SPSA parameters
        x_pp_ = np.array([x + eps * (delta1 + delta2) for delta1, delta2 in zip(deltas1, deltas2)])
        x_pm_ = np.array([x + eps * delta1 for delta1 in deltas1])
        x_mp_ = np.array([x - eps * (delta1 - delta2) for delta1, delta2 in zip(deltas1, deltas2)])
        x_mm_ = np.array([x - eps * delta1 for delta1 in deltas1])
        y_ = np.array([x for _ in deltas1])

        # build dictionary
        values_dict = {}

        for params, value_matrix in zip(
            self.grad_params + self.hessian_params,
            [theta_p_, theta_m_, x_pp_, x_pm_, x_mp_, x_mm_, y_],
        ):
            values_dict.update({
                params[i]: value_matrix[:, i].tolist() for i in range(num_parameters)
            })

        # execute at once
        results = np.array(self._sampler.convert(self.gradient_expressions,
                                                 params=values_dict).eval()).real

        # put results together
        gradient_estimate = np.zeros(x.size)
        fx_estimate = 0
        for i in range(resamplings):
            gradient_estimate += (results[i, 0] - results[i, 1]) / (2 * eps) * deltas1[0]
            fx_estimate += (results[i, 0] + results[i, 1]) / 2

        hessian_estimate = np.zeros((x.size, x.size))
        for i in range(resamplings):
            diff = results[i, 2] - results[i, 3]
            diff -= results[i, 4] - results[i, 5]
            diff /= 2 * eps ** 2

            rank_one = np.outer(deltas1[i], deltas2[i])
            hessian_estimate += diff * (rank_one + rank_one.T) / 2

        return (gradient_estimate / resamplings,
                hessian_estimate / resamplings,
                fx_estimate / resamplings)

    def _compute_update(self, loss, x, k, eps):
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            avg = self.resamplings.get(k, 1)
        else:
            avg = self.resamplings

        gradient = np.zeros(x.size)
        preconditioner = np.zeros((x.size, x.size))

        # accumulate the number of samples
        deltas1 = [bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(avg)]
        deltas2 = [bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(avg)]

        if callable(loss):
            gradient, preconditioner, fx = self._point_samples_blackbox(loss, x, eps, deltas1,
                                                                        deltas2)
        else:
            gradient, preconditioner, fx = self._point_samples_circuits(loss, x, eps, deltas1,
                                                                        deltas2)

        # update the exponentially smoothed average
        if self.second_order:
            smoothed = k / (k + 1) * self._moving_avg + 1 / (k + 1) * preconditioner
            self._moving_avg = smoothed

            if k > self.hessian_delay:
                # make the preconditioner SPD
                spd_preconditioner = _make_spd(smoothed, self.regularization)

                # solve for the gradient update
                gradient = np.real(self.lse_solver(spd_preconditioner, gradient))

        return gradient, fx

    def _minimize(self, loss, initial_point):
        # handle circuits case
        if not callable(loss):
            # sorted loss parameters
            sorted_params = sorted(loss.parameters, key=lambda p: p.name)

            def loss_callable(x):
                value_dict = dict(zip(sorted_params, x))
                return self._sampler.convert(loss, params=value_dict).eval().real

        else:
            loss_callable = loss

        # ensure learning rate and perturbation are set
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_learning_rate, get_perturbation = self.calibrate(loss_callable, initial_point)
            eta = get_learning_rate()
            eps = get_perturbation()
        elif self.learning_rate is None or self.perturbation is None:
            raise ValueError('If one of learning rate or perturbation is set, both must be set.')
        else:
            if isinstance(self.learning_rate, float):
                eta = constant(self.learning_rate)
            else:
                eta = self.learning_rate()

            if isinstance(self.perturbation, float):
                eps = constant(self.perturbation)
            else:
                eps = self.perturbation()

        # prepare some initials
        x = np.asarray(initial_point)

        if self.initial_hessian is None:
            self._moving_avg = np.identity(x.size)
        else:
            self._moving_avg = self.initial_hessian

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss_callable(x)

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
            update, fx_next = self._compute_update(loss, x, k, next(eps))

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
                fx_next = loss_callable(x_next)

                self._nfev += 1
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

        return x, loss_callable(x), self._nfev

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            'gradient': OptimizerSupportLevel.ignored,  # could be supported though
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        return self._minimize(objective_function, initial_point)


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return np.array([1 - 2 * np.random.binomial(1, 0.5) for _ in range(dim)])

    pert = np.array([1 - 2 * np.random.binomial(1, 0.5)
                     for _ in range(perturbation_dims)])
    indices = np.random.choice(list(range(dim)), size=perturbation_dims, replace=False)
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


def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return (1 - bias) * psd + bias * identity
