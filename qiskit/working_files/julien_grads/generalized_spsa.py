"""A generalized SPSA optimizer including support for Hessians."""

from typing import Iterator, Optional, Union, Callable, Tuple
import inspect

import scipy
import numpy as np

from qiskit.aqua.components.optimizers import Optimizer, OptimizerSupportLevel

# a preconditioner can either be a function (e.g. loss function to obtain the Hessian)
# or a metric (e.g. Fubini-Study metric to obtain the quantum Fisher information)
PRECONDITIONER = Union[Callable[[float], float],
                       Callable[[float, float], float]]


class GSPSA(Optimizer):
    """A generalized SPSA optimizer including support for Hessians."""

    def __init__(self, maxiter: int = 100,
                 blocking: bool = False,
                 trust_region: Union[bool, str] = True,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
                 tolerance: float = 1e-7,
                 preconditioner: Optional[PRECONDITIONER] = None,
                 preconditioner_delay: int = 0,
                 preconditioner_avg: int = 1,
                 lse_solver: Optional[Union[str,
                                            Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 spd_bias: Optional[float] = None,
                 perturbation_dims: Optional[int] = None,
                 ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            blocking: If True, only accepts updates that improve the loss.
            trust_region: If True, restricts norm of the random direction to be <= 1.
            learning_rate: A generator yielding learning rates for the parameter updates,
                :math:`a_k`.
            perturbation: A generator yielding the perturbation magnitudes :math:`c_k`.
            tolerance: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
            preconditioner: Precondition the gradient update with the inverse of a matrix. If
                None, no preconditioning is used (standard SPSA). Set to the loss function to
                multiply with the inverse of the approximated Hessian (2-SPSA) or to the
                Fubini-study metric to precondition with the Fisher information (natural gradient).
            preconditioner_delay: Start preconditioning only after a certain number of iterations.
                Can be useful to first get a stable average over the last iterations before using
                the preconditioner.
            lse_solver: The method to solve for the inverse of the preconditioner. If set to
                'sherman-morrison' the inverse is computed by using two applications of the
                Sherman-Morrison formula in each iteration. Other must be a callable that takes the
                matrix and right-hand side and returns the solution of the LSE. Per default an exact
                LSE solver is used, but can e.g. be overwritten by a minimization routine.
            spd_bias: To ensure the preconditioner is symmetric and positive definite, the identity
                times a small coefficient is added to it. This generator yields that coefficient.
            perturbation_dims: The number of dimensions to perturb at once. Per default all
                dimensions are perturbed simulatneously.
        """
        super().__init__()

        if spd_bias is None:
            spd_bias = 0.01

        if isinstance(learning_rate, float):
            self.learning_rate = lambda: constant(learning_rate)
        else:
            self.learning_rate = learning_rate

        if isinstance(perturbation, float):
            self.perturbation = lambda: constant(perturbation)
        else:
            self.perturbation = perturbation

        if lse_solver is None:
            lse_solver = np.linalg.solve

        if trust_region is True:
            trust_region = 'soft'

        self.maxiter = maxiter
        self.blocking = blocking
        self.trust_region = trust_region
        self.preconditioner = preconditioner  # more logic included in the setter
        self.tolerance = tolerance
        self.preconditioner_delay = preconditioner_delay
        self.preconditioner_avg = preconditioner_avg
        self.lse_solver = lse_solver
        self.spd_bias = spd_bias
        self.perturbation_dims = perturbation_dims

        self.history = None  # data of the last optimization run
        self._moving_avg = None  # moving average of the preconditioner
        self._moving_avg_inv = None  # inverse used for shermann-morrison update

    @staticmethod
    def calibrate(loss: Callable[[np.ndarray], float],
                  initial_point: np.ndarray,
                  c: float = 0.2,
                  stability_constant: float = 0,
                  target_magnitude: float = 1,
                  alpha: float = 0.602,
                  gamma: float = 0.101) -> Tuple[Iterator[float], Iterator[float]]:
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

        Returns:
            tuple(generator, generator): A tuple of powerseries generators, the first one for the
                learning rate and the second one for the perturbation.
        """
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
            delta /= 2 * c * pert

            # get the update step
            update = delta / (stability_constant + 1) ** alpha

            # update the average
            avg_magnitudes += np.abs(update) / steps

        # take a mean over all elements
        magnitude = np.linalg.norm(avg_magnitudes) / dim

        # compute the rescaling factor for correct first learning rate
        if magnitude > 1e-10:
            a = target_magnitude / magnitude
        else:
            print('calibration might not have worked')
            a = target_magnitude

        # set up the powerseries
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation

    @property
    def preconditioner(self) -> Optional[PRECONDITIONER]:
        """Get the preconditioner.

        Returns:
            The preconditioner.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, preconditioner: Optional[PRECONDITIONER]) -> None:
        """Set the preconditioner.

        Args:
            preconditioner: The preconditioner or None.
        """
        self._preconditioner = preconditioner

        # bring the preconditioner in a common format where the first argument is the current
        # parameter value and the second the perturbation
        if preconditioner is not None:
            parameters = inspect.signature(preconditioner).parameters
            is_metric = len(parameters) == 2

            if is_metric:
                def eval_preconditioner(x, y):
                    return preconditioner(x, x + y)
            else:
                def eval_preconditioner(x, y):
                    return preconditioner(x + y)
        else:
            def eval_preconditioner(x, y):  # pylint: disable=unused-argument
                return None

        self._eval_preconditioner = eval_preconditioner

    def _make_spd(self, matrix):
        # return self._make_spd_eig(matrix)
        return self._make_spd_sqrt(matrix)

    def _make_spd_sqrt(self, matrix):
        identity = np.identity(matrix.shape[0])
        psd = scipy.linalg.sqrtm(matrix.dot(matrix))
        return psd + self.spd_bias * identity

    def _make_spd_eig(self, matrix):
        identity = np.identity(matrix.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        clipped_eigenvalues = np.clip(eigenvalues, 0, np.inf)

        # positive semidefinite matrix
        psd = eigenvectors.dot(np.diag(clipped_eigenvalues)).dot(eigenvectors.T)

        return psd + self.spd_bias * identity

    def _compute_gradient(self, loss, x, eps, delta):
        # compute the gradient approximation and additionally return the loss function evaluations
        plus, minus = loss(x + eps * delta), loss(x - eps * delta)
        self.history['nfev'] += 2
        return (plus - minus) / (2 * eps) * delta, plus, minus

    def _compute_precond_delta(self, x, eps, delta1, delta2, plus, minus):
        pert1, pert2 = eps * delta1, eps * delta2
        # if the loss is the preconditioner we can save two evaluations
        # if loss is not self.preconditioner:
        plus = self._eval_preconditioner(x, pert1)
        minus = self._eval_preconditioner(x, -pert1)
        self.history['nfev'] += 2

        # compute the preconditioner point estimate
        diff = self._eval_preconditioner(x, pert1 + pert2) - plus
        diff -= self._eval_preconditioner(x, -pert1 + pert2) - minus
        diff /= 2 * eps ** 2
        self.history['nfev'] += 2

        return diff

    def _preconditioner_via_solver(self, x, k, eps, delta1, delta2, plus, minus, gradient):
        estimate = np.zeros((x.size, x.size))
        rank_one = np.outer(delta1, delta2)

        for _ in range(self.preconditioner_avg):
            diff = self._compute_precond_delta(x, eps, delta1, delta2, plus, minus)
            estimate = estimate + diff * (rank_one + rank_one.T) / 2

        estimate = estimate / self.preconditioner_avg

        # update moving average
        self._moving_avg = k / (k + 1) * self._moving_avg + 1 / (k + 1) * estimate
        self.history['metric'].append(self._moving_avg)

        # make moving average spd
        if self.lse_solver == 'eigenvalues':
            eigenvalues, eigenvectors = np.linalg.eigh(self._moving_avg)
            clipped = np.clip(eigenvalues, 0, np.inf)
            biased = clipped + self.spd_bias
            inverse = eigenvectors.dot(np.diag(1 / biased)).dot(eigenvectors.T)
            return inverse.dot(gradient)

        spd = self._make_spd(self._moving_avg).real

        # solve for the update if we're past the delay
        return self.lse_solver(spd, gradient)

    def _preconditioner_sherman_morrison(self, x, k, eps, delta1, delta2, plus, minus, gradient):
        moving_avg = k / (k + 1) * self._moving_avg
        moving_avg_inv = (k + 1) / k * self._moving_avg_inv

        for _ in range(self.preconditioner_avg):
            delta = self._compute_precond_delta(x, eps, delta1, delta2, plus, minus)
            # inverse plus some extra factors
            pert1_inv = delta1 / (2 * eps) * 1 / (k + 1) * 1 / 2 / self.preconditioner_avg

            tmp = moving_avg + np.outer(pert1_inv, delta)
            tmp_inv = sherman_morrison(moving_avg_inv, pert1_inv, delta)

            moving_avg = tmp + np.outer(delta, pert1_inv)
            moving_avg_inv = sherman_morrison(tmp_inv, delta, pert1_inv)

        self._moving_avg = moving_avg
        self._moving_avg_inv = moving_avg_inv

        return moving_avg_inv.dot(gradient)

    def _compute_update(self, loss, x, k, eps):
        # compute the perturbations
        delta1, delta2 = [bernoulli_perturbation(x.size, self.perturbation_dims)
                          for _ in range(2)]

        # compute the gradient
        gradient, plus, minus = self._compute_gradient(loss, x, eps, delta1)

        # compute the preconditioner
        if self.preconditioner is not None:
            if isinstance(self.lse_solver, str) and self.lse_solver == 'sherman-morrison':
                precond_gradient = self._preconditioner_sherman_morrison(
                    x, k, eps, delta1, delta2, plus, minus, gradient
                )
            elif self.lse_solver == 'eigenvalues':
                precond_gradient = self._preconditioner_via_solver(
                    x, k, eps, delta1, delta2, plus, minus, gradient
                )
            elif callable(self.lse_solver):
                precond_gradient = self._preconditioner_via_solver(
                    x, k, eps, delta1, delta2, plus, minus, gradient
                )
            else:
                raise NotImplementedError(
                    'Invalid setting for lse_solver:', self.lse_solver)

            if k > self.preconditioner_delay:
                gradient = precond_gradient

        return gradient

    def _minimize(self, loss, initial_point):
        # ensure learning rate and perturbation are set
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_learning_rate, get_perturbation = self.calibrate(
                loss, initial_point)
            self.learning_rate = get_learning_rate()
            self.perturbation = get_perturbation()

        if self.learning_rate is None:
            self.learning_rate = stepseries()

        if self.perturbation is None:
            self.perturbation = powerseries()

        # get iterator
        eta = self.learning_rate()
        eps = self.perturbation()

        # prepare some initials
        x = np.asarray(initial_point)
        self._moving_avg = np.identity(x.size)
        self._moving_avg_inv = np.identity(x.size)
        self.history = {'nfev': 0,  # number of function evaluations
                        # number of function evaluations per iteration
                        'nfevs': [0],
                        'fx': [loss(x)],  # function values
                        'x': [x],  # the parameter values
                        'accepted': [True],  # if the update step was accepted
                        'converged': False,  # whether the algorithm converged
                        'stepsizes': [],
                        'metric': [],
                        }

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss(x)
            self.history['nfev'] += 1

        for k in range(1, self.maxiter + 1):
            # compute update
            update = self._compute_update(loss, x, k, next(eps))

            # trust region
            if self.trust_region == 'hard':
                norm = np.linalg.norm(update)
                if norm > 0:  # stop from dividing by 0
                    update = update / norm
            elif self.trust_region == 'soft':
                norm = np.linalg.norm(update)
                if norm > 1:  # only apply if larger than 1
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update
            self.history['x'].append(x_next)
            self.history['fx'].append(loss(x_next))
            self.history['stepsizes'].append(np.linalg.norm(update))

            # blocking
            if self.blocking:
                fx_next = loss(x_next)
                self.history['nfev'] += 1
                if fx <= fx_next:  # discard update if it didn't improve the loss
                    self.history['accepted'].append(False)
                    self.history['nfevs'].append(
                        self.history['nfev'] - self.history['nfevs'][-1])
                    continue
                fx = fx_next

            self.history['nfevs'].append(
                self.history['nfev'] - self.history['nfevs'][-1])
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


def sherman_morrison(a_inv, u, v):  # pylint: disable=invalid-name
    """Use the Sherman-Morrison formula to compute (A + uv^T)."""
    return a_inv - np.outer(a_inv.dot(u), v.T.dot(a_inv)) / (1 + v.T.dot(a_inv).dot(u))
