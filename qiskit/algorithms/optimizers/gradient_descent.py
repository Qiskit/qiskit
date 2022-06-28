# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A standard gradient descent optimizer."""

from dataclasses import dataclass
from typing import Dict, Any, Union, Callable, Optional, Tuple, List, Iterator, Generator
import numpy as np
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT
from .steppable_optimizer import AskData, TellData, OptimizerState, SteppableOptimizer

CALLBACK = Callable[[int, np.ndarray, float, float], None]


def constant_generator(eta: float = 0.01) -> Generator[float, None, None]:
    """Returns a python generator that always yields the same value.

    Args:
        eta: The value to yield.

    Yields:
        The learning rate for the next iteration.
    """

    while True:
        yield eta


@dataclass
class GradientDescentState(OptimizerState):
    """State of :class:`~.GradientDescent`."""

    eta: Iterator
    stepsize: Optional[float]


class GradientDescent(SteppableOptimizer):
    r"""The gradient descent minimization routine.
    For a function :math:`f` and an initial point :math:`\vec\theta_0`, the standard (or "vanilla")
    gradient descent method is an iterative scheme to find the minimum :math:`\vec\theta^*` of
    :math:`f` by updating the parameters in the direction of the negative gradient of :math:`f`

    .. math::

        \vec\theta_{n+1} = \vec\theta_{n} - \vec\eta\nabla f(\vec\theta_{n}),

    for a small learning rate :math:`\eta > 0`.

    You can either provide the analytic gradient :math:`\vec\nabla f` as ``gradient_function``
    in the :meth:`~.minimize` method, or, if you do not provide it, use a finite difference approximation
    of the gradient. To adapt the size of the perturbation in the finite difference gradients,
    set the ``perturbation`` property in the initializer.

    This optimizer supports a callback function. If provided in the initializer, the optimizer
    will call the callback in each iteration with the following information in this order:
    current number of function values, current parameters, current function value, norm of current
    gradient.

    Examples:

        A minimum example that will use finite difference gradients with a default perturbation
        of 0.01 and a default learning rate of 0.01.

        .. code-block:: python

            from qiskit.algorithms.optimizers import GradientDescent

            def f(x):
                return (np.linalg.norm(x) - 1) ** 2

            initial_point = np.array([1, 0.5, -0.2])

            optimizer = GradientDescent(maxiter=100)

            result = optimizer.minimize(fun=fun, x0=initial_point)

            print(f"Found minimum {result.x} at a value"
                "of {result.fun} using {result.nfev} evaluations.")

        An example where the learning rate is an iterator and we supply the analytic gradient.
        Note how much faster this convergences (i.e. less ``nfev``) compared to the previous
        example.

        .. code-block:: python

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
            result = optimizer.minimize(fun=fun, jac=grad_f, x0=initial_point)

            print(f"Found minimum {result.x} at a value"
            "of {result.fun} using {result.nfev} evaluations.")


    An other example where the evaluation of the function has a chance of failing. The user, with
    specific knowledge about his function can catch this errors and handle them before passing the
    result to the optimizer.

        .. code-block:: python

            import random
            import numpy as np
            from qiskit.algorithms.optimizers import GradientDescent

            def objective(x):
                if random.choice([True, False]):
                    return None
                else:
                    return (np.linalg.norm(x) - 1) ** 2

            def grad(x):
                if random.choice([True, False]):
                    return None
                else:
                    return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)


            initial_point = np.random.normal(0, 1, size=(100,))

            optimizer = GradientDescent(maxiter=20)
            optimizer.start(x0=initial_point, fun=objective, jac=grad)

            for _ in range(20):
                ask_data = optimizer.ask()
                evaluated_gradient = None

                while evaluated_gradient is None:
                    evaluated_gradient = grad(ask_data.x_center)
                    optimizer.state.njev += 1

                optmizer.state.nit += 1

                tell_data = TellData(eval_jac=evaluated_gradient)
                optimizer.tell(ask_data=ask_data, tell_data=tell_data)

            result = optimizer.create_result()

    In case the user isn't dealing with complicated functI am not 100% sure about it, so if on and is more familiar with step by step
    optimization algorithms, :meth:`~.step` has been created to acts as a wrapper for :meth:`~.ask`
    and :meth:`~.tell`.
    In the same spirit the method :meth:`~.minimize` will optimize the function and return the result
    directly.

    To see other libraries that use this interface one can visit:
    https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html


    """

    def __init__(
        self,
        maxiter: int = 100,
        learning_rate: Union[float, Callable[[], Iterator]] = 0.01,
        tol: float = 1e-7,
        callback: Optional[CALLBACK] = None,
        perturbation: Optional[float] = None,
    ) -> None:
        """
        Args:
            maxiter: The maximum number of iterations.
            learning_rate: A constant or generator yielding learning rates for the parameter
                updates. See the docstring for an example.
            tol: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
            perturbation: If no gradient is passed to :meth:`~.minimize` the gradient is
                approximated with a symmetric finite difference scheme with ``perturbation``
                perturbation in both directions (defaults to 1e-2 if required).
                Ignored when we have an explicit function for the gradient.
        """
        super().__init__(maxiter=maxiter)
        self.callback = callback
        self._state: Optional[GradientDescentState] = None
        self._learning_rate = learning_rate
        self._perturbation = perturbation
        self._tol = tol

    @property
    def learning_rate(self) -> Union[float, Callable[[], Iterator]]:
        """Returns the learning rate. It can be either a constant value or a function that returns
        generators."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: Union[float, Callable[[], Iterator]]) -> None:
        """Set the learning rate.
        The learning rate provided needs to be a function that returns a generator.
        If a constant is passed, the optimizer will use it for all iterations.
        """
        self._learning_rate = learning_rate

    @property
    def tol(self) -> float:
        """Returns the tolerance of the optimizer.
        Any step with smaller stepsize than this value will stop the optimization."""
        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        """Set the tolerance."""
        self._tol = tol

    @property
    def perturbation(self) -> Optional[float]:
        """Returns the perturbation.
        This is the perturbation used in the finite difference gradient approximation.
        """
        return self._perturbation

    @perturbation.setter
    def perturbation(self, perturbation: Optional[float]) -> None:
        """Set the perturbation."""
        self._perturbation = perturbation

    def _callback_wrapper(self) -> None:
        """
        Wraps the callback function to accomodate GradientDescent.
        Will call :attr:`~.callback` and pass the following arguments:
        current number of function values, current parameters, current function value,
        norm of current gradient.
        """
        if self.callback is not None:
            self.callback(
                self.state.nfev,
                self.state.x,
                self.state.fun(self.state.x),
                self.state.stepsize,
            )

    @property
    def settings(self) -> Dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self._learning_rate()
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

    def ask(self) -> AskData:
        """
        Returns an object with the data needed in order to evaluate the gradient.
        If this object contains a gradient function the gradient can be evaluated directly. Otherwise
        approximate it with a finite difference scheme.
        """
        return AskData(
            x_jac=self.state.x,
        )

    def tell(self, ask_data: AskData, tell_data: TellData) -> None:
        """
        Updates :attr:`.~GradientDescentState.x` by an ammount proportional to the learning
        rate and value of the gradient at that point.

        Args:
            ask_data: The data used to evaluate the function.
            tell_data: The data from the function evaluation.
        """
        self.state.x = self.state.x - next(self.state.eta) * tell_data.eval_jac
        self.state.stepsize = np.linalg.norm(tell_data.eval_jac)
        self.state.nit += 1

    def evaluate(self, ask_data: AskData) -> TellData:
        """Evaluates the gradient.
        It does so either by evaluating an analitic gradient or by approximating it with a
        finite difference scheme. It will either add ``1`` to the number of gradient evaluations or add
        ``N+1`` to the number of function evaluations (Where N is the dimension of the gradient).

        Args:
            ask_data: It contains the point where the gradient is to be evaluated and the gradient
                      function or in it's defect the objective function to perform a finite difference
                      approximation.

        Returns:
        The data containing the gradient evaluation.

        """
        if self.state.jac is None:
            eps = 0.01 if self.perturbation is None else self.perturbation
            grad = Optimizer.gradient_num_diff(
                x_center=ask_data.x_jac,
                f=self.state.fun,
                epsilon=eps,
                max_evals_grouped=self._max_evals_grouped,
            )
            self.state.nfev += 1 + len(ask_data.x_jac)
        else:
            grad = self.state.jac(ask_data.x_jac)
            self.state.njev += 1

        return TellData(eval_jac=grad)

    def create_result(self) -> OptimizerResult:
        """Creates a result of the optimization process.
        This result contains the best point, the best function value, the number of function/gradient
        evaluations and the number of iterations.

        Returns:
            The result of the optimization process.
        """
        result = OptimizerResult()
        result.x = self.state.x
        result.fun = self.state.fun(self.state.x)
        result.nfev = self.state.nfev
        result.njev = self.state.njev
        result.nit = self.state.nit
        return result

    def start(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:

        if isinstance(self.learning_rate, float):
            eta = constant_generator(self.learning_rate)
        else:
            eta = self._learning_rate()

        self.state = GradientDescentState(
            fun=fun,
            jac=jac,
            x=np.asarray(x0),
            nit=0,
            nfev=0,
            njev=0,
            eta=eta,
            stepsize=None,
        )

    def continue_condition(self) -> bool:
        """
        Condition that indicates the optimization process should come to an end.
        When the stepsize is smaller than the tolerance, the optimization process is considered
        finished.

        Returns:
            ``True`` if the optimization process should continue, ``False`` otherwise.
        """
        if self.state.stepsize is None:
            return True
        else:
            return (self.state.stepsize > self.tol) and super().continue_condition()

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }
