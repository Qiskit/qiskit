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
from typing import Dict, Any, Union, Callable, Optional, Tuple, List, Iterator
import numpy as np
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT
from .steppable_optimizer import AskObject, TellObject, OptimizerState, SteppableOptimizer

CALLBACK = Callable[[int, np.ndarray, float, float], None]


def constant(eta=0.01):
    """Yield a constant."""

    while True:
        yield eta


@dataclass
class GradientDescentState(OptimizerState):
    """State of :class:`qiskit.algorithms.optimizers.GradientDescent`."""

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
        maxiter: int = 1000,
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
        super().__init__(maxiter=maxiter, callback=callback)
        self._state: GradientDescentState = None
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.tol = tol

    def _callback_wrapper(self, ask_object: AskObject, tell_object: TellObject) -> None:
        """
        Callback method for GradientDescent.
        Will call :attr:`~.callback` and pass the following arguments:
        current number of function values, current parameters, current function value,
        norm of current gradient.
        """
        if self.callback is not None:
            self.callback(
                self._state.nfev,
                self._state.x,
                self._state.fun(self._state.x),  # This could also come from the tell_object.
                self._state.stepsize,
            )

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

    def ask(self) -> AskObject:
        """
        For gradient descent this method simply returns an
        :class:`qiskit.algorithms.optimizers.AskObject` containing the current point for the
        gradient to be evaluated, a parameter epsilon in case the gradient has to be approximated
        and a bool flag telling the prefered way of evaluating the gradient.
        """
        return AskObject(
            x_jac=self._state.x,
        )

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        """
        For gradient descent this method updates self._state.x by an ammount proportional
        to the learning rate and the gradient at that point.
        """
        update = tell_object.eval_jac
        self._state.x = self._state.x - next(self._state.eta) * update
        self._state.stepsize = np.linalg.norm(update)
        self._state.nit += 1

    def evaluate(self, ask_object: AskObject) -> TellObject:
        """
        For gradient descent we are going to check how to evaluate the gradient, either by evaluating
        an analitic gradient or by approximating it with a finite difference scheme.
        The value of the gradient is returned as a :class:`qiskit.algorithms.optimizers.TellObject`.
        """
        if self._state.jac is None:
            eps = 0.01 if self.perturbation is None else self.perturbation
            grad = Optimizer.gradient_num_diff(
                x_center=ask_object.x_jac,
                f=self._state.fun,
                epsilon=eps,
                max_evals_grouped=self._max_evals_grouped,
            )
            self._state.nfev += 1 + len(ask_object.x_jac)
        else:
            grad = self._state.jac(ask_object.x_jac)
            self._state.njev += 1

        return TellObject(eval_jac=grad)

    def create_result(self) -> OptimizerResult:
        """
        Creates a result of the optimization process using the values from :attr:`~.state`.
        """
        result = OptimizerResult()
        result.x = self._state.x
        result.fun = self._state.fun(self._state.x)
        result.nfev = self._state.nfev
        result.njev = self._state.njev
        return result

    def initialize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        This method will initialize the state of the optimizer so that an optimization can be performed.
        It will always setup the initial point and will restart the counter for function evaluations.
        This method is left blank because every optimizer has a different kind of state.
        """
        if isinstance(self.learning_rate, float):
            eta = constant(self.learning_rate)
        else:
            eta = self.learning_rate()

        self._state = GradientDescentState(
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
            True if the optimization process should continue, False otherwise.
        """
        if self._state.stepsize is None:
            return True
        else:
            return (self._state.stepsize > self.tol) and super().continue_condition()

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }
