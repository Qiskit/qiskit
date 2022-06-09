from dataclasses import dataclass
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Any, Union, Callable, Optional, Tuple, List, Iterator
import numpy as np
from numpy.lib.function_base import gradient

from qiskit.algorithms.algorithm_result import AlgorithmResult


from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT
from .steppable_optimizer import AskObject, TellObject, OptimizerState, SteppableOptimizer, CALLBACK


def constant(eta=0.01):
    """Yield a constant."""

    while True:
        yield eta


@dataclass
class GD_OptimizerState(OptimizerState):

    eta: Union[float, Callable[[], Iterator]]
    stepsize: float


class GradientDescent(SteppableOptimizer):
    def __init__(
        self,
        maxiter: int = 1000,
        callback: Optional[CALLBACK] = None,
        learning_rate: Union[float, Callable[[], Iterator]] = 0.01,
        tol: float = 1e-7,
        perturbation: Optional[float] = None,
        **kwargs,
    ) -> None:

        super().__init__(maxiter=maxiter, callback=callback)
        self._state: GD_OptimizerState = None
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.tol = tol

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
        This method is the part of the interface of the optimizer that asks the
        user/quantum_circuit how and where the function to optimize needs to be evaluated. It is the
        first method inside of a "step" in the optimization process.
        For gradient descent this method simply returns an AskObject containing the current point for
        the gradient to be evaluated, a parameter epsilon in case the gradient has to be approximated
        and a bool flag telling the prefered way of evaluating the gradient.
        """
        return AskObject(
            x_jac=self._state.x,
        )

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        """
        This method is the part of the interface of the optimizer that tells the
        user/quantum_circuit what is the next point that minimizes the function (with respect to last
        step). In this case it is not going to return anything since instead it is just going to update
        state of the optimizer.It is the last method called inside of a "step" in the optimization process.
        For gradient descent this method updates self._state.x by an ammount proportional to the learning rate
        and the gradient at that point.
        """
        update = tell_object.eval_jac
        self._state.x -= next(self._state.eta) * update
        self._state.stepsize = np.linalg.norm(update)
        self._state.nit += 1

    def evaluate(self, ask_object: AskObject) -> TellObject:
        """
        This is the default way of evaluating the function given the request by self.ask().
        For gradient descent we are going to check how to evaluate the gradient, evaluate and
        return a TellObject.
        """
        if self._state.jac is None:
            grad = Optimizer.gradient_num_diff(
                x_center=ask_object.x_fun,
                f=self._state.fun,
                epsilon=self.perturbation,
                max_evals_grouped=1,  # Here there was some extra logic I am just neglecting for now.
            )
            self._state.nfev += 1 + len(ask_object.x_jac)
        else:
            grad = self._state.jac(ask_object.x_jac)
            self._state.njev += 1

        return TellObject(eval_jac=grad)

    def create_result(self) -> OptimizerResult:
        """
        Creates a result of the optimization process using the values from self.state.
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

        self._state = GD_OptimizerState(
            fun=fun,
            jac=jac,
            x=np.asarray(x0),
            nit=0,
            nfev=0,
            njev=0,
            eta=eta,
            stepsize=np.inf,
        )

    def continue_condition(self) -> bool:
        """
        Condition that indicates the optimization process should come to an end.
        When the stepsize is smaller than the tolerance, the optimization process is considered
        finished.
        Returns:
            True if the optimization process should continue, False otherwise.
        """

        cont_condition = self._state.stepsize > self.tol
        cont_condition &= super().continue_condition()
        return cont_condition

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }
