from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Callable, Optional, Tuple, List

from .optimizer import Optimizer, POINT, OptimizerResult
import numpy as np

CALLBACK = Callable[[int, np.ndarray, float, float], None]


@dataclass
class AskObject(ABC):
    """
    Base class for return type of method :meth:`~qiskit.algorithms.optimizers.SteppableOptimizer.ask`.
    Args:
        x_fun: Point or list of points where the function needs to be evaluated to compute the
            next state of the optimizer.
        x_jac: Point or list of points where the gradient/jacobian needs to be evaluated to compute the
            next state of the optimizer.
    """

    x_fun: Union[POINT, List[POINT]]
    x_jac: Union[POINT, List[POINT]]


@dataclass
class TellObject(ABC):
    """
    Base class for argument type of method SteppableOptimizer.tell().
    Args:
        eval_fun: Image of the function at
            :attr:`~qiskit.algorithms.optimizers.SteppableOptimizer.Ask_Object.x_fun`.
        eval_jac: Image of the gradient-jacobian at
            :attr:`~qiskit.algorithms.optimizers.SteppableOptimizer.Ask_Object.x_fun`.
    """

    eval_fun: Union[float, List[float]]
    eval_jac: Union[POINT, List[POINT]]


@dataclass
class OptimizerState:
    """
    Base class representing the state of the optmiizer.
    Any variable that changes during the optimization process and is needed for the next step
    of optimization should be stored in this dataclass.
    """

    x: POINT  # pylint: disable=invalid-name
    fun: Optional[Callable[[POINT], float]]  # Make optional
    jac: Optional[Callable[[POINT], POINT]]
    nfev: Optional[int]
    njev: Optional[int]
    nit: Optional[int]


class SteppableOptimizer(Optimizer):
    """
    Base class for a steppable optimizer.

    This family of optimizers will be using the 
    `ask and tell interface <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html>`_.
    When using this interface the user has to call the function ask() in order to get information about how to evaluate the fucntion
    (we are asking the optimizer about how to do the evaluation). This information will be mostly about at what point should we evaluate
    the function next, but depending on the optimizer it can also be about whether we should evaluate the function itself or its gradient.

    Once the function has been evaluated, the user calls the method tell() to tell the optimizer what has been the result of the function evaluation.
    The optimizer then updates its state accordingly and the user can decide whether to stop the optimization process or to repeat a step.

    This interface is more customizable, and allows the user to have full control over the evaluation of the function.

    For example:
    .. code-block::python
        import random
        import numpy as np
        from qiskit.algorithms.optimizers import SteppableGradientDescent

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

        optimizer = SteppableGradientDescent(maxiter=20)
        optimizer.initialize(x0=initial_point, fun=objective, jac=grad)

        for _ in range(20):
            ask_object = optimizer.ask()
            evaluated_gradient = None

            while evaluated_gradient is None:
                evaluated_gradient = grad(ask_object.x_center)
                optimizer._state.njev += 1

            tell_object = GD_TellObject(gradient=evaluated_gradient)
            optimizer.tell(ask_object=ask_object, tell_object=tell_object)

        result = optimizer.create_result()

    In this case the evaluation of the function has a chance of failing. The user, with specific knowledge about his function can catch this errors and handle before
    passing the result to the optimizer.

    In case the user isn't dealing with complicated function and is more familiar with step by step optimization algorithms, a method step() has been created that
    acts as a wrapper for ask() and tell().
    In the same spirit the method minimize() will optimize the function and return the result without the user having to worry about the optimization process.

    To see other libraries that use this interface one can visit: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html


    """

    def __init__(
        self,
        maxiter: int = None,
        callback: Optional[CALLBACK] = None,
        

    ):
        """
        Args:
            maxiter: Number of steps in the optimization process before ending the loop.
            callback: Function to be called after each iteration.
        """
        super().__init__()
        self._state : OptimizerState = None
        self.callback = callback
        self.maxiter = maxiter  # Remove maxiter

    def ask(self) -> AskObject:
        """Ask the optimizer for a set of points to evaluate.
        
        This method asks the optimizer which are the next points to evaluate. 
        These points can, e.g., correspond to function values and/or its derivative. 
        It may also correspond to variables that let the user infer which points to evaluate.
        It is the first method inside of a "step" in the optimization process.
        
        Returns:
            Since the way to evaluate the function can vary much with different
            optimization algorithms, the object will be a custom dataclass for each optimizer.
        """
        raise NotImplementedError

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        """
        Updates the optimization state once the objective function has been evaluated.
        Canonical optimization workflow using ask() and tell() can be seen in SteppableOptimizer.step().
        Args:
            ask_object: Contains the information on how the evaluation was done.
            tell_object: Contains all relevant information about the evaluation of the objective function.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, ask_object: AskObject) -> TellObject:
        """
        Evaluates the function according to the instructions contained in ask_object.
        If the user decides to use step() instead of ask() and tell() this function will contain the logic on how to evaluate
        the function.
        Args:
            ask_object: Contains the information on how to do the evaluation.
        Return:
            Contains all relevant information about the evaluation of the objective function.
        """
        raise NotImplementedError

    def user_evaluate(
        self, *args, **kwargs
    ) -> TellObject:  # Maybe this function should be abstract. For some optimizers (The return type with a custom tell object the function needs to be overwriten)
        """
        Constructs TellObject.
        Used when the user manually evaluates the function.
        """
        return TellObject(*args, **kwargs)

    def step(self) -> None:
        """
        Performs one step in the optimization process.
        This method composes ask(), evaluate(), and tell() to make a step in the optimization process.
        """
        ask_object = self.ask()
        tell_object = self.evaluate(ask_object=ask_object)
        self.tell(ask_object=ask_object, tell_object=tell_object)
        if self.callback is not None:
            self.callback(state=self._state)
    @abstractmethod
    def initialize(
        self,
        x0: POINT,
        fun: Callable[[POINT], float],
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        This method sets (or restarts) the optimization state and the parameters to perform a new optimization.
        Needs to be called before starting the optimization loop and also will need.
        Args:
            fun: Function to minimize.
            x0: Initial point.
            jac: Function to compute the gradient.
            bounds: Bounds of the search space.
            **kwargs: Additional arguments for the minimization algorithm.
        """
        raise NotImplementedError

    def minimize(
        self,
        x0: POINT,
        fun: Callable[[POINT], float],
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        """
        For well behaved functions the user can call this method to minimize a function. If the user wants more control
        on how to evaluate the function a custom loop can be created using ask() and tell() and evaluating the function
        manually.
        Args:
            fun: Function to minimize.
            x0: Initial point.
            jac: Function to compute the gradient.
            bounds: Bounds of the search space.
            **kwargs: Additional arguments for the minimization algorithm.
        """
        self.initialize(x0=x0, fun=fun, jac=jac,bounds=bounds)
        while self.continue_condition():
            self.step()
        return self.create_result()

    @abstractmethod
    def create_result(self) -> OptimizerResult:
        """
        Returns the result of the optimization.
        All the information needed to create the result should be stored in the optimizer state.
        Returns:
            The result of the optimization process.
        """
        raise NotImplementedError


    @abstractmethod
    def continue_condition(self) -> bool:
        """
        Condition that indicates the optimization process should come to an end.
        Returns:
            True if the optimization process should continue, False otherwise.
        """
        return self._state.nit < self.maxiter
