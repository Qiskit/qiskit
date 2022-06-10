from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Callable, Optional, Tuple, List

from .optimizer import Optimizer, POINT, OptimizerResult


@dataclass
class AskObject(ABC):
    """Base class for return type of method SteppableOptimizer.ask()"""


@dataclass
class TellObject(ABC):
    """Base class for argument type of method SteppableOptimizer.tell()"""


@dataclass
class OptimizerState:
    """
    Base class representing the state of the optmiizer. On top of that it
    will also store the function and the jacobians to be evaluated.
    """

    x: POINT  # pylint: disable=invalid-name
    fun: Callable[[POINT], float]
    jac: Callable[[POINT], POINT] = None
    nfev: int = 0
    njev: int = 0
    nit: int = 0


class SteppableOptimizer(Optimizer):
    """Base class for a steppable optimizer."""

    def __init__(self, maxiter: int = 100):
        super().__init__()
        self._state = None
        self.maxiter = maxiter

    def ask(self) -> AskObject:
        """
        Abstract method ask. This method is the part of the interface of the optimizer that asks the
        user/quantum_circuit how and where the function to optimize needs to be evaluated. It is the
        first method inside of a "step" in the optimization process.
        """
        raise NotImplementedError

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        """
        Abstract method tell. This method is the part of the interface of the optimizer that tells the
        user/quantum_circuit what is the next point that minimizes the function (with respect to last
        step). In this case it is not going to return anything since instead it is just going to update
        state of the optimizer.It is the last method called inside of a "step" in the optimization process.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, ask_object: AskObject) -> TellObject:
        """
        This is the default way of evaluating the function given the request by self.ask()
        """
        raise NotImplementedError

    def step(self) -> None:
        """
        This mehtod composes ask, evaluate, and tell to make a step in the optimization process.
        This method uses the default logic from evaluate.
        """
        ask_object = self.ask()
        tell_object = self.evaluate(ask_object=ask_object)
        self.tell(ask_object=ask_object, tell_object=tell_object)

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> OptimizerResult:
        self.initialize(x0=x0, fun=fun, jac=jac)
        for _ in range(self.maxiter):
            self.step()
            if self.stop_condition():
                break

        return self.create_result()

    @abstractmethod
    def create_result(self) -> OptimizerResult:
        """
        Creates a result of the optimization process using the values from self.state.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        """
        This method will initialize the state of the optimizer so that an optimization can be performed.
        It will always setup the initial point and will restart the counter for function evaluations.
        This method is left blank because every optimizer has a different kind of state.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_condition(self) -> bool:
        """
        This is the condition that will be checked after each step to stop the optimization process.
        """
        raise NotImplementedError
