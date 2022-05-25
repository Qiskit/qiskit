from dataclasses import dataclass
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Callable, Optional, Tuple, List, Iterator
import numpy as np
from numpy.lib.function_base import gradient

from qiskit.algorithms.algorithm_result import AlgorithmResult


from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT
from .steppable_optimizer import AskObject, TellObject, OptimizerState, SteppableOptimizer

CALLBACK = Callable[[int, np.ndarray, float, float], None]

# random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)
# numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None)


@dataclass
class CMAES_AskObject(AskObject):
    """
    Args:
        cloud: List of randomly sampled points close to CMAES_OptmizerState.x .
        variation_cloud: Sampling from a multivariate normal distribution used to create CMAES_AskObject.cloud.
    """

    cloud: List[POINT] # Change to x and put in base class
    variation_cloud: List[POINT] # Change the name


@dataclass
class CMAES_TellObject(TellObject):
    """
    Args:
        cloud_evaluated: List of the value of the function evaluated at each point of the sample.
    """

    cloud_evaluated: List[POINT]


@dataclass
class CMAES_OptimizerState(OptimizerState):
    """
    Args:
        p_sigma: Used to store
        p_c:
        C:
        B:
        D:
        generation:
        sigma:
        best_x:
    """

    # x will be treated as the mean
    p_sigma: POINT = None
    p_c: POINT = None
    C: np.ndarray = None  # pylint: disable=invalid-name
    B: np.ndarray = None
    D: POINT = None  # Will store the sqrt of the diagonal elements
    generation: int = 0
    sigma: float = 0.5
    best_x: Optional[Tuple[POINT, float]] = None

    def __post_init__(self):
        if self.C is None:
            self.C = np.eye(self.x.size)
            self.B = np.eye(self.x.size)
            self.D = np.ones((self.x.size))
            self.p_sigma = np.zeros(self.x.size)
            self.p_c = np.zeros(self.x.size)

    def __str__(self):
        # print(self.generation,self.fun,self.x,self.nfev,self.best_x)
        funeval = self.fun(self.x)
        funeval = funeval if funeval is not None else 0
        bestx = self.best_x[0] if self.best_x is not None else 0
        return f"(Generation:{self.generation} ; sigma:{self.sigma:.4E} ; objective:{funeval:.4E} ; nfev:{self.nfev} ; best_case: {bestx:.4E})"


class SteppableCMAES(SteppableOptimizer):
    def __init__(
        self,
        tol: float = 1e-3,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.N = None
        self.lmbda = None
        self.weights = None
        self.mu = None
        self.mueff = None
        self.tol = tol

    # @property
    # def settings(self) -> Dict[str, Any]:
    #     # if learning rate or perturbation are custom iterators expand them
    #     if callable(self.learning_rate):
    #         iterator = self.learning_rate()
    #         learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
    #     else:
    #         learning_rate = self.learning_rate

    #     return {
    #         "maxiter": self.maxiter,
    #         "tol": self.tol,
    #         "learning_rate": learning_rate,
    #         "perturbation": self.perturbation,
    #         "callback": self.callback,
    #     }

    def ask(self) -> CMAES_AskObject:
        """ """
        z = np.random.normal(0, 1, size=(self.lmbda, self.N))
        y = np.einsum("ij,j,kj->ki", self._state.B, self._state.D, z)  # ij or ji???
        x = self._state.x + self._state.sigma * y
        return CMAES_AskObject(cloud=x, variation_cloud=z)

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        self._state.generation += 1
        sorting_indexes = np.argsort(tell_object.cloud_evaluated)
        # print(tell_object.cloud_evaluated)
        # print(sorting_indexes)
        sorted_x = ask_object.cloud[sorting_indexes][: self.mu]
        self._state.best_x = (tell_object.cloud_evaluated[sorting_indexes[0]], sorted_x[0])
        sorted_z = ask_object.variation_cloud[sorting_indexes][: self.mu]

        self._state.x = np.dot(sorted_x.T, self.weights)
        mean_z = np.dot(sorted_z.T, self.weights)

        bdz = np.einsum("ij,j,kj->ik", self._state.B, self._state.D, sorted_z)

        self._state.p_sigma *= 1 - self.cs
        self._state.p_sigma += np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(
            self._state.B, mean_z
        )  # Check algebra
        hsig = np.linalg.norm(self._state.p_sigma)
        hsig /= np.sqrt(1 - (1 - self.cs) ** (2 * self._state.generation / self.lmbda))
        hsig = hsig < 1.4 + 2 / (self.N + 1) * self.chiN
        self._state.p_c *= 1 - self.cc
        self._state.p_c += hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * mean_z

        self._state.C = (
            (1 - self.c1 - self.cmu) * self._state.C
            + self.c1
            * (
                np.einsum("i,j->ij", self._state.p_c, self._state.p_c)
                + (1 - hsig) * self.cc * (2 - self.cc) * self._state.C
            )
            + self.cmu * np.einsum("ij,j,kj->ik", bdz, self.weights, bdz)  # Need to check indexes
        )

        self._state.sigma *= np.exp(
            self.cs / self.damps * (np.linalg.norm(self._state.p_sigma) / self.chiN - 1)
        )

        if (
            True
        ):  # Here the paper only diagonalizes the matrix every certain amount of iterations. We prefer to be as precise as possible even if we need to sacrifice efficiency in the optimizer.
            self._state.C = (self._state.C + self._state.C.T) / 2  # We guarantee symmetry
            self._state.D, self._state.B = np.linalg.eig(self._state.C)
            self._state.D = np.real(np.sqrt(self._state.D))
            self._state.B = np.real(self._state.B)

    def evaluate(self, ask_object: AskObject) -> TellObject:
        cloud_eval = [self._state.fun(x) for x in ask_object.cloud]
        self._state.nfev += len(ask_object.cloud)
        return CMAES_TellObject(cloud_evaluated=cloud_eval)

    def create_result(self) -> OptimizerResult:
        """
        Creates a result of the optimization process using the values from self.state.
        """
        result = OptimizerResult()
        result.x = self._state.best_x[1]
        result.fun = self._state.best_x[0]
        result.nfev = self._state.nfev
        return result

    def initialize(
        self,
        x0: POINT,
        fun: Callable[[POINT], float],
        jac: Callable[[POINT], POINT] = None,
        tol: float = 1e-3,
    ) -> None:
        """
        This method will initialize the state of the optimizer so that an optimization can be performed.
        It will always setup the initial point and will restart the counter for function evaluations.
        This method is left blank because every optimizer has a different kind of state.
        """
        # initialize state
        self._state = CMAES_OptimizerState(x=x0, fun=fun, generation=0)
        # Initialize static variables
        self.N = self._state.x.size
        self.lmbda = 4 + int(3 * np.log(self.N))
        self.mu = int(self.lmbda / 2)
        self.weights = np.log((self.lmbda + 1) / 2) - np.log(np.arange(1, self.mu + 1))

        self.weights = self.weights / self.weights.sum()
        print(self.weights)
        self.mueff = 1.0 / (self.weights**2).sum()

        self.cc = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)
        self.cs = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1.0 / self.mueff) / ((self.N + 2) ** 2 + self.mueff)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cs
        self.chiN = self.N**0.5 * (1 - 1 / (4 * self.N) + 1 / (21 * self.N**2))

        self.tol = tol

    def stop_condition(self) -> bool:
        """
        This is the condition that will be checked after each step to stop the optimization process.
        """
        return self._state.best_x[0] < self.tol

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
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
        super().optimize(
            num_vars, objective_function, gradient_function, variable_bounds, initial_point
        )
        result = self.minimize(objective_function, initial_point, gradient_function)
        return result.x, result.fun, result.nfev
