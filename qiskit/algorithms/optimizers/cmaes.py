from dataclasses import dataclass
from typing import Union, Callable, Optional, List
import numpy as np
from .optimizer import OptimizerSupportLevel, OptimizerResult, POINT
from .steppable_optimizer import AskObject, TellObject, OptimizerState, SteppableOptimizer

CALLBACK = Callable[[int, np.ndarray, float, float], None]


@dataclass
class CMAES_AskObject(AskObject):
    """
    Args:
        x_fun_translation: Sampling from a multivariate normal distribution used to create CMAES_AskObject.cloud.
    """

    x_fun_translation: Union[POINT, List[POINT]]


@dataclass
class CMAES_OptimizerState(OptimizerState):
    """
    Args:
        p_sigma: Used to store
        p_c:
        C:
        B:
        D:
        nit:
        sigma:
        best_x:
    """

    # x will be treated as the mean
    p_sigma: POINT
    p_c: POINT
    C: np.ndarray  # pylint: disable=invalid-name
    B: np.ndarray
    D: POINT  # Will store the sqrt of the diagonal elements
    sigma: float


class CMAES(SteppableOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy minimization routine.
    """

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
        self.cc = None
        self.cs = None
        self.c1 = None
        self.cmu = None
        self.damps = None
        self.chiN = None

    def ask(self) -> CMAES_AskObject:
        """ """
        z = np.random.normal(0, 1, size=(self.lmbda, self.N))
        y = np.einsum("ij,j,kj->ki", self._state.B, self._state.D, z)  # ij or ji???
        x = self._state.x + self._state.sigma * y
        return CMAES_AskObject(x_fun=x, x_jac=None, x_fun_translation=z)

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        self._state.nit += 1
        sorting_indexes = np.argsort(tell_object.eval_fun)
        sorted_x = ask_object.x_fun[sorting_indexes][: self.mu]
        sorted_z = ask_object.x_fun_translation[sorting_indexes][: self.mu]

        self._state.x = np.dot(sorted_x.T, self.weights)
        mean_z = np.dot(sorted_z.T, self.weights)

        bdz = np.einsum("ij,j,kj->ik", self._state.B, self._state.D, sorted_z)

        self._state.p_sigma *= 1 - self.cs
        self._state.p_sigma += np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(
            self._state.B, mean_z
        )  # Check algebra
        hsig = np.linalg.norm(self._state.p_sigma)
        hsig /= np.sqrt(1 - (1 - self.cs) ** (2 * self._state.nit / self.lmbda))
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
        eval_fun = [self._state.fun(x) for x in ask_object.x_fun]
        self._state.nfev += len(ask_object.x_fun)
        return TellObject(eval_fun=eval_fun, eval_jac=None)

    def user_evaluate(self, eval_fun: List[float]) -> TellObject:
        return TellObject(eval_fun=eval_fun, eval_jac=None)

    def create_result(self) -> OptimizerResult:
        """
        Creates a result of the optimization process using the values from self.state.
        """
        result = OptimizerResult()
        result.x = self._state.x
        result.fun = self._state.fun(self._state.x)
        result.nfev = self._state.nfev
        result.nit = self._state.nit
        return result

    def initialize(
        self,
        x0: POINT,
        fun: Callable[[POINT], float],
        jac: Callable[[POINT], POINT] = None,
        tol: float = 1e-3,
        population_size: Optional[int] = None,
    ) -> None:
        """
        This method will initialize the state of the optimizer so that an optimization can be performed.
        It will always setup the initial point and will restart the counter for function evaluations.
        This method is left blank because every optimizer has a different kind of state.
        """
        # initialize state
        self._state = CMAES_OptimizerState(
            x=x0,
            fun=fun,
            jac=jac,
            nfev=0,
            njev=None,
            nit=0,
            C=np.eye(x0.size),
            B=np.eye(x0.size),
            D=np.ones((x0.size)),
            p_sigma=np.zeros(x0.size),
            p_c=np.zeros(x0.size),
            sigma=0.5,
        )

        # Initialize static variables
        self.N = self._state.x.size
        self.lmbda = 4 + int(3 * np.log(self.N)) if population_size is None else population_size
        self.mu = int(self.lmbda / 2)
        self.weights = np.log((self.lmbda + 1) / 2) - np.log(np.arange(1, self.mu + 1))

        self.weights = self.weights / self.weights.sum()
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
        return self._state.fun(self._state.x) < self.tol

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
