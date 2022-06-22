from dataclasses import dataclass
import inspect
import warnings
from typing import Union, Callable, Optional, List, Tuple, Dict
import numpy as np
from qiskit.algorithms.optimizers.optimizer import OptimizerSupportLevel, OptimizerResult, POINT
from qiskit.algorithms.optimizers.steppable_optimizer import (
    AskObject,
    TellObject,
    OptimizerState,
    SteppableOptimizer,
)

# from .optimizer import OptimizerSupportLevel, OptimizerResult, POINT
# from .steppable_optimizer import AskObject, TellObject, OptimizerState, SteppableOptimizer

CALLBACK = Callable[[], None]


@dataclass
class CMAES_AskObject(AskObject):
    """
    Args:
        x_fun_translation:
    """

    x_fun_translation: Optional[Union[POINT, List[POINT]]] = None


@dataclass
class CMAESState(OptimizerState):
    """
    Args:
        p_sigma: Used to store
        p_c:
        cov_matrix:
        eigenvectors:
        std_vector:
        nit:
        sigma:
        best_x:
    """

    # x will be treated as the mean
    p_sigma: POINT
    p_c: POINT
    cov_matrix: np.ndarray  # pylint: disable=invalid-name
    eigenvectors: np.ndarray
    std_vector: POINT  # Will store the sqrt of the diagonal elements
    sigma: float

    # Parameters for the stoping criteria
    _best_func_hist: List[float] = []
    _median_func_hist: List[float] = []


class CMAES(SteppableOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy minimization routine.
    """

    def __init__(
        self,
        maxiter: int = 1000,
        callback: Optional[CALLBACK] = None,
        sigma_0: float = 0.5,
        weights: Optional[np.ndarray] = None,
        termination_criteria: Optional[Dict[str, float]] = None,
    ) -> None:

        super().__init__(maxiter=maxiter, callback=callback)
        self._state: CMAESState = None
        self.N = None
        self.lmbda = None
        self.weights = weights
        self.mu = None
        self.mueff = None
        self.sigma_0 = sigma_0
        self.cc = None
        self.cs = None
        self.c1 = None
        self.cmu = None
        self.damps = None
        self.chiN = None
        self.termination_criteria = termination_criteria

    def ask(self) -> CMAES_AskObject:
        """ """
        z = np.random.normal(0, 1, size=(self.lmbda, self.N))
        y = np.einsum(
            "ij,j,kj->ki", self._state.eigenvectors, self._state.std_vector, z
        )  # ij or ji???
        x = self._state.x + self._state.sigma * y
        return CMAES_AskObject(x_fun=x, x_jac=None, x_fun_translation=z)

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        self._state.nit += 1
        sorting_indexes = np.argsort(tell_object.eval_fun)
        sorted_x = ask_object.x_fun[sorting_indexes][: self.mu]
        sorted_z = ask_object.x_fun_translation[sorting_indexes][: self.mu]

        self._state.x = np.dot(sorted_x.T, self.weights)
        mean_z = np.dot(sorted_z.T, self.weights)

        bdz = np.einsum("ij,j,kj->ik", self._state.eigenvectors, self._state.std_vector, sorted_z)

        self._state.p_sigma *= 1 - self.cs
        self._state.p_sigma += np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(
            self._state.eigenvectors, mean_z
        )  # Check algebra
        hsig = np.linalg.norm(self._state.p_sigma)
        hsig /= np.sqrt(1 - (1 - self.cs) ** (2 * self._state.nit / self.lmbda))
        hsig = hsig < (1.4 + (2 / (self.N + 1)) * self.chiN)
        self._state.p_c *= 1 - self.cc
        self._state.p_c += hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * mean_z

        self._state.cov_matrix = (
            (1 - self.c1 - self.cmu) * self._state.cov_matrix
            + self.c1
            * (
                np.einsum("i,j->ij", self._state.p_c, self._state.p_c)
                + (1 - hsig) * self.cc * (2 - self.cc) * self._state.cov_matrix
            )
            + self.cmu * np.einsum("ij,j,kj->ik", bdz, self.weights, bdz)  # Need to check indexes
        )

        self._state.sigma *= np.exp(
            self.cs / self.damps * (np.linalg.norm(self._state.p_sigma) / self.chiN - 1)
        )

        if (
            True
        ):  # Here the paper only diagonalizes the matrix every certain amount of iterations. We prefer to be as precise as possible even if we need to sacrifice efficiency in the optimizer.
            self._state.cov_matrix = (
                self._state.cov_matrix + self._state.cov_matrix.T
            ) / 2  # We guarantee symmetry
            self._state.std_vector, self._state.eigenvectors = np.linalg.eig(self._state.cov_matrix)
            negative_mask = self._state.std_vector < 0
            if negative_mask.any():
                warnings.warn("Casting negative eigenvalues to zero.")
                self._state.std_vector[negative_mask] = 0

            self._state.std_vector = np.real(np.sqrt(self._state.std_vector))
            self._state.eigenvectors = np.real(self._state.eigenvectors)

        # Now updating the stopping criteria
        complete_list = self._state.nit > 10 + 30 * self.N / self.lmbda
        self._state._best_func_hist = self._state._best_func_hist[complete_list:] + [
            np.max(tell_object.eval_fun)
        ]
        self._state._median_func_hist = self._state._median_func_hist[complete_list:] + [
            np.median(tell_object.eval_fun)
        ]

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
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        This method will initialize the state of the optimizer so that an optimization can be performed.
        It will always setup the initial point and will restart the counter for function evaluations.
        This method is left blank because every optimizer has a different kind of state.
        """
        # initialize state
        self._state = CMAESState(
            x=x0,
            fun=fun,
            jac=jac,
            nfev=0,
            njev=None,
            nit=0,
            cov_matrix=np.eye(x0.size),
            eigenvectors=np.eye(x0.size),
            std_vector=np.ones((x0.size)),
            p_sigma=np.zeros(x0.size),
            p_c=np.zeros(x0.size),
            sigma=self.sigma_0,
        )

        # Initialize static variables
        self.N = self._state.x.size
        self.lmbda = 4 + int(3 * np.log(self.N))

        self.mu = int(self.lmbda / 2)

        self._initialize_weights()
        self.mueff = 1.0 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)
        self.cs = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1.0 / self.mueff) / ((self.N + 2) ** 2 + self.mueff)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cs
        self.chiN = self.N**0.5 * (1 - 1 / (4 * self.N) + 1 / (21 * self.N**2))

    def _initialize_weights(self) -> None:
        """
        Initializes the weights of the CMA-ES optimizer.
        """
        # Default value for weights.
        if self.weights is None:
            self.weights = np.log((self.lmbda + 1) / 2) - np.log(np.arange(1, self.mu + 1))

        # Check if weights are in descending order.
        if np.any(self.weights[:-1] < self.weights[1:]):
            raise ValueError("Weights must be in descending order.")

        # Check that all weights are positive.
        if np.any(self.weights < 0):
            raise ValueError("Weights must be positive.")

        # Normalize weights.
        self.weights = self.weights / np.sum(self.weights)

    def _NoEffectAxis(self) -> bool:
        """Termination criteria if the the covariance of a principal axis gets too small."""
        return (
            0.1 * np.max(self._state.std_vector)
            > np.linalg.norm(self._state.x) * self.termination_criteria["NoEffectAxis"]
        )

    def _NoEffectCoord(self) -> bool:
        """Termination criteria if the covariance over a cartesian axis gets too small."""
        return (
            0.1 * np.max(np.diag(self._state.cov_matrix))
            > np.linalg.norm(self._state.x) * self.termination_criteria["NoEffectCoord"]
        )

    def _ConditionCov(self) -> bool:
        """What is condition number???"""
        pass

    def _EqualFunValues(self) -> bool:
        """Termination criteria if the range of the best evaluation during the last 10+ 30*N/lmbda
        generations is smaller than a given threshold.
        """
        return (
            self._state._best_func_hist.max() - self._state._best_func_hist.min()
            < self.termination_criteria["EqualFunValues"]
        )

    def _Stagnation(self) -> bool:
        """
        WIP
        """
        pass

    def _TolXUp(self) -> bool:
        """
        Termination criteria if the covariance increases too fast.
        This usually indicates that the initial standard deviation is too small.
        """
        pass

    def _TolFun(self) -> bool:
        """
        Termination criteria if the objective function stops decreasing during too many generations.
        We take 10+ 30*dim/population generations as the number of generations to wait before
        we stop the optimization.
        """

        pass

    def _TolX(self) -> bool:
        """
        Termination criteria if the covariance matrix and p_sigma are too small.
        """
        pass

    def continue_condition(self) -> bool:
        """
        This is the condition that will be checked after each step to stop the optimization process.
        """
        cont_cond = True

        for key, value in self.termination_criteria.items():
            cont_cond &= getattr(self, "_" + key)(value)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values) < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(self._sigma * self._pc < self._tolx):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False

        cont_cond = True
        # stop if adding a 0.1-standard deviation vector in any principal axis direction of C does not change m.
        # TolXUp:The initial std is probably too small.
        if self._state.sigma * np.max(self._state.std_vector) > self.tolxup:
            print("TolXUp")
            return False
        # Tolerance.
        cont_cond &= self._state.fun(self._state.x) > self.tol

        return cont_cond and super().continue_condition()

    def _callback_wrapper(self) -> None:
        """
        This is the wrapper for the callback function.
        """
        if self.callback is not None:
            kwargs = {}
            for argument in inspect.getfullargspec(self.callback)[0]:
                if argument not in self._state.__dict__:
                    raise ValueError(
                        f"The callback function requires the argument {argument} to be in the state."
                    )
                kwargs[argument] = getattr(self._state, argument)
            self.callback(**kwargs)

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }
