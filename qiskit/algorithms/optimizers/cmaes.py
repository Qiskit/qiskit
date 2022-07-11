from dataclasses import dataclass
import inspect
from tracemalloc import stop
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

CALLBACK = Callable[[], None]


@dataclass
class CMAES_AskObject(AskObject):
    """
    Args:
        x_fun_translation: The translation between the points to be evaluated and the current center
        of the distribution.
    """

    x_fun_translation: Optional[Union[POINT, List[POINT]]] = None


@dataclass
class CMAESState(OptimizerState):
    """
    Args:
        p_sigma: Evolution path for the standard deviation of the distribution.
        p_c: Evolution path for the covariance matrix.
        cov_matrix: Covariance matrix.
        eigenvectors:   Eigenvectors of the covariance matrix.
        std_vector: Standard deviation in each principal axis of the covariance matrix.
        They can also be tought as the square root of the eigenvalues of the covariance matrix.
        sigma: The standard deviation of the distribution.
    """

    # x will be treated as the mean
    p_sigma: POINT
    p_c: POINT
    cov_matrix: np.ndarray  # pylint: disable=invalid-name
    eigenvectors: np.ndarray
    std_vector: POINT  # Will store the sqrt of the diagonal elements
    sigma: float

    # Parameters for the stoping criteria
    _best_func_hist: List[float]
    _median_func_hist: List[float]
    _worst_func_eval: Optional[float]


class CMAES(SteppableOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy minimization routine.
    This Optimizer is based on the paper: `arXiv:1604.00772 <https://arxiv.org/abs/1604.00772>`_.
    CMA-ES is an evolution strategy based on sampling points from a multivariate normal distribution
    with a given centroid and covariance matrix.
    From the :math:`\lambda\ points sampled in generation ``g``, a new centroid is calculated as:
    .. math::
        x^{g+1} = \sum_{i=1}^{\mu} w_i x_{i:\lambda}^{g}
        where :math:`\sum_{i=1}^{\mu} w_i = 1` and :math:`x_{i:\lambda}^{g}` are the best :math:`\mu`
        points from generation ``g`` ordered in function of their objective function image.

    A new covariance matrix is also computed taking into account not only the current population,
    but also the previous covariance matrix.

    The optimization is stopped either at the begining of the optimization process if the initial
    parameters were not well chosen, or once the objective function stops decreasing for a given
    number of generations. There are several criteria that can be manually set by the user, and
    are well described in the paper.

    One evaluation with a funciton evaluation outlier with a significantly low value is not
    enough to stop the optimization process or significantly disturb the position of the centroid,
    and an outlier with a significantly high value would simply be ignored by the algorithm.
    Therefore, this optimizer is resistant to noise in the objective function and can be used with
    non-smooth functions.
    """

    def __init__(
        self,
        maxiter: int = 100,
        callback: Optional[CALLBACK] = None,
        sigma_0: float = 0.5,
        weights: Optional[np.ndarray] = None,
        termination_criteria: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations.
            callback: A callback function to be called after each iteration when using :meth:`~.minimize`.
            sigma_0: Standard deviation for the sampling of the first generation.
            weights: Weights used to compute the next centroid and covariance matrix. If ``None``, they
            will take a default value. All components must be positive, arranged in decreasing order,
            and add up to one.
            termination_criteria: Dictionary with the terimnation criteria to be used and the tolerances
            for each of them. The different options and their default values are:

            * NoEffectAxis: 0
            * NoeffectCoord: 1e4
            * TolFun: 1e-12
            * TolX: :math:`\sigma`
            * NoEffectCoord: 0
            * ConditionCov: 1e14
            * EqualFunValues: 0
            * Stagnation : WIP
        """

        super().__init__(maxiter=maxiter, callback=callback)
        self._state: CMAESState = None
        self.N = None
        self.lmbda = None
        self.weights = weights
        self.mu = None
        self.mueff = None
        self.sigma_0 = sigma_0
        self.cc = None  # Learing rate for evolution path pc.
        self.cs = None
        self.c1 = None  # Learning rate for rank 1 updates. This updates make the next sampling more likely in the direction of the best sampled point from last generateion.
        self.cmu = None  # Covariance Matrix Learning rate
        self.damps = None
        self.chiN = None
        self.termination_criteria = termination_criteria

    def ask(self) -> CMAES_AskObject:
        """ """
        z = np.random.normal(0, 1, size=(self.lmbda, self.N))
        y = np.einsum("ij,j,kj->ki", self._state.eigenvectors, self._state.std_vector, z)
        x = self._state.x + self._state.sigma * y
        return CMAES_AskObject(x_fun=x, x_jac=None, x_fun_translation=z)

    def tell(self, ask_object: AskObject, tell_object: TellObject) -> None:
        self._state.nit += 1
        # We sort the sampled points by the function value. And keep the mu best points.
        sorting_indexes = np.argsort(tell_object.eval_fun)
        sorted_x = ask_object.x_fun[sorting_indexes][: self.mu]
        sorted_z = ask_object.x_fun_translation[sorting_indexes][: self.mu]

        # Update the mean of the distribution.
        self._state.x = np.dot(sorted_x.T, self.weights)
        mean_z = np.dot(sorted_z.T, self.weights)

        # Update all the evolution paths.
        # Sigma
        self._state.p_sigma = (1 - self.cs) * self._state.p_sigma + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * np.dot(self._state.eigenvectors, mean_z)

        hsig = np.linalg.norm(self._state.p_sigma) / np.sqrt(
            1 - (1 - self.cs) ** (2 * self._state.nit / self.lmbda)
        ) < (1.4 + (2 / (self.N + 1)) * self.chiN)

        self._state.p_c = (1 - self.cc) * self._state.p_c + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * mean_z

        # Now we update the covariance matrix
        rank_one_update = self.c1 * (
            np.einsum("i,j->ij", self._state.p_c, self._state.p_c)
            + (1 - hsig) * self.cc * (2 - self.cc) * self._state.cov_matrix
        )

        bdz = np.einsum("ij,j,kj->ik", self._state.eigenvectors, self._state.std_vector, sorted_z)
        rank_mu_update = self.cmu * np.einsum("ij,j,kj->ik", bdz, self.weights, bdz)

        self._state.cov_matrix = (
            (1 - self.c1 - self.cmu) * self._state.cov_matrix + rank_one_update + rank_mu_update
        )

        # We update sigma
        self._state.sigma *= np.exp(
            self.cs / self.damps * (np.linalg.norm(self._state.p_sigma) / self.chiN - 1)
        )

        # Now we will diagonalize the covariance matrix.
        # Here the paper only diagonalizes the matrix every certain amount of iterations.
        # We prefer to be as precise as possible even if we need to sacrifice classical
        # efficiency in the optimizer.
        self._state.cov_matrix = (
            self._state.cov_matrix + self._state.cov_matrix.T
        ) / 2  # We guarantee that the covariance matrix is symmetrical
        self._state.std_vector, self._state.eigenvectors = np.linalg.eig(self._state.cov_matrix)

        negative_mask = self._state.std_vector <= 0
        if negative_mask.any():  # The matrix should be positive definite
            warnings.warn("Casting negative eigenvalues to zero.")
            self._state.std_vector[negative_mask] = 0

        self._state.std_vector = np.real(np.sqrt(self._state.std_vector))
        self._state.eigenvectors = np.real(self._state.cov_matrix)

        # Storing data for the stopping criteria.
        complete_list = self._state.nit > 10 + 30 * self.N / self.lmbda
        self._state._best_func_hist = self._state._best_func_hist[complete_list:] + [
            np.min(tell_object.eval_fun)
        ]
        self._state._median_func_hist = self._state._median_func_hist[complete_list:] + [
            np.median(tell_object.eval_fun)
        ]
        self._state._worst_func_eval = np.max(tell_object.eval_fun)

    def evaluate(self, ask_object: AskObject) -> TellObject:
        eval_fun = [self._state.fun(x) for x in ask_object.x_fun]
        self._state.nfev += len(ask_object.x_fun)
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
        It will also set some parameters needed for the optimization.
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
            _best_func_hist=[],
            _worst_func_eval=None,
            _median_func_hist=[],
        )
        self._initialize_termination_criteria()
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

    def _initialize_termination_criteria(self) -> None:
        """
        Sets the default values for the termination criteria if they haven't been
        specified by the user. Also removes any entries on the dictionary that don't correspond
        to a termination criteria.
        """
        default_term_crit = {
            "NoEffectAxis": 0,
            "TolXUp": 1e4,
            "TolFun": 1e-12,
            "TolX": self.sigma_0 * 1e-12,
            "NoEffectCoord": 0,
            "ConditionCov": 1e14,
            "EqualFunValues": 0,
            "Stagnation": 0,#WIP
        }
        # Create default choice of termination criteria if None is passed.
        if self.termination_criteria is None:
            self.termination_criteria = default_term_crit
        # Giving default tolerances to the criteria with None as a parameter.
        for key, value in self.termination_criteria.items():
            if key not in default_term_crit:
                self.termination_criteria.pop(key)
            if value is None:
                self.termination_criteria[key] = default_term_crit[key]

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

    def _NoEffectAxis(self, tol: float) -> bool:
        """Termination criteria if the the covariance of a principal axis gets too small."""
        return 0.1 * np.max(self._state.std_vector) >= np.linalg.norm(self._state.x) * tol

    def _NoEffectCoord(self, tol: float) -> bool:
        """Termination criteria if the covariance over a cartesian axis gets too small."""
        return 0.2 * np.max(np.diag(self._state.cov_matrix)) >= np.linalg.norm(self._state.x) * tol

    def _ConditionCov(self, tol: float) -> bool:
        """What is condition number???"""
        return True

    def _EqualFunValues(self, tol: float) -> bool:
        """Termination criteria if the range of the best evaluation during the last 10+ 30*N/lmbda
        generations is smaller than a given threshold.
        """
        if self._state.nit < 10 + 30 * self.N / self.lmbda:
            return True
        return max(self._state._best_func_hist) - min(self._state._best_func_hist) > tol

    def _Stagnation(self, tol: float) -> bool:
        """
        WIP
        """
        return True

    def _TolXUp(self, tol: float) -> bool:
        """
        Termination criteria if the covariance increases too fast.
        This usually indicates that the initial standard deviation is too small.
        """
        return self._state.sigma * self._state.std_vector.max() < tol

    def _TolFun(self, tol: float) -> bool:
        """
        Termination criteria if the best evaluation of the objective function has been below a
        threshold during ``10+ 30*N/lmbda`` generations and the worst evaluation on the current
        generation is also below that threshold.
        """
        if self._state.nit < 10 + 30 * self.N / self.lmbda:
            return True
        return (
            self._state._worst_func_eval > tol
            or np.max(self._state._best_func_hist) - np.min(self._state._best_func_hist) > tol
        )

    def _TolX(self, tol: float) -> bool:
        """
        Termination criteria if the covariance matrix and p_sigma are too small.
        """
        return np.any(self._state.sigma * np.diag(self._state.cov_matrix) > tol) or np.any(
            self._state.sigma * self._state.p_c > tol
        )

    def get_stopping_condition(self) -> Dict[str,bool]:
        stop_dict = {}
        for key, value in self.termination_criteria.items():
            stop_dict[key] = getattr(self, "_" + key)(value)
        return stop_dict
    def continue_condition(self) -> bool:
        """
        This is the condition that will be checked after each step to stop the optimization process.
        """
        cont_cond = True
        for key, value in self.termination_criteria.items():
            cont_cond &= getattr(self, "_" + key)(value)
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
