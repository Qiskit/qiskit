# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Natural Gradient. """

from collections.abc import Iterable
from typing import List, Tuple, Callable, Optional, Union
import functools
import numpy as np

from qiskit.circuit.quantumcircuit import _compare_parameters
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.exceptions import MissingOptionalLibraryError
from ..operator_base import OperatorBase
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..state_fns.circuit_state_fn import CircuitStateFn
from .circuit_gradients import CircuitGradient
from .circuit_qfis import CircuitQFI
from .gradient import Gradient
from .gradient_base import GradientBase
from .qfi import QFI


class NaturalGradient(GradientBase):
    r"""Convert an operator expression to the first-order gradient.

    Given an ill-posed inverse problem

        x = arg min{||Ax-C||^2} (1)

    one can use regularization schemes can be used to stabilize the system and find a numerical
    solution

        x_lambda = arg min{||Ax-C||^2 + lambda*R(x)} (2)

    where R(x) represents the penalization term.
    """

    def __init__(
        self,
        grad_method: Union[str, CircuitGradient] = "lin_comb",
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        regularization: Optional[str] = None,
        **kwargs,
    ):
        r"""
        Args:
            grad_method: The method used to compute the state gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
            regularization: Use the following regularization with a least square method to solve the
                underlying system of linear equations
                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search
                If regularization is None but the metric is ill-conditioned or singular then
                a least square solver is used without regularization
            kwargs (dict): Optional parameters for a CircuitGradient
        """
        super().__init__(grad_method)

        self._qfi_method = QFI(qfi_method)
        self._regularization = regularization
        self._epsilon = kwargs.get("epsilon", 1e-6)

    # pylint: disable=signature-differs
    def convert(
        self,
        operator: OperatorBase,
        params: Optional[
            Union[ParameterVector, ParameterExpression, List[ParameterExpression]]
        ] = None,
    ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of.
            params: The parameters we are taking the gradient with respect to. If not explicitly
                passed, they are inferred from the operator and sorted by name.

        Returns:
            An operator whose evaluation yields the NaturalGradient.

        Raises:
            TypeError: If ``operator`` does not represent an expectation value or the quantum
                state is not ``CircuitStateFn``.
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            ValueError: If ``operator`` is not parameterized.
        """
        if not isinstance(operator, ComposedOp):
            if not (isinstance(operator, ListOp) and len(operator.oplist) == 1):
                raise TypeError(
                    "Please provide the operator either as ComposedOp or as ListOp of "
                    "a CircuitStateFn potentially with a combo function."
                )

        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError(
                "Please make sure that the operator for which you want to compute "
                "Quantum Fisher Information represents an expectation value or a "
                "loss function and that the quantum state is given as "
                "CircuitStateFn."
            )
        if len(operator.parameters) == 0:
            raise ValueError("The operator we are taking the gradient of is not parameterized!")
        if params is None:
            params = sorted(operator.parameters, key=functools.cmp_to_key(_compare_parameters))
        if not isinstance(params, Iterable):
            params = [params]
        # Instantiate the gradient
        grad = Gradient(self._grad_method, epsilon=self._epsilon).convert(operator, params)
        # Instantiate the QFI metric which is used to re-scale the gradient
        metric = self._qfi_method.convert(operator[-1], params) * 0.25

        # Define the function which compute the natural gradient from the gradient and the QFI.
        def combo_fn(x):
            c = np.real(x[0])
            a = np.real(x[1])
            if self.regularization:
                # If a regularization method is chosen then use a regularized solver to
                # construct the natural gradient.
                nat_grad = NaturalGradient._regularized_sle_solver(
                    a, c, regularization=self.regularization
                )
            else:
                try:
                    # Try to solve the system of linear equations Ax = C.
                    nat_grad = np.linalg.solve(a, c)
                except np.linalg.LinAlgError:  # singular matrix
                    nat_grad = np.linalg.lstsq(a, c)[0]
            return np.real(nat_grad)

        # Define the ListOp which combines the gradient and the QFI according to the combination
        # function defined above.
        return ListOp([grad, metric], combo_fn=combo_fn)

    @property
    def qfi_method(self) -> CircuitQFI:
        """Returns ``CircuitQFI``.

        Returns: ``CircuitQFI``

        """
        return self._qfi_method.qfi_method

    @property
    def regularization(self) -> Optional[str]:
        """Returns the regularization option.

        Returns: the regularization option.

        """
        return self._regularization

    @staticmethod
    def _reg_term_search(
        a: np.ndarray,
        c: np.ndarray,
        reg_method: Callable[[np.ndarray, np.ndarray, float], float],
        lambda1: float = 1e-3,
        lambda4: float = 1.0,
        tol: float = 1e-8,
    ) -> Tuple[float, np.ndarray]:
        """
        This method implements a search for a regularization parameter lambda by finding for the
        corner of the L-curve
        More explicitly, one has to evaluate a suitable lambda by finding a compromise between
        the error in the solution and the norm of the regularization.
        This function implements a method presented in
        `A simple algorithm to find the L-curve corner in the regularization of inverse problems
         <https://arxiv.org/pdf/1608.04571.pdf>`
        Args:
            a: see (1) and (2)
            c: see (1) and (2)
            reg_method: Given A, C and lambda the regularization method must return x_lambda
            - see (2)
            lambda1: left starting point for L-curve corner search
            lambda4: right starting point for L-curve corner search
            tol: termination threshold

        Returns:
            regularization coefficient, solution to the regularization inverse problem
        """

        def _get_curvature(x_lambda: List) -> float:
            """Calculate Menger curvature

            Menger, K. (1930).  Untersuchungen  ̈uber Allgemeine Metrik. Math. Ann.,103(1), 466–501

            Args:
                x_lambda: [[x_lambdaj], [x_lambdak], [x_lambdal]]
                    lambdaj < lambdak < lambdal

            Returns:
                Menger Curvature

            """
            eps = []
            eta = []
            for x in x_lambda:
                try:
                    eps.append(np.log(np.linalg.norm(np.matmul(a, x) - c) ** 2))
                except ValueError:
                    eps.append(np.log(np.linalg.norm(np.matmul(a, np.transpose(x)) - c) ** 2))
                eta.append(np.log(max(np.linalg.norm(x) ** 2, 1e-6)))
            p_temp = 1
            c_k = 0
            for i in range(3):
                p_temp *= (eps[np.mod(i + 1, 3)] - eps[i]) ** 2 + (
                    eta[np.mod(i + 1, 3)] - eta[i]
                ) ** 2
                c_k += eps[i] * eta[np.mod(i + 1, 3)] - eps[np.mod(i + 1, 3)] * eta[i]
            c_k = 2 * c_k / max(1e-4, np.sqrt(p_temp))
            return c_k

        def get_lambda2_lambda3(lambda1, lambda4):
            gold_sec = (1 + np.sqrt(5)) / 2.0
            lambda2 = 10 ** ((np.log10(lambda4) + np.log10(lambda1) * gold_sec) / (1 + gold_sec))
            lambda3 = 10 ** (np.log10(lambda1) + np.log10(lambda4) - np.log10(lambda2))
            return lambda2, lambda3

        lambda2, lambda3 = get_lambda2_lambda3(lambda1, lambda4)
        lambda_ = [lambda1, lambda2, lambda3, lambda4]
        x_lambda = []
        for lam in lambda_:
            x_lambda.append(reg_method(a, c, lam))
        counter = 0
        while (lambda_[3] - lambda_[0]) / lambda_[3] >= tol:
            counter += 1
            c_2 = _get_curvature(x_lambda[:-1])
            c_3 = _get_curvature(x_lambda[1:])
            while c_3 < 0:
                lambda_[3] = lambda_[2]
                x_lambda[3] = x_lambda[2]
                lambda_[2] = lambda_[1]
                x_lambda[2] = x_lambda[1]
                lambda2, _ = get_lambda2_lambda3(lambda_[0], lambda_[3])
                lambda_[1] = lambda2
                x_lambda[1] = reg_method(a, c, lambda_[1])
                c_3 = _get_curvature(x_lambda[1:])

            if c_2 > c_3:
                lambda_mc = lambda_[1]
                x_mc = x_lambda[1]
                lambda_[3] = lambda_[2]
                x_lambda[3] = x_lambda[2]
                lambda_[2] = lambda_[1]
                x_lambda[2] = x_lambda[1]
                lambda2, _ = get_lambda2_lambda3(lambda_[0], lambda_[3])
                lambda_[1] = lambda2
                x_lambda[1] = reg_method(a, c, lambda_[1])
            else:
                lambda_mc = lambda_[2]
                x_mc = x_lambda[2]
                lambda_[0] = lambda_[1]
                x_lambda[0] = x_lambda[1]
                lambda_[1] = lambda_[2]
                x_lambda[1] = x_lambda[2]
                _, lambda3 = get_lambda2_lambda3(lambda_[0], lambda_[3])
                lambda_[2] = lambda3
                x_lambda[2] = reg_method(a, c, lambda_[2])
        return lambda_mc, x_mc

    @staticmethod
    def _ridge(
        a: np.ndarray,
        c: np.ndarray,
        lambda_: float = 1.0,
        lambda1: float = 1e-4,
        lambda4: float = 1e-1,
        tol_search: float = 1e-8,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_a: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        solver: str = "auto",
        random_state: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Ridge Regression with automatic search for a good regularization term lambda
        x_lambda = arg min{||Ax-C||^2 + lambda*||x||_2^2} (3)
        `Scikit Learn Ridge Regression
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`
        Args:
            a: see (1) and (2)
            c: see (1) and (2)
            lambda_ : regularization parameter used if auto_search = False
            lambda1: left starting point for L-curve corner search
            lambda4: right starting point for L-curve corner search
            tol_search: termination threshold for regularization parameter search
            fit_intercept: if True calculate intercept
            normalize: deprecated if fit_intercept=False, if True normalize A for regression
            copy_a: if True A is copied, else overwritten
            max_iter: max. number of iterations if solver is CG
            tol: precision of the regression solution
            solver: solver {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
            random_state: seed for the pseudo random number generator used when data is shuffled

        Returns:
           regularization coefficient, solution to the regularization inverse problem

        Raises:
            MissingOptionalLibraryError: scikit-learn not installed

        """
        try:
            from sklearn.linear_model import Ridge
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="scikit-learn", name="_ridge", pip_install="pip install scikit-learn"
            ) from ex

        reg = Ridge(
            alpha=lambda_,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_a,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state,
        )

        def reg_method(a, c, alpha):
            reg.set_params(alpha=alpha)
            reg.fit(a, c)
            return reg.coef_

        lambda_mc, x_mc = NaturalGradient._reg_term_search(
            a, c, reg_method, lambda1=lambda1, lambda4=lambda4, tol=tol_search
        )
        return lambda_mc, np.transpose(x_mc)

    @staticmethod
    def _lasso(
        a: np.ndarray,
        c: np.ndarray,
        lambda_: float = 1.0,
        lambda1: float = 1e-4,
        lambda4: float = 1e-1,
        tol_search: float = 1e-8,
        fit_intercept: bool = True,
        normalize: bool = False,
        precompute: Union[bool, Iterable] = False,
        copy_a: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        warm_start: bool = False,
        positive: bool = False,
        random_state: Optional[int] = None,
        selection: str = "random",
    ) -> Tuple[float, np.ndarray]:
        """
        Lasso Regression with automatic search for a good regularization term lambda
        x_lambda = arg min{||Ax-C||^2/(2*n_samples) + lambda*||x||_1} (4)
        `Scikit Learn Lasso Regression
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`

        Args:
            a: mxn matrix
            c: m vector
            lambda_ : regularization parameter used if auto_search = False
            lambda1: left starting point for L-curve corner search
            lambda4: right starting point for L-curve corner search
            tol_search: termination threshold for regularization parameter search
            fit_intercept: if True calculate intercept
            normalize: deprecated if fit_intercept=False, if True normalize A for regression
            precompute: If True compute and use Gram matrix to speed up calculations.
                                             Gram matrix can also be given explicitly
            copy_a: if True A is copied, else overwritten
            max_iter: max. number of iterations if solver is CG
            tol: precision of the regression solution
            warm_start: if True reuse solution from previous fit as initialization
            positive: if True force positive coefficients
            random_state: seed for the pseudo random number generator used when data is shuffled
            selection: {'cyclic', 'random'}

        Returns:
            regularization coefficient, solution to the regularization inverse problem

        Raises:
            MissingOptionalLibraryError: scikit-learn not installed

        """
        try:
            from sklearn.linear_model import Lasso
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="scikit-learn", name="_lasso", pip_install="pip install scikit-learn"
            ) from ex

        reg = Lasso(
            alpha=lambda_,
            fit_intercept=fit_intercept,
            normalize=normalize,
            precompute=precompute,
            copy_X=copy_a,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection,
        )

        def reg_method(a, c, alpha):
            reg.set_params(alpha=alpha)
            reg.fit(a, c)
            return reg.coef_

        lambda_mc, x_mc = NaturalGradient._reg_term_search(
            a, c, reg_method, lambda1=lambda1, lambda4=lambda4, tol=tol_search
        )

        return lambda_mc, x_mc

    @staticmethod
    def _regularized_sle_solver(
        a: np.ndarray,
        c: np.ndarray,
        regularization: str = "perturb_diag",
        lambda1: float = 1e-3,
        lambda4: float = 1.0,
        alpha: float = 0.0,
        tol_norm_x: Tuple[float, float] = (1e-8, 5.0),
        tol_cond_a: float = 1000.0,
    ) -> np.ndarray:
        """
        Solve a linear system of equations with a regularization method and automatic lambda fitting
        Args:
            a: mxn matrix
            c: m vector
            regularization: Regularization scheme to be used: 'ridge', 'lasso',
                'perturb_diag_elements' or 'perturb_diag'
            lambda1: left starting point for L-curve corner search (for 'ridge' and 'lasso')
            lambda4: right starting point for L-curve corner search (for 'ridge' and 'lasso')
            alpha: perturbation coefficient for 'perturb_diag_elements' and 'perturb_diag'
            tol_norm_x: tolerance for the norm of x
            tol_cond_a: tolerance for the condition number of A

        Returns:
            solution to the regularized system of linear equations

        """
        if regularization == "ridge":
            _, x = NaturalGradient._ridge(a, c, lambda1=lambda1)
        elif regularization == "lasso":
            _, x = NaturalGradient._lasso(a, c, lambda1=lambda1)
        elif regularization == "perturb_diag_elements":
            alpha = 1e-7
            while np.linalg.cond(a + alpha * np.diag(a)) > tol_cond_a:
                alpha *= 10
            # include perturbation in A to avoid singularity
            x, _, _, _ = np.linalg.lstsq(a + alpha * np.diag(a), c, rcond=None)
        elif regularization == "perturb_diag":
            alpha = 1e-7
            while np.linalg.cond(a + alpha * np.eye(len(c))) > tol_cond_a:
                alpha *= 10
            # include perturbation in A to avoid singularity
            x, _, _, _ = np.linalg.lstsq(a + alpha * np.eye(len(c)), c, rcond=None)
        else:
            # include perturbation in A to avoid singularity
            x, _, _, _ = np.linalg.lstsq(a, c, rcond=None)

        if np.linalg.norm(x) > tol_norm_x[1] or np.linalg.norm(x) < tol_norm_x[0]:
            if regularization == "ridge":
                lambda1 = lambda1 / 10.0
                _, x = NaturalGradient._ridge(a, c, lambda1=lambda1, lambda4=lambda4)
            elif regularization == "lasso":
                lambda1 = lambda1 / 10.0
                _, x = NaturalGradient._lasso(a, c, lambda1=lambda1)
            elif regularization == "perturb_diag_elements":
                while np.linalg.cond(a + alpha * np.diag(a)) > tol_cond_a:
                    if alpha == 0:
                        alpha = 1e-7
                    else:
                        alpha *= 10
                # include perturbation in A to avoid singularity
                x, _, _, _ = np.linalg.lstsq(a + alpha * np.diag(a), c, rcond=None)
            else:
                if alpha == 0:
                    alpha = 1e-7
                else:
                    alpha *= 10
                while np.linalg.cond(a + alpha * np.eye(len(c))) > tol_cond_a:
                    # include perturbation in A to avoid singularity
                    x, _, _, _ = np.linalg.lstsq(a + alpha * np.eye(len(c)), c, rcond=None)
                    alpha *= 10
        return x
