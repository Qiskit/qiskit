# This code is part of Mthree.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=no-name-in-module
"""mthree one-norm estimators"""

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from .exceptions import M3Error


def ainv_onenorm_est_lu(A, LU=None):
    """
    Estimates the one-norm of the inverse A**-1 of the input
    matrix A using direct LU factorization.

    Parameters:
        A (ndarray): Input square matrix.
        LU (ndarray): Perfactored LU of A, if any.

    Returns:
        float: Estimate of one-norm for A**-1.

    Notes:
        This uses the modified Hager's method (Alg. 4.1) from
        N. J. Higham, ACM Trans. Math. Software, Vol. 14, 381 (1988).
    """
    dims = A.shape[0]

    # Starting vec
    v = (1.0/dims)*np.ones(dims, dtype=float)

    # Factor A and A.T
    if LU is None:
        LU = la.lu_factor(A, check_finite=False)
    LU_T = la.lu_factor(A.T, check_finite=False)

    # Initial solve
    v = la.lu_solve(LU, v, check_finite=False)
    gamma = la.norm(v, 1)
    eta = np.sign(v)
    x = la.lu_solve(LU_T, eta, check_finite=False)

    # loop over reasonable number of trials
    k = 2
    while k < 6:
        x_nrm = la.norm(x, np.inf)
        idx = np.where(np.abs(x) == x_nrm)[0][0]
        v = np.zeros(dims, dtype=float)
        v[idx] = 1
        v = la.lu_solve(LU, v, check_finite=False)

        gamma_prime = gamma
        gamma = la.norm(v, 1)

        if gamma <= gamma_prime or np.allclose(np.sign(v), eta):
            break

        eta = np.sign(v)
        x = la.lu_solve(LU_T, eta, check_finite=False)
        if la.norm(x, np.inf) == x[idx]:
            break
        k += 1

    # After loop do Higham's check for cancellations.
    x = np.arange(1, dims+1)
    x = (-1)**(x+1)*(1+(x-1)/(dims-1))

    x = la.lu_solve(LU, x, check_finite=False)

    temp = 2*la.norm(x, 1)/(3*dims)

    if temp > gamma:
        gamma = temp

    return gamma


def ainv_onenorm_est_iter(M, tol=1e-5, max_iter=25):
    """
    Estimates the one-norm of the inverse A**-1 of the input
    matrix A using itertive GMRES.

    Parameters:
        M (M3MatVec): M3 matrix-vector multiplication container.
        tol (float): Tolerance of iterative solver.
        max_iter (int): Number of max iterations to perform.

    Returns:
        float: Estimate of one-norm for A**-1.

    Notes:
        This uses the modified Hager's method (Alg. 4.1) from
        N. J. Higham, ACM Trans. Math. Software, Vol. 14, 381 (1988).

    Raises:
        M3Error: Error in iterative solver.
    """
    # Setup linear operator interfaces
    L = spla.LinearOperator((M.num_elems, M.num_elems),
                            matvec=M.matvec)

    LT = spla.LinearOperator((M.num_elems, M.num_elems),
                             matvec=M.rmatvec)

    diags = M.get_diagonal()

    def precond_matvec(x):
        out = x / diags
        return out

    P = spla.LinearOperator((M.num_elems, M.num_elems), precond_matvec)

    dims = M.num_elems

    # Starting vec
    v = (1.0/dims)*np.ones(dims, dtype=float)

    # Initial solve
    v, error = spla.gmres(L, v, tol=tol, atol=tol, maxiter=max_iter,
                          M=P)
    if error:
        raise M3Error('Iterative solver error {}'.format(error))
    gamma = la.norm(v, 1)
    eta = np.sign(v)
    x, error = spla.gmres(LT, eta, tol=tol, atol=tol, maxiter=max_iter,
                          M=P)
    if error:
        raise M3Error('Iterative solver error {}'.format(error))
    # loop over reasonable number of trials
    k = 2
    while k < 6:
        x_nrm = la.norm(x, np.inf)
        idx = np.where(np.abs(x) == x_nrm)[0][0]
        v = np.zeros(dims, dtype=float)
        v[idx] = 1
        v, error = spla.gmres(L, v, tol=tol, atol=tol, maxiter=max_iter,
                              M=P)
        if error:
            raise M3Error('Iterative solver error {}'.format(error))
        gamma_prime = gamma
        gamma = la.norm(v, 1)

        if gamma <= gamma_prime or np.allclose(np.sign(v), eta, atol=tol):
            break

        eta = np.sign(v)
        x, error = spla.gmres(LT, eta, tol=tol, atol=tol, maxiter=max_iter,
                              M=P)
        if error:
            raise M3Error('Iterative solver error {}'.format(error))
        if la.norm(x, np.inf) == x[idx]:
            break
        k += 1

    # After loop do Higham's check for cancellations.
    x = np.arange(1, dims+1)
    x = (-1)**(x+1)*(1+(x-1)/(dims-1))

    x, error = spla.gmres(L, x, tol=tol, atol=tol, maxiter=max_iter,
                          M=P)
    if error:
        raise M3Error('Iterative solver error {}'.format(error))

    temp = 2*la.norm(x, 1)/(3*dims)

    if temp > gamma:
        gamma = temp

    return gamma
