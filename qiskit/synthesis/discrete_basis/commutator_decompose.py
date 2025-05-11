# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to compute the decomposition of an SO(3) matrix as balanced commutator."""

from __future__ import annotations

import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence


def _compute_trace_so3(matrix: np.ndarray) -> float:
    """Computes trace of an SO(3)-matrix.

    Args:
        matrix: an SO(3)-matrix

    Returns:
        Trace of ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    _check_is_so3(matrix)

    trace = np.matrix.trace(matrix)
    trace_rounded = min(trace, 3)
    return trace_rounded


def _compute_rotation_axis(matrix: np.ndarray) -> np.ndarray:
    """Computes rotation axis of SO(3)-matrix.

    Args:
        matrix: The SO(3)-matrix for which rotation angle needs to be computed.

    Returns:
        The rotation axis of the SO(3)-matrix ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    _check_is_so3(matrix)

    # If theta represents the rotation angle, then trace = 1 + 2cos(theta).
    trace = _compute_trace_so3(matrix)

    if trace >= 3 - 1e-10:
        # The matrix is the identity (rotation by 0)
        x = 1.0
        y = 0.0
        z = 0.0

    elif trace <= -1 + 1e-10:
        # The matrix is the 180-degree rotation
        squares = (1 + np.diagonal(matrix)) / 2
        index_of_max = np.argmax(squares)

        if index_of_max == 0:
            x = math.sqrt(squares[0])
            y = matrix[0][1] / (2 * x)
            z = matrix[0][2] / (2 * x)
        elif index_of_max == 1:
            y = math.sqrt(squares[1])
            x = matrix[0][1] / (2 * y)
            z = matrix[1][2] / (2 * y)
        else:
            z = math.sqrt(squares[2])
            x = matrix[0][2] / (2 * z)
            y = matrix[1][2] / (2 * z)

    else:
        # The matrix is the rotation by theta with sin(theta)!=0
        theta = math.acos(0.5 * (trace - 1))
        x = 1 / (2 * math.sin(theta)) * (matrix[2][1] - matrix[1][2])
        y = 1 / (2 * math.sin(theta)) * (matrix[0][2] - matrix[2][0])
        z = 1 / (2 * math.sin(theta)) * (matrix[1][0] - matrix[0][1])

    return np.array([x, y, z])


def _solve_decomposition_angle(matrix: np.ndarray) -> float:
    """Computes angle for balanced commutator of SO(3)-matrix ``matrix``.

    Computes angle a so that the SO(3)-matrix ``matrix`` can be decomposed
    as commutator [v,w] where v and w are both rotations of a about some axis.
    The computation is done by solving a trigonometric equation using scipy.optimize.fsolve.

    Args:
        matrix: The SO(3)-matrix for which the decomposition angle needs to be computed.

    Returns:
        Angle a so that matrix = [v,w] with v and w rotations of a about some axis.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    from scipy.optimize import fsolve

    _check_is_so3(matrix)

    trace = _compute_trace_so3(matrix)
    angle = math.acos((1 / 2) * (trace - 1))

    lhs = math.sin(angle / 2)

    def objective(phi):
        sin_sq = math.sin(phi.item() / 2) ** 2
        return 2 * sin_sq * math.sqrt(1 - sin_sq**2) - lhs

    decomposition_angle = fsolve(objective, angle)[0]
    return decomposition_angle


def _compute_rotation_from_angle_and_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """Computes the SO(3)-matrix corresponding to the rotation of ``angle`` about ``axis``.

    Args:
        angle: The angle of the rotation.
        axis: The axis of the rotation.

    Returns:
        SO(3)-matrix that represents a rotation of ``angle`` about ``axis``.

    Raises:
        ValueError: if ``axis`` is not a 3-dim unit vector.
    """
    if axis.shape != (3,):
        raise ValueError(f"Axis must be a 1d array of length 3, but has shape {axis.shape}.")

    if abs(np.linalg.norm(axis) - 1.0) > 1e-4:
        raise ValueError(f"Axis must have a norm of 1, but has {np.linalg.norm(axis)}.")

    res = math.cos(angle) * np.identity(3) + math.sin(angle) * _cross_product_matrix(axis)
    res += (1 - math.cos(angle)) * np.outer(axis, axis)
    return res


def _compute_rotation_between(from_vector: np.ndarray, to_vector: np.ndarray) -> np.ndarray:
    """Computes the SO(3)-matrix for rotating ``from_vector`` to ``to_vector``.

    Args:
        from_vector: unit vector of size 3
        to_vector: unit vector of size 3

    Returns:
        SO(3)-matrix that brings ``from_vector`` to ``to_vector``.

    Raises:
        ValueError: if at least one of ``from_vector`` of ``to_vector`` is not a 3-dim unit vector.
    """
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    dot = np.dot(from_vector, to_vector)
    cross = _cross_product_matrix(np.cross(from_vector, to_vector))
    rotation_matrix = np.identity(3) + cross + np.dot(cross, cross) / (1 + dot)
    return rotation_matrix


def _cross_product_matrix(v: np.ndarray) -> np.ndarray:
    """Computes cross product matrix from vector.

    Args:
        v: Vector for which cross product matrix needs to be computed.

    Returns:
        The cross product matrix corresponding to vector ``v``.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _compute_commutator_so3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the commutator of the SO(3)-matrices ``a`` and ``b``.

    The computation uses the fact that the inverse of an SO(3)-matrix is equal to its transpose.

    Args:
        a: SO(3)-matrix
        b: SO(3)-matrix

    Returns:
        The commutator [a,b] of ``a`` and ``b`` w

    Raises:
        ValueError: if at least one of ``a`` or ``b`` is not an SO(3)-matrix.
    """
    _check_is_so3(a)
    _check_is_so3(b)

    a_dagger = np.conj(a).T
    b_dagger = np.conj(b).T

    return np.dot(np.dot(np.dot(a, b), a_dagger), b_dagger)


def commutator_decompose(
    u_so3: np.ndarray, check_input: bool = True
) -> tuple[GateSequence, GateSequence]:
    r"""Decompose an :math:`SO(3)`-matrix, :math:`U` as a balanced commutator.

    This function finds two :math:`SO(3)` matrices :math:`V, W` such that the input matrix
    equals

    .. math::

        U = V^\dagger W^\dagger V W.

    For this decomposition, the following statement holds


    .. math::

        ||V - I||_F, ||W - I||_F \leq \frac{\sqrt{||U - I||_F}}{2},

    where :math:`I` is the identity and :math:`||\cdot ||_F` is the Frobenius norm.

    Args:
        u_so3: SO(3)-matrix that needs to be decomposed as balanced commutator.
        check_input: If True, checks whether the input matrix is actually SO(3).

    Returns:
        Tuple of GateSequences from SO(3)-matrices :math:`V, W`.

    Raises:
        ValueError: if ``u_so3`` is not an SO(3)-matrix.
    """
    if check_input:
        # assert that the input matrix is really SO(3)
        _check_is_so3(u_so3)

        if not is_identity_matrix(u_so3.dot(u_so3.T)):
            raise ValueError("Input matrix is not orthogonal.")

    angle = _solve_decomposition_angle(u_so3)

    # Compute rotation about x-axis with angle 'angle'
    vx = _compute_rotation_from_angle_and_axis(angle, np.array([1, 0, 0]))

    # Compute rotation about y-axis with angle 'angle'
    wy = _compute_rotation_from_angle_and_axis(angle, np.array([0, 1, 0]))

    commutator = _compute_commutator_so3(vx, wy)

    u_so3_axis = _compute_rotation_axis(u_so3)
    commutator_axis = _compute_rotation_axis(commutator)

    sim_matrix = _compute_rotation_between(commutator_axis, u_so3_axis)
    sim_matrix_dagger = np.conj(sim_matrix).T

    v = np.dot(np.dot(sim_matrix, vx), sim_matrix_dagger)
    w = np.dot(np.dot(sim_matrix, wy), sim_matrix_dagger)

    return GateSequence.from_matrix(v), GateSequence.from_matrix(w)
