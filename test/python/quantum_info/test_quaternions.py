# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests quaternion conversion"""

import math
import numpy as np
import scipy.linalg as la

from qiskit.quantum_info.operators.quaternion import quaternion_from_euler, Quaternion

from qiskit.test import QiskitTestCase


class TestQuaternions(QiskitTestCase):
    """Tests qiskit.quantum_info.operators.quaternion"""

    def setUp(self):
        self.rndArray = np.array([0.5, 0.8, 0.9, -0.3])
        self.norm = la.norm(self.rndArray)
        self.quatUnnormalized = Quaternion(self.rndArray)
        axes = ['x', 'y', 'z']
        rnd = np.array([-0.92545003, -2.19985357, 6.01761209])
        idx = np.array([0, 2, 1])
        self.mat1 = rotation_matrix(rnd[0], axes[idx[0]]).dot(
            rotation_matrix(rnd[1], axes[idx[1]]).dot(
                rotation_matrix(rnd[2], axes[idx[2]])))
        axes_str = ''.join(axes[i] for i in idx)
        quat = quaternion_from_euler(rnd, axes_str)
        self.mat2 = quat.to_matrix()

    def test_norm(self):
        """Quaternions should give correct norm."""
        self.assertEqual(self.norm, self.quatUnnormalized.norm())

    def test_normalize(self):
        """Quaternions should be normalizable"""
        self.assertAlmostEqual(self.quatUnnormalized.normalize().norm(), 1, places=5)

    def test_random_euler(self):
        """Quaternion from Euler rotations."""
        self.assertTrue(np.allclose(self.mat1, self.mat2))

    def test_orthogonality(self):
        """Quaternion rotation matrix orthogonality"""
        self.assertTrue(np.allclose(self.mat2.dot(self.mat2.T), np.identity(3, dtype=float)))

    def test_det(self):
        """Quaternion det = 1"""
        self.assertTrue(np.allclose(la.det(self.mat2), 1))

    def test_equiv_quaternions(self):
        """Different Euler rotations give same quaternion, up to sign."""
        # Check if euler angles from to_zyz return same quaternion
        # up to a sign (2pi rotation)
        rot = ['xyz', 'xyx', 'xzy', 'xzx', 'yzx', 'yzy', 'yxz', 'yxy', 'zxy', 'zxz', 'zyx', 'zyz']
        for value in rot:
            rnd = np.array([-1.57657536, 5.66384302, 2.91532185])
            quat1 = quaternion_from_euler(rnd, value)
            euler = quat1.to_zyz()
            quat2 = quaternion_from_euler(euler, 'zyz')
            self.assertTrue(np.allclose(abs(quat1.data.dot(quat2.data)), 1))

    def test_mul(self):
        """Quarternions should multiply correctly."""
        # multiplication of quarternions is equivalent to the
        # multiplication of corresponding rotation matrices.
        other_quat = Quaternion(np.array([0.4, 0.2, -0.7, 0.8]))
        other_mat = other_quat.to_matrix()
        product_quat = self.quatUnnormalized * other_quat
        product_mat = (self.quatUnnormalized.to_matrix()).dot(other_mat)
        self.assertTrue(np.allclose(product_quat.to_matrix(), product_mat))

    def test_rotation(self):
        """Multiplication by -1 should give the same rotation."""
        neg_quat = Quaternion(self.quatUnnormalized.data * -1)
        self.assertTrue(np.allclose(neg_quat.to_matrix(), self.quatUnnormalized.to_matrix()))


def rotation_matrix(angle, axis):
    """Generates a rotation matrix for a given angle and axis.

    Args:
        angle (float): Rotation angle in radians.
        axis (str): Axis for rotation: 'x', 'y', 'z'

    Returns:
        ndarray: Rotation matrix.

    Raises:
        ValueError: Invalid input axis.
    """
    direction = np.zeros(3, dtype=float)
    if axis == 'x':
        direction[0] = 1
    elif axis == 'y':
        direction[1] = 1
    elif axis == 'z':
        direction[2] = 1
    else:
        raise ValueError('Invalid axis.')
    direction = np.asarray(direction, dtype=float)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    rot = np.diag([cos_angle, cos_angle, cos_angle])
    rot += np.outer(direction, direction) * (1.0 - cos_angle)
    direction *= sin_angle
    rot += np.array([
        [0, -direction[2], direction[1]],
        [direction[2], 0, -direction[0]],
        [-direction[1], direction[0], 0]
    ])
    return rot
