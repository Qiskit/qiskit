# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests qiskit/mapper/_quaternion"""
import numpy as np
import scipy.linalg as la
from qiskit.mapper._quaternion import (quaternion_from_euler, _rotm)
from .common import QiskitTestCase


class TestQuaternions(QiskitTestCase):
    """Tests qiskit/mapper_quaternion"""

    def test_random_euler(self):
        """Quaternion from random Euler rotations."""
        # Random angles and axes
        axes = {0: 'x', 1: 'y', 2: 'z'}
        for _ in range(1000):
            rnd = 4*np.pi*(np.random.random(3)-0.5)
            idx = np.random.randint(3, size=3)
            mat1 = _rotm(rnd[0], axes[idx[0]]).dot(
                _rotm(rnd[1], axes[idx[1]]).dot(_rotm(rnd[2], axes[idx[2]])))
            axes_str = ''.join(axes[i] for i in idx)
            quat = quaternion_from_euler(rnd, axes_str)
            mat2 = quat.to_matrix()
            self.assertTrue(np.allclose(mat1, mat2))

    def test_orthogonality(self):
        """Quaternion rotation matrix orthogonality"""
        # Check orthogonality of generated rotation matrix
        axes = {0: 'x', 1: 'y', 2: 'z'}
        for _ in range(1000):
            rnd = 4*np.pi*(np.random.random(3)-0.5)
            idx = np.random.randint(3, size=3)
            axes_str = ''.join(axes[i] for i in idx)
            quat = quaternion_from_euler(rnd, axes_str)
            mat = quat.to_matrix()
            self.assertTrue(np.allclose(mat.dot(mat.T),
                                        np.identity(3, dtype=float)))

    def test_det(self):
        """Quaternion det = 1"""
        # Check det for rotation and not reflection
        axes = {0: 'x', 1: 'y', 2: 'z'}
        for _ in range(1000):
            rnd = 4*np.pi*(np.random.random(3)-0.5)
            idx = np.random.randint(3, size=3)
            axes_str = ''.join(axes[i] for i in idx)
            quat = quaternion_from_euler(rnd, axes_str)
            mat = quat.to_matrix()
            self.assertTrue(np.allclose(la.det(mat), 1))

    def test_equiv_quaternions(self):
        """Different Euler rotations give same quaternion, up to sign."""
        # Check if euler angles from to_zyz return same quaternion
        # up to a sign (2pi rotation)
        rot = {0: 'xyz', 1: 'xyx', 2: 'xzy',
               3: 'xzx', 4: 'yzx', 5: 'yzy',
               6: 'yxz', 7: 'yxy', 8: 'zxy',
               9: 'zxz', 10: 'zyx', 11: 'zyz'}

        for _ in range(1000):
            rnd = 4*np.pi*(np.random.random(3)-0.5)
            idx = np.random.randint(12)
            quat1 = quaternion_from_euler(rnd, rot[idx])
            euler = quat1.to_zyz()
            quat2 = quaternion_from_euler(euler, 'zyz')
            self.assertTrue(np.allclose(abs(quat1.data.dot(quat2.data)), 1))
