# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A module for common unitary matrix.
"""
import math
import numpy as np


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
