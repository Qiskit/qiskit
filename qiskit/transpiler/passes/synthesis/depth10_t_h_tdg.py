# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Basic set of {H, T, T_dg} compositions and their according SO(3) representation."""

from numpy import array, sqrt

ISQRT2 = 1 / sqrt(2)
SIN_PI8 = (1 - ISQRT2) / 2
COS_PI8 = (1 + ISQRT2) / 2
ISQRT2_P = ISQRT2 + 0.25  # or sqrt(15/16 - sin^4(pi/8))
ISQRT2_M = ISQRT2 - 0.25  # or sqrt(15/16 - cos^4(pi/8))
TWO_M = (2 - sqrt(2)) / 8
TWO_P = (2 + sqrt(2)) / 8
SIX_M = (6 - sqrt(2)) / 8
SIX_P = (6 + sqrt(2)) / 8
THREE_SQRT2_M = (3 * sqrt(2) - 2) / 8
THREE_SQRT2_P = (3 * sqrt(2) + 2) / 8

# pylint: disable=invalid-name
decompositions = {
    "h": array(
        [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "t": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "tdg": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [-ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "h t": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [0.0, -ISQRT2, -ISQRT2],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "h tdg": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [0.0, -ISQRT2, ISQRT2],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "t h": array(
        [
            [0.0, 0.0, -1.0],
            [-ISQRT2, -ISQRT2, 0.0],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t": array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "tdg h": array(
        [
            [0.0, 0.0, -1.0],
            [ISQRT2, -ISQRT2, 0.0],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg tdg": array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "h t h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ISQRT2, ISQRT2],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t t": array(
        [
            [0.0, 1.0e00, 0.0],
            [0.0, 0.0, -1.0e00],
            [-1.0e00, 0.0, 0.0],
        ]
    ),
    "h tdg h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ISQRT2, -ISQRT2],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg tdg": array(
        [
            [0.0, -1.0e00, 0.0],
            [0.0, 0.0, 1.0e00],
            [-1.0e00, 0.0, 0.0],
        ]
    ),
    "t h t": array(
        [
            [0.5, 0.5, -ISQRT2],
            [-0.5, -0.5, -ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [-0.5, -0.5, ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t h": array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "t t t": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "tdg h t": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [0.5, -0.5, -ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h tdg": array(
        [
            [0.5, -0.5, -ISQRT2],
            [0.5, -0.5, ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg tdg h": array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "tdg tdg tdg": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [-ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "h t h t": array(
        [
            [ISQRT2, -0.5, -0.5],
            [ISQRT2, 0.5, 0.5],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t h tdg": array(
        [
            [ISQRT2, 0.5, 0.5],
            [-ISQRT2, 0.5, 0.5],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t t h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "h t t t": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [0.0, ISQRT2, -ISQRT2],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "h tdg h t": array(
        [
            [ISQRT2, -0.5, 0.5],
            [ISQRT2, 0.5, -0.5],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg h tdg": array(
        [
            [ISQRT2, 0.5, -0.5],
            [-ISQRT2, 0.5, -0.5],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg tdg h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "h tdg tdg tdg": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [0.0, ISQRT2, ISQRT2],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "t h t h": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [0.5, 0.5, ISQRT2],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t t": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg h": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [0.5, 0.5, -ISQRT2],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg tdg": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t h t": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [-ISQRT2, 0.0, -ISQRT2],
            [0.0, 1.0, 0.0],
        ]
    ),
    "t t h tdg": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [-ISQRT2, 0.0, ISQRT2],
            [0.0, 1.0, 0.0],
        ]
    ),
    "t t t h": array(
        [
            [0.0, 0.0, -1.0],
            [-ISQRT2, ISQRT2, 0.0],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t t t": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "tdg h t h": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [-0.5, 0.5, ISQRT2],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "tdg h t t": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h tdg h": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [-0.5, 0.5, -ISQRT2],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "tdg h tdg tdg": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg tdg h t": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [ISQRT2, 0.0, -ISQRT2],
            [0.0, -1.0, 0.0],
        ]
    ),
    "tdg tdg h tdg": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [ISQRT2, 0.0, ISQRT2],
            [0.0, -1.0, 0.0],
        ]
    ),
    "tdg tdg tdg h": array(
        [
            [0.0, 0.0, -1.0],
            [ISQRT2, ISQRT2, 0.0],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "h t h t h": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [-ISQRT2, -0.5, -0.5],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t t": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [1.0, 0.0, 0.0],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t h tdg h": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [ISQRT2, -0.5, -0.5],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg tdg": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [-1.0, 0.0, 0.0],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t t h tdg": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [-ISQRT2, 0.0, ISQRT2],
            [0.0, -1.0e00, 0.0],
        ]
    ),
    "h t t t h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -ISQRT2, ISQRT2],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t t t t": array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "h tdg h t h": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [-ISQRT2, -0.5, 0.5],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [1.0, 0.0, 0.0],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg h tdg h": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [ISQRT2, -0.5, 0.5],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg tdg": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [-1.0, 0.0, 0.0],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg tdg h t": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [ISQRT2, 0.0, -ISQRT2],
            [0.0, 1.0e00, 0.0],
        ]
    ),
    "h tdg tdg tdg h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -ISQRT2, -ISQRT2],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "t h t h t": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [COS_PI8, -SIN_PI8, 0.5],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t h tdg": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [-SIN_PI8, COS_PI8, 0.5],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t t t": array(
        [
            [0.5, 0.5, ISQRT2],
            [0.5, 0.5, -ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg h t": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [COS_PI8, -SIN_PI8, -0.5],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg h tdg": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [-SIN_PI8, COS_PI8, -0.5],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg tdg h": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg tdg tdg": array(
        [
            [-0.5, -0.5, ISQRT2],
            [0.5, 0.5, ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t h t h": array(
        [
            [0.0, -1.0, 0.0],
            [ISQRT2, 0.0, ISQRT2],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg h": array(
        [
            [0.0, -1.0, 0.0],
            [ISQRT2, 0.0, -ISQRT2],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg tdg": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "t t t h t": array(
        [
            [0.5, -0.5, -ISQRT2],
            [-0.5, 0.5, -ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t t h tdg": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [-0.5, 0.5, ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t t t h": array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
    "tdg h t h t": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [SIN_PI8, COS_PI8, 0.5],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "tdg h t h tdg": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [-COS_PI8, -SIN_PI8, 0.5],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "tdg h t t h": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h t t t": array(
        [
            [-0.5, 0.5, ISQRT2],
            [-0.5, 0.5, -ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h tdg h t": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [SIN_PI8, COS_PI8, -0.5],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "tdg h tdg h tdg": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [-COS_PI8, -SIN_PI8, -0.5],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "tdg h tdg tdg tdg": array(
        [
            [0.5, -0.5, ISQRT2],
            [-0.5, 0.5, ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg tdg h t h": array(
        [
            [0.0, 1.0, 0.0],
            [-ISQRT2, 0.0, ISQRT2],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "tdg tdg h t t": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "tdg tdg h tdg h": array(
        [
            [0.0, 1.0, 0.0],
            [-ISQRT2, 0.0, -ISQRT2],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "tdg tdg tdg h t": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [0.5, 0.5, -ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg tdg tdg h tdg": array(
        [
            [0.5, 0.5, -ISQRT2],
            [0.5, 0.5, ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "h t h t h t": array(
        [
            [0.5, COS_PI8, -SIN_PI8],
            [-0.5, SIN_PI8, -COS_PI8],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t h tdg": array(
        [
            [-0.5, SIN_PI8, -COS_PI8],
            [-0.5, -COS_PI8, SIN_PI8],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t t t": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [ISQRT2, -0.5, -0.5],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t h tdg h t": array(
        [
            [-0.5, COS_PI8, -SIN_PI8],
            [0.5, SIN_PI8, -COS_PI8],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg h tdg": array(
        [
            [0.5, SIN_PI8, -COS_PI8],
            [0.5, -COS_PI8, SIN_PI8],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg tdg h": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [1.0, 0.0, 0.0],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t h tdg tdg tdg": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [-ISQRT2, -0.5, -0.5],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t t h tdg h": array(
        [
            [0.0, 1.0, 0.0],
            [ISQRT2, 0.0, -ISQRT2],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h t t h tdg tdg": array(
        [
            [0.0, 0.0, 1.0e00],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "h t t t h t": array(
        [
            [ISQRT2, 0.5, -0.5],
            [ISQRT2, -0.5, 0.5],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t t t h tdg": array(
        [
            [ISQRT2, -0.5, 0.5],
            [-ISQRT2, -0.5, 0.5],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t t t t h": array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "h tdg h t h t": array(
        [
            [0.5, -SIN_PI8, -COS_PI8],
            [-0.5, -COS_PI8, -SIN_PI8],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t h tdg": array(
        [
            [-0.5, -COS_PI8, -SIN_PI8],
            [-0.5, SIN_PI8, COS_PI8],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t h": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [-1.0, 0.0, 0.0],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "h tdg h t t t": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [ISQRT2, -0.5, 0.5],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg h tdg h t": array(
        [
            [-0.5, -SIN_PI8, -COS_PI8],
            [0.5, -COS_PI8, -SIN_PI8],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg h tdg": array(
        [
            [0.5, -COS_PI8, -SIN_PI8],
            [0.5, SIN_PI8, COS_PI8],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg tdg tdg": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [-ISQRT2, -0.5, 0.5],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg tdg h t h": array(
        [
            [0.0, -1.0, 0.0],
            [-ISQRT2, 0.0, ISQRT2],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h tdg tdg h t t": array(
        [
            [0.0, 0.0, 1.0e00],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "h tdg tdg tdg h t": array(
        [
            [ISQRT2, 0.5, 0.5],
            [ISQRT2, -0.5, -0.5],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "h tdg tdg tdg h tdg": array(
        [
            [ISQRT2, -0.5, -0.5],
            [-ISQRT2, -0.5, -0.5],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "t h t h t h": array(
        [
            [0.5, 0.5, -ISQRT2],
            [-COS_PI8, SIN_PI8, -0.5],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t t": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t h tdg h": array(
        [
            [0.5, 0.5, -ISQRT2],
            [SIN_PI8, -COS_PI8, -0.5],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg tdg": array(
        [
            [0.5, 0.5, ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t t t h": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [-0.5, -0.5, ISQRT2],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "t h t t t t": array(
        [
            [0.0, 0.0, 1.0],
            [ISQRT2, ISQRT2, 0.0],
            [-ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg h t h": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [-COS_PI8, SIN_PI8, 0.5],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h tdg h t t": array(
        [
            [-0.5, -0.5, ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg h tdg h": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [SIN_PI8, -COS_PI8, 0.5],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg tdg": array(
        [
            [0.5, 0.5, -ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg tdg h t": array(
        [
            [0.5, -0.5, ISQRT2],
            [0.5, -0.5, -ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg tdg tdg h": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [-0.5, -0.5, -ISQRT2],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t t h t h t": array(
        [
            [-0.5, -ISQRT2, -0.5],
            [0.5, -ISQRT2, 0.5],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h t h tdg": array(
        [
            [0.5, -ISQRT2, 0.5],
            [0.5, ISQRT2, 0.5],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg h t": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [0.5, -ISQRT2, -0.5],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg h tdg": array(
        [
            [0.5, -ISQRT2, -0.5],
            [0.5, ISQRT2, -0.5],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg tdg h": array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    ),
    "t t h tdg tdg tdg": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [ISQRT2, 0.0, ISQRT2],
            [0.0, 1.0, 0.0],
        ]
    ),
    "t t t h t h": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [0.5, -0.5, ISQRT2],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "t t t h tdg h": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [0.5, -0.5, -ISQRT2],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "t t t h tdg tdg": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, 1.0],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t t t t h t": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [0.0, ISQRT2, -ISQRT2],
            [1.0, 0.0, 0.0],
        ]
    ),
    "t t t t h tdg": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [0.0, ISQRT2, ISQRT2],
            [1.0, 0.0, 0.0],
        ]
    ),
    "tdg h t h t h": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [-SIN_PI8, -COS_PI8, -0.5],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h t t": array(
        [
            [0.5, -0.5, -ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "tdg h t h tdg h": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [COS_PI8, SIN_PI8, -0.5],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg tdg": array(
        [
            [-0.5, 0.5, ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "tdg h t t h tdg": array(
        [
            [0.5, 0.5, ISQRT2],
            [-0.5, -0.5, ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h t t t h": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [0.5, -0.5, ISQRT2],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t t t t": array(
        [
            [0.0, 0.0, 1.0],
            [-ISQRT2, ISQRT2, 0.0],
            [-ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h tdg h t h": array(
        [
            [0.5, -0.5, -ISQRT2],
            [-SIN_PI8, -COS_PI8, 0.5],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h tdg h t t": array(
        [
            [0.5, -0.5, ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "tdg h tdg h tdg h": array(
        [
            [0.5, -0.5, -ISQRT2],
            [COS_PI8, SIN_PI8, 0.5],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h tdg h tdg tdg": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "tdg h tdg tdg tdg h": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [0.5, -0.5, -ISQRT2],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "tdg tdg h t h t": array(
        [
            [0.5, ISQRT2, -0.5],
            [-0.5, ISQRT2, 0.5],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "tdg tdg h t h tdg": array(
        [
            [-0.5, ISQRT2, 0.5],
            [-0.5, -ISQRT2, 0.5],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "tdg tdg h t t h": array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    ),
    "tdg tdg h t t t": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [-ISQRT2, 0.0, -ISQRT2],
            [0.0, -1.0, 0.0],
        ]
    ),
    "tdg tdg h tdg h t": array(
        [
            [0.5, ISQRT2, 0.5],
            [-0.5, ISQRT2, -0.5],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "tdg tdg h tdg h tdg": array(
        [
            [-0.5, ISQRT2, -0.5],
            [-0.5, -ISQRT2, -0.5],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "tdg tdg tdg h t h": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [-0.5, -0.5, ISQRT2],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "tdg tdg tdg h t t": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg tdg tdg h tdg h": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [-0.5, -0.5, -ISQRT2],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "h t h t h t h": array(
        [
            [ISQRT2, -0.5, -0.5],
            [0.5, -SIN_PI8, COS_PI8],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t t": array(
        [
            [ISQRT2, 0.5, 0.5],
            [0.0, ISQRT2, -ISQRT2],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t h tdg h": array(
        [
            [ISQRT2, -0.5, -0.5],
            [0.5, COS_PI8, -SIN_PI8],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg tdg": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [0.0, -ISQRT2, ISQRT2],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t t t h": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [-ISQRT2, 0.5, 0.5],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t t t t": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -ISQRT2, -ISQRT2],
            [0.0, -ISQRT2, ISQRT2],
        ]
    ),
    "h t h tdg h t h": array(
        [
            [ISQRT2, 0.5, 0.5],
            [-0.5, -SIN_PI8, COS_PI8],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h tdg h t t": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [0.0, ISQRT2, -ISQRT2],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg h tdg h": array(
        [
            [ISQRT2, 0.5, 0.5],
            [-0.5, COS_PI8, -SIN_PI8],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h tdg h tdg tdg": array(
        [
            [ISQRT2, -0.5, -0.5],
            [0.0, -ISQRT2, ISQRT2],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg tdg h t": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [ISQRT2, 0.5, -0.5],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t h tdg tdg tdg h": array(
        [
            [0.0, ISQRT2, -ISQRT2],
            [ISQRT2, 0.5, 0.5],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t t h tdg h t": array(
        [
            [-0.5, ISQRT2, 0.5],
            [0.5, ISQRT2, -0.5],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h t t h tdg h tdg": array(
        [
            [0.5, ISQRT2, -0.5],
            [0.5, -ISQRT2, -0.5],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h t t h tdg tdg h": array(
        [
            [0.0, 1.0e00, 0.0],
            [1.0e00, 0.0, 0.0],
            [0.0, 0.0, -1.0e00],
        ]
    ),
    "h t t t h t h": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [-ISQRT2, 0.5, -0.5],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h t t t h tdg h": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [ISQRT2, 0.5, -0.5],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h t t t h tdg tdg": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [-1.0, 0.0, 0.0],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t t t t h t": array(
        [
            [ISQRT2, ISQRT2, 0.0],
            [ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "h t t t t h tdg": array(
        [
            [ISQRT2, -ISQRT2, 0.0],
            [-ISQRT2, -ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "h tdg h t h t h": array(
        [
            [ISQRT2, -0.5, 0.5],
            [0.5, COS_PI8, SIN_PI8],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h t t": array(
        [
            [ISQRT2, 0.5, -0.5],
            [0.0, -ISQRT2, -ISQRT2],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t h tdg h": array(
        [
            [ISQRT2, -0.5, 0.5],
            [0.5, -SIN_PI8, -COS_PI8],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg tdg": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [0.0, ISQRT2, ISQRT2],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t h tdg": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [-ISQRT2, 0.5, 0.5],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "h tdg h t t t h": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [-ISQRT2, 0.5, -0.5],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t t t": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -ISQRT2, ISQRT2],
            [0.0, ISQRT2, ISQRT2],
        ]
    ),
    "h tdg h tdg h t h": array(
        [
            [ISQRT2, 0.5, -0.5],
            [-0.5, COS_PI8, SIN_PI8],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h tdg h t t": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [0.0, -ISQRT2, -ISQRT2],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg h tdg h": array(
        [
            [ISQRT2, 0.5, -0.5],
            [-0.5, -SIN_PI8, -COS_PI8],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h tdg h tdg tdg": array(
        [
            [ISQRT2, -0.5, 0.5],
            [0.0, ISQRT2, ISQRT2],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg tdg tdg h": array(
        [
            [0.0, -ISQRT2, -ISQRT2],
            [ISQRT2, 0.5, -0.5],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg tdg h t h t": array(
        [
            [0.5, -ISQRT2, -0.5],
            [-0.5, -ISQRT2, 0.5],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h tdg tdg h t h tdg": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [-0.5, ISQRT2, 0.5],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h tdg tdg h t t h": array(
        [
            [0.0, -1.0e00, 0.0],
            [-1.0e00, 0.0, 0.0],
            [0.0, 0.0, -1.0e00],
        ]
    ),
    "h tdg tdg tdg h t h": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [-ISQRT2, 0.5, 0.5],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h tdg tdg tdg h t t": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [1.0, 0.0, 0.0],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "h tdg tdg tdg h tdg h": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [ISQRT2, 0.5, 0.5],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "t h t h t h t": array(
        [
            [ISQRT2_P, 0.25, -SIN_PI8],
            [-0.25, ISQRT2_M, -COS_PI8],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t h tdg": array(
        [
            [-0.25, ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, -0.25, SIN_PI8],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t t t": array(
        [
            [-COS_PI8, SIN_PI8, -0.5],
            [SIN_PI8, -COS_PI8, -0.5],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t h tdg h t": array(
        [
            [0.25, ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, -0.25, -COS_PI8],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg h tdg": array(
        [
            [ISQRT2_M, -0.25, -COS_PI8],
            [-0.25, -ISQRT2_P, SIN_PI8],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg tdg h": array(
        [
            [0.5, 0.5, -ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "t h t h tdg tdg tdg": array(
        [
            [-SIN_PI8, COS_PI8, 0.5],
            [-COS_PI8, SIN_PI8, -0.5],
            [-0.5, -0.5, ISQRT2],
        ]
    ),
    "t h t t t h t": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [SIN_PI8, -COS_PI8, 0.5],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "t h t t t h tdg": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [-COS_PI8, SIN_PI8, 0.5],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "t h tdg h t h t": array(
        [
            [0.25, -ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h tdg h t h tdg": array(
        [
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [-0.25, ISQRT2_M, COS_PI8],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h tdg h t t h": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg h t t t": array(
        [
            [-COS_PI8, SIN_PI8, 0.5],
            [SIN_PI8, -COS_PI8, 0.5],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg h tdg h t": array(
        [
            [-ISQRT2_M, 0.25, -COS_PI8],
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg h tdg": array(
        [
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, -0.25, COS_PI8],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg tdg tdg": array(
        [
            [-SIN_PI8, COS_PI8, -0.5],
            [-COS_PI8, SIN_PI8, 0.5],
            [0.5, 0.5, ISQRT2],
        ]
    ),
    "t h tdg tdg h t h": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [-0.5, 0.5, ISQRT2],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg tdg h t t": array(
        [
            [0.0, 0.0, 1.0],
            [ISQRT2, -ISQRT2, 0.0],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "t h tdg tdg tdg h t": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [SIN_PI8, -COS_PI8, -0.5],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg tdg tdg h tdg": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [-COS_PI8, SIN_PI8, -0.5],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t t h t h t h": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [-0.5, ISQRT2, -0.5],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h t t": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [0.0, -1.0, 0.0],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h t h tdg h": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [-0.5, -ISQRT2, -0.5],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h t h tdg tdg": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [0.0, 1.0, 0.0],
            [-ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg h t h": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [-0.5, ISQRT2, 0.5],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t t": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [0.0, -1.0, 0.0],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg h tdg h": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [-0.5, -ISQRT2, 0.5],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h tdg h tdg tdg": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [0.0, 1.0, 0.0],
            [ISQRT2, 0.0, ISQRT2],
        ]
    ),
    "t t h tdg tdg h t": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [0.0, -ISQRT2, -ISQRT2],
            [1.0, 0.0, 0.0],
        ]
    ),
    "t t h tdg tdg tdg h": array(
        [
            [0.0, -1.0, 0.0],
            [-ISQRT2, 0.0, -ISQRT2],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "t t t h t h t": array(
        [
            [-COS_PI8, -SIN_PI8, -0.5],
            [-SIN_PI8, -COS_PI8, 0.5],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "t t t h t h tdg": array(
        [
            [-SIN_PI8, -COS_PI8, 0.5],
            [COS_PI8, SIN_PI8, 0.5],
            [-0.5, 0.5, ISQRT2],
        ]
    ),
    "t t t h tdg h t": array(
        [
            [-COS_PI8, -SIN_PI8, 0.5],
            [-SIN_PI8, -COS_PI8, -0.5],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "t t t h tdg h tdg": array(
        [
            [-SIN_PI8, -COS_PI8, -0.5],
            [COS_PI8, SIN_PI8, -0.5],
            [0.5, -0.5, ISQRT2],
        ]
    ),
    "t t t h tdg tdg tdg": array(
        [
            [-0.5, 0.5, ISQRT2],
            [0.5, -0.5, ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
        ]
    ),
    "tdg h t h t h t": array(
        [
            [-0.25, ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, -0.25, -COS_PI8],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h t h tdg": array(
        [
            [-ISQRT2_M, -0.25, -COS_PI8],
            [0.25, -ISQRT2_P, SIN_PI8],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h tdg h t": array(
        [
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [0.25, ISQRT2_M, -COS_PI8],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg h tdg": array(
        [
            [0.25, ISQRT2_M, -COS_PI8],
            [ISQRT2_P, -0.25, SIN_PI8],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg tdg h": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t t h tdg h": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [0.5, 0.5, -ISQRT2],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t t h tdg tdg": array(
        [
            [0.0, 0.0, 1.0],
            [-ISQRT2, -ISQRT2, 0.0],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "tdg h t t t h t": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [COS_PI8, SIN_PI8, 0.5],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t t t h tdg": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [-SIN_PI8, -COS_PI8, 0.5],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h tdg h t h t": array(
        [
            [ISQRT2_M, 0.25, -COS_PI8],
            [0.25, -ISQRT2_P, -SIN_PI8],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h tdg h t h tdg": array(
        [
            [0.25, -ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, -0.25, COS_PI8],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h tdg h t t h": array(
        [
            [0.5, -0.5, -ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "tdg h tdg h tdg h t": array(
        [
            [-0.25, -ISQRT2_M, -COS_PI8],
            [ISQRT2_P, -0.25, -SIN_PI8],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h tdg h tdg h tdg": array(
        [
            [ISQRT2_P, -0.25, -SIN_PI8],
            [0.25, ISQRT2_M, COS_PI8],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h tdg tdg tdg h t": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [COS_PI8, SIN_PI8, -0.5],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "tdg h tdg tdg tdg h tdg": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [-SIN_PI8, -COS_PI8, -0.5],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "tdg tdg h t h t h": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [0.5, -ISQRT2, -0.5],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t h tdg h": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [0.5, ISQRT2, -0.5],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h t t h tdg": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [0.0, -ISQRT2, ISQRT2],
            [1.0, 0.0, 0.0],
        ]
    ),
    "tdg tdg h t t t h": array(
        [
            [0.0, 1.0, 0.0],
            [ISQRT2, 0.0, ISQRT2],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "tdg tdg h tdg h t h": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [0.5, -ISQRT2, 0.5],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h tdg h": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [0.5, ISQRT2, 0.5],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg tdg h t t t": array(
        [
            [-0.5, -0.5, ISQRT2],
            [-0.5, -0.5, -ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
        ]
    ),
    "h t h t h t h t": array(
        [
            [SIN_PI8, -0.25, -ISQRT2_P],
            [COS_PI8, -ISQRT2_M, 0.25],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t h tdg": array(
        [
            [COS_PI8, -ISQRT2_M, 0.25],
            [-SIN_PI8, 0.25, ISQRT2_P],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t t t": array(
        [
            [0.5, -SIN_PI8, COS_PI8],
            [0.5, COS_PI8, -SIN_PI8],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t h tdg h t": array(
        [
            [SIN_PI8, -ISQRT2_P, -0.25],
            [COS_PI8, 0.25, -ISQRT2_M],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg h tdg": array(
        [
            [COS_PI8, 0.25, -ISQRT2_M],
            [-SIN_PI8, ISQRT2_P, 0.25],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg tdg h": array(
        [
            [ISQRT2, -0.5, -0.5],
            [0.0, ISQRT2, -ISQRT2],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t h tdg tdg tdg": array(
        [
            [-0.5, -COS_PI8, SIN_PI8],
            [0.5, -SIN_PI8, COS_PI8],
            [-ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t t t h t": array(
        [
            [0.5, SIN_PI8, -COS_PI8],
            [-0.5, COS_PI8, -SIN_PI8],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t t t h tdg": array(
        [
            [-0.5, COS_PI8, -SIN_PI8],
            [-0.5, -SIN_PI8, COS_PI8],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h tdg h t h t": array(
        [
            [COS_PI8, ISQRT2_M, -0.25],
            [SIN_PI8, 0.25, ISQRT2_P],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h tdg h t h tdg": array(
        [
            [SIN_PI8, 0.25, ISQRT2_P],
            [-COS_PI8, -ISQRT2_M, 0.25],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h tdg h t t h": array(
        [
            [ISQRT2, 0.5, 0.5],
            [0.0, -ISQRT2, ISQRT2],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg h t t t": array(
        [
            [-0.5, -SIN_PI8, COS_PI8],
            [-0.5, COS_PI8, -SIN_PI8],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg h tdg h t": array(
        [
            [COS_PI8, -0.25, ISQRT2_M],
            [SIN_PI8, ISQRT2_P, 0.25],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h tdg h tdg h tdg": array(
        [
            [SIN_PI8, ISQRT2_P, 0.25],
            [-COS_PI8, 0.25, -ISQRT2_M],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h tdg h tdg tdg tdg": array(
        [
            [0.5, -COS_PI8, SIN_PI8],
            [-0.5, -SIN_PI8, COS_PI8],
            [-ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg tdg h t h": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [-ISQRT2, -0.5, 0.5],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h t h tdg tdg h t t": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, ISQRT2, -ISQRT2],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h t h tdg tdg tdg h t": array(
        [
            [-0.5, SIN_PI8, -COS_PI8],
            [0.5, COS_PI8, -SIN_PI8],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg tdg tdg h tdg": array(
        [
            [0.5, COS_PI8, -SIN_PI8],
            [0.5, -SIN_PI8, COS_PI8],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t t h tdg h t h": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [-0.5, -ISQRT2, 0.5],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "h t t h tdg h t t": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [0.0, 1.0, 0.0],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h t t h tdg h tdg h": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [-0.5, ISQRT2, 0.5],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "h t t h tdg h tdg tdg": array(
        [
            [ISQRT2, 0.0, -ISQRT2],
            [0.0, -1.0, 0.0],
            [-ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h t t h tdg tdg h t": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "h t t t h t h t": array(
        [
            [0.5, SIN_PI8, COS_PI8],
            [-0.5, COS_PI8, SIN_PI8],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h t t t h t h tdg": array(
        [
            [-0.5, COS_PI8, SIN_PI8],
            [-0.5, -SIN_PI8, -COS_PI8],
            [-ISQRT2, -0.5, 0.5],
        ]
    ),
    "h t t t h tdg h t": array(
        [
            [-0.5, SIN_PI8, COS_PI8],
            [0.5, COS_PI8, SIN_PI8],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h t t t h tdg h tdg": array(
        [
            [0.5, COS_PI8, SIN_PI8],
            [0.5, -SIN_PI8, -COS_PI8],
            [-ISQRT2, 0.5, -0.5],
        ]
    ),
    "h t t t h tdg tdg tdg": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [-ISQRT2, 0.5, -0.5],
            [0.0, -ISQRT2, -ISQRT2],
        ]
    ),
    "h tdg h t h t h t": array(
        [
            [SIN_PI8, -ISQRT2_P, 0.25],
            [COS_PI8, 0.25, ISQRT2_M],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h t h tdg": array(
        [
            [COS_PI8, 0.25, ISQRT2_M],
            [-SIN_PI8, ISQRT2_P, -0.25],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h tdg h t": array(
        [
            [SIN_PI8, -0.25, ISQRT2_P],
            [COS_PI8, -ISQRT2_M, -0.25],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg h tdg": array(
        [
            [COS_PI8, -ISQRT2_M, -0.25],
            [-SIN_PI8, 0.25, -ISQRT2_P],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg tdg h": array(
        [
            [ISQRT2, -0.5, 0.5],
            [0.0, -ISQRT2, -ISQRT2],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t h tdg h": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [ISQRT2, -0.5, -0.5],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h tdg h t t h tdg tdg": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, ISQRT2, ISQRT2],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "h tdg h t t t h t": array(
        [
            [0.5, -COS_PI8, -SIN_PI8],
            [-0.5, -SIN_PI8, -COS_PI8],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t t h tdg": array(
        [
            [-0.5, -SIN_PI8, -COS_PI8],
            [-0.5, COS_PI8, SIN_PI8],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h tdg h t h t": array(
        [
            [COS_PI8, -0.25, -ISQRT2_M],
            [SIN_PI8, ISQRT2_P, -0.25],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h tdg h t h tdg": array(
        [
            [SIN_PI8, ISQRT2_P, -0.25],
            [-COS_PI8, 0.25, ISQRT2_M],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h tdg h t t h": array(
        [
            [ISQRT2, 0.5, -0.5],
            [0.0, ISQRT2, ISQRT2],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg h tdg h t": array(
        [
            [COS_PI8, ISQRT2_M, 0.25],
            [SIN_PI8, 0.25, -ISQRT2_P],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h tdg h tdg h tdg": array(
        [
            [SIN_PI8, 0.25, -ISQRT2_P],
            [-COS_PI8, -ISQRT2_M, -0.25],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h tdg tdg tdg h t": array(
        [
            [-0.5, -COS_PI8, -SIN_PI8],
            [0.5, -SIN_PI8, -COS_PI8],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg tdg tdg h tdg": array(
        [
            [0.5, -SIN_PI8, -COS_PI8],
            [0.5, COS_PI8, SIN_PI8],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg tdg h t h t h": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [0.5, ISQRT2, -0.5],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "h tdg tdg h t h tdg h": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [0.5, -ISQRT2, -0.5],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "h tdg tdg h t t h tdg": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [-ISQRT2, ISQRT2, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "h tdg tdg tdg h t t t": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [ISQRT2, 0.5, 0.5],
            [0.0, ISQRT2, -ISQRT2],
        ]
    ),
    "t h t h t h t h": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [0.25, -ISQRT2_M, COS_PI8],
            [-ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t h t h t h t t": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [0.5, 0.5, -ISQRT2],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t h tdg h": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [ISQRT2_P, 0.25, -SIN_PI8],
            [0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t h t h t h tdg tdg": array(
        [
            [-COS_PI8, SIN_PI8, -0.5],
            [-0.5, -0.5, ISQRT2],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t t t h": array(
        [
            [0.5, 0.5, -ISQRT2],
            [-SIN_PI8, COS_PI8, 0.5],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h t h tdg h t h": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [-ISQRT2_M, 0.25, COS_PI8],
            [-0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h t h tdg h t t": array(
        [
            [-SIN_PI8, COS_PI8, 0.5],
            [0.5, 0.5, -ISQRT2],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg h tdg h": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [0.25, ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t h t h tdg h tdg tdg": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [-0.5, -0.5, ISQRT2],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg tdg h t": array(
        [
            [-SIN_PI8, COS_PI8, -0.5],
            [COS_PI8, -SIN_PI8, -0.5],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "t h t h tdg tdg tdg h": array(
        [
            [0.5, 0.5, -ISQRT2],
            [COS_PI8, -SIN_PI8, 0.5],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t h t t t h t h": array(
        [
            [0.5, 0.5, ISQRT2],
            [-SIN_PI8, COS_PI8, -0.5],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h t t t h tdg h": array(
        [
            [0.5, 0.5, ISQRT2],
            [COS_PI8, -SIN_PI8, -0.5],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h t t t h tdg tdg": array(
        [
            [-0.5, -0.5, ISQRT2],
            [-ISQRT2, ISQRT2, 0.0],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "t h tdg h t h t h": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [ISQRT2_P, 0.25, SIN_PI8],
            [-0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t h tdg h t h t t": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [-0.5, -0.5, -ISQRT2],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h tdg h t h tdg h": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [0.25, -ISQRT2_M, -COS_PI8],
            [ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t h tdg h t h tdg tdg": array(
        [
            [-COS_PI8, SIN_PI8, 0.5],
            [0.5, 0.5, ISQRT2],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h tdg h t t h tdg": array(
        [
            [-COS_PI8, SIN_PI8, -0.5],
            [-SIN_PI8, COS_PI8, 0.5],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg h t t t h": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [-SIN_PI8, COS_PI8, -0.5],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "t h tdg h tdg h t h": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [0.25, ISQRT2_P, SIN_PI8],
            [ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t h tdg h tdg h t t": array(
        [
            [-SIN_PI8, COS_PI8, -0.5],
            [-0.5, -0.5, -ISQRT2],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg h tdg h": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [-ISQRT2_M, 0.25, -COS_PI8],
            [0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h tdg h tdg h tdg tdg": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [0.5, 0.5, ISQRT2],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg tdg tdg h": array(
        [
            [-0.5, -0.5, -ISQRT2],
            [COS_PI8, -SIN_PI8, -0.5],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h tdg tdg h t h t": array(
        [
            [-SIN_PI8, -COS_PI8, -0.5],
            [-COS_PI8, -SIN_PI8, 0.5],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg tdg h t h tdg": array(
        [
            [-COS_PI8, -SIN_PI8, 0.5],
            [SIN_PI8, COS_PI8, 0.5],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg tdg tdg h t h": array(
        [
            [-0.5, -0.5, ISQRT2],
            [-SIN_PI8, COS_PI8, 0.5],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h tdg tdg tdg h t t": array(
        [
            [0.5, 0.5, ISQRT2],
            [ISQRT2, -ISQRT2, 0.0],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t h tdg tdg tdg h tdg h": array(
        [
            [-0.5, -0.5, ISQRT2],
            [COS_PI8, -SIN_PI8, 0.5],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t t h t h t h t": array(
        [
            [COS_PI8, -0.5, -SIN_PI8],
            [SIN_PI8, 0.5, -COS_PI8],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h t h tdg": array(
        [
            [SIN_PI8, 0.5, -COS_PI8],
            [-COS_PI8, 0.5, SIN_PI8],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h tdg h t": array(
        [
            [COS_PI8, 0.5, -SIN_PI8],
            [SIN_PI8, -0.5, -COS_PI8],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h t h tdg h tdg": array(
        [
            [SIN_PI8, -0.5, -COS_PI8],
            [-COS_PI8, -0.5, SIN_PI8],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t h t": array(
        [
            [-SIN_PI8, -0.5, -COS_PI8],
            [-COS_PI8, 0.5, -SIN_PI8],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t h tdg": array(
        [
            [-COS_PI8, 0.5, -SIN_PI8],
            [SIN_PI8, 0.5, COS_PI8],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t t h": array(
        [
            [-ISQRT2, 0.0, -ISQRT2],
            [0.0, 1.0, 0.0],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "t t h tdg h tdg h t": array(
        [
            [-SIN_PI8, 0.5, -COS_PI8],
            [-COS_PI8, -0.5, -SIN_PI8],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h tdg h tdg h tdg": array(
        [
            [-COS_PI8, -0.5, -SIN_PI8],
            [SIN_PI8, -0.5, COS_PI8],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h tdg tdg h t t": array(
        [
            [0.0, 0.0, 1.0e00],
            [0.0, -1.0, 0.0],
            [1.0e00, 0.0, 0.0],
        ]
    ),
    "t t h tdg tdg tdg h t": array(
        [
            [0.5, -ISQRT2, 0.5],
            [-0.5, -ISQRT2, -0.5],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "t t h tdg tdg tdg h tdg": array(
        [
            [-0.5, -ISQRT2, -0.5],
            [-0.5, ISQRT2, -0.5],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "t t t h t h t h": array(
        [
            [0.5, -0.5, -ISQRT2],
            [SIN_PI8, COS_PI8, -0.5],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t t t h t h tdg h": array(
        [
            [0.5, -0.5, -ISQRT2],
            [-COS_PI8, -SIN_PI8, -0.5],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t t t h tdg h t h": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [SIN_PI8, COS_PI8, 0.5],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t t t h tdg h tdg h": array(
        [
            [-0.5, 0.5, -ISQRT2],
            [-COS_PI8, -SIN_PI8, 0.5],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t t t h tdg tdg tdg h": array(
        [
            [-ISQRT2, -ISQRT2, 0.0],
            [-0.5, 0.5, -ISQRT2],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t h t h t h": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [ISQRT2_M, 0.25, COS_PI8],
            [0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h t h t h t t": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [-0.5, 0.5, -ISQRT2],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h t h tdg h": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [-0.25, ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "tdg h t h t h tdg tdg": array(
        [
            [-SIN_PI8, -COS_PI8, -0.5],
            [0.5, -0.5, ISQRT2],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h tdg h t h": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [-0.25, -ISQRT2_M, COS_PI8],
            [ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "tdg h t h tdg h t t": array(
        [
            [-COS_PI8, -SIN_PI8, 0.5],
            [-0.5, 0.5, -ISQRT2],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg h tdg h": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [-0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h t h tdg h tdg tdg": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [0.5, -0.5, ISQRT2],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg tdg h t": array(
        [
            [-COS_PI8, -SIN_PI8, -0.5],
            [SIN_PI8, COS_PI8, -0.5],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t t h tdg h t": array(
        [
            [-COS_PI8, SIN_PI8, 0.5],
            [-SIN_PI8, COS_PI8, -0.5],
            [-0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t t t h t h": array(
        [
            [-0.5, 0.5, ISQRT2],
            [-COS_PI8, -SIN_PI8, -0.5],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h t t t h tdg h": array(
        [
            [-0.5, 0.5, ISQRT2],
            [SIN_PI8, COS_PI8, -0.5],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h t t t h tdg tdg": array(
        [
            [0.5, -0.5, ISQRT2],
            [-ISQRT2, -ISQRT2, 0.0],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h tdg h t h t h": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [-0.25, ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "tdg h tdg h t h t t": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [0.5, -0.5, -ISQRT2],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h tdg h t h tdg h": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [ISQRT2_M, 0.25, -COS_PI8],
            [-0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h tdg h t h tdg tdg": array(
        [
            [-SIN_PI8, -COS_PI8, 0.5],
            [-0.5, 0.5, ISQRT2],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h tdg h tdg h t h": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [-ISQRT2_P, 0.25, SIN_PI8],
            [0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h tdg h tdg h t t": array(
        [
            [-COS_PI8, -SIN_PI8, -0.5],
            [0.5, -0.5, -ISQRT2],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h tdg h tdg h tdg h": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [-0.25, -ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "tdg h tdg h tdg h tdg tdg": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [-0.5, 0.5, ISQRT2],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h tdg tdg tdg h t h": array(
        [
            [0.5, -0.5, ISQRT2],
            [-COS_PI8, -SIN_PI8, 0.5],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h tdg tdg tdg h t t": array(
        [
            [-0.5, 0.5, ISQRT2],
            [ISQRT2, ISQRT2, 0.0],
            [-0.5, 0.5, -ISQRT2],
        ]
    ),
    "tdg h tdg tdg tdg h tdg h": array(
        [
            [0.5, -0.5, ISQRT2],
            [SIN_PI8, COS_PI8, 0.5],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg tdg h t h t h t": array(
        [
            [-COS_PI8, 0.5, -SIN_PI8],
            [-SIN_PI8, -0.5, -COS_PI8],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t h t h tdg": array(
        [
            [-SIN_PI8, -0.5, -COS_PI8],
            [COS_PI8, -0.5, SIN_PI8],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t h tdg h t": array(
        [
            [-COS_PI8, -0.5, -SIN_PI8],
            [-SIN_PI8, 0.5, -COS_PI8],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h t h tdg h tdg": array(
        [
            [-SIN_PI8, 0.5, -COS_PI8],
            [COS_PI8, 0.5, SIN_PI8],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h t t t h t": array(
        [
            [-0.5, ISQRT2, -0.5],
            [0.5, ISQRT2, 0.5],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "tdg tdg h t t t h tdg": array(
        [
            [0.5, ISQRT2, 0.5],
            [0.5, -ISQRT2, 0.5],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "tdg tdg h tdg h t h t": array(
        [
            [SIN_PI8, 0.5, -COS_PI8],
            [COS_PI8, -0.5, -SIN_PI8],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h t h tdg": array(
        [
            [COS_PI8, -0.5, -SIN_PI8],
            [-SIN_PI8, -0.5, COS_PI8],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h tdg h t": array(
        [
            [SIN_PI8, -0.5, -COS_PI8],
            [COS_PI8, 0.5, -SIN_PI8],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h tdg h tdg h tdg": array(
        [
            [COS_PI8, 0.5, -SIN_PI8],
            [-SIN_PI8, 0.5, COS_PI8],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg tdg h t t t h": array(
        [
            [-ISQRT2, ISQRT2, 0.0],
            [0.5, 0.5, ISQRT2],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "h t h t h t h t h": array(
        [
            [0.5, COS_PI8, -SIN_PI8],
            [-COS_PI8, ISQRT2_M, -0.25],
            [-SIN_PI8, 0.25, ISQRT2_P],
        ]
    ),
    "h t h t h t h t t": array(
        [
            [-0.5, SIN_PI8, -COS_PI8],
            [ISQRT2, -0.5, -0.5],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t h tdg h": array(
        [
            [0.5, COS_PI8, -SIN_PI8],
            [SIN_PI8, -0.25, -ISQRT2_P],
            [-COS_PI8, ISQRT2_M, -0.25],
        ]
    ),
    "h t h t h t h tdg tdg": array(
        [
            [0.5, -SIN_PI8, COS_PI8],
            [-ISQRT2, 0.5, 0.5],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t t t h": array(
        [
            [ISQRT2, -0.5, -0.5],
            [-0.5, -COS_PI8, SIN_PI8],
            [-0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h t h tdg h t h": array(
        [
            [-0.5, SIN_PI8, -COS_PI8],
            [-COS_PI8, -0.25, ISQRT2_M],
            [-SIN_PI8, ISQRT2_P, 0.25],
        ]
    ),
    "h t h t h tdg h t t": array(
        [
            [-0.5, -COS_PI8, SIN_PI8],
            [ISQRT2, -0.5, -0.5],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg h tdg h": array(
        [
            [-0.5, SIN_PI8, -COS_PI8],
            [SIN_PI8, -ISQRT2_P, -0.25],
            [-COS_PI8, -0.25, ISQRT2_M],
        ]
    ),
    "h t h t h tdg h tdg tdg": array(
        [
            [0.5, COS_PI8, -SIN_PI8],
            [-ISQRT2, 0.5, 0.5],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg tdg h t": array(
        [
            [0.5, -COS_PI8, SIN_PI8],
            [0.5, SIN_PI8, -COS_PI8],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h t h tdg tdg tdg h": array(
        [
            [ISQRT2, -0.5, -0.5],
            [-0.5, SIN_PI8, -COS_PI8],
            [0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h t t t h t h": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [0.5, -COS_PI8, SIN_PI8],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t t t h tdg h": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [0.5, SIN_PI8, -COS_PI8],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t t t h tdg tdg": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [0.0, -ISQRT2, ISQRT2],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h t h tdg h t h t h": array(
        [
            [-0.5, COS_PI8, -SIN_PI8],
            [-SIN_PI8, -0.25, -ISQRT2_P],
            [-COS_PI8, -ISQRT2_M, 0.25],
        ]
    ),
    "h t h tdg h t h t t": array(
        [
            [0.5, SIN_PI8, -COS_PI8],
            [ISQRT2, 0.5, 0.5],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h tdg h t h tdg h": array(
        [
            [-0.5, COS_PI8, -SIN_PI8],
            [COS_PI8, ISQRT2_M, -0.25],
            [-SIN_PI8, -0.25, -ISQRT2_P],
        ]
    ),
    "h t h tdg h t h tdg tdg": array(
        [
            [-0.5, -SIN_PI8, COS_PI8],
            [-ISQRT2, -0.5, -0.5],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h tdg h t t h tdg": array(
        [
            [0.5, -SIN_PI8, COS_PI8],
            [-0.5, -COS_PI8, SIN_PI8],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg h t t t h": array(
        [
            [ISQRT2, 0.5, 0.5],
            [0.5, -COS_PI8, SIN_PI8],
            [0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h tdg h tdg h t h": array(
        [
            [0.5, SIN_PI8, -COS_PI8],
            [-SIN_PI8, -ISQRT2_P, -0.25],
            [-COS_PI8, 0.25, -ISQRT2_M],
        ]
    ),
    "h t h tdg h tdg h t t": array(
        [
            [0.5, -COS_PI8, SIN_PI8],
            [ISQRT2, 0.5, 0.5],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h tdg h tdg h tdg h": array(
        [
            [0.5, SIN_PI8, -COS_PI8],
            [COS_PI8, -0.25, ISQRT2_M],
            [-SIN_PI8, -ISQRT2_P, -0.25],
        ]
    ),
    "h t h tdg h tdg h tdg tdg": array(
        [
            [-0.5, COS_PI8, -SIN_PI8],
            [-ISQRT2, -0.5, -0.5],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h tdg h tdg tdg tdg h": array(
        [
            [ISQRT2, 0.5, 0.5],
            [0.5, SIN_PI8, -COS_PI8],
            [-0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h tdg tdg h t h t": array(
        [
            [0.5, COS_PI8, SIN_PI8],
            [-0.5, SIN_PI8, COS_PI8],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h t h tdg tdg h t h tdg": array(
        [
            [-0.5, SIN_PI8, COS_PI8],
            [-0.5, -COS_PI8, -SIN_PI8],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h t h tdg tdg tdg h t h": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [-0.5, -COS_PI8, SIN_PI8],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h tdg tdg tdg h t t": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [0.0, ISQRT2, -ISQRT2],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t h tdg tdg tdg h tdg h": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [-0.5, SIN_PI8, -COS_PI8],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t t h tdg h t h t": array(
        [
            [COS_PI8, 0.5, SIN_PI8],
            [SIN_PI8, -0.5, COS_PI8],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "h t t h tdg h t h tdg": array(
        [
            [SIN_PI8, -0.5, COS_PI8],
            [-COS_PI8, -0.5, -SIN_PI8],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "h t t h tdg h t t h": array(
        [
            [ISQRT2, 0.0, ISQRT2],
            [0.0, -1.0, 0.0],
            [ISQRT2, 0.0, -ISQRT2],
        ]
    ),
    "h t t h tdg h tdg h t": array(
        [
            [COS_PI8, -0.5, SIN_PI8],
            [SIN_PI8, 0.5, COS_PI8],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "h t t h tdg h tdg h tdg": array(
        [
            [SIN_PI8, 0.5, COS_PI8],
            [-COS_PI8, 0.5, -SIN_PI8],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "h t t h tdg tdg h t t": array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "h t t t h t h t h": array(
        [
            [ISQRT2, 0.5, -0.5],
            [0.5, -COS_PI8, -SIN_PI8],
            [-0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t t t h t h tdg h": array(
        [
            [ISQRT2, 0.5, -0.5],
            [0.5, SIN_PI8, COS_PI8],
            [0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t t t h tdg h t h": array(
        [
            [ISQRT2, -0.5, 0.5],
            [-0.5, -COS_PI8, -SIN_PI8],
            [0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t t t h tdg h tdg h": array(
        [
            [ISQRT2, -0.5, 0.5],
            [-0.5, SIN_PI8, COS_PI8],
            [-0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t t t h tdg tdg tdg h": array(
        [
            [0.0, ISQRT2, ISQRT2],
            [ISQRT2, -0.5, 0.5],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t h t h t h": array(
        [
            [0.5, -SIN_PI8, -COS_PI8],
            [-COS_PI8, -0.25, -ISQRT2_M],
            [-SIN_PI8, ISQRT2_P, -0.25],
        ]
    ),
    "h tdg h t h t h t t": array(
        [
            [-0.5, -COS_PI8, -SIN_PI8],
            [ISQRT2, -0.5, 0.5],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h t h tdg h": array(
        [
            [0.5, -SIN_PI8, -COS_PI8],
            [SIN_PI8, -ISQRT2_P, 0.25],
            [-COS_PI8, -0.25, -ISQRT2_M],
        ]
    ),
    "h tdg h t h t h tdg tdg": array(
        [
            [0.5, COS_PI8, SIN_PI8],
            [-ISQRT2, 0.5, -0.5],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h tdg h t h": array(
        [
            [-0.5, -COS_PI8, -SIN_PI8],
            [-COS_PI8, ISQRT2_M, 0.25],
            [-SIN_PI8, 0.25, -ISQRT2_P],
        ]
    ),
    "h tdg h t h tdg h t t": array(
        [
            [-0.5, SIN_PI8, COS_PI8],
            [ISQRT2, -0.5, 0.5],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg h tdg h": array(
        [
            [-0.5, -COS_PI8, -SIN_PI8],
            [SIN_PI8, -0.25, ISQRT2_P],
            [-COS_PI8, ISQRT2_M, 0.25],
        ]
    ),
    "h tdg h t h tdg h tdg tdg": array(
        [
            [0.5, -SIN_PI8, -COS_PI8],
            [-ISQRT2, 0.5, -0.5],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg tdg h t": array(
        [
            [0.5, SIN_PI8, COS_PI8],
            [0.5, -COS_PI8, -SIN_PI8],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t t h tdg h t": array(
        [
            [-0.5, -SIN_PI8, COS_PI8],
            [0.5, -COS_PI8, SIN_PI8],
            [ISQRT2, 0.5, 0.5],
        ]
    ),
    "h tdg h t t t h t h": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [0.5, SIN_PI8, COS_PI8],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t t t h tdg h": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [0.5, -COS_PI8, -SIN_PI8],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t t t h tdg tdg": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [0.0, ISQRT2, ISQRT2],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h tdg h t h t h": array(
        [
            [-0.5, -SIN_PI8, -COS_PI8],
            [-SIN_PI8, -ISQRT2_P, 0.25],
            [-COS_PI8, 0.25, ISQRT2_M],
        ]
    ),
    "h tdg h tdg h t h t t": array(
        [
            [0.5, -COS_PI8, -SIN_PI8],
            [ISQRT2, 0.5, -0.5],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h tdg h t h tdg h": array(
        [
            [-0.5, -SIN_PI8, -COS_PI8],
            [COS_PI8, -0.25, -ISQRT2_M],
            [-SIN_PI8, -ISQRT2_P, 0.25],
        ]
    ),
    "h tdg h tdg h t h tdg tdg": array(
        [
            [-0.5, COS_PI8, SIN_PI8],
            [-ISQRT2, -0.5, 0.5],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h tdg h tdg h t h": array(
        [
            [0.5, -COS_PI8, -SIN_PI8],
            [-SIN_PI8, -0.25, ISQRT2_P],
            [-COS_PI8, -ISQRT2_M, -0.25],
        ]
    ),
    "h tdg h tdg h tdg h t t": array(
        [
            [0.5, SIN_PI8, COS_PI8],
            [ISQRT2, 0.5, -0.5],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h tdg h tdg h tdg h": array(
        [
            [0.5, -COS_PI8, -SIN_PI8],
            [COS_PI8, ISQRT2_M, 0.25],
            [-SIN_PI8, -0.25, ISQRT2_P],
        ]
    ),
    "h tdg h tdg h tdg h tdg tdg": array(
        [
            [-0.5, -SIN_PI8, -COS_PI8],
            [-ISQRT2, -0.5, 0.5],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h tdg tdg tdg h t h": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [-0.5, SIN_PI8, COS_PI8],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h tdg tdg tdg h t t": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [0.0, -ISQRT2, -ISQRT2],
            [ISQRT2, -0.5, 0.5],
        ]
    ),
    "h tdg h tdg tdg tdg h tdg h": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [-0.5, -COS_PI8, -SIN_PI8],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg tdg h t h t h t": array(
        [
            [SIN_PI8, -0.5, COS_PI8],
            [COS_PI8, 0.5, SIN_PI8],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "h tdg tdg h t h t h tdg": array(
        [
            [COS_PI8, 0.5, SIN_PI8],
            [-SIN_PI8, 0.5, -COS_PI8],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "h tdg tdg h t h tdg h t": array(
        [
            [SIN_PI8, 0.5, COS_PI8],
            [COS_PI8, -0.5, SIN_PI8],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "h tdg tdg h t h tdg h tdg": array(
        [
            [COS_PI8, -0.5, SIN_PI8],
            [-SIN_PI8, -0.5, -COS_PI8],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "h tdg tdg tdg h t t t h": array(
        [
            [0.0, -ISQRT2, ISQRT2],
            [-ISQRT2, -0.5, -0.5],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "t h t h t h t h t": array(
        [
            [-TWO_M, -THREE_SQRT2_M, -ISQRT2_P],
            [THREE_SQRT2_M, -SIX_P, 0.25],
            [-ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t h t h t h t h tdg": array(
        [
            [THREE_SQRT2_M, -SIX_P, 0.25],
            [TWO_M, THREE_SQRT2_M, ISQRT2_P],
            [-ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t h t h t h t t t": array(
        [
            [0.25, -ISQRT2_M, COS_PI8],
            [ISQRT2_P, 0.25, -SIN_PI8],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t h tdg h t": array(
        [
            [-SIX_M, -THREE_SQRT2_P, -0.25],
            [THREE_SQRT2_P, -TWO_P, -ISQRT2_M],
            [0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t h t h t h tdg h tdg": array(
        [
            [THREE_SQRT2_P, -TWO_P, -ISQRT2_M],
            [SIX_M, THREE_SQRT2_P, 0.25],
            [0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t h t h t h tdg tdg h": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [0.5, 0.5, -ISQRT2],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h t h t h tdg tdg tdg": array(
        [
            [-ISQRT2_P, -0.25, SIN_PI8],
            [0.25, -ISQRT2_M, COS_PI8],
            [-SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h t h t t t h t": array(
        [
            [ISQRT2_M, -0.25, -COS_PI8],
            [0.25, ISQRT2_P, -SIN_PI8],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h t h t t t h tdg": array(
        [
            [0.25, ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, 0.25, COS_PI8],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h t h tdg h t h t": array(
        [
            [SIX_P, -THREE_SQRT2_M, -0.25],
            [THREE_SQRT2_M, TWO_M, ISQRT2_P],
            [-0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h t h tdg h t h tdg": array(
        [
            [THREE_SQRT2_M, TWO_M, ISQRT2_P],
            [-SIX_P, THREE_SQRT2_M, 0.25],
            [-0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h t h tdg h t t h": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [-0.5, -0.5, ISQRT2],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t h t h tdg h t t t": array(
        [
            [-ISQRT2_M, 0.25, COS_PI8],
            [0.25, ISQRT2_P, -SIN_PI8],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg h tdg h t": array(
        [
            [TWO_P, -THREE_SQRT2_P, ISQRT2_M],
            [THREE_SQRT2_P, SIX_M, 0.25],
            [-ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t h t h tdg h tdg h tdg": array(
        [
            [THREE_SQRT2_P, SIX_M, 0.25],
            [-TWO_P, THREE_SQRT2_P, -ISQRT2_M],
            [-ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t h t h tdg h tdg tdg tdg": array(
        [
            [-0.25, -ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, 0.25, COS_PI8],
            [-COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h t h tdg tdg h t h": array(
        [
            [0.5, 0.5, ISQRT2],
            [-COS_PI8, SIN_PI8, 0.5],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h t h tdg tdg tdg h t": array(
        [
            [-0.25, ISQRT2_M, -COS_PI8],
            [ISQRT2_P, 0.25, -SIN_PI8],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t h t h tdg tdg tdg h tdg": array(
        [
            [ISQRT2_P, 0.25, -SIN_PI8],
            [0.25, -ISQRT2_M, COS_PI8],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t h t t t h t h t": array(
        [
            [ISQRT2_M, -0.25, COS_PI8],
            [0.25, ISQRT2_P, SIN_PI8],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h t t t h t h tdg": array(
        [
            [0.25, ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, 0.25, -COS_PI8],
            [-COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t h t t t h tdg h t": array(
        [
            [-0.25, ISQRT2_M, COS_PI8],
            [ISQRT2_P, 0.25, SIN_PI8],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h t t t h tdg h tdg": array(
        [
            [ISQRT2_P, 0.25, SIN_PI8],
            [0.25, -ISQRT2_M, -COS_PI8],
            [-SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t h tdg h t h t h t": array(
        [
            [-SIX_M, -THREE_SQRT2_P, 0.25],
            [THREE_SQRT2_P, -TWO_P, ISQRT2_M],
            [-0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t h tdg h t h t h tdg": array(
        [
            [THREE_SQRT2_P, -TWO_P, ISQRT2_M],
            [SIX_M, THREE_SQRT2_P, -0.25],
            [-0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t h tdg h t h tdg h t": array(
        [
            [-TWO_M, -THREE_SQRT2_M, ISQRT2_P],
            [THREE_SQRT2_M, -SIX_P, -0.25],
            [ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t h tdg h t h tdg h tdg": array(
        [
            [THREE_SQRT2_M, -SIX_P, -0.25],
            [TWO_M, THREE_SQRT2_M, -ISQRT2_P],
            [ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t h tdg h t h tdg tdg h": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [-0.5, -0.5, -ISQRT2],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "t h tdg h t t h tdg h": array(
        [
            [-0.5, -0.5, ISQRT2],
            [SIN_PI8, -COS_PI8, -0.5],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h t t t h t": array(
        [
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, 0.25, -COS_PI8],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "t h tdg h t t t h tdg": array(
        [
            [-ISQRT2_M, 0.25, -COS_PI8],
            [0.25, ISQRT2_P, SIN_PI8],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "t h tdg h tdg h t h t": array(
        [
            [TWO_P, -THREE_SQRT2_P, -ISQRT2_M],
            [THREE_SQRT2_P, SIX_M, -0.25],
            [ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t h tdg h tdg h t h tdg": array(
        [
            [THREE_SQRT2_P, SIX_M, -0.25],
            [-TWO_P, THREE_SQRT2_P, ISQRT2_M],
            [ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t h tdg h tdg h t t h": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [0.5, 0.5, ISQRT2],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg h tdg h t": array(
        [
            [SIX_P, -THREE_SQRT2_M, 0.25],
            [THREE_SQRT2_M, TWO_M, -ISQRT2_P],
            [0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h tdg h tdg h tdg h tdg": array(
        [
            [THREE_SQRT2_M, TWO_M, -ISQRT2_P],
            [-SIX_P, THREE_SQRT2_M, -0.25],
            [0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h tdg h tdg tdg tdg h t": array(
        [
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [0.25, -ISQRT2_M, -COS_PI8],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg tdg tdg h tdg": array(
        [
            [0.25, -ISQRT2_M, -COS_PI8],
            [ISQRT2_P, 0.25, SIN_PI8],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h tdg tdg h t h t h": array(
        [
            [0.5, -0.5, ISQRT2],
            [COS_PI8, SIN_PI8, -0.5],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h tdg tdg h t h tdg h": array(
        [
            [0.5, -0.5, ISQRT2],
            [-SIN_PI8, -COS_PI8, -0.5],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h tdg tdg tdg h t t t": array(
        [
            [-SIN_PI8, COS_PI8, 0.5],
            [COS_PI8, -SIN_PI8, 0.5],
            [0.5, 0.5, -ISQRT2],
        ]
    ),
    "t t h t h t h t h": array(
        [
            [-0.5, -ISQRT2, -0.5],
            [-SIN_PI8, -0.5, COS_PI8],
            [-COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "t t h t h t h t t": array(
        [
            [0.5, -ISQRT2, 0.5],
            [ISQRT2, 0.0, -ISQRT2],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h t h tdg h": array(
        [
            [-0.5, -ISQRT2, -0.5],
            [COS_PI8, -0.5, -SIN_PI8],
            [-SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "t t h t h t h tdg tdg": array(
        [
            [-0.5, ISQRT2, -0.5],
            [-ISQRT2, 0.0, ISQRT2],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h tdg h t h": array(
        [
            [0.5, -ISQRT2, 0.5],
            [-SIN_PI8, 0.5, COS_PI8],
            [-COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "t t h t h tdg h t t": array(
        [
            [0.5, ISQRT2, 0.5],
            [ISQRT2, 0.0, -ISQRT2],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h t h tdg h tdg h": array(
        [
            [0.5, -ISQRT2, 0.5],
            [COS_PI8, 0.5, -SIN_PI8],
            [-SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "t t h t h tdg h tdg tdg": array(
        [
            [-0.5, -ISQRT2, -0.5],
            [-ISQRT2, 0.0, ISQRT2],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t h t h": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [COS_PI8, -0.5, SIN_PI8],
            [SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "t t h tdg h t h t t": array(
        [
            [0.5, -ISQRT2, -0.5],
            [-ISQRT2, 0.0, -ISQRT2],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t h tdg h": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [-SIN_PI8, -0.5, -COS_PI8],
            [COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "t t h tdg h t h tdg tdg": array(
        [
            [-0.5, ISQRT2, 0.5],
            [ISQRT2, 0.0, ISQRT2],
            [0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h tdg h t h": array(
        [
            [0.5, -ISQRT2, -0.5],
            [COS_PI8, 0.5, SIN_PI8],
            [SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "t t h tdg h tdg h t t": array(
        [
            [0.5, ISQRT2, -0.5],
            [-ISQRT2, 0.0, -ISQRT2],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h tdg h tdg h tdg h": array(
        [
            [0.5, -ISQRT2, -0.5],
            [-SIN_PI8, 0.5, -COS_PI8],
            [COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "t t h tdg h tdg h tdg tdg": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [ISQRT2, 0.0, ISQRT2],
            [-0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h tdg tdg tdg h t h": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [0.5, ISQRT2, 0.5],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg tdg tdg h tdg h": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [0.5, -ISQRT2, 0.5],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t t h t h t h t": array(
        [
            [0.25, -ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, 0.25, -COS_PI8],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t t t h t h t h tdg": array(
        [
            [ISQRT2_M, 0.25, -COS_PI8],
            [-0.25, ISQRT2_P, SIN_PI8],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t t t h t h tdg h t": array(
        [
            [ISQRT2_P, -0.25, -SIN_PI8],
            [-0.25, -ISQRT2_M, -COS_PI8],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t t t h t h tdg h tdg": array(
        [
            [-0.25, -ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, 0.25, SIN_PI8],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t t t h tdg h t h t": array(
        [
            [-ISQRT2_M, -0.25, -COS_PI8],
            [-0.25, ISQRT2_P, -SIN_PI8],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t t t h tdg h t h tdg": array(
        [
            [-0.25, ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, 0.25, COS_PI8],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t t t h tdg h tdg h t": array(
        [
            [0.25, ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t t t h tdg h tdg h tdg": array(
        [
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [-0.25, -ISQRT2_M, COS_PI8],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t t t h tdg tdg tdg h t": array(
        [
            [-SIN_PI8, -COS_PI8, 0.5],
            [-COS_PI8, -SIN_PI8, -0.5],
            [0.5, -0.5, -ISQRT2],
        ]
    ),
    "tdg h t h t h t h t": array(
        [
            [THREE_SQRT2_M, -TWO_M, -ISQRT2_P],
            [SIX_P, THREE_SQRT2_M, 0.25],
            [0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h t h t h t h tdg": array(
        [
            [SIX_P, THREE_SQRT2_M, 0.25],
            [-THREE_SQRT2_M, TWO_M, ISQRT2_P],
            [0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h t h t h t t t": array(
        [
            [ISQRT2_M, 0.25, COS_PI8],
            [-0.25, ISQRT2_P, -SIN_PI8],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h t h tdg h t": array(
        [
            [THREE_SQRT2_P, -SIX_M, -0.25],
            [TWO_P, THREE_SQRT2_P, -ISQRT2_M],
            [ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "tdg h t h t h tdg h tdg": array(
        [
            [TWO_P, THREE_SQRT2_P, -ISQRT2_M],
            [-THREE_SQRT2_P, SIX_M, 0.25],
            [ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "tdg h t h t h tdg tdg h": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [-0.5, 0.5, -ISQRT2],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "tdg h t h t h tdg tdg tdg": array(
        [
            [0.25, -ISQRT2_P, SIN_PI8],
            [ISQRT2_M, 0.25, COS_PI8],
            [-COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h tdg h t h t": array(
        [
            [THREE_SQRT2_M, SIX_P, -0.25],
            [-TWO_M, THREE_SQRT2_M, ISQRT2_P],
            [ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "tdg h t h tdg h t h tdg": array(
        [
            [-TWO_M, THREE_SQRT2_M, ISQRT2_P],
            [-THREE_SQRT2_M, -SIX_P, 0.25],
            [ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "tdg h t h tdg h t t h": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [0.5, -0.5, ISQRT2],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg h t t t": array(
        [
            [-0.25, -ISQRT2_M, COS_PI8],
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg h tdg h t": array(
        [
            [THREE_SQRT2_P, TWO_P, ISQRT2_M],
            [-SIX_M, THREE_SQRT2_P, 0.25],
            [-0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h t h tdg h tdg h tdg": array(
        [
            [-SIX_M, THREE_SQRT2_P, 0.25],
            [-THREE_SQRT2_P, -TWO_P, -ISQRT2_M],
            [-0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h t h tdg h tdg tdg tdg": array(
        [
            [ISQRT2_P, -0.25, SIN_PI8],
            [-0.25, -ISQRT2_M, COS_PI8],
            [-SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "tdg h t h tdg tdg h t h": array(
        [
            [-0.5, 0.5, ISQRT2],
            [-SIN_PI8, -COS_PI8, 0.5],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "tdg h t t h tdg h t h": array(
        [
            [0.5, 0.5, ISQRT2],
            [SIN_PI8, -COS_PI8, 0.5],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h t t t h t h t": array(
        [
            [0.25, ISQRT2_M, COS_PI8],
            [-ISQRT2_P, 0.25, SIN_PI8],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h t t t h t h tdg": array(
        [
            [-ISQRT2_P, 0.25, SIN_PI8],
            [-0.25, -ISQRT2_M, -COS_PI8],
            [-SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "tdg h t t t h tdg h t": array(
        [
            [-ISQRT2_M, -0.25, COS_PI8],
            [-0.25, ISQRT2_P, SIN_PI8],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h t t t h tdg h tdg": array(
        [
            [-0.25, ISQRT2_P, SIN_PI8],
            [ISQRT2_M, 0.25, -COS_PI8],
            [-COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h tdg h t h t h t": array(
        [
            [THREE_SQRT2_P, -SIX_M, 0.25],
            [TWO_P, THREE_SQRT2_P, ISQRT2_M],
            [-ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "tdg h tdg h t h t h tdg": array(
        [
            [TWO_P, THREE_SQRT2_P, ISQRT2_M],
            [-THREE_SQRT2_P, SIX_M, -0.25],
            [-ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "tdg h tdg h t h tdg h t": array(
        [
            [THREE_SQRT2_M, -TWO_M, ISQRT2_P],
            [SIX_P, THREE_SQRT2_M, -0.25],
            [-0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h tdg h t h tdg h tdg": array(
        [
            [SIX_P, THREE_SQRT2_M, -0.25],
            [-THREE_SQRT2_M, TWO_M, -ISQRT2_P],
            [-0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h tdg h t h tdg tdg h": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [0.5, -0.5, -ISQRT2],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "tdg h tdg h tdg h t h t": array(
        [
            [THREE_SQRT2_P, TWO_P, -ISQRT2_M],
            [-SIX_M, THREE_SQRT2_P, -0.25],
            [0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h tdg h tdg h t h tdg": array(
        [
            [-SIX_M, THREE_SQRT2_P, -0.25],
            [-THREE_SQRT2_P, -TWO_P, ISQRT2_M],
            [0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h tdg h tdg h t t h": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [-0.5, 0.5, ISQRT2],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "tdg h tdg h tdg h tdg h t": array(
        [
            [THREE_SQRT2_M, SIX_P, 0.25],
            [-TWO_M, THREE_SQRT2_M, -ISQRT2_P],
            [-ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "tdg h tdg h tdg h tdg h tdg": array(
        [
            [-TWO_M, THREE_SQRT2_M, -ISQRT2_P],
            [-THREE_SQRT2_M, -SIX_P, -0.25],
            [-ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "tdg tdg h t h t h t h": array(
        [
            [0.5, ISQRT2, -0.5],
            [SIN_PI8, 0.5, COS_PI8],
            [COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h t h t h t t": array(
        [
            [-0.5, ISQRT2, 0.5],
            [-ISQRT2, 0.0, -ISQRT2],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t h t h tdg h": array(
        [
            [0.5, ISQRT2, -0.5],
            [-COS_PI8, 0.5, -SIN_PI8],
            [SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "tdg tdg h t h t h tdg tdg": array(
        [
            [0.5, -ISQRT2, -0.5],
            [ISQRT2, 0.0, ISQRT2],
            [-0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t h tdg h t h": array(
        [
            [-0.5, ISQRT2, 0.5],
            [SIN_PI8, -0.5, COS_PI8],
            [COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h t h tdg h t t": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [-ISQRT2, 0.0, -ISQRT2],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h t h tdg h tdg h": array(
        [
            [-0.5, ISQRT2, 0.5],
            [-COS_PI8, -0.5, -SIN_PI8],
            [SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "tdg tdg h t h tdg h tdg tdg": array(
        [
            [0.5, ISQRT2, -0.5],
            [ISQRT2, 0.0, ISQRT2],
            [0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h t t t h t h": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [-0.5, -ISQRT2, -0.5],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t t t h tdg h": array(
        [
            [-ISQRT2, 0.0, ISQRT2],
            [-0.5, ISQRT2, -0.5],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h t h t h": array(
        [
            [0.5, ISQRT2, 0.5],
            [-COS_PI8, 0.5, SIN_PI8],
            [-SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "tdg tdg h tdg h t h t t": array(
        [
            [-0.5, ISQRT2, -0.5],
            [ISQRT2, 0.0, -ISQRT2],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h t h tdg h": array(
        [
            [0.5, ISQRT2, 0.5],
            [SIN_PI8, 0.5, -COS_PI8],
            [-COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h tdg h t h tdg tdg": array(
        [
            [0.5, -ISQRT2, 0.5],
            [-ISQRT2, 0.0, ISQRT2],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h tdg h t h": array(
        [
            [-0.5, ISQRT2, -0.5],
            [-COS_PI8, -0.5, SIN_PI8],
            [-SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "tdg tdg h tdg h tdg h t t": array(
        [
            [-0.5, -ISQRT2, -0.5],
            [ISQRT2, 0.0, -ISQRT2],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h tdg h tdg h tdg h": array(
        [
            [-0.5, ISQRT2, -0.5],
            [SIN_PI8, -0.5, -COS_PI8],
            [-COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h tdg h tdg h tdg tdg": array(
        [
            [0.5, ISQRT2, 0.5],
            [-ISQRT2, 0.0, ISQRT2],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "h t h t h t h t h t": array(
        [
            [ISQRT2_P, THREE_SQRT2_M, TWO_M],
            [-0.25, SIX_P, -THREE_SQRT2_M],
            [-SIN_PI8, 0.25, ISQRT2_P],
        ]
    ),
    "h t h t h t h t h tdg": array(
        [
            [-0.25, SIX_P, -THREE_SQRT2_M],
            [-ISQRT2_P, -THREE_SQRT2_M, -TWO_M],
            [-SIN_PI8, 0.25, ISQRT2_P],
        ]
    ),
    "h t h t h t h t t t": array(
        [
            [-COS_PI8, ISQRT2_M, -0.25],
            [SIN_PI8, -0.25, -ISQRT2_P],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t h tdg h t": array(
        [
            [0.25, THREE_SQRT2_P, SIX_M],
            [ISQRT2_M, TWO_P, -THREE_SQRT2_P],
            [-COS_PI8, ISQRT2_M, -0.25],
        ]
    ),
    "h t h t h t h tdg h tdg": array(
        [
            [ISQRT2_M, TWO_P, -THREE_SQRT2_P],
            [-0.25, -THREE_SQRT2_P, -SIX_M],
            [-COS_PI8, ISQRT2_M, -0.25],
        ]
    ),
    "h t h t h t h tdg tdg h": array(
        [
            [0.5, COS_PI8, -SIN_PI8],
            [ISQRT2, -0.5, -0.5],
            [-0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h t h t h tdg tdg tdg": array(
        [
            [-SIN_PI8, 0.25, ISQRT2_P],
            [-COS_PI8, ISQRT2_M, -0.25],
            [-0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t h t t t h t": array(
        [
            [COS_PI8, 0.25, -ISQRT2_M],
            [SIN_PI8, -ISQRT2_P, -0.25],
            [-0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h t h t t t h tdg": array(
        [
            [SIN_PI8, -ISQRT2_P, -0.25],
            [-COS_PI8, -0.25, ISQRT2_M],
            [-0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h t h tdg h t h t": array(
        [
            [0.25, THREE_SQRT2_M, -SIX_P],
            [-ISQRT2_P, -TWO_M, -THREE_SQRT2_M],
            [-SIN_PI8, ISQRT2_P, 0.25],
        ]
    ),
    "h t h t h tdg h t h tdg": array(
        [
            [-ISQRT2_P, -TWO_M, -THREE_SQRT2_M],
            [-0.25, -THREE_SQRT2_M, SIX_P],
            [-SIN_PI8, ISQRT2_P, 0.25],
        ]
    ),
    "h t h t h tdg h t t h": array(
        [
            [-0.5, SIN_PI8, -COS_PI8],
            [-ISQRT2, 0.5, 0.5],
            [0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h t h tdg h t t t": array(
        [
            [-COS_PI8, -0.25, ISQRT2_M],
            [SIN_PI8, -ISQRT2_P, -0.25],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg h tdg h t": array(
        [
            [-ISQRT2_M, THREE_SQRT2_P, -TWO_P],
            [-0.25, -SIX_M, -THREE_SQRT2_P],
            [-COS_PI8, -0.25, ISQRT2_M],
        ]
    ),
    "h t h t h tdg h tdg h tdg": array(
        [
            [-0.25, -SIX_M, -THREE_SQRT2_P],
            [ISQRT2_M, -THREE_SQRT2_P, TWO_P],
            [-COS_PI8, -0.25, ISQRT2_M],
        ]
    ),
    "h t h t h tdg h tdg tdg tdg": array(
        [
            [-SIN_PI8, ISQRT2_P, 0.25],
            [-COS_PI8, -0.25, ISQRT2_M],
            [0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t h tdg tdg h t h": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [-0.5, -SIN_PI8, COS_PI8],
            [-0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h t h tdg tdg tdg h t": array(
        [
            [COS_PI8, -ISQRT2_M, 0.25],
            [SIN_PI8, -0.25, -ISQRT2_P],
            [0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h t h tdg tdg tdg h tdg": array(
        [
            [SIN_PI8, -0.25, -ISQRT2_P],
            [-COS_PI8, ISQRT2_M, -0.25],
            [0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h t t t h t h t": array(
        [
            [-COS_PI8, 0.25, -ISQRT2_M],
            [-SIN_PI8, -ISQRT2_P, -0.25],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t t t h t h tdg": array(
        [
            [-SIN_PI8, -ISQRT2_P, -0.25],
            [COS_PI8, -0.25, ISQRT2_M],
            [-0.5, -SIN_PI8, COS_PI8],
        ]
    ),
    "h t h t t t h tdg h t": array(
        [
            [-COS_PI8, -ISQRT2_M, 0.25],
            [-SIN_PI8, -0.25, -ISQRT2_P],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h t t t h tdg h tdg": array(
        [
            [-SIN_PI8, -0.25, -ISQRT2_P],
            [COS_PI8, ISQRT2_M, -0.25],
            [0.5, -COS_PI8, SIN_PI8],
        ]
    ),
    "h t h tdg h t h t h t": array(
        [
            [-0.25, THREE_SQRT2_P, SIX_M],
            [-ISQRT2_M, TWO_P, -THREE_SQRT2_P],
            [-COS_PI8, -ISQRT2_M, 0.25],
        ]
    ),
    "h t h tdg h t h t h tdg": array(
        [
            [-ISQRT2_M, TWO_P, -THREE_SQRT2_P],
            [0.25, -THREE_SQRT2_P, -SIX_M],
            [-COS_PI8, -ISQRT2_M, 0.25],
        ]
    ),
    "h t h tdg h t h tdg h t": array(
        [
            [-ISQRT2_P, THREE_SQRT2_M, TWO_M],
            [0.25, SIX_P, -THREE_SQRT2_M],
            [-SIN_PI8, -0.25, -ISQRT2_P],
        ]
    ),
    "h t h tdg h t h tdg h tdg": array(
        [
            [0.25, SIX_P, -THREE_SQRT2_M],
            [ISQRT2_P, -THREE_SQRT2_M, -TWO_M],
            [-SIN_PI8, -0.25, -ISQRT2_P],
        ]
    ),
    "h t h tdg h t h tdg tdg h": array(
        [
            [-0.5, COS_PI8, -SIN_PI8],
            [ISQRT2, 0.5, 0.5],
            [0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h tdg h t t h tdg h": array(
        [
            [-ISQRT2, 0.5, 0.5],
            [0.5, COS_PI8, -SIN_PI8],
            [-0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h tdg h t t t h t": array(
        [
            [SIN_PI8, ISQRT2_P, 0.25],
            [COS_PI8, -0.25, ISQRT2_M],
            [0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h tdg h t t t h tdg": array(
        [
            [COS_PI8, -0.25, ISQRT2_M],
            [-SIN_PI8, -ISQRT2_P, -0.25],
            [0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h tdg h tdg h t h t": array(
        [
            [ISQRT2_M, THREE_SQRT2_P, -TWO_P],
            [0.25, -SIX_M, -THREE_SQRT2_P],
            [-COS_PI8, 0.25, -ISQRT2_M],
        ]
    ),
    "h t h tdg h tdg h t h tdg": array(
        [
            [0.25, -SIX_M, -THREE_SQRT2_P],
            [-ISQRT2_M, -THREE_SQRT2_P, TWO_P],
            [-COS_PI8, 0.25, -ISQRT2_M],
        ]
    ),
    "h t h tdg h tdg h t t h": array(
        [
            [0.5, SIN_PI8, -COS_PI8],
            [-ISQRT2, -0.5, -0.5],
            [-0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h tdg h tdg h tdg h t": array(
        [
            [-0.25, THREE_SQRT2_M, -SIX_P],
            [ISQRT2_P, -TWO_M, -THREE_SQRT2_M],
            [-SIN_PI8, -ISQRT2_P, -0.25],
        ]
    ),
    "h t h tdg h tdg h tdg h tdg": array(
        [
            [ISQRT2_P, -TWO_M, -THREE_SQRT2_M],
            [0.25, -THREE_SQRT2_M, SIX_P],
            [-SIN_PI8, -ISQRT2_P, -0.25],
        ]
    ),
    "h t h tdg h tdg tdg tdg h t": array(
        [
            [SIN_PI8, 0.25, ISQRT2_P],
            [COS_PI8, ISQRT2_M, -0.25],
            [-0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h tdg h tdg tdg tdg h tdg": array(
        [
            [COS_PI8, ISQRT2_M, -0.25],
            [-SIN_PI8, -0.25, -ISQRT2_P],
            [-0.5, COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h tdg tdg h t h t h": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [0.5, -SIN_PI8, -COS_PI8],
            [-0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t h tdg tdg h t h tdg h": array(
        [
            [-ISQRT2, 0.5, -0.5],
            [0.5, COS_PI8, SIN_PI8],
            [0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t h tdg tdg tdg h t t t": array(
        [
            [-0.5, -COS_PI8, SIN_PI8],
            [-0.5, SIN_PI8, -COS_PI8],
            [ISQRT2, -0.5, -0.5],
        ]
    ),
    "h t t h tdg h t h t h": array(
        [
            [-0.5, ISQRT2, 0.5],
            [-SIN_PI8, 0.5, -COS_PI8],
            [-COS_PI8, -0.5, -SIN_PI8],
        ]
    ),
    "h t t h tdg h t h tdg h": array(
        [
            [-0.5, ISQRT2, 0.5],
            [COS_PI8, 0.5, SIN_PI8],
            [-SIN_PI8, 0.5, -COS_PI8],
        ]
    ),
    "h t t h tdg h tdg h t h": array(
        [
            [0.5, ISQRT2, -0.5],
            [-SIN_PI8, -0.5, -COS_PI8],
            [-COS_PI8, 0.5, -SIN_PI8],
        ]
    ),
    "h t t h tdg h tdg h tdg h": array(
        [
            [0.5, ISQRT2, -0.5],
            [COS_PI8, -0.5, SIN_PI8],
            [-SIN_PI8, -0.5, -COS_PI8],
        ]
    ),
    "h t t t h t h t h t": array(
        [
            [SIN_PI8, ISQRT2_P, -0.25],
            [COS_PI8, -0.25, -ISQRT2_M],
            [-0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t t t h t h t h tdg": array(
        [
            [COS_PI8, -0.25, -ISQRT2_M],
            [-SIN_PI8, -ISQRT2_P, 0.25],
            [-0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t t t h t h tdg h t": array(
        [
            [SIN_PI8, 0.25, -ISQRT2_P],
            [COS_PI8, ISQRT2_M, 0.25],
            [0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t t t h t h tdg h tdg": array(
        [
            [COS_PI8, ISQRT2_M, 0.25],
            [-SIN_PI8, -0.25, ISQRT2_P],
            [0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t t t h tdg h t h t": array(
        [
            [COS_PI8, 0.25, ISQRT2_M],
            [SIN_PI8, -ISQRT2_P, 0.25],
            [0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t t t h tdg h t h tdg": array(
        [
            [SIN_PI8, -ISQRT2_P, 0.25],
            [-COS_PI8, -0.25, -ISQRT2_M],
            [0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h t t t h tdg h tdg h t": array(
        [
            [COS_PI8, -ISQRT2_M, -0.25],
            [SIN_PI8, -0.25, ISQRT2_P],
            [-0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t t t h tdg h tdg h tdg": array(
        [
            [SIN_PI8, -0.25, ISQRT2_P],
            [-COS_PI8, ISQRT2_M, 0.25],
            [-0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h t t t h tdg tdg tdg h t": array(
        [
            [-0.5, COS_PI8, SIN_PI8],
            [0.5, SIN_PI8, COS_PI8],
            [ISQRT2, 0.5, -0.5],
        ]
    ),
    "h tdg h t h t h t h t": array(
        [
            [ISQRT2_P, TWO_M, -THREE_SQRT2_M],
            [-0.25, -THREE_SQRT2_M, -SIX_P],
            [-SIN_PI8, ISQRT2_P, -0.25],
        ]
    ),
    "h tdg h t h t h t h tdg": array(
        [
            [-0.25, -THREE_SQRT2_M, -SIX_P],
            [-ISQRT2_P, -TWO_M, THREE_SQRT2_M],
            [-SIN_PI8, ISQRT2_P, -0.25],
        ]
    ),
    "h tdg h t h t h t t t": array(
        [
            [-COS_PI8, -0.25, -ISQRT2_M],
            [SIN_PI8, -ISQRT2_P, 0.25],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h t h tdg h t": array(
        [
            [0.25, SIX_M, -THREE_SQRT2_P],
            [ISQRT2_M, -THREE_SQRT2_P, -TWO_P],
            [-COS_PI8, -0.25, -ISQRT2_M],
        ]
    ),
    "h tdg h t h t h tdg h tdg": array(
        [
            [ISQRT2_M, -THREE_SQRT2_P, -TWO_P],
            [-0.25, -SIX_M, THREE_SQRT2_P],
            [-COS_PI8, -0.25, -ISQRT2_M],
        ]
    ),
    "h tdg h t h t h tdg tdg h": array(
        [
            [0.5, -SIN_PI8, -COS_PI8],
            [ISQRT2, -0.5, 0.5],
            [-0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h tdg h t h t h tdg tdg tdg": array(
        [
            [-SIN_PI8, ISQRT2_P, -0.25],
            [-COS_PI8, -0.25, -ISQRT2_M],
            [-0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t h tdg h t h t": array(
        [
            [0.25, -SIX_P, -THREE_SQRT2_M],
            [-ISQRT2_P, -THREE_SQRT2_M, TWO_M],
            [-SIN_PI8, 0.25, -ISQRT2_P],
        ]
    ),
    "h tdg h t h tdg h t h tdg": array(
        [
            [-ISQRT2_P, -THREE_SQRT2_M, TWO_M],
            [-0.25, SIX_P, THREE_SQRT2_M],
            [-SIN_PI8, 0.25, -ISQRT2_P],
        ]
    ),
    "h tdg h t h tdg h t t h": array(
        [
            [-0.5, -COS_PI8, -SIN_PI8],
            [-ISQRT2, 0.5, -0.5],
            [0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h tdg h t h tdg h t t t": array(
        [
            [-COS_PI8, ISQRT2_M, 0.25],
            [SIN_PI8, -0.25, ISQRT2_P],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg h tdg h t": array(
        [
            [-ISQRT2_M, -TWO_P, -THREE_SQRT2_P],
            [-0.25, -THREE_SQRT2_P, SIX_M],
            [-COS_PI8, ISQRT2_M, 0.25],
        ]
    ),
    "h tdg h t h tdg h tdg h tdg": array(
        [
            [-0.25, -THREE_SQRT2_P, SIX_M],
            [ISQRT2_M, TWO_P, THREE_SQRT2_P],
            [-COS_PI8, ISQRT2_M, 0.25],
        ]
    ),
    "h tdg h t h tdg h tdg tdg tdg": array(
        [
            [-SIN_PI8, 0.25, -ISQRT2_P],
            [-COS_PI8, ISQRT2_M, 0.25],
            [0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t h tdg tdg h t h": array(
        [
            [-ISQRT2, -0.5, 0.5],
            [-0.5, COS_PI8, SIN_PI8],
            [-0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h tdg h t t h tdg h t h": array(
        [
            [-ISQRT2, -0.5, -0.5],
            [-0.5, COS_PI8, -SIN_PI8],
            [0.5, SIN_PI8, -COS_PI8],
        ]
    ),
    "h tdg h t t t h t h t": array(
        [
            [-COS_PI8, -ISQRT2_M, -0.25],
            [-SIN_PI8, -0.25, ISQRT2_P],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t t t h t h tdg": array(
        [
            [-SIN_PI8, -0.25, ISQRT2_P],
            [COS_PI8, ISQRT2_M, 0.25],
            [-0.5, COS_PI8, SIN_PI8],
        ]
    ),
    "h tdg h t t t h tdg h t": array(
        [
            [-COS_PI8, 0.25, ISQRT2_M],
            [-SIN_PI8, -ISQRT2_P, 0.25],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h t t t h tdg h tdg": array(
        [
            [-SIN_PI8, -ISQRT2_P, 0.25],
            [COS_PI8, -0.25, -ISQRT2_M],
            [0.5, SIN_PI8, COS_PI8],
        ]
    ),
    "h tdg h tdg h t h t h t": array(
        [
            [-0.25, SIX_M, -THREE_SQRT2_P],
            [-ISQRT2_M, -THREE_SQRT2_P, -TWO_P],
            [-COS_PI8, 0.25, ISQRT2_M],
        ]
    ),
    "h tdg h tdg h t h t h tdg": array(
        [
            [-ISQRT2_M, -THREE_SQRT2_P, -TWO_P],
            [0.25, -SIX_M, THREE_SQRT2_P],
            [-COS_PI8, 0.25, ISQRT2_M],
        ]
    ),
    "h tdg h tdg h t h tdg h t": array(
        [
            [-ISQRT2_P, TWO_M, -THREE_SQRT2_M],
            [0.25, -THREE_SQRT2_M, -SIX_P],
            [-SIN_PI8, -ISQRT2_P, 0.25],
        ]
    ),
    "h tdg h tdg h t h tdg h tdg": array(
        [
            [0.25, -THREE_SQRT2_M, -SIX_P],
            [ISQRT2_P, -TWO_M, THREE_SQRT2_M],
            [-SIN_PI8, -ISQRT2_P, 0.25],
        ]
    ),
    "h tdg h tdg h t h tdg tdg h": array(
        [
            [-0.5, -SIN_PI8, -COS_PI8],
            [ISQRT2, 0.5, -0.5],
            [0.5, -COS_PI8, -SIN_PI8],
        ]
    ),
    "h tdg h tdg h tdg h t h t": array(
        [
            [ISQRT2_M, -TWO_P, -THREE_SQRT2_P],
            [0.25, -THREE_SQRT2_P, SIX_M],
            [-COS_PI8, -ISQRT2_M, -0.25],
        ]
    ),
    "h tdg h tdg h tdg h t h tdg": array(
        [
            [0.25, -THREE_SQRT2_P, SIX_M],
            [-ISQRT2_M, TWO_P, THREE_SQRT2_P],
            [-COS_PI8, -ISQRT2_M, -0.25],
        ]
    ),
    "h tdg h tdg h tdg h t t h": array(
        [
            [0.5, -COS_PI8, -SIN_PI8],
            [-ISQRT2, -0.5, 0.5],
            [-0.5, -SIN_PI8, -COS_PI8],
        ]
    ),
    "h tdg h tdg h tdg h tdg h t": array(
        [
            [-0.25, -SIX_P, -THREE_SQRT2_M],
            [ISQRT2_P, -THREE_SQRT2_M, TWO_M],
            [-SIN_PI8, -0.25, ISQRT2_P],
        ]
    ),
    "h tdg h tdg h tdg h tdg h tdg": array(
        [
            [ISQRT2_P, -THREE_SQRT2_M, TWO_M],
            [0.25, SIX_P, THREE_SQRT2_M],
            [-SIN_PI8, -0.25, ISQRT2_P],
        ]
    ),
    "h tdg tdg h t h t h t h": array(
        [
            [0.5, -ISQRT2, -0.5],
            [-COS_PI8, -0.5, -SIN_PI8],
            [-SIN_PI8, 0.5, -COS_PI8],
        ]
    ),
    "h tdg tdg h t h t h tdg h": array(
        [
            [0.5, -ISQRT2, -0.5],
            [SIN_PI8, -0.5, COS_PI8],
            [-COS_PI8, -0.5, -SIN_PI8],
        ]
    ),
    "h tdg tdg h t h tdg h t h": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [-COS_PI8, 0.5, -SIN_PI8],
            [-SIN_PI8, -0.5, -COS_PI8],
        ]
    ),
    "h tdg tdg h t h tdg h tdg h": array(
        [
            [-0.5, -ISQRT2, 0.5],
            [SIN_PI8, 0.5, COS_PI8],
            [-COS_PI8, 0.5, -SIN_PI8],
        ]
    ),
    "t h t h t h t h t h": array(
        [
            [ISQRT2_P, 0.25, -SIN_PI8],
            [-THREE_SQRT2_M, SIX_P, -0.25],
            [TWO_M, THREE_SQRT2_M, ISQRT2_P],
        ]
    ),
    "t h t h t h t h t t": array(
        [
            [-0.25, ISQRT2_M, -COS_PI8],
            [SIN_PI8, -COS_PI8, -0.5],
            [-ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t h t h t h t h tdg h": array(
        [
            [ISQRT2_P, 0.25, -SIN_PI8],
            [-TWO_M, -THREE_SQRT2_M, -ISQRT2_P],
            [-THREE_SQRT2_M, SIX_P, -0.25],
        ]
    ),
    "t h t h t h t h tdg tdg": array(
        [
            [0.25, -ISQRT2_M, COS_PI8],
            [-SIN_PI8, COS_PI8, 0.5],
            [-ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t h t h t h t t t h": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [-ISQRT2_P, -0.25, SIN_PI8],
            [-0.25, ISQRT2_M, -COS_PI8],
        ]
    ),
    "t h t h t h tdg h t h": array(
        [
            [-0.25, ISQRT2_M, -COS_PI8],
            [-THREE_SQRT2_P, TWO_P, ISQRT2_M],
            [SIX_M, THREE_SQRT2_P, 0.25],
        ]
    ),
    "t h t h t h tdg h t t": array(
        [
            [-ISQRT2_P, -0.25, SIN_PI8],
            [SIN_PI8, -COS_PI8, -0.5],
            [0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t h t h t h tdg h tdg h": array(
        [
            [-0.25, ISQRT2_M, -COS_PI8],
            [-SIX_M, -THREE_SQRT2_P, -0.25],
            [-THREE_SQRT2_P, TWO_P, ISQRT2_M],
        ]
    ),
    "t h t h t h tdg h tdg tdg": array(
        [
            [ISQRT2_P, 0.25, -SIN_PI8],
            [-SIN_PI8, COS_PI8, 0.5],
            [0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t h t h t h tdg tdg h t": array(
        [
            [-0.25, -ISQRT2_P, SIN_PI8],
            [ISQRT2_M, -0.25, -COS_PI8],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h t h t h tdg tdg tdg h": array(
        [
            [SIN_PI8, -COS_PI8, -0.5],
            [-0.25, ISQRT2_M, -COS_PI8],
            [ISQRT2_P, 0.25, -SIN_PI8],
        ]
    ),
    "t h t h t t t h t h": array(
        [
            [-COS_PI8, SIN_PI8, -0.5],
            [-0.25, -ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t h t h t t t h tdg h": array(
        [
            [-COS_PI8, SIN_PI8, -0.5],
            [ISQRT2_M, -0.25, -COS_PI8],
            [-0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h t h t t t h tdg tdg": array(
        [
            [-SIN_PI8, COS_PI8, 0.5],
            [-0.5, -0.5, ISQRT2],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h t h tdg h t h t h": array(
        [
            [0.25, ISQRT2_P, -SIN_PI8],
            [-THREE_SQRT2_M, -TWO_M, -ISQRT2_P],
            [-SIX_P, THREE_SQRT2_M, 0.25],
        ]
    ),
    "t h t h tdg h t h t t": array(
        [
            [ISQRT2_M, -0.25, -COS_PI8],
            [COS_PI8, -SIN_PI8, 0.5],
            [-0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h t h tdg h t h tdg h": array(
        [
            [0.25, ISQRT2_P, -SIN_PI8],
            [SIX_P, -THREE_SQRT2_M, -0.25],
            [-THREE_SQRT2_M, -TWO_M, -ISQRT2_P],
        ]
    ),
    "t h t h tdg h t h tdg tdg": array(
        [
            [-ISQRT2_M, 0.25, COS_PI8],
            [-COS_PI8, SIN_PI8, -0.5],
            [-0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h t h tdg h t t h tdg": array(
        [
            [0.25, -ISQRT2_M, COS_PI8],
            [-ISQRT2_P, -0.25, SIN_PI8],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t h t h tdg h t t t h": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [-0.25, -ISQRT2_P, SIN_PI8],
            [ISQRT2_M, -0.25, -COS_PI8],
        ]
    ),
    "t h t h tdg h tdg h t h": array(
        [
            [ISQRT2_M, -0.25, -COS_PI8],
            [-THREE_SQRT2_P, -SIX_M, -0.25],
            [-TWO_P, THREE_SQRT2_P, -ISQRT2_M],
        ]
    ),
    "t h t h tdg h tdg h t t": array(
        [
            [-0.25, -ISQRT2_P, SIN_PI8],
            [COS_PI8, -SIN_PI8, 0.5],
            [-ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t h t h tdg h tdg h tdg h": array(
        [
            [ISQRT2_M, -0.25, -COS_PI8],
            [TWO_P, -THREE_SQRT2_P, ISQRT2_M],
            [-THREE_SQRT2_P, -SIX_M, -0.25],
        ]
    ),
    "t h t h tdg h tdg h tdg tdg": array(
        [
            [0.25, ISQRT2_P, -SIN_PI8],
            [-COS_PI8, SIN_PI8, -0.5],
            [-ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t h t h tdg h tdg tdg tdg h": array(
        [
            [COS_PI8, -SIN_PI8, 0.5],
            [ISQRT2_M, -0.25, -COS_PI8],
            [0.25, ISQRT2_P, -SIN_PI8],
        ]
    ),
    "t h t h tdg tdg h t h t": array(
        [
            [ISQRT2_P, 0.25, SIN_PI8],
            [-0.25, ISQRT2_M, COS_PI8],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h t h tdg tdg h t h tdg": array(
        [
            [-0.25, ISQRT2_M, COS_PI8],
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h t h tdg tdg tdg h t h": array(
        [
            [-SIN_PI8, COS_PI8, 0.5],
            [-ISQRT2_P, -0.25, SIN_PI8],
            [0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t h t h tdg tdg tdg h t t": array(
        [
            [-COS_PI8, SIN_PI8, -0.5],
            [0.5, 0.5, -ISQRT2],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t h t h tdg tdg tdg h tdg h": array(
        [
            [-SIN_PI8, COS_PI8, 0.5],
            [-0.25, ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t h t t t h t h t h": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, 0.25, -COS_PI8],
        ]
    ),
    "t h t t t h t h tdg h": array(
        [
            [COS_PI8, -SIN_PI8, -0.5],
            [ISQRT2_M, -0.25, COS_PI8],
            [-0.25, -ISQRT2_P, -SIN_PI8],
        ]
    ),
    "t h t t t h tdg h t h": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [0.25, -ISQRT2_M, -COS_PI8],
        ]
    ),
    "t h t t t h tdg h tdg h": array(
        [
            [SIN_PI8, -COS_PI8, 0.5],
            [-0.25, ISQRT2_M, COS_PI8],
            [-ISQRT2_P, -0.25, -SIN_PI8],
        ]
    ),
    "t h tdg h t h t h t h": array(
        [
            [0.25, -ISQRT2_M, -COS_PI8],
            [-THREE_SQRT2_P, TWO_P, -ISQRT2_M],
            [SIX_M, THREE_SQRT2_P, -0.25],
        ]
    ),
    "t h tdg h t h t h t t": array(
        [
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [SIN_PI8, -COS_PI8, 0.5],
            [-0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t h tdg h t h t h tdg h": array(
        [
            [0.25, -ISQRT2_M, -COS_PI8],
            [-SIX_M, -THREE_SQRT2_P, 0.25],
            [-THREE_SQRT2_P, TWO_P, -ISQRT2_M],
        ]
    ),
    "t h tdg h t h t h tdg tdg": array(
        [
            [ISQRT2_P, 0.25, SIN_PI8],
            [-SIN_PI8, COS_PI8, -0.5],
            [-0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t h tdg h t h tdg h t h": array(
        [
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [-THREE_SQRT2_M, SIX_P, 0.25],
            [TWO_M, THREE_SQRT2_M, -ISQRT2_P],
        ]
    ),
    "t h tdg h t h tdg h t t": array(
        [
            [-0.25, ISQRT2_M, COS_PI8],
            [SIN_PI8, -COS_PI8, 0.5],
            [ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t h tdg h t h tdg h tdg h": array(
        [
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [-TWO_M, -THREE_SQRT2_M, ISQRT2_P],
            [-THREE_SQRT2_M, SIX_P, 0.25],
        ]
    ),
    "t h tdg h t h tdg h tdg tdg": array(
        [
            [0.25, -ISQRT2_M, -COS_PI8],
            [-SIN_PI8, COS_PI8, -0.5],
            [ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t h tdg h t h tdg tdg h t": array(
        [
            [ISQRT2_M, -0.25, COS_PI8],
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "t h tdg h t t h tdg h t": array(
        [
            [-ISQRT2_M, 0.25, COS_PI8],
            [-0.25, -ISQRT2_P, SIN_PI8],
            [COS_PI8, -SIN_PI8, 0.5],
        ]
    ),
    "t h tdg h t t t h t h": array(
        [
            [-COS_PI8, SIN_PI8, 0.5],
            [ISQRT2_M, -0.25, COS_PI8],
            [0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h tdg h t t t h tdg h": array(
        [
            [-COS_PI8, SIN_PI8, 0.5],
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t h tdg h t t t h tdg tdg": array(
        [
            [-SIN_PI8, COS_PI8, -0.5],
            [0.5, 0.5, ISQRT2],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "t h tdg h tdg h t h t h": array(
        [
            [-ISQRT2_M, 0.25, -COS_PI8],
            [-THREE_SQRT2_P, -SIX_M, 0.25],
            [-TWO_P, THREE_SQRT2_P, ISQRT2_M],
        ]
    ),
    "t h tdg h tdg h t h t t": array(
        [
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [COS_PI8, -SIN_PI8, -0.5],
            [ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t h tdg h tdg h t h tdg h": array(
        [
            [-ISQRT2_M, 0.25, -COS_PI8],
            [TWO_P, -THREE_SQRT2_P, -ISQRT2_M],
            [-THREE_SQRT2_P, -SIX_M, 0.25],
        ]
    ),
    "t h tdg h tdg h t h tdg tdg": array(
        [
            [0.25, ISQRT2_P, SIN_PI8],
            [-COS_PI8, SIN_PI8, 0.5],
            [ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t h tdg h tdg h tdg h t h": array(
        [
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [-THREE_SQRT2_M, -TWO_M, ISQRT2_P],
            [-SIX_P, THREE_SQRT2_M, -0.25],
        ]
    ),
    "t h tdg h tdg h tdg h t t": array(
        [
            [ISQRT2_M, -0.25, COS_PI8],
            [COS_PI8, -SIN_PI8, -0.5],
            [0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h tdg h tdg h tdg h tdg h": array(
        [
            [-0.25, -ISQRT2_P, -SIN_PI8],
            [SIX_P, -THREE_SQRT2_M, 0.25],
            [-THREE_SQRT2_M, -TWO_M, ISQRT2_P],
        ]
    ),
    "t h tdg h tdg h tdg h tdg tdg": array(
        [
            [-ISQRT2_M, 0.25, -COS_PI8],
            [-COS_PI8, SIN_PI8, 0.5],
            [0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t h tdg h tdg tdg tdg h t h": array(
        [
            [-SIN_PI8, COS_PI8, -0.5],
            [-0.25, ISQRT2_M, COS_PI8],
            [ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t h tdg h tdg tdg tdg h t t": array(
        [
            [-COS_PI8, SIN_PI8, 0.5],
            [-0.5, -0.5, -ISQRT2],
            [SIN_PI8, -COS_PI8, 0.5],
        ]
    ),
    "t h tdg h tdg tdg tdg h tdg h": array(
        [
            [-SIN_PI8, COS_PI8, -0.5],
            [-ISQRT2_P, -0.25, -SIN_PI8],
            [-0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t h tdg tdg h t h t h t": array(
        [
            [-0.25, -ISQRT2_M, COS_PI8],
            [ISQRT2_P, -0.25, SIN_PI8],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h tdg tdg h t h t h tdg": array(
        [
            [ISQRT2_P, -0.25, SIN_PI8],
            [0.25, ISQRT2_M, -COS_PI8],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t h tdg tdg h t h tdg h t": array(
        [
            [ISQRT2_M, 0.25, COS_PI8],
            [0.25, -ISQRT2_P, SIN_PI8],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h tdg tdg h t h tdg h tdg": array(
        [
            [0.25, -ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, -0.25, -COS_PI8],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t h tdg tdg tdg h t t t h": array(
        [
            [-0.5, -0.5, ISQRT2],
            [-COS_PI8, SIN_PI8, -0.5],
            [SIN_PI8, -COS_PI8, -0.5],
        ]
    ),
    "t t h t h t h t h t": array(
        [
            [-0.25, -SIN_PI8, -ISQRT2_P],
            [-ISQRT2_M, -COS_PI8, 0.25],
            [-COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "t t h t h t h t h tdg": array(
        [
            [-ISQRT2_M, -COS_PI8, 0.25],
            [0.25, SIN_PI8, ISQRT2_P],
            [-COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "t t h t h t h t t t": array(
        [
            [-SIN_PI8, -0.5, COS_PI8],
            [COS_PI8, -0.5, -SIN_PI8],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h t h tdg h t": array(
        [
            [-ISQRT2_P, -SIN_PI8, -0.25],
            [0.25, -COS_PI8, -ISQRT2_M],
            [-SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "t t h t h t h tdg h tdg": array(
        [
            [0.25, -COS_PI8, -ISQRT2_M],
            [ISQRT2_P, SIN_PI8, 0.25],
            [-SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "t t h t h t h tdg tdg tdg": array(
        [
            [-COS_PI8, 0.5, SIN_PI8],
            [-SIN_PI8, -0.5, COS_PI8],
            [0.5, ISQRT2, 0.5],
        ]
    ),
    "t t h t h tdg h t h t": array(
        [
            [ISQRT2_M, -COS_PI8, -0.25],
            [0.25, -SIN_PI8, ISQRT2_P],
            [-COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "t t h t h tdg h t h tdg": array(
        [
            [0.25, -SIN_PI8, ISQRT2_P],
            [-ISQRT2_M, COS_PI8, 0.25],
            [-COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "t t h t h tdg h t t t": array(
        [
            [-SIN_PI8, 0.5, COS_PI8],
            [COS_PI8, 0.5, -SIN_PI8],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h t h tdg h tdg h t": array(
        [
            [-0.25, -COS_PI8, ISQRT2_M],
            [ISQRT2_P, -SIN_PI8, 0.25],
            [-SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "t t h t h tdg h tdg h tdg": array(
        [
            [ISQRT2_P, -SIN_PI8, 0.25],
            [0.25, COS_PI8, -ISQRT2_M],
            [-SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "t t h t h tdg h tdg tdg tdg": array(
        [
            [-COS_PI8, -0.5, SIN_PI8],
            [-SIN_PI8, 0.5, COS_PI8],
            [-0.5, ISQRT2, -0.5],
        ]
    ),
    "t t h tdg h t h t h t": array(
        [
            [-ISQRT2_P, -SIN_PI8, 0.25],
            [0.25, -COS_PI8, ISQRT2_M],
            [SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "t t h tdg h t h t h tdg": array(
        [
            [0.25, -COS_PI8, ISQRT2_M],
            [ISQRT2_P, SIN_PI8, -0.25],
            [SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "t t h tdg h t h tdg h t": array(
        [
            [-0.25, -SIN_PI8, ISQRT2_P],
            [-ISQRT2_M, -COS_PI8, -0.25],
            [COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "t t h tdg h t h tdg h tdg": array(
        [
            [-ISQRT2_M, -COS_PI8, -0.25],
            [0.25, SIN_PI8, -ISQRT2_P],
            [COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "t t h tdg h tdg h t h t": array(
        [
            [-0.25, -COS_PI8, -ISQRT2_M],
            [ISQRT2_P, -SIN_PI8, -0.25],
            [SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "t t h tdg h tdg h t h tdg": array(
        [
            [ISQRT2_P, -SIN_PI8, -0.25],
            [0.25, COS_PI8, ISQRT2_M],
            [SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "t t h tdg h tdg h tdg h t": array(
        [
            [ISQRT2_M, -COS_PI8, 0.25],
            [0.25, -SIN_PI8, -ISQRT2_P],
            [COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "t t h tdg h tdg h tdg h tdg": array(
        [
            [0.25, -SIN_PI8, -ISQRT2_P],
            [-ISQRT2_M, COS_PI8, -0.25],
            [COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "t t t h t h t h t h": array(
        [
            [-COS_PI8, -SIN_PI8, -0.5],
            [-ISQRT2_M, -0.25, COS_PI8],
            [-0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "t t t h t h t h t t": array(
        [
            [-SIN_PI8, -COS_PI8, 0.5],
            [0.5, -0.5, -ISQRT2],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "t t t h t h t h tdg h": array(
        [
            [-COS_PI8, -SIN_PI8, -0.5],
            [0.25, -ISQRT2_P, -SIN_PI8],
            [-ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "t t t h t h tdg h t h": array(
        [
            [-SIN_PI8, -COS_PI8, 0.5],
            [0.25, ISQRT2_M, COS_PI8],
            [-ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "t t t h t h tdg h tdg h": array(
        [
            [-SIN_PI8, -COS_PI8, 0.5],
            [ISQRT2_P, -0.25, -SIN_PI8],
            [0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "t t t h t h tdg h tdg tdg": array(
        [
            [-COS_PI8, -SIN_PI8, -0.5],
            [-0.5, 0.5, ISQRT2],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "t t t h tdg h t h t h": array(
        [
            [-COS_PI8, -SIN_PI8, 0.5],
            [0.25, -ISQRT2_P, SIN_PI8],
            [ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "t t t h tdg h t h t t": array(
        [
            [-SIN_PI8, -COS_PI8, -0.5],
            [-0.5, 0.5, -ISQRT2],
            [COS_PI8, SIN_PI8, -0.5],
        ]
    ),
    "t t t h tdg h t h tdg h": array(
        [
            [-COS_PI8, -SIN_PI8, 0.5],
            [-ISQRT2_M, -0.25, -COS_PI8],
            [0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "t t t h tdg h tdg h t h": array(
        [
            [-SIN_PI8, -COS_PI8, -0.5],
            [ISQRT2_P, -0.25, SIN_PI8],
            [-0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "t t t h tdg h tdg h tdg h": array(
        [
            [-SIN_PI8, -COS_PI8, -0.5],
            [0.25, ISQRT2_M, -COS_PI8],
            [ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "t t t h tdg h tdg h tdg tdg": array(
        [
            [-COS_PI8, -SIN_PI8, 0.5],
            [0.5, -0.5, ISQRT2],
            [SIN_PI8, COS_PI8, 0.5],
        ]
    ),
    "t t t h tdg tdg tdg h t h": array(
        [
            [-0.5, 0.5, ISQRT2],
            [COS_PI8, SIN_PI8, 0.5],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "tdg h t h t h t h t h": array(
        [
            [-0.25, ISQRT2_P, -SIN_PI8],
            [-SIX_P, -THREE_SQRT2_M, -0.25],
            [-THREE_SQRT2_M, TWO_M, ISQRT2_P],
        ]
    ),
    "tdg h t h t h t h t t": array(
        [
            [-ISQRT2_M, -0.25, -COS_PI8],
            [COS_PI8, SIN_PI8, -0.5],
            [0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h t h t h t h tdg h": array(
        [
            [-0.25, ISQRT2_P, -SIN_PI8],
            [THREE_SQRT2_M, -TWO_M, -ISQRT2_P],
            [-SIX_P, -THREE_SQRT2_M, -0.25],
        ]
    ),
    "tdg h t h t h t h tdg tdg": array(
        [
            [ISQRT2_M, 0.25, COS_PI8],
            [-COS_PI8, -SIN_PI8, 0.5],
            [0.25, -ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h t h t h t t t h": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [0.25, -ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, -0.25, -COS_PI8],
        ]
    ),
    "tdg h t h t h tdg h t h": array(
        [
            [-ISQRT2_M, -0.25, -COS_PI8],
            [-TWO_P, -THREE_SQRT2_P, ISQRT2_M],
            [-THREE_SQRT2_P, SIX_M, 0.25],
        ]
    ),
    "tdg h t h t h tdg h t t": array(
        [
            [0.25, -ISQRT2_P, SIN_PI8],
            [COS_PI8, SIN_PI8, -0.5],
            [ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "tdg h t h t h tdg h tdg h": array(
        [
            [-ISQRT2_M, -0.25, -COS_PI8],
            [THREE_SQRT2_P, -SIX_M, -0.25],
            [-TWO_P, -THREE_SQRT2_P, ISQRT2_M],
        ]
    ),
    "tdg h t h t h tdg h tdg tdg": array(
        [
            [-0.25, ISQRT2_P, -SIN_PI8],
            [-COS_PI8, -SIN_PI8, 0.5],
            [ISQRT2_M, 0.25, COS_PI8],
        ]
    ),
    "tdg h t h t h tdg tdg tdg h": array(
        [
            [COS_PI8, SIN_PI8, -0.5],
            [-ISQRT2_M, -0.25, -COS_PI8],
            [-0.25, ISQRT2_P, -SIN_PI8],
        ]
    ),
    "tdg h t h tdg h t h t h": array(
        [
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [TWO_M, -THREE_SQRT2_M, -ISQRT2_P],
            [-THREE_SQRT2_M, -SIX_P, 0.25],
        ]
    ),
    "tdg h t h tdg h t h t t": array(
        [
            [0.25, ISQRT2_M, -COS_PI8],
            [SIN_PI8, COS_PI8, 0.5],
            [ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "tdg h t h tdg h t h tdg h": array(
        [
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [THREE_SQRT2_M, SIX_P, -0.25],
            [TWO_M, -THREE_SQRT2_M, -ISQRT2_P],
        ]
    ),
    "tdg h t h tdg h t h tdg tdg": array(
        [
            [-0.25, -ISQRT2_M, COS_PI8],
            [-SIN_PI8, -COS_PI8, -0.5],
            [ISQRT2_P, -0.25, SIN_PI8],
        ]
    ),
    "tdg h t h tdg h t t t h": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [ISQRT2_P, -0.25, SIN_PI8],
            [0.25, ISQRT2_M, -COS_PI8],
        ]
    ),
    "tdg h t h tdg h tdg h t h": array(
        [
            [0.25, ISQRT2_M, -COS_PI8],
            [SIX_M, -THREE_SQRT2_P, -0.25],
            [-THREE_SQRT2_P, -TWO_P, -ISQRT2_M],
        ]
    ),
    "tdg h t h tdg h tdg h t t": array(
        [
            [ISQRT2_P, -0.25, SIN_PI8],
            [SIN_PI8, COS_PI8, 0.5],
            [-0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h t h tdg h tdg h tdg h": array(
        [
            [0.25, ISQRT2_M, -COS_PI8],
            [THREE_SQRT2_P, TWO_P, ISQRT2_M],
            [SIX_M, -THREE_SQRT2_P, -0.25],
        ]
    ),
    "tdg h t h tdg h tdg h tdg tdg": array(
        [
            [-ISQRT2_P, 0.25, -SIN_PI8],
            [-SIN_PI8, -COS_PI8, -0.5],
            [-0.25, -ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h t h tdg h tdg tdg tdg h": array(
        [
            [SIN_PI8, COS_PI8, 0.5],
            [0.25, ISQRT2_M, -COS_PI8],
            [-ISQRT2_P, 0.25, -SIN_PI8],
        ]
    ),
    "tdg h t h tdg tdg h t h t": array(
        [
            [-0.25, ISQRT2_P, SIN_PI8],
            [-ISQRT2_M, -0.25, COS_PI8],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "tdg h t h tdg tdg h t h tdg": array(
        [
            [-ISQRT2_M, -0.25, COS_PI8],
            [0.25, -ISQRT2_P, -SIN_PI8],
            [COS_PI8, SIN_PI8, 0.5],
        ]
    ),
    "tdg h t t h tdg h t h t": array(
        [
            [0.25, ISQRT2_P, SIN_PI8],
            [ISQRT2_M, -0.25, COS_PI8],
            [COS_PI8, -SIN_PI8, -0.5],
        ]
    ),
    "tdg h t t t h t h t h": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [ISQRT2_P, -0.25, -SIN_PI8],
            [-0.25, -ISQRT2_M, -COS_PI8],
        ]
    ),
    "tdg h t t t h t h tdg h": array(
        [
            [SIN_PI8, COS_PI8, -0.5],
            [0.25, ISQRT2_M, COS_PI8],
            [ISQRT2_P, -0.25, -SIN_PI8],
        ]
    ),
    "tdg h t t t h tdg h t h": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [0.25, -ISQRT2_P, -SIN_PI8],
            [ISQRT2_M, 0.25, -COS_PI8],
        ]
    ),
    "tdg h t t t h tdg h tdg h": array(
        [
            [COS_PI8, SIN_PI8, 0.5],
            [-ISQRT2_M, -0.25, COS_PI8],
            [0.25, -ISQRT2_P, -SIN_PI8],
        ]
    ),
    "tdg h tdg h t h t h t h": array(
        [
            [ISQRT2_M, 0.25, -COS_PI8],
            [-TWO_P, -THREE_SQRT2_P, -ISQRT2_M],
            [-THREE_SQRT2_P, SIX_M, -0.25],
        ]
    ),
    "tdg h tdg h t h t h t t": array(
        [
            [0.25, -ISQRT2_P, -SIN_PI8],
            [COS_PI8, SIN_PI8, 0.5],
            [-ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "tdg h tdg h t h t h tdg h": array(
        [
            [ISQRT2_M, 0.25, -COS_PI8],
            [THREE_SQRT2_P, -SIX_M, 0.25],
            [-TWO_P, -THREE_SQRT2_P, -ISQRT2_M],
        ]
    ),
    "tdg h tdg h t h t h tdg tdg": array(
        [
            [-0.25, ISQRT2_P, SIN_PI8],
            [-COS_PI8, -SIN_PI8, -0.5],
            [-ISQRT2_M, -0.25, COS_PI8],
        ]
    ),
    "tdg h tdg h t h tdg h t h": array(
        [
            [0.25, -ISQRT2_P, -SIN_PI8],
            [-SIX_P, -THREE_SQRT2_M, 0.25],
            [-THREE_SQRT2_M, TWO_M, -ISQRT2_P],
        ]
    ),
    "tdg h tdg h t h tdg h t t": array(
        [
            [-ISQRT2_M, -0.25, COS_PI8],
            [COS_PI8, SIN_PI8, 0.5],
            [-0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h tdg h t h tdg h tdg h": array(
        [
            [0.25, -ISQRT2_P, -SIN_PI8],
            [THREE_SQRT2_M, -TWO_M, ISQRT2_P],
            [-SIX_P, -THREE_SQRT2_M, 0.25],
        ]
    ),
    "tdg h tdg h t h tdg h tdg tdg": array(
        [
            [ISQRT2_M, 0.25, -COS_PI8],
            [-COS_PI8, -SIN_PI8, -0.5],
            [-0.25, ISQRT2_P, SIN_PI8],
        ]
    ),
    "tdg h tdg h t h tdg tdg h t": array(
        [
            [0.25, ISQRT2_M, COS_PI8],
            [ISQRT2_P, -0.25, -SIN_PI8],
            [SIN_PI8, COS_PI8, -0.5],
        ]
    ),
    "tdg h tdg h tdg h t h t h": array(
        [
            [-0.25, -ISQRT2_M, -COS_PI8],
            [SIX_M, -THREE_SQRT2_P, 0.25],
            [-THREE_SQRT2_P, -TWO_P, ISQRT2_M],
        ]
    ),
    "tdg h tdg h tdg h t h t t": array(
        [
            [ISQRT2_P, -0.25, -SIN_PI8],
            [SIN_PI8, COS_PI8, -0.5],
            [0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h tdg h tdg h t h tdg h": array(
        [
            [-0.25, -ISQRT2_M, -COS_PI8],
            [THREE_SQRT2_P, TWO_P, -ISQRT2_M],
            [SIX_M, -THREE_SQRT2_P, 0.25],
        ]
    ),
    "tdg h tdg h tdg h t h tdg tdg": array(
        [
            [-ISQRT2_P, 0.25, SIN_PI8],
            [-SIN_PI8, -COS_PI8, 0.5],
            [0.25, ISQRT2_M, COS_PI8],
        ]
    ),
    "tdg h tdg h tdg h tdg h t h": array(
        [
            [ISQRT2_P, -0.25, -SIN_PI8],
            [TWO_M, -THREE_SQRT2_M, ISQRT2_P],
            [-THREE_SQRT2_M, -SIX_P, -0.25],
        ]
    ),
    "tdg h tdg h tdg h tdg h t t": array(
        [
            [0.25, ISQRT2_M, COS_PI8],
            [SIN_PI8, COS_PI8, -0.5],
            [-ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "tdg h tdg h tdg h tdg h tdg h": array(
        [
            [ISQRT2_P, -0.25, -SIN_PI8],
            [THREE_SQRT2_M, SIX_P, 0.25],
            [TWO_M, -THREE_SQRT2_M, ISQRT2_P],
        ]
    ),
    "tdg h tdg h tdg h tdg h tdg tdg": array(
        [
            [-0.25, -ISQRT2_M, -COS_PI8],
            [-SIN_PI8, -COS_PI8, 0.5],
            [-ISQRT2_P, 0.25, SIN_PI8],
        ]
    ),
    "tdg tdg h t h t h t h t": array(
        [
            [0.25, SIN_PI8, -ISQRT2_P],
            [ISQRT2_M, COS_PI8, 0.25],
            [COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h t h t h t h tdg": array(
        [
            [ISQRT2_M, COS_PI8, 0.25],
            [-0.25, -SIN_PI8, ISQRT2_P],
            [COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h t h t h tdg h t": array(
        [
            [ISQRT2_P, SIN_PI8, -0.25],
            [-0.25, COS_PI8, -ISQRT2_M],
            [SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "tdg tdg h t h t h tdg h tdg": array(
        [
            [-0.25, COS_PI8, -ISQRT2_M],
            [-ISQRT2_P, -SIN_PI8, 0.25],
            [SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "tdg tdg h t h tdg h t h t": array(
        [
            [-ISQRT2_M, COS_PI8, -0.25],
            [-0.25, SIN_PI8, ISQRT2_P],
            [COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h t h tdg h t h tdg": array(
        [
            [-0.25, SIN_PI8, ISQRT2_P],
            [ISQRT2_M, -COS_PI8, 0.25],
            [COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h t h tdg h tdg h t": array(
        [
            [0.25, COS_PI8, ISQRT2_M],
            [-ISQRT2_P, SIN_PI8, 0.25],
            [SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "tdg tdg h t h tdg h tdg h tdg": array(
        [
            [-ISQRT2_P, SIN_PI8, 0.25],
            [-0.25, -COS_PI8, -ISQRT2_M],
            [SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "tdg tdg h t t t h t h t": array(
        [
            [-SIN_PI8, 0.5, COS_PI8],
            [-COS_PI8, -0.5, SIN_PI8],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t t t h t h tdg": array(
        [
            [-COS_PI8, -0.5, SIN_PI8],
            [SIN_PI8, -0.5, -COS_PI8],
            [0.5, -ISQRT2, 0.5],
        ]
    ),
    "tdg tdg h t t t h tdg h t": array(
        [
            [-SIN_PI8, -0.5, COS_PI8],
            [-COS_PI8, 0.5, SIN_PI8],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h t t t h tdg h tdg": array(
        [
            [-COS_PI8, 0.5, SIN_PI8],
            [SIN_PI8, 0.5, -COS_PI8],
            [-0.5, -ISQRT2, -0.5],
        ]
    ),
    "tdg tdg h tdg h t h t h t": array(
        [
            [ISQRT2_P, SIN_PI8, 0.25],
            [-0.25, COS_PI8, ISQRT2_M],
            [-SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "tdg tdg h tdg h t h t h tdg": array(
        [
            [-0.25, COS_PI8, ISQRT2_M],
            [-ISQRT2_P, -SIN_PI8, -0.25],
            [-SIN_PI8, -0.5, COS_PI8],
        ]
    ),
    "tdg tdg h tdg h t h tdg h t": array(
        [
            [0.25, SIN_PI8, ISQRT2_P],
            [ISQRT2_M, COS_PI8, -0.25],
            [-COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h tdg h t h tdg h tdg": array(
        [
            [ISQRT2_M, COS_PI8, -0.25],
            [-0.25, -SIN_PI8, -ISQRT2_P],
            [-COS_PI8, 0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h tdg h tdg h t h t": array(
        [
            [0.25, COS_PI8, -ISQRT2_M],
            [-ISQRT2_P, SIN_PI8, -0.25],
            [-SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "tdg tdg h tdg h tdg h t h tdg": array(
        [
            [-ISQRT2_P, SIN_PI8, -0.25],
            [-0.25, -COS_PI8, ISQRT2_M],
            [-SIN_PI8, 0.5, COS_PI8],
        ]
    ),
    "tdg tdg h tdg h tdg h tdg h t": array(
        [
            [-ISQRT2_M, COS_PI8, 0.25],
            [-0.25, SIN_PI8, -ISQRT2_P],
            [-COS_PI8, -0.5, SIN_PI8],
        ]
    ),
    "tdg tdg h tdg h tdg h tdg h tdg": array(
        [
            [-0.25, SIN_PI8, -ISQRT2_P],
            [ISQRT2_M, -COS_PI8, -0.25],
            [-COS_PI8, -0.5, SIN_PI8],
        ]
    ),
}
