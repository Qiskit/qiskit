# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Free Quaternion Selection (FQS) algorithm."""

from typing import List, Tuple

import numpy as np

from .sequential_optimizer import SequentialOptimizer, _Paulis


class FQS(SequentialOptimizer):
    """
    Free Quaternion Selection (FQS) algorithm [1].

    This optimizer optimizes parameterized U gates one by one in the order they appear in the ansatz.
    Once the last parameterized U gate is optimized, it returns to the first one.

    .. note::

        This optimizer only works with U gates as parameterized gates.

    .. note::

        This optimizer evaluates the energy function 10 times for each U gate.

    References:
      [1] "Sequential optimal selection of a single-qubit gate and its relation to barren plateau
          in parameterized quantum circuits,"
          K. Wada, R. Raymond, Y. Sato, H.C. Watanabe,
          `arXiv:2209.08535 <https://arxiv.org/abs/2209.08535>`__
    """

    _MATRICES = [
        _Paulis.I,
        _Paulis.X,
        _Paulis.Y,
        _Paulis.Z,
        _Paulis.IX,
        _Paulis.IY,
        _Paulis.IZ,
        _Paulis.XY,
        _Paulis.YZ,
        _Paulis.ZX,
    ]
    _ANGLES = [_Paulis.DECOMPOSER.angles(mat) for mat in _MATRICES]

    @property
    def _angles(self) -> List[Tuple[float, float, float]]:
        return self._ANGLES

    def _cost_matrix(self, vals: list[float]) -> np.ndarray:
        r_id, r_x, r_y, r_z, r_ix, r_iy, r_iz, r_xy, r_yz, r_zx = vals
        mat = np.array(
            [
                [
                    r_id / 2,
                    r_ix - r_x / 2 - r_id / 2,
                    r_iy - r_y / 2 - r_id / 2,
                    r_iz - r_z / 2 - r_id / 2,
                ],
                [0, r_x / 2, r_xy - r_x / 2 - r_y / 2, r_zx - r_x / 2 - r_z / 2],
                [0, 0, r_y / 2, r_yz - r_y / 2 - r_z / 2],
                [0, 0, 0, r_z / 2],
            ]
        )
        return mat + mat.T
