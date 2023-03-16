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

"""Free-Axis Selection (Fraxis) algorithm."""

from typing import List, Tuple

import numpy as np

from qiskit.circuit.library import UGate

from .sequential_optimizer import SequentialOptimizer, _Angles, _Paulis


class Fraxis(SequentialOptimizer):
    """
    Free-Axis Selection (Fraxis) algorithm [1].

    More precisely, this class implements π-Fraxis algorithm in Algorithm 1 of [1].
    This optimizer optimizes parameterized U gates one by one in the order they appear in the ansatz.
    Once the last parameterized U gate is optimized, it returns to the first one.

    .. note::

        This optimizer only works with U gates as parameterized gates.

    .. note::

        This optimizer evaluates the energy function 6 times for each U gate.

    References:
      [1] "Optimizing Parameterized Quantum Circuits with Free-Axis Selection,"
          H.C. Watanabe, R. Raymond, Y. Ohnishi, E. Kaminishi, M. Sugawara
          `arXiv:2104.14875 <https://arxiv.org/abs/2104.14875>`__
    """

    def _initialize(self, x0: np.ndarray) -> np.ndarray:
        for idx in range(0, x0.size, 3):
            # Fraxis cannot handle some U3 rotations such as identity(=U3(0,0,0)).
            # The following converts such rotations into ones that Fraxis can handle.
            mat = UGate(*x0[idx : idx + 3]).to_matrix()
            n_x = mat[1, 0].real
            n_y = mat[1, 0].imag
            n_z = mat[0, 0]
            vec = np.array([0, n_x, n_y, n_z])
            if np.allclose(vec, 0):
                vec[0] = 1
            vec /= np.linalg.norm(vec)
            x0[idx : idx + 3] = _Paulis.angles(vec)
        return x0

    @property
    def _angles(self) -> List[Tuple[float, float, float]]:
        return [_Angles.X, _Angles.Y, _Angles.Z, _Angles.XY, _Angles.YZ, _Angles.ZX]

    def _energy_matrix(self, vals: List[float]) -> np.ndarray:
        r_x, r_y, r_z, r_xy, r_yz, r_zx = vals
        mat = np.array(
            [
                [0, 0, 0, 0],
                [0, r_x / 2, r_xy - r_x / 2 - r_y / 2, r_zx - r_x / 2 - r_z / 2],
                [0, 0, r_y / 2, r_yz - r_y / 2 - r_z / 2],
                [0, 0, 0, r_z / 2],
            ]
        )
        return mat + mat.T
