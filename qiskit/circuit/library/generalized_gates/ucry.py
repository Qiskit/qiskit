# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Uniformly controlled Pauli-Y rotations."""

from __future__ import annotations

from .uc_pauli_rot import UCPauliRotGate


class UCRYGate(UCPauliRotGate):
    r"""Uniformly controlled Pauli-Y rotations.

    Implements the :class:`.UCGate` for the special case that all unitaries are Pauli-Y rotations,
    :math:`U_i = R_Y(a_i)` where :math:`a_i \in \mathbb{R}` is the rotation angle.
    """

    def __init__(self, angle_list: list[float]) -> None:
        r"""
        Args:
            angle_list: List of rotation angles :math:`[a_0, ..., a_{2^{k-1}}]`.
        """
        super().__init__(angle_list, "Y")
