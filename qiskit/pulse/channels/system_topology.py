# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Topology of system.
"""
from typing import List

from .channels import AcquireChannel, DriveChannel, ControlChannel, MeasureChannel
from .qubit import Qubit


class SystemTopology:
    """Helper class to store the system mapping between qubits and channels.
    """

    def __init__(self,
                 drives: List[DriveChannel],
                 controls: List[ControlChannel],
                 measures: List[MeasureChannel],
                 acquires: List[AcquireChannel]):
        """Create new system topology.

        Args:
            drives: List of drive channels.
            controls: List of control channels.
            measures: List of measure channels.
            acquires: List of acquire channels.
        """

        # TODO: allow for more flexible mapping of channels by using device Hamiltonian.
        # TODO: assign proper control channels to each qubit
        self._qubits = [
            Qubit(ii, drive, measure, acquire, controls) for ii, (drive, measure, acquire)
            in enumerate(zip(drives, measures, acquires))
        ]

    @property
    def qubits(self) -> List[Qubit]:
        """Return list of qubit in this system."""
        return self._qubits
