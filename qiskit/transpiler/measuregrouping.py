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

"""
MeasureGrouping
"""
from typing import List


class MeasureGrouping:
    """
    MeasureGrouping

    """

    __slots__ = (
        "description",
        "meas_map",
    )
    def __init__(self, meas_map=None, description=None):
        """
        Create a new "MeasureGrouping" object.
        """
        self.description = description

        self.meas_map = meas_map
    
    @meas_map.setter
    def meas_map(self, meas_map):
        # if specify meas_map, is it necessary to convert the type of meas_map?
        self.meas_map = meas_map
    
    def get_qubit_groups(self, *qubits: List):
        """
        Gets qubit groups including at least one qubit of `qubits` from meas_map. 
        """
        pass

    def constraints(self):
        """
        Return constraints of a hardware backend.
        """
        return NotImplemented