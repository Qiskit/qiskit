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

"""
MeasureGrouping
"""
from typing import Dict, List, Optional, Union
from qiskit.pulse import utils


class MeasureGrouping:
    """
    MeasureGrouping
    """

    __slots__ = (
        "description",
        "_meas_map",
    )

    def __init__(
        self,
        meas_map: Optional[Union[List[List[int]], Dict[int, List[int]]]] = None,
        description=None,
    ):
        """
        Create a new "MeasureGrouping" object.

        Args:
            meas_map (list): List of sets of qubits that must be measured together.
            description (str): An optional string to describe the MeasureGrouping.
        """
        self.description = description
        if isinstance(meas_map, list):
            meas_map = utils.format_meas_map(meas_map)
        self._meas_map = meas_map

    @property
    def meas_map(self):
        return self._meas_map

    def get_qubit_groups(self, qubits: List) -> List:
        """
        Gets qubits list including at least one qubit of `qubits` from meas_map.

        Args:
            qubits (list): List of qubits to be measured.
        Returns:
            list: Sorted list with qubits measured simultaneously.
        """
        if self.meas_map is None:
            return sorted(qubits)

        qubits_group = set()
        for qubit in qubits:
            qubits_group |= set(self.meas_map[qubit])
        return sorted(list(qubits_group))

    def constraints(self):
        """
        Return constraints of a hardware backend.
        """
        return NotImplemented
