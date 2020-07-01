# -*- coding: utf-8 -*-

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

"""Validation pass module for pulse programs."""

from typing import List

from qiskit import pulse
from qiskit.pulse import analysis
from qiskit.pulse.basepasses import ValidationPass
from qiskit.pulse.exceptions import CompilerError


class ValidateMeasMap(ValidationPass):
    """Validate all acquires in the qobj obey the measurement map."""

    def __init__(
        self,
        meas_map: List[List[int]]
    ):
        """
        Args:
            meas_map: List of groups of qubits that must be acquired together.
        """
        super().__init__()
        self.meas_map = meas_map
        self.requires.append(analysis.AmalgamatedAcquires())

    def run(self, program: pulse.Program) -> pulse.Program:
        acquire_instruction_maps = self.analysis.acquire_instruction_maps
        if acquire_instruction_maps:
            meas_map_sets = [set(m) for m in self.meas_map]
            # Check each acquisition time individually
            for acquire_instruction_map in acquire_instruction_maps:
                for _, instrs in acquire_instruction_map.items():
                    measured_qubits = set()
                    for inst in instrs:
                        measured_qubits.add(inst.channel.index)

                    for meas_set in meas_map_sets:
                        intersection = measured_qubits.intersection(meas_set)
                        if intersection and intersection != meas_set:
                            raise CompilerError(
                                'Qubits to be acquired: {0} do not satisfy required qubits '
                                'in measurement map: {1}'.format(measured_qubits, meas_set),
                                )
