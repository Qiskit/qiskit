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

"""
Command definition module. Relates circuit gates to pulse commands.
"""
import warnings

from typing import List, Tuple, Iterable, Union, Dict, Optional

from qiskit.qobj import PulseQobjInstruction
from .commands import SamplePulse



class CmdDef:
    """Command definition class. Relates `Gate`s to `Schedule`s."""

    def __new__(cls, schedules: Optional[Dict] = None):
        warnings.warn("The CmdDef is being deprecated. All CmdDef methods are now supported by "
                      "`PulseDefaults` accessible as backend.defaults() for any Pulse enabled "
                      "system. Use defaults instead.",
                      DeprecationWarning)
        from qiskit.providers.models.pulsedefaults import CircuitOperationToScheduleMap
        replacement = CircuitOperationToScheduleMap([], [])
        if schedules:
            for key, schedule in schedules.items():
                replacement.add(key[0], key[1:], schedule)
        return replacement

    @classmethod
    def from_defaults(cls, flat_cmd_def: List[PulseQobjInstruction],
                      pulse_library: Dict[str, SamplePulse],
                      buffer: int = 0) -> 'CmdDef':
        """Create command definition from backend defaults output.

        Args:
            flat_cmd_def: Command definition list returned by backend
            pulse_library: Dictionary of `SamplePulse`s
            buffer: Buffer between instructions on channel
        """
        if buffer:
            warnings.warn("Buffers are no longer supported. Please use an explicit Delay.")
        return CircuitOperationToScheduleMap(flat_cmd_def, pulse_library)
