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

"""The ``instruction`` module holds the various ``Instruction`` s which are supported by
Qiskit Pulse. Instructions accept a list of operands unique to instructions of that type.
Instructions typically include at least one :py:class:`~qiskit.pulse.channels.Channel` as an
operand specifying where the instruction will be applied, and every instruction has a duration,
whether implicitly or explicitly defined.

For instance, a :py:class:`~qiskit.pulse.instructions.ShiftPhase` instruction can be instantiated
with operands *phase* and *channel*, for some float ``phase`` and a
:py:class`~qiskit.pulse.channels.Channel` ``channel``::

    ShiftPhase(phase, channel)

The duration of this instruction is implicitly zero.

An instruction can be added to a :py:class:`~qiskit.pulse.Schedule`, which is a
sequence of scheduled Pulse ``Instruction`` s over many channels.
"""
from .acquire import Acquire
from .delay import Delay
from .instruction import Instruction
from .frequency import SetFrequency
from .phase import ShiftPhase
from .play import Play
from .snapshot import Snapshot
