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

r"""
.. _pulse-insts:

===============================================
Instructions (:mod:`qiskit.pulse.instructions`)
===============================================

The ``instructions`` module holds the various :obj:`Instruction`\ s which are supported by
Qiskit Pulse. Instructions have operands, which typically include at least one
:py:class:`~qiskit.pulse.channels.Channel` specifying where the instruction will be applied.

Every instruction has a duration, whether explicitly included as an operand or implicitly defined.
For instance, a :py:class:`~qiskit.pulse.instructions.ShiftPhase` instruction can be instantiated
with operands *phase* and *channel*, for some float ``phase`` and a
:py:class:`~qiskit.pulse.channels.Channel` ``channel``::

    ShiftPhase(phase, channel)

The duration of this instruction is implicitly zero. On the other hand, the
:py:class:`~qiskit.pulse.instructions.Delay` instruction takes an explicit duration::

    Delay(duration, channel)

An instruction can be added to a :py:class:`~qiskit.pulse.Schedule`, which is a
sequence of scheduled Pulse ``Instruction`` s over many channels. ``Instruction`` s and
``Schedule`` s implement the same interface.

.. autosummary::
   :toctree: ../stubs/

   Acquire
   Call
   Reference
   Delay
   Play
   RelativeBarrier
   SetFrequency
   ShiftFrequency
   SetPhase
   ShiftPhase
   Snapshot
   TimeBlockade

These are all instances of the same base class:

.. autoclass:: Instruction
"""
from .acquire import Acquire
from .delay import Delay
from .directives import Directive, RelativeBarrier, TimeBlockade
from .call import Call
from .instruction import Instruction
from .frequency import SetFrequency, ShiftFrequency
from .phase import ShiftPhase, SetPhase
from .play import Play
from .snapshot import Snapshot
from .reference import Reference
