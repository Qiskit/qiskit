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
.. _pulse-logical-elements-frames:

=======================================
Logical Elements & Frames (:mod:`qiskit.pulse.logical_elements_frames`)
=======================================

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Qiskit Pulse's logical element and frames are meant to create a flexible framework
to define where pulses instructions are applied, and what would be their carrier frequency and phase
(because typically AC pulses are used). Each :class:`LogicalElement` represents a separate component
in the quantum computing system on which instructions could be applied. On the other hand, each :class:`Frame`
represents a frequency and phase duo for the carrier of the pulse.

This logical and virtual representation allows the user to write template pulse
programs without worrying about the exact details of the HW implementation (are the pulses to be played
via the same port? Which NCO is used?), while still allowing for effective utilization of the quantum
HW. The burden of mapping the different combinations of :class:`LogicalElement`s and :class:`Frame`s
to HW aware objects is left to the Pulse Compiler.

LogicalElement
=============
:class:`LogicalElement`s are identified by their type and index. Currently, the most prominent example
is the :class:`Qubit`.

.. autosummary::
   :toctree: ../stubs/

   Qubit
   Coupler

Frame
=============
:class:`Frame`s are identified by their name. A :class:`GenericFrame` is used to specify custom frequency
and phase duos, while :class:`QubitFrame` and :class:`MeasurementFrame` are used to indicate that backend
defaults are to be used (for the qubit's driving frequency and measurement frequency respectively).

.. autosummary::
   :toctree: ../stubs/

   GenericFrame
   QubitFrame
   MeasurementFrame


MixedFrame
=============
The combination of a :class:`LogicalElement` and :class:`Frame` is dubbed a :class:`MixedFrame`.

.. autosummary::
   :toctree: ../stubs/

   MixedFrame
"""

from .logical_elements import (
    LogicalElement,
    Qubit,
    Coupler,
)

from .frames import (
    Frame,
    GenericFrame,
    QubitFrame,
    MeasurementFrame,
)

from .mixed_frames import (
    MixedFrame,
)
