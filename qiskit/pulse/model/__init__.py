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

r"""
==========================================================
Logical Elements & Frames (:mod:`qiskit.pulse.model`)
==========================================================

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Qiskit Pulse's logical element and frames create a flexible framework
to define where pulse instructions are applied, and what would be their carrier frequency and phase
(because typically AC pulses are used). Each :class:`.LogicalElement` represents a separate component
in the quantum computing system on which instructions could be applied. On the other hand,
each :class:`.Frame` represents a frequency and phase duo for the carrier of the pulse.

This logical and virtual representation allows the user to write template pulse
programs without worrying about the exact details of the hardware implementation
(are the pulses to be played via the same port? Which NCO is used?), while still
allowing for effective utilization of the quantum hardware. The burden of mapping
the different combinations of :class:`.LogicalElement` and :class:`.Frame`
to hardware aware objects is left to the Pulse Compiler.

.. _logical_elements:

LogicalElement
================
:class:`.LogicalElement` s are identified by their type and index. Currently, the most prominent example
is the :class:`~.pulse.Qubit`.

.. autosummary::
   :toctree: ../stubs/

   Qubit
   Coupler


.. _frames:

Frame
=============
:class:`.Frame` s are identified by their type and unique identifier. A :class:`.GenericFrame` is used to
specify custom frequency
and phase duos, while :class:`.QubitFrame` and :class:`.MeasurementFrame` are used to indicate that
backend defaults are to be used (for the qubit's driving frequency and measurement frequency
respectively).

.. autosummary::
   :toctree: ../stubs/

   GenericFrame
   QubitFrame
   MeasurementFrame


.. _mixed_frames:

MixedFrame
=============
The combination of a :class:`.LogicalElement` and :class:`.Frame` is dubbed a :class:`.MixedFrame`.

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
