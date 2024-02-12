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

"""The phase instructions update the modulation phase of pulses played on a channel.
This includes ``SetPhase`` instructions which lock the modulation to a particular phase
at that moment, and ``ShiftPhase`` instructions which increase the existing phase by a
relative amount.
"""
from __future__ import annotations

from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import FrameUpdate
from qiskit.pulse.model import Frame, MixedFrame, PulseTarget


class ShiftPhase(FrameUpdate):
    r"""The shift phase instruction updates the modulation phase of proceeding pulses associated with
    the frame the instruction is acting upon. It is a relative increase in phase determined
    by the ``phase`` operand.

    In particular, played pulses take the form:

    .. math::
        Re[\exp(i 2\pi f jdt + \phi) d_j].

    The ``ShiftPhase`` instruction causes :math:`\phi` to be increased by the instruction's
    ``phase`` operand. This will affect all pulses following on the same frame.

    The qubit phase is tracked in software, enabling instantaneous, nearly error-free Z-rotations
    by using a ShiftPhase to update the frame tracking the qubit state.
    """

    def __init__(
        self,
        phase: ParameterValueType,
        *,
        frame: Frame | None = None,
        target: PulseTarget | None = None,
        mixed_frame: MixedFrame | None = None,
        channel: PulseChannel | None = None,
        name: str | None = None,
    ):
        """Instantiate a shift phase instruction, increasing the output signal phase on a given
        mixed frame (=channel) or frame by ``phase`` [radians].

        The instruction can be set on a ``MixedFrame`` (=``Channel``) or a ``Frame``. For the latter,
        provide only the ``frame`` argument, and the instruction will be broadcasted to all
        ``MixedFrame``s involving the ``Frame``. For the former, provide exactly one of
        ``mixed_frame``, ``channel`` or the duo ``target`` and ``frame``, and the instruction
        will apply only to the specified ``MixedFrame``.

        Args:
            phase: The rotation angle in radians.
            frame: The frame the instruction will apply to.
            target: The target which in conjunction with ``frame`` defines the mixed frame that
                instruction will apply to.
            mixed_frame: The mixed_frame the instruction will apply to.
            channel: The channel the instruction will apply to.
            name: Display name for this instruction.
        """
        inst_target = self._validate_and_format_frame(target, frame, mixed_frame, channel)
        super().__init__(operands=(phase, inst_target), name=name)

    @property
    def phase(self) -> ParameterValueType:
        """Return the rotation angle enacted by this instruction in radians."""
        return self.operands[0]


class SetPhase(FrameUpdate):
    r"""The set phase instruction updates the modulation phase of proceeding pulses associated with
    the frame the instruction is acting upon. It sets the phase to the ``phase`` operand.

    In particular, played pulses take the form:

    .. math::

        Re[\exp(i 2\pi f jdt + \phi) d_j]

    The ``SetPhase`` instruction sets :math:`\phi` to the instruction's ``phase`` operand.
    """

    def __init__(
        self,
        phase: ParameterValueType,
        *,
        frame: Frame | None = None,
        target: PulseTarget | None = None,
        mixed_frame: MixedFrame | None = None,
        channel: PulseChannel | None = None,
        name: str = None,
    ):
        """Instantiate a set phase instruction, setting the output signal phase on a given
        mixed frame (=channel) or frame to ``phase`` [radians]..

        The instruction can be set on a ``MixedFrame`` (=``Channel``) or a ``Frame``. For the latter,
        provide only the ``frame`` argument, and the instruction will be broadcasted to all
        ``MixedFrame``s involving the ``Frame``. For the former, provide exactly one of
        ``mixed_frame``, ``channel`` or the duo ``target`` and ``frame``, and the instruction
        will apply only to the specified ``MixedFrame``.

        Args:
            phase: The rotation angle in radians.
            frame: The frame the instruction will apply to.
            target: The target which in conjunction with ``frame`` defines the mixed frame that
                instruction will apply to.
            mixed_frame: The mixed_frame the instruction will apply to.
            channel: The channel the instruction will apply to.
            name: Display name for this instruction.
        """
        inst_target = self._validate_and_format_frame(target, frame, mixed_frame, channel)
        super().__init__(operands=(phase, inst_target), name=name)

    @property
    def phase(self) -> ParameterValueType:
        """Return the rotation angle enacted by this instruction in radians."""
        return self.operands[0]
