# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""RZX calibration builders."""

import enum
import warnings
from math import pi, erf  # pylint: disable=no-name-in-module
from typing import List, Tuple, Union

import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    Play,
    Schedule,
    ScheduleBlock,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    Waveform,
)
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap

from .base_builder import CalibrationBuilder
from .exceptions import CalibrationNotAvailable


class CXCalType(enum.Enum):
    """Estimated calibration type of backend CX gate."""

    ECR = "Echoed Cross Resonance"
    DIRECT_CX = "Direct CX"


class RZXCalibrationBuilder(CalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate. This is done by retrieving (for a given pair of
    qubits) the CX schedule in the instruction schedule map of the backend defaults.
    The CX schedule must be an echoed cross-resonance gate optionally with rotary tones.
    The cross-resonance drive tones and rotary pulses must be Gaussian square pulses.
    The width of the Gaussian square pulse is adjusted so as to match the desired rotation angle.
    If the rotation angle is small such that the width disappears then the amplitude of the
    zero width Gaussian square pulse (i.e. a Gaussian) is reduced to reach the target rotation
    angle. Additional details can be found in https://arxiv.org/abs/2012.11660.
    """

    def __init__(
        self,
        instruction_schedule_map: InstructionScheduleMap = None,
        qubit_channel_mapping: List[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Initializes a RZXGate calibration builder.

        Args:
            instruction_schedule_map: The :obj:`InstructionScheduleMap` object representing the
                default pulse calibrations for the target backend
            qubit_channel_mapping: The list mapping qubit indices to the list of
                channel names that apply on that qubit.
            verbose: Set True to raise a user warning when RZX schedule cannot be built.

        Raises:
            QiskitError: Instruction schedule map is not provided.
        """
        super().__init__()

        if instruction_schedule_map is None:
            raise QiskitError("Calibrations can only be added to Pulse-enabled backends")

        if qubit_channel_mapping:
            warnings.warn(
                "'qubit_channel_mapping' is no longer used. This value is ignored.",
                DeprecationWarning,
            )

        self._inst_map = instruction_schedule_map
        self._verbose = verbose

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, RZXGate) and self._inst_map.has("cx", qubits)

    @staticmethod
    @builder.macro
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16) -> int:
        """A builder macro to play stretched pulse.

        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            Duration of stretched pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        # This method is called for instructions which are guaranteed to play GaussianSquare pulse
        params = instruction.pulse.parameters.copy()
        risefall_sigma_ratio = (params["duration"] - params["width"]) / params["sigma"]

        # The error function is used because the Gaussian may have chopped tails.
        # Area is normalized by amplitude.
        # This makes widths robust to the rounding error.
        risefall_area = params["sigma"] * np.sqrt(2 * pi) * erf(risefall_sigma_ratio)
        full_area = params["width"] + risefall_area

        # Get estimate of target area. Assume this is pi/2 controlled rotation.
        cal_angle = pi / 2
        target_area = abs(theta) / cal_angle * full_area
        new_width = target_area - risefall_area

        if new_width >= 0:
            width = new_width
            params["amp"] *= np.sign(theta)
        else:
            width = 0
            params["amp"] *= np.sign(theta) * target_area / risefall_area

        round_duration = (
            round((width + risefall_sigma_ratio * params["sigma"]) / sample_mult) * sample_mult
        )
        params["duration"] = round_duration
        params["width"] = width

        stretched_pulse = GaussianSquare(**params)
        builder.play(stretched_pulse, instruction.channel)

        return round_duration

    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Builds the calibration schedule for the RZXGate(theta) with echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]

        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        if np.isclose(theta, 0.0):
            return ScheduleBlock(name="rzx(0.000)")

        cx_sched = self._inst_map.get("cx", qubits=qubits)
        cal_type, cr_tones, comp_tones = _check_calibration_type(cx_sched)

        if cal_type != CXCalType.ECR:
            if self._verbose:
                warnings.warn(
                    f"CX instruction for qubits {qubits} is likely {cal_type.value} sequence. "
                    "Pulse stretch for this calibration is not currently implemented. "
                    "RZX schedule is not generated for this qubit pair.",
                    UserWarning,
                )
            raise CalibrationNotAvailable

        if len(comp_tones) == 0:
            raise QiskitError(
                f"{repr(cx_sched)} has no target compensation tones. "
                "Native CR direction cannot be determined."
            )

        # Determine native direction, assuming only single drive channel per qubit.
        # This guarantees channel and qubit index equality.
        if comp_tones[0].channel.index == qubits[1]:
            xgate = self._inst_map.get("x", qubits[0])
            with builder.build(
                default_alignment="sequential", name="rzx(%.3f)" % theta
            ) as rzx_theta_native:
                for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                    with builder.align_left():
                        self.rescale_cr_inst(cr_tone, theta)
                        self.rescale_cr_inst(comp_tone, theta)
                    builder.call(xgate)
            return rzx_theta_native

        # The direction is not native. Add Hadamard gates to flip the direction.
        xgate = self._inst_map.get("x", qubits[1])
        szc = self._inst_map.get("rz", qubits[1], pi / 2)
        sxc = self._inst_map.get("sx", qubits[1])
        szt = self._inst_map.get("rz", qubits[0], pi / 2)
        sxt = self._inst_map.get("sx", qubits[0])
        with builder.build(name="hadamard") as hadamard:
            # Control qubit
            builder.call(szc, name="szc")
            builder.call(sxc, name="sxc")
            builder.call(szc, name="szc")
            # Target qubit
            builder.call(szt, name="szt")
            builder.call(sxt, name="sxt")
            builder.call(szt, name="szt")

        with builder.build(
            default_alignment="sequential", name="rzx(%.3f)" % theta
        ) as rzx_theta_flip:
            builder.call(hadamard, name="hadamard")
            for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                with builder.align_left():
                    self.rescale_cr_inst(cr_tone, theta)
                    self.rescale_cr_inst(comp_tone, theta)
                builder.call(xgate)
            builder.call(hadamard, name="hadamard")
        return rzx_theta_flip


class RZXCalibrationBuilderNoEcho(RZXCalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate.

    The ``RZXCalibrationBuilderNoEcho`` is a variation of the
    :class:`~qiskit.transpiler.passes.RZXCalibrationBuilder` pass
    that creates calibrations for the cross-resonance pulses without inserting
    the echo pulses in the pulse schedule. This enables exposing the echo in
    the cross-resonance sequence as gates so that the transpiler can simplify them.
    The ``RZXCalibrationBuilderNoEcho`` only supports the hardware-native direction
    of the CX gate.
    """

    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Builds the calibration schedule for the RZXGate(theta) without echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified,
                or the backend does not natively support the specified direction of the cx.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]

        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        if np.isclose(theta, 0.0):
            return ScheduleBlock(name="rzx(0.000)")

        cx_sched = self._inst_map.get("cx", qubits=qubits)
        cal_type, cr_tones, comp_tones = _check_calibration_type(cx_sched)

        if cal_type != CXCalType.ECR:
            if self._verbose:
                warnings.warn(
                    f"CX instruction for qubits {qubits} is likely {cal_type.value} sequence. "
                    "Pulse stretch for this calibration is not currently implemented. "
                    "RZX schedule is not generated for this qubit pair.",
                    UserWarning,
                )
            raise CalibrationNotAvailable

        if len(comp_tones) == 0:
            raise QiskitError(
                f"{repr(cx_sched)} has no target compensation tones. "
                "Native CR direction cannot be determined."
            )

        # Determine native direction, assuming only single drive channel per qubit.
        # This guarantees channel and qubit index equality.
        if comp_tones[0].channel.index == qubits[1]:
            with builder.build(default_alignment="left", name="rzx(%.3f)" % theta) as rzx_theta:
                stretched_dur = self.rescale_cr_inst(cr_tones[0], 2 * theta)
                self.rescale_cr_inst(comp_tones[0], 2 * theta)
                # Placeholder to make pulse gate work
                builder.delay(stretched_dur, DriveChannel(qubits[0]))
            return rzx_theta

        raise QiskitError("RZXCalibrationBuilderNoEcho only supports hardware-native RZX gates.")


def _filter_cr_tone(time_inst_tup):
    """A helper function to filter pulses on control channels."""
    valid_types = ["GaussianSquare"]

    _, inst = time_inst_tup
    if isinstance(inst, Play) and isinstance(inst.channel, ControlChannel):
        pulse = inst.pulse
        if isinstance(pulse, Waveform) or pulse.pulse_type in valid_types:
            return True
    return False


def _filter_comp_tone(time_inst_tup):
    """A helper function to filter pulses on drive channels."""
    valid_types = ["GaussianSquare"]

    _, inst = time_inst_tup
    if isinstance(inst, Play) and isinstance(inst.channel, DriveChannel):
        pulse = inst.pulse
        if isinstance(pulse, Waveform) or pulse.pulse_type in valid_types:
            return True
    return False


def _check_calibration_type(cx_sched) -> Tuple[CXCalType, List[Play], List[Play]]:
    """A helper function to check type of CR calibration.

    Args:
        cx_sched: A target schedule to stretch.

    Returns:
        Filtered instructions and most-likely type of calibration.

    Raises:
        QiskitError: Unknown calibration type is detected.
    """
    cr_tones = list(
        map(lambda t: t[1], filter_instructions(cx_sched, [_filter_cr_tone]).instructions)
    )
    comp_tones = list(
        map(lambda t: t[1], filter_instructions(cx_sched, [_filter_comp_tone]).instructions)
    )

    if len(cr_tones) == 2 and len(comp_tones) in (0, 2):
        # ECR can be implemented without compensation tone at price of lower fidelity.
        # Remarkable noisy terms are usually eliminated by echo.
        return CXCalType.ECR, cr_tones, comp_tones

    if len(cr_tones) == 1 and len(comp_tones) == 1:
        # Direct CX must have compensation tone on target qubit.
        # Otherwise, it cannot eliminate IX interaction.
        return CXCalType.DIRECT_CX, cr_tones, comp_tones

    raise QiskitError(
        f"{repr(cx_sched)} is undefined pulse sequence. "
        "Check if this is a calibration for CX gate."
    )
