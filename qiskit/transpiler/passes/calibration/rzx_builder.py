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

import math
import warnings
from typing import List, Tuple, Union

import enum
import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    Play,
    Delay,
    Schedule,
    ScheduleBlock,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    Waveform,
)
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap, CalibrationPublisher

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
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16) -> Play:
        """
        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            qiskit.pulse.Play: The play instruction with the stretched compressed
                GaussianSquare pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        # This method is called for instructions which are guaranteed to play GaussianSquare pulse
        amp = instruction.pulse.amp
        width = instruction.pulse.width
        sigma = instruction.pulse.sigma
        n_sigmas = (instruction.pulse.duration - width) / sigma

        # The error function is used because the Gaussian may have chopped tails.
        gaussian_area = abs(amp) * sigma * np.sqrt(2 * np.pi) * math.erf(n_sigmas)
        area = gaussian_area + abs(amp) * width

        target_area = abs(theta) / (np.pi / 2.0) * area
        sign = np.sign(theta)

        if target_area > gaussian_area:
            width = (target_area - gaussian_area) / abs(amp)
            duration = round((width + n_sigmas * sigma) / sample_mult) * sample_mult
            return Play(
                GaussianSquare(amp=sign * amp, width=width, sigma=sigma, duration=duration),
                channel=instruction.channel,
            )
        else:
            amp_scale = sign * target_area / gaussian_area
            duration = round(n_sigmas * sigma / sample_mult) * sample_mult
            return Play(
                GaussianSquare(amp=amp * amp_scale, width=0, sigma=sigma, duration=duration),
                channel=instruction.channel,
            )

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

        rzx_theta = Schedule(name="rzx(%.3f)" % theta)
        rzx_theta.metadata["publisher"] = CalibrationPublisher.QISKIT

        if np.isclose(theta, 0.0):
            return rzx_theta

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
        is_native = comp_tones[0].channel.index == qubits[1]

        stretched_cr_tones = list(map(lambda p: self.rescale_cr_inst(p, theta), cr_tones))
        stretched_comp_tones = list(map(lambda p: self.rescale_cr_inst(p, theta), comp_tones))

        if is_native:
            xgate = self._inst_map.get("x", qubits[0])

            for cr, comp in zip(stretched_cr_tones, stretched_comp_tones):
                current_dur = rzx_theta.duration
                rzx_theta.insert(current_dur, cr, inplace=True)
                rzx_theta.insert(current_dur, comp, inplace=True)
                rzx_theta.append(xgate, inplace=True)

        else:
            # Add hadamard gate to flip
            xgate = self._inst_map.get("x", qubits[1])
            szc = self._inst_map.get("rz", qubits[1], np.pi / 2)
            sxc = self._inst_map.get("sx", qubits[1])
            szt = self._inst_map.get("rz", qubits[0], np.pi / 2)
            sxt = self._inst_map.get("sx", qubits[0])

            # Hadamard to control
            rzx_theta.insert(0, szc, inplace=True)
            rzx_theta.insert(0, sxc, inplace=True)
            rzx_theta.insert(sxc.duration, szc, inplace=True)

            # Hadamard to target
            rzx_theta.insert(0, szt, inplace=True)
            rzx_theta.insert(0, sxt, inplace=True)
            rzx_theta.insert(sxt.duration, szt, inplace=True)

            for cr, comp in zip(stretched_cr_tones, stretched_comp_tones):
                current_dur = rzx_theta.duration
                rzx_theta.insert(current_dur, cr, inplace=True)
                rzx_theta.insert(current_dur, comp, inplace=True)
                rzx_theta.append(xgate, inplace=True)

            current_dur = rzx_theta.duration

            # Hadamard to control
            rzx_theta.insert(current_dur, szc, inplace=True)
            rzx_theta.insert(current_dur, sxc, inplace=True)
            rzx_theta.insert(current_dur + sxc.duration, szc, inplace=True)

            # Hadamard to target
            rzx_theta.insert(current_dur, szt, inplace=True)
            rzx_theta.insert(current_dur, sxt, inplace=True)
            rzx_theta.insert(current_dur + sxt.duration, szt, inplace=True)

        return rzx_theta


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

        rzx_theta = Schedule(name="rzx(%.3f)" % theta)
        rzx_theta.metadata["publisher"] = CalibrationPublisher.QISKIT

        if np.isclose(theta, 0.0):
            return rzx_theta

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
        is_native = comp_tones[0].channel.index == qubits[1]

        stretched_cr_tone = self.rescale_cr_inst(cr_tones[0], 2 * theta)
        stretched_comp_tone = self.rescale_cr_inst(comp_tones[0], 2 * theta)

        if is_native:
            # Placeholder to make pulse gate work
            delay = Delay(stretched_cr_tone.duration, DriveChannel(qubits[0]))

            # This doesn't remove unwanted instruction such as ZI
            # These terms are eliminated along with other gates around the pulse gate.
            rzx_theta = rzx_theta.insert(0, stretched_cr_tone, inplace=True)
            rzx_theta = rzx_theta.insert(0, stretched_comp_tone, inplace=True)
            rzx_theta = rzx_theta.insert(0, delay, inplace=True)

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
