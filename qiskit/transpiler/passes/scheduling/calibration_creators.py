# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Calibration creators."""

import math
from typing import List, Union
from abc import abstractmethod
import numpy as np

from qiskit.pulse import (
    Play,
    Delay,
    ShiftPhase,
    Schedule,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
)
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.providers import basebackend
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.transpiler.basepasses import TransformationPass


class CalibrationCreator(TransformationPass):
    """Abstract base class to inject calibrations into circuits."""

    @abstractmethod
    def supported(self, node_op: DAGNode) -> bool:
        """Determine if a given name supports the calibration."""

    @abstractmethod
    def get_calibration(self, params: List, qubits: List) -> Schedule:
        """Gets the calibrated schedule for the given qubits and parameters."""

    def run(self, dag):
        """Run the calibration adder pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A DAG with calibrations added to it.
        """
        bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}

        for node in dag.nodes():
            if node.type == "op":
                if self.supported(node.op):
                    params = node.op.params
                    qubits = [bit_indices[qarg] for qarg in node.qargs]

                    schedule = self.get_calibration(params, qubits)

                    dag.add_calibration(node.op, qubits, schedule, params=params)

        return dag


class RZXCalibrationBuilder(CalibrationCreator):
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

    def __init__(self, backend: basebackend):
        """
        Initializes a RZXGate calibration builder.

        Args:
            backend: Backend for which to construct the gates.

        Raises:
            QiskitError: if open pulse is not supported by the backend.
        """
        super().__init__()
        if not backend.configuration().open_pulse:
            raise QiskitError(
                "Calibrations can only be added to Pulse-enabled backends, "
                "but {} is not enabled with Pulse.".format(backend.name())
            )

        self._inst_map = backend.defaults().instruction_schedule_map
        self._config = backend.configuration()
        self._channel_map = backend.configuration().qubit_channel_mapping

    def supported(self, node_op: DAGNode) -> bool:
        """
        Args:
            node_op: The node from the dag dep.

        Returns:
            match: True if the node is a RZXGate.
        """
        return isinstance(node_op, RZXGate)

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
            QiskitError: if the pulses are not GaussianSquare.
        """
        pulse_ = instruction.pulse
        if isinstance(pulse_, GaussianSquare):
            amp = pulse_.amp
            width = pulse_.width
            sigma = pulse_.sigma
            n_sigmas = (pulse_.duration - width) / sigma

            # The error function is used because the Gaussian may have chopped tails.
            gaussian_area = abs(amp) * sigma * np.sqrt(2 * np.pi) * math.erf(n_sigmas)
            area = gaussian_area + abs(amp) * width

            target_area = abs(theta) / (np.pi / 2.0) * area
            sign = theta / abs(theta)

            if target_area > gaussian_area:
                width = (target_area - gaussian_area) / abs(amp)
                duration = math.ceil((width + n_sigmas * sigma) / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(amp=sign * amp, width=width, sigma=sigma, duration=duration),
                    channel=instruction.channel,
                )
            else:
                amp_scale = sign * target_area / gaussian_area
                duration = math.ceil(n_sigmas * sigma / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(amp=amp * amp_scale, width=0, sigma=sigma, duration=duration),
                    channel=instruction.channel,
                )
        else:
            raise QiskitError("RZXCalibrationBuilder only stretches/compresses GaussianSquare.")

    def get_calibration(self, params: List, qubits: List) -> Schedule:
        """
        Args:
            params: Parameters of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if the control and target qubits cannot be identified or the backend
                does not support cx between the qubits.
        """
        theta = params[0]
        q1, q2 = qubits[0], qubits[1]

        if not self._inst_map.has("cx", qubits):
            raise QiskitError(
                "This transpilation pass requires the backend to support cx "
                "between qubits %i and %i." % (q1, q2)
            )

        cx_sched = self._inst_map.get("cx", qubits=(q1, q2))
        rzx_theta = Schedule(name="rzx(%.3f)" % theta)

        if theta == 0.0:
            return rzx_theta

        crs, comp_tones = [], []
        control, target = None, None

        for time, inst in cx_sched.instructions:

            # Identify the CR pulses.
            if isinstance(inst, Play) and not isinstance(inst, ShiftPhase):
                if isinstance(inst.channel, ControlChannel):
                    crs.append((time, inst))

            # Identify the compensation tones.
            if isinstance(inst.channel, DriveChannel) and not isinstance(inst, ShiftPhase):
                if isinstance(inst.pulse, GaussianSquare):
                    comp_tones.append((time, inst))
                    target = inst.channel.index
                    control = q1 if target == q2 else q2

        if control is None:
            raise QiskitError("Control qubit is None.")
        if target is None:
            raise QiskitError("Target qubit is None.")

        echo_x = self._inst_map.get("x", qubits=control)

        # Build the schedule

        # Stretch/compress the CR gates and compensation tones
        cr1 = self.rescale_cr_inst(crs[0][1], theta)
        cr2 = self.rescale_cr_inst(crs[1][1], theta)

        if len(comp_tones) == 0:
            comp1, comp2 = None, None
        elif len(comp_tones) == 2:
            comp1 = self.rescale_cr_inst(comp_tones[0][1], theta)
            comp2 = self.rescale_cr_inst(comp_tones[1][1], theta)
        else:
            raise QiskitError(
                "CX must have either 0 or 2 rotary tones between qubits %i and %i "
                "but %i were found." % (control, target, len(comp_tones))
            )

        # Build the schedule for the RZXGate
        rzx_theta = rzx_theta.insert(0, cr1)

        if comp1 is not None:
            rzx_theta = rzx_theta.insert(0, comp1)

        rzx_theta = rzx_theta.insert(comp1.duration, echo_x)
        time = comp1.duration + echo_x.duration
        rzx_theta = rzx_theta.insert(time, cr2)

        if comp2 is not None:
            rzx_theta = rzx_theta.insert(time, comp2)

        time = 2 * comp1.duration + echo_x.duration
        rzx_theta = rzx_theta.insert(time, echo_x)

        # Reverse direction of the ZX with Hadamard gates
        if control == qubits[0]:
            return rzx_theta
        else:
            rzc = self._inst_map.get("rz", [control], np.pi / 2)
            sxc = self._inst_map.get("sx", [control])
            rzt = self._inst_map.get("rz", [target], np.pi / 2)
            sxt = self._inst_map.get("sx", [target])
            h_sched = Schedule(name="hadamards")
            h_sched = h_sched.insert(0, rzc)
            h_sched = h_sched.insert(0, sxc)
            h_sched = h_sched.insert(sxc.duration, rzc)
            h_sched = h_sched.insert(0, rzt)
            h_sched = h_sched.insert(0, sxt)
            h_sched = h_sched.insert(sxc.duration, rzt)
            rzx_theta = h_sched.append(rzx_theta)
            return rzx_theta.append(h_sched)


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

    @staticmethod
    def _filter_control(inst: (int, Union["Schedule", Instruction])) -> bool:
        """
        Looks for Gaussian square pulses applied to control channels.

        Args:
            inst: Instructions to be filtered.

        Returns:
            match: True if the instruction is a Play instruction with
                a Gaussian square pulse on the ControlChannel.
        """
        if isinstance(inst[1], Play):
            if isinstance(inst[1].pulse, GaussianSquare) and isinstance(
                inst[1].channel, ControlChannel
            ):
                return True

        return False

    @staticmethod
    def _filter_drive(inst: (int, Union["Schedule", Instruction])) -> bool:
        """
        Looks for Gaussian square pulses applied to drive channels.

        Args:
            inst: Instructions to be filtered.

        Returns:
            match: True if the instruction is a Play instruction with
                a Gaussian square pulse on the DriveChannel.
        """
        if isinstance(inst[1], Play):
            if isinstance(inst[1].pulse, GaussianSquare) and isinstance(
                inst[1].channel, DriveChannel
            ):
                return True

        return False

    def get_calibration(self, params: List, qubits: List) -> Schedule:
        """
        Builds the calibration schedule for the RZXGate(theta) without echos.

        Args:
            params: Parameters of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: If the control and target qubits cannot be identified, or the backend
                does not support a cx gate between the qubits, or the backend does not natively
                support the specified direction of the cx.
        """
        theta = params[0]
        q1, q2 = qubits[0], qubits[1]

        if not self._inst_map.has("cx", qubits):
            raise QiskitError(
                "This transpilation pass requires the backend to support cx "
                "between qubits %i and %i." % (q1, q2)
            )

        cx_sched = self._inst_map.get("cx", qubits=(q1, q2))
        rzx_theta = Schedule(name="rzx(%.3f)" % theta)

        if theta == 0.0:
            return rzx_theta

        control, target = None, None

        for _, inst in cx_sched.instructions:
            # Identify the compensation tones.
            if isinstance(inst.channel, DriveChannel) and isinstance(inst, Play):
                if isinstance(inst.pulse, GaussianSquare):
                    target = inst.channel.index
                    control = q1 if target == q2 else q2

        if control is None:
            raise QiskitError("Control qubit is None.")
        if target is None:
            raise QiskitError("Target qubit is None.")

        if control != qubits[0]:
            raise QiskitError(
                "RZXCalibrationBuilderNoEcho only supports hardware-native RZX gates."
            )

        # Get the filtered Schedule instructions for the CR gates and compensation tones.
        crs = cx_sched.filter(*[self._filter_control]).instructions
        rotaries = cx_sched.filter(*[self._filter_drive]).instructions

        # Stretch/compress the CR gates and compensation tones.
        cr = self.rescale_cr_inst(crs[0][1], 2 * theta)
        rot = self.rescale_cr_inst(rotaries[0][1], 2 * theta)

        # Build the schedule for the RZXGate without the echos.
        rzx_theta = rzx_theta.insert(0, cr)
        rzx_theta = rzx_theta.insert(0, rot)
        rzx_theta = rzx_theta.insert(0, Delay(cr.duration, DriveChannel(control)))

        return rzx_theta
