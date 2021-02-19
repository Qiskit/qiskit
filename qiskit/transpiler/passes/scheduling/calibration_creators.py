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
from typing import List
from abc import abstractmethod
import numpy as np

from qiskit.pulse import Play, ShiftPhase, Schedule, ControlChannel, DriveChannel, GaussianSquare
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
        for node in dag.nodes():
            if node.type == 'op':
                if self.supported(node.op):
                    params = node.op.params
                    qubits = [_.index for _ in node.qargs]

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
            raise QiskitError('Calibrations can only be added to Pulse-enabled backends, '
                              'but {0} is not enabled with Pulse.'.format(backend.name()))

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
            Play: The play instruction with the stretched compressed GaussianSquare pulse.

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

            target_area = abs(theta) / (np.pi / 2.) * area
            sign = theta / abs(theta)

            if target_area > gaussian_area:
                width = (target_area - gaussian_area) / abs(amp)
                duration = math.ceil((width + n_sigmas * sigma) / sample_mult) * sample_mult
                return Play(GaussianSquare(amp=sign*amp, width=width, sigma=sigma,
                                           duration=duration), channel=instruction.channel)
            else:
                amp_scale = sign * target_area / gaussian_area
                duration = math.ceil(n_sigmas * sigma / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(amp=amp * amp_scale, width=0, sigma=sigma, duration=duration),
                    channel=instruction.channel)
        else:
            raise QiskitError('RZXCalibrationBuilder only stretches/compresses GaussianSquare.')

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

        if not self._inst_map.has('cx', qubits):
            raise QiskitError('This transpilation pass requires the backend to support cx '
                              'between qubits %i and %i.' % (q1, q2))

        cx_sched = self._inst_map.get('cx', qubits=(q1, q2))
        rzx_theta = Schedule(name='rzx(%.3f)' % theta)

        if theta == 0.0:
            return rzx_theta

        crs, comp_tones, shift_phases = [], [], []
        control, target = None, None

        for time, inst in cx_sched.instructions:

            if isinstance(inst, ShiftPhase) and time == 0:
                shift_phases.append(ShiftPhase(-theta, inst.channel))

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
            raise QiskitError('Control qubit is None.')
        if target is None:
            raise QiskitError('Target qubit is None.')

        echo_x = self._inst_map.get('x', qubits=control)

        # Build the schedule
        for inst in shift_phases:
            rzx_theta = rzx_theta.insert(0, inst)

        # Stretch/compress the CR gates and compensation tones
        cr1 = self.rescale_cr_inst(crs[0][1], theta)
        cr2 = self.rescale_cr_inst(crs[1][1], theta)

        if len(comp_tones) == 0:
            comp1, comp2 = None, None
        elif len(comp_tones) == 2:
            comp1 = self.rescale_cr_inst(comp_tones[0][1], theta)
            comp2 = self.rescale_cr_inst(comp_tones[1][1], theta)
        else:
            raise QiskitError('CX must have either 0 or 2 rotary tones between qubits %i and %i '
                              'but %i were found.' % (control, target, len(comp_tones)))

        # Build the schedule for the RZXGate
        rzx_theta = rzx_theta.insert(0, cr1)

        if comp1 is not None:
            rzx_theta = rzx_theta.insert(0, comp1)

        rzx_theta = rzx_theta.insert(comp1.duration, echo_x)
        time = comp1.duration + echo_x.duration
        rzx_theta = rzx_theta.insert(time, cr2)

        if comp2 is not None:
            rzx_theta = rzx_theta.insert(time, comp2)

        time = 2*comp1.duration + echo_x.duration
        rzx_theta = rzx_theta.insert(time, echo_x)

        # Reverse direction of the ZX with Hadamard gates
        if control == qubits[0]:
            return rzx_theta
        else:
            rzc = self._inst_map.get('rz', [control], np.pi / 2)
            sxc = self._inst_map.get('sx', [control])
            rzt = self._inst_map.get('rz', [target], np.pi / 2)
            sxt = self._inst_map.get('sx', [target])
            h_sched = Schedule(name='hadamards')
            h_sched = h_sched.insert(0, rzc)
            h_sched = h_sched.insert(0, sxc)
            h_sched = h_sched.insert(sxc.duration, rzc)
            h_sched = h_sched.insert(0, rzt)
            h_sched = h_sched.insert(0, sxt)
            h_sched = h_sched.insert(sxc.duration, rzt)
            rzx_theta = h_sched.append(rzx_theta)
            return rzx_theta.append(h_sched)
