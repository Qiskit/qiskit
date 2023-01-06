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

"""Module for common pulse programming macros."""

import math
import warnings
from typing import Dict, List, Union, Optional

from qiskit import pulse
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.pulse import channels, exceptions, instructions, utils
from qiskit.pulse.library.symbolic_pulses import GaussianRiseEdge, GaussianFallEdge
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms import AlignSequential


def measure(
    qubits: List[int],
    backend=None,
    inst_map: Optional[InstructionScheduleMap] = None,
    meas_map: Optional[Union[List[List[int]], Dict[int, List[int]]]] = None,
    qubit_mem_slots: Optional[Dict[int, int]] = None,
    measure_name: str = "measure",
) -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    instruction mapping and measure map, or by using the defaults provided by the backend.

    By default, the measurement results for each qubit are trivially mapped to the qubit
    index. This behavior is overridden by qubit_mem_slots. For instance, to measure
    qubit 0 into MemorySlot(1), qubit_mem_slots can be provided as {0: 1}.

    Args:
        qubits: List of qubits to be measured.
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``instruction_schedule_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.

    Returns:
        A measurement schedule corresponding to the inputs provided.

    Raises:
        PulseError: If both ``inst_map`` or ``meas_map``, and ``backend`` is None.
    """
    schedule = Schedule(name=f"Default measurement schedule for qubits {qubits}")
    try:
        inst_map = inst_map or backend.defaults().instruction_schedule_map
        meas_map = meas_map or backend.configuration().meas_map
    except AttributeError as ex:
        raise exceptions.PulseError(
            "inst_map or meas_map, and backend cannot be None simultaneously"
        ) from ex
    if isinstance(meas_map, list):
        meas_map = utils.format_meas_map(meas_map)

    measure_groups = set()
    for qubit in qubits:
        measure_groups.add(tuple(meas_map[qubit]))
    for measure_group_qubits in measure_groups:
        if qubit_mem_slots is not None:
            unused_mem_slots = set(measure_group_qubits) - set(qubit_mem_slots.values())
        try:
            default_sched = inst_map.get(measure_name, measure_group_qubits)
        except exceptions.PulseError as ex:
            raise exceptions.PulseError(
                "We could not find a default measurement schedule called '{}'. "
                "Please provide another name using the 'measure_name' keyword "
                "argument. For assistance, the instructions which are defined are: "
                "{}".format(measure_name, inst_map.instructions)
            ) from ex
        for time, inst in default_sched.instructions:
            if inst.channel.index not in qubits:
                continue
            if qubit_mem_slots and isinstance(inst, instructions.Acquire):
                if inst.channel.index in qubit_mem_slots:
                    mem_slot = channels.MemorySlot(qubit_mem_slots[inst.channel.index])
                else:
                    mem_slot = channels.MemorySlot(unused_mem_slots.pop())
                inst = instructions.Acquire(inst.duration, inst.channel, mem_slot=mem_slot)
            # Measurement pulses should only be added if its qubit was measured by the user
            schedule = schedule.insert(time, inst)

    return schedule


def measure_all(backend) -> Schedule:
    """
    Return a Schedule which measures all qubits of the given backend.

    Args:
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.

    Returns:
        A schedule corresponding to the inputs provided.
    """
    return measure(qubits=list(range(backend.configuration().n_qubits)), backend=backend)


def chunking_gaussian_square(
    duration: int,
    amp: ParameterValueType,
    angle: ParameterValueType,
    sigma: ParameterValueType,
    risefall_sigma_ratio: ParameterValueType,
    channel: PulseChannel,
    granularity: int,
    name: Optional[str] = None,
    limit_amplitude: Optional[bool] = None,
    chunk_size: int = 256,
    min_chunk_number: int = 10,
) -> ScheduleBlock:
    """A macro to build long GaussianSquare pulse in the chunk divided fashion.

    When we play a very long Gaussian Square pulse, such a pulse may quickly consume
    the waveform memory resource of the waveform generator of the corresponding channel,
    depending on how the microarchitecture of the pulse controller is designed.

    This macro will provide a memory-efficient encoding of the pulse envelope, in other words,
    it splits the flat-top part of the pulse into multiple short constant envelopes,
    and concatenates all of them together with the rise and fall edges to
    express the entire pulse envelope. This chunking allows the pulse controller to compress
    the very long flat-top part into a single short constant pulse in the waveform memory,
    and thus it can reduce waveform memory footprint for a single job.

    Chunked pulse can be parameterized except for the `duration`.

    .. note::

        A chunked pulse may increase the payload size because rising and falling edges
        are expressed by the raw samples rather than in the parametric form.
        When required chunk, i.e. number of repeated constant pulse, is less than the
        `min_chunk_number` limit, this macro function plays a single
        :class:`.GaussianSquare` pulse instead of generating a chunked pulse schedule.

    .. see_also::
        :class:`.GaussianSquare` for definition of the pulse envelope.

    Args:
        duration: Pulse length in terms of the sampling period `dt`.
            This duration must be an integer numer. Parameterization is not acceptable.
            Duration may be rounded to match the chunk size and ramp durations
            with the device timing constraint of the pulse granularity.
        amp: The magnitude of the amplitude of the Gaussian and square pulse.
            Complex amp support will be deprecated.
        angle: The angle of the complex amplitude of the pulse.
        sigma: A measure of how wide or narrow the Gaussian risefall is.
        risefall_sigma_ratio: The ratio of each risefall duration to sigma.
        channel: A pulse channel to play this pulse.
        granularity: A device timing constraint for the pulse granularity.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        chunk_size: Size of a single chunk in units of dt. This must satisfy the
            device timing constraint for the pulse granularity.
        min_chunk_number: Minimum chunk number to enable the chunk division.
            This macro plays a single :class:`.GaussianSquare` pulse when
            the pulse duration is sufficiently short.

    Returns:
        ScheduleBlock of the chunked GaussianSquare pulse sequence.

    Raises:
        PulseError: When duration is parameterized.
    """
    # TODO this must be called from proper pulse compiler once implemented.
    #  i.e. Standard GaussianSquare is implicitly replaced with this based on a compiler option.
    #  Experimentalist doesn't need to take care of chunking like this.

    if not isinstance(duration, int):
        raise PulseError(
            "Parameterized duration cannot be applied to the chunked pulse. "
            "Specify an integer duration in units of dt."
        )

    if chunk_size % granularity != 0:
        _new_size = granularity * int(chunk_size / granularity)
        warnings.warn(
            f"Chunk size of {chunk_size} dt doesn't match with the device timing constraint. "
            f"The value is rounded to the nearest valid size of {_new_size} dt.",
            UserWarning,
        )
        chunk_size = _new_size
    width = duration - 2 * sigma * risefall_sigma_ratio
    n_chunks = int(width / chunk_size)

    schedule = ScheduleBlock(name=name, alignment_context=AlignSequential())

    if n_chunks < min_chunk_number:
        # Pulse is very short. No need to apply chunk division.
        # Usually chunk division increases payload size because rising and falling edges
        # are submitted down to the device in the waveform format.
        # Switch back to the standard GaussianSquare pulse to compress payload.
        valid_duration = granularity * int(duration / granularity)
        schedule.append(
            instructions.Play(
                pulse.GaussianSquare(
                    duration=valid_duration,
                    amp=amp,
                    sigma=sigma,
                    risefall_sigma_ratio=risefall_sigma_ratio,
                    name=name,
                    limit_amplitude=limit_amplitude,
                ),
                channel=channel,
            ),
            inplace=True,
        )
    else:
        # Extra flat-top duration absorbed by ramps, i.e. waveform
        chunked_flat_top_size = n_chunks * chunk_size
        total_slack = width - chunked_flat_top_size

        # Effective pulse edge duration
        # This must guarantee the duration > sigma * risefall_sigma_ratio
        # It rounds up duration rather than truncating.
        edge_size = granularity * math.ceil(
            (sigma * risefall_sigma_ratio + total_slack / 2) / granularity
        )

        # Rising edge
        schedule.append(
            instructions.Play(
                GaussianRiseEdge(
                    amp=amp,
                    angle=angle,
                    duration=edge_size,
                    sigma=sigma,
                    risefall_sigma_ratio=risefall_sigma_ratio,
                    limit_amplitude=limit_amplitude,
                    name=name + "_rise" if name else None,
                ),
                channel=channel,
            ),
            inplace=True,
        )
        # Flat-top part
        flat_top_pulse = pulse.Constant(
            duration=chunk_size,
            amp=amp,
            angle=angle,
            limit_amplitude=limit_amplitude,
            name=name + "_flat" if name else None,
        )
        for _ in range(n_chunks):
            schedule.append(
                instructions.Play(flat_top_pulse, channel=channel),
                inplace=True,
            )
        # Falling edge
        schedule.append(
            instructions.Play(
                GaussianFallEdge(
                    amp=amp,
                    angle=angle,
                    duration=edge_size,
                    sigma=sigma,
                    risefall_sigma_ratio=risefall_sigma_ratio,
                    limit_amplitude=limit_amplitude,
                    name=name + "_fall" if name else None,
                ),
                channel=channel,
            ),
            inplace=True,
        )

    return schedule
