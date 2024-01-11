# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
===========================
Pulse (:mod:`qiskit.pulse`)
===========================

.. currentmodule:: qiskit.pulse

Qiskit-Pulse is a pulse-level quantum programming kit. This lower level of
programming offers the user more control than programming with
:py:class:`~qiskit.circuit.QuantumCircuit`\ s.

Extracting the greatest performance from quantum hardware requires real-time
pulse-level instructions. Pulse answers that need: it enables the quantum
physicist *user* to specify the exact time dynamics of an experiment.
It is especially powerful for error mitigation techniques.

The input is given as arbitrary, time-ordered signals (see: :ref:`Instructions <pulse-insts>`)
scheduled in parallel over multiple virtual hardware or simulator resources
(see: :ref:`Channels <pulse-channels>`). The system also allows the user to recover the
time dynamics of the measured output.

This is sufficient to allow the quantum physicist to explore and correct for
noise in a quantum system.

.. automodule:: qiskit.pulse.instructions
.. automodule:: qiskit.pulse.library
.. automodule:: qiskit.pulse.channels
.. automodule:: qiskit.pulse.schedule
.. automodule:: qiskit.pulse.transforms
.. automodule:: qiskit.pulse.builder

.. currentmodule:: qiskit.pulse

Configuration
=============

.. autosummary::
   :toctree: ../stubs/

   InstructionScheduleMap

Exceptions
==========

.. autoexception:: PulseError
.. autoexception:: BackendNotSet
.. autoexception:: NoActiveBuilder
.. autoexception:: UnassignedDurationError
.. autoexception:: UnassignedReferenceError
"""

# Builder imports.
from qiskit.pulse.builder import (
    # Construction methods.
    active_backend,
    active_transpiler_settings,
    active_circuit_scheduler_settings,
    build,
    num_qubits,
    qubit_channels,
    samples_to_seconds,
    seconds_to_samples,
    # Instructions.
    acquire,
    barrier,
    call,
    delay,
    play,
    reference,
    set_frequency,
    set_phase,
    shift_frequency,
    shift_phase,
    snapshot,
    # Channels.
    acquire_channel,
    control_channels,
    drive_channel,
    measure_channel,
    # Contexts.
    align_equispaced,
    align_func,
    align_left,
    align_right,
    align_sequential,
    circuit_scheduler_settings,
    frequency_offset,
    phase_offset,
    transpiler_settings,
    # Macros.
    macro,
    measure,
    measure_all,
    delay_qubits,
    # Circuit instructions.
    cx,
    u1,
    u2,
    u3,
    x,
)
from qiskit.pulse.channels import (
    AcquireChannel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
    MemorySlot,
    RegisterSlot,
    SnapshotChannel,
)
from qiskit.pulse.configuration import (
    Discriminator,
    Kernel,
    LoConfig,
    LoRange,
)
from qiskit.pulse.exceptions import (
    PulseError,
    BackendNotSet,
    NoActiveBuilder,
    UnassignedDurationError,
    UnassignedReferenceError,
)
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import (
    Acquire,
    Call,
    Delay,
    Instruction,
    Play,
    SetFrequency,
    SetPhase,
    ShiftFrequency,
    ShiftPhase,
    Snapshot,
)
from qiskit.pulse.library import (
    Constant,
    Drag,
    Gaussian,
    GaussianSquare,
    GaussianSquareDrag,
    gaussian_square_echo,
    Sin,
    Cos,
    Sawtooth,
    Triangle,
    Square,
    GaussianDeriv,
    Sech,
    SechDeriv,
    ParametricPulse,
    SymbolicPulse,
    ScalableSymbolicPulse,
    Waveform,
)
from qiskit.pulse.library.samplers.decorators import functional_pulse
from qiskit.pulse.schedule import Schedule, ScheduleBlock
