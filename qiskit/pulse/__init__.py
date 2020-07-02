# -*- coding: utf-8 -*-

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

"""

.. currentmodule:: qiskit.pulse


Getting Started with OpenPulse
==============================

Qiskit Pulse programs give you more control than :py:class:`~qiskit.circuit.QuantumCircuit` s. At
this level, you can specify the exact time dynamics of the input signals
across all input channels. Most quantum algorithms can be described with
circuits alone using gate operations – typically, it’s when you want to
apply error mitigation techniques or characterize a time-dependent
quality of a quantum system that pulse-level control becomes useful.

On this page, we will discover how to build and execute a very basic
Pulse program, which is called a schedule.

Initialize
~~~~~~~~~~

We import the :py:class:`~qiskit.pulse.Schedule` class from the :py:mod:`qiskit.pulse` module. To
start, we are going to initialize a :py:class:`~qiskit.pulse.Schedule` with the name
``"getting_started"``.

.. code:: ipython3

    from qiskit.pulse import Schedule

    sched = Schedule(name='getting_started')

Easy! We have an empty schedule now.

Build waveforms
~~~~~~~~~~~~~~~

The next thing we will want to do is create a pulse waveform to add to our schedule.

One of the core features of building schedules are ``Pulse`` s.
Here, we will build a certain kind of pulse called a
:py:class:`~qiskit.pulse.pulse_lib.SamplePulse`, which specifies a pulse signal as an array of
time-ordered complex amplitudes, or *samples*. Each sample is played for
one cycle, a timestep ``dt``, determined by the backend. If we want to
know the real-time dynamics of our program, we need to know the value of
``dt``. For now, let’s focus on how to build the pulse.

.. code:: ipython3

    from qiskit.pulse.pulse_lib import SamplePulse

    my_pulse = SamplePulse([0.00043, 0.0007 , 0.00112, 0.00175, 0.00272, 0.00414, 0.00622,
                            0.00919, 0.01337, 0.01916, 0.02702, 0.03751, 0.05127, 0.06899,
                            0.09139, 0.1192 , 0.15306, 0.19348, 0.24079, 0.29502, 0.35587,
                            0.4226 , 0.49407, 0.56867, 0.64439, 0.71887, 0.78952, 0.85368,
                            0.90873, 0.95234, 0.98258, 0.99805, 0.99805, 0.98258, 0.95234,
                            0.90873, 0.85368, 0.78952, 0.71887, 0.64439, 0.56867, 0.49407,
                            0.4226 , 0.35587, 0.29502, 0.24079, 0.19348, 0.15306, 0.1192 ,
                            0.09139, 0.06899, 0.05127, 0.03751, 0.02702, 0.01916, 0.01337,
                            0.00919, 0.00622, 0.00414, 0.00272, 0.00175, 0.00112, 0.0007 ,
                            0.00043],
                           name="short_gaussian_pulse")

There are multiple ways to build pulses, which you can learn about on
later pages. This time, we’ve simply passed the exact amplitudes of the
pulse envelope we want to play as an array. The array above is a
Gaussian function evaluated at 64 points, with an amplitude of 1 and a
standard deviation of 8. The (zero-indexed) :math:`i^{th}` sample will
play from time ``i*dt`` up to ``(i + 1)*dt``, modulated by the qubit
frequency. Think of this like an arbitrary waveform generator (AWG),
playing the samples you give to the ``SamplePulse``, mixed with a
continuous sine wave generator outputting a tone at the qubit frequency.

The values above happen to be real, but they can also be complex. The
amplitude norm of any pulse signal is arbitrarily limited to 1. Each
backend system may also impose further constraints – for instance, a
minimum pulse size of 64. Find out more by checking out the `Backend`
documentation: :py:mod:`qiskit.providers.models`.

Schedule instructions
~~~~~~~~~~~~~~~~~~~~~

Next, we have to add an instruction to execute the pulse signal we just
built. This means specifying not only the *time* that the pulse should
be played, but also *where* it should be played. When we build circuits,
we specify which qubit a gate operation should be applied to. In Pulse,
every qubit has multiple *channels*.

We will *play* our pulse on the *drive* channel of qubit 0. The drive
channel lets us enact single qubit operations.

.. code:: ipython3

    from qiskit.pulse import Play, DriveChannel

    qubit_idx = 0

    sched = sched.insert(0, Play(my_pulse, DriveChannel(qubit_idx)))

Note that the pulse we defined operates on the :py:class:`~qiskit.pulse.channels.DriveChannel`,
which
in turn is initialized with the qubit index. We use :py:func:`~qiskit.pulse.Schedule.insert`
to play the pulse at timestep ``t = 0``.

Let’s review what we’ve done, using :py:func:`~qiskit.pulse.Schedule.draw`:

.. code:: ipython3

    sched.draw(label=True)




.. image:: ../../docs/source_images/pulse_imgs/getting_started_with_pulse_7_0.png



The ways in which schedules can be composed is covered in detail in the
:py:class:`qiskit.pulse.Schedule` documentation.

This pulse will drive qubit 0. It is modulated at qubit 0’s resonant
frequency, so it will drive the :math:`|0\rangle` to :math:`|1\rangle`
transition. It is not calibrated to stop at a particular state, so we
won’t know what state we’ve prepared until we look at the results. For
our purposes, we don’t mind what state we end up in.

All that’s left to do is to add a measurement. There is a convenient
utility function for adding measurements, but it requires data from the
backend system that the program will be running on. We will also need
the backend to execute the program.

Grab a backend
~~~~~~~~~~~~~~

.. code:: ipython3

    from qiskit.test.mock import FakeAlmaden

    backend = FakeAlmaden()

Add measurements
~~~~~~~~~~~~~~~~

Now we can use this backend to add the measurement instructions for us.

.. code:: ipython3

    from qiskit.scheduler.utils import measure_all

    sched = sched.insert(sched.duration, measure_all(backend))

Let’s see what the convenience function has added for us, using draw
again. The acquisition and measurement pulses are very long compared to
our initial pulse, so we can use the ``plot_range`` argument to clip the
schedule.

.. code:: ipython3

    sched.draw(plot_range=[0, 1000])




.. image:: ../../docs/source_images/pulse_imgs/getting_started_with_pulse_13_0.png



There is a new pulse on :py:class:`~qiskit.pulse.channels.MeasureChannel` ``m0``, a shorthand
name for
``MeasureChannel(0)``. This channel stimulates readout on qubit 0.
Likewise, ``a0`` is shorthand for ``AcquireChannel(0)``. The
:py:class:`~qiskit.pulse.instructions.Acquire`
instruction on ``a0`` tells the measurement devices when to begin
collecting data on their analog-to-digital converters (ADC), and for how
long. It is drawn as an amplitude 1 constant pulse.

Execute
~~~~~~~

Our schedule is done! We can’t use our mocked backend to execute
programs, but if you have an IBM Quantum account, you could use
``backend = IBMQ.load_account().get_backend(open_pulse=True)`` to see if
you have an OpenPulse enabled backend. Once you have such a backend, we
can execute it the same way we execute circuits:

::

   job = execute(sched, backend)

That’s it! To get the results, use ``result = job.result()``. You’ve
created a Pulse program schedule, containing an operation on qubit 0
followed by a measurement, executed the experiment on the backend and
retrieved the results. A good next step in learning about Pulse would be
to check out how to build :py:class:`qiskit.pulse.Instruction` s.


Pulse API (:mod:`qiskit.pulse`)
===============================

Qiskit-Pulse is a pulse-level quantum programming kit. This lower level of programming offers the
user more control than programming with :py:class:`~qiskit.circuit.QuantumCircuit` s.

Extracting the greatest performance from quantum hardware requires real-time pulse-level
instructions. Pulse answers that need: it enables the quantum physicist *user* to specify the
exact time dynamics of an experiment. It is especially powerful for error mitigation techniques.

The input is given as arbitrary, time-ordered signals (see: :ref:`pulse-insts`) scheduled in
parallel over multiple virtual hardware or simulator resources (see: :ref:`pulse-channels`). The
system also allows the user to recover the time dynamics of the measured output.

This is sufficient to allow the quantum physicist to explore and correct for noise in a quantum
system.

.. _pulse-insts:

Instructions (:mod:`~qiskit.pulse.instructions`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.instructions

   Acquire
   Delay
   Play
   SetFrequency
   ShiftFrequency
   SetPhase
   ShiftPhase
   Snapshot

Pulse Library (waveforms :mod:`~qiskit.pulse.pulse_lib`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.pulse_lib

   ~qiskit.pulse.pulse_lib.discrete
   SamplePulse
   Constant
   Drag
   Gaussian
   GaussianSquare

.. _pulse-channels:

Channels (:mod:`~qiskit.pulse.channels`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Therefore, our signal channels are  *virtual* hardware channels. The backend
which executes our programs is responsible for mapping these virtual channels to the proper
physical channel within the quantum control hardware.

Channels are characterized by their type and their index. See each channel type below to learn
more.

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.channels

   DriveChannel
   MeasureChannel
   AcquireChannel
   ControlChannel
   RegisterSlot
   MemorySlot

Schedules
~~~~~~~~~

Schedules are Pulse programs. They describe instruction sequences for the control hardware.

.. autosummary::
   :toctree: ../stubs/

   Schedule
   Instruction

Configuration
~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../stubs/

   InstructionScheduleMap

Schedule Transforms
~~~~~~~~~~~~~~~~~~~

These functions take :class:`~qiskit.pulse.Schedule` s as input and return modified
:class:`~qiskit.pulse.Schedule` s.

.. autosummary::
   :toctree: ../stubs/

   ~transforms.align_measures
   ~transforms.add_implicit_acquires
   ~transforms.pad

Exceptions
~~~~~~~~~~

.. autosummary::
   :toctree: ../stubs/

   PulseError

"""

from .channels import (DriveChannel, MeasureChannel, AcquireChannel,
                       ControlChannel, RegisterSlot, MemorySlot)
from .commands import AcquireInstruction, FrameChange, PersistentValue
from .configuration import LoConfig, LoRange, Kernel, Discriminator
from .exceptions import PulseError
from .instruction_schedule_map import InstructionScheduleMap
from .instructions import (Acquire, Instruction, Delay, Play, ShiftPhase, Snapshot,
                           SetFrequency, ShiftFrequency, SetPhase)
from .interfaces import ScheduleComponent
from .pulse_lib import (SamplePulse, Gaussian, GaussianSquare, Drag,
                        Constant, ConstantPulse, ParametricPulse)
from .pulse_lib.samplers.decorators import functional_pulse
from .schedule import Schedule
