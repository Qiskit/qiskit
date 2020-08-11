# -*- coding: utf-8 -*-

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

r"""Use the pulse builder DSL to write pulse programs with an imperative syntax.

.. warning::
    The pulse builder interface is still in active development. It may have
    breaking API changes without deprecation warnings in future releases until
    otherwise indicated.

To begin pulse programming we must first initialize our program builder
context with :func:`build`, after which we can begin adding program
statements. For example, below we write a simple program that :func:`play`\s
a pulse:

.. jupyter-execute::

    from qiskit import execute, pulse

    d0 = pulse.DriveChannel(0)

    with pulse.build() as pulse_prog:
        pulse.play(pulse.Constant(100, 1.0), d0)

    pulse_prog.draw()

The builder initializes a :class:`pulse.Schedule`, ``pulse_prog``
and then begins to construct the program within the context. The output pulse
schedule will survive after the context is exited and can be executed like a
normal Qiskit schedule using ``qiskit.execute(pulse_prog, backend)``.

Pulse programming has a simple imperative style. This leaves the programmer
to worry about the raw experimental physics of pulse programming and not
constructing cumbersome data structures.

We can optionally pass a :class:`~qiskit.providers.BaseBackend` to
:func:`build` to enable enhanced functionality. Below, we prepare a Bell state
by automatically compiling the required pulses from their gate-level
representations, while simultaneously applying a long decoupling pulse to a
neighboring qubit. We terminate the experiment with a measurement to observe the
state we prepared. This program which mixes circuits and pulses will be
automatically lowered to be run as a pulse program:

.. jupyter-execute::

    import math

    from qiskit import pulse
    from qiskit.test.mock import FakeOpenPulse3Q

    # TODO: This example should use a real mock backend.
    backend = FakeOpenPulse3Q()

    d2 = pulse.DriveChannel(2)

    with pulse.build(backend) as bell_prep:
        pulse.u2(0, math.pi, 0)
        pulse.cx(0, 1)

    with pulse.build(backend) as decoupled_bell_prep_and_measure:
        # We call our bell state preparation schedule constructed above.
        with pulse.align_right():
            pulse.call(bell_prep)
            pulse.play(pulse.Constant(bell_prep.duration, 0.02), d2)
            pulse.barrier(0, 1, 2)
            registers = pulse.measure_all()

    decoupled_bell_prep_and_measure.draw()

With the pulse builder we are able to blend programming on qubits and channels.
While the pulse schedule is based on instructions that operate on
channels, the pulse builder automatically handles the mapping from qubits to
channels for you.

In the example below we demonstrate some more features of the pulse builder:

.. jupyter-execute::

    import math

    from qiskit import pulse, QuantumCircuit
    from qiskit.pulse import library
    from qiskit.test.mock import FakeOpenPulse2Q

    backend = FakeOpenPulse2Q()

    with pulse.build(backend) as pulse_prog:
        # Create a pulse.
        gaussian_pulse = library.gaussian(10, 1.0, 2)
        # Get the qubit's corresponding drive channel from the backend.
        d0 = pulse.drive_channel(0)
        d1 = pulse.drive_channel(1)
        # Play a pulse at t=0.
        pulse.play(gaussian_pulse, d0)
        # Play another pulse directly after the previous pulse at t=10.
        pulse.play(gaussian_pulse, d0)
        # The default scheduling behavior is to schedule pulses in parallel
        # across channels. For example, the statement below
        # plays the same pulse on a different channel at t=0.
        pulse.play(gaussian_pulse, d1)

        # We also provide pulse scheduling alignment contexts.
        # The default alignment context is align_left.

        # The sequential context schedules pulse instructions sequentially in time.
        # This context starts at t=10 due to earlier pulses above.
        with pulse.align_sequential():
            pulse.play(gaussian_pulse, d0)
            # Play another pulse after at t=20.
            pulse.play(gaussian_pulse, d1)

            # We can also nest contexts as each instruction is
            # contained in its local scheduling context.
            # The output of a child context is a context-schedule
            # with the internal instructions timing fixed relative to
            # one another. This is schedule is then called in the parent context.

            # Context starts at t=30.
            with pulse.align_left():
                # Start at t=30.
                pulse.play(gaussian_pulse, d0)
                # Start at t=30.
                pulse.play(gaussian_pulse, d1)
            # Context ends at t=40.

            # Alignment context where all pulse instructions are
            # aligned to the right, ie., as late as possible.
            with pulse.align_right():
                # Shift the phase of a pulse channel.
                pulse.shift_phase(math.pi, d1)
                # Starts at t=40.
                pulse.delay(100, d0)
                # Ends at t=140.

                # Starts at t=130.
                pulse.play(gaussian_pulse, d1)
                # Ends at t=140.

            # Acquire data for a qubit and store in a memory slot.
            pulse.acquire(100, 0, pulse.MemorySlot(0))

            # We also support a variety of macros for common operations.

            # Measure all qubits.
            pulse.measure_all()

            # Delay on some qubits.
            # This requires knowledge of which channels belong to which qubits.
            # delay for 100 cycles on qubits 0 and 1.
            pulse.delay_qubits(100, 0, 1)

            # Call a quantum circuit. The pulse builder lazily constructs a quantum
            # circuit which is then transpiled and scheduled before inserting into
            # a pulse schedule.
            # NOTE: Quantum register indices correspond to physical qubit indices.
            qc = QuantumCircuit(2, 2)
            qc.cx(0, 1)
            pulse.call(qc)
            # Calling a small set of standard gates and decomposing to pulses is
            # also supported with more natural syntax.
            pulse.u3(0, math.pi, 0, 0)
            pulse.cx(0, 1)


            # It is also be possible to call a preexisting schedule
            tmp_sched = pulse.Schedule()
            tmp_sched += pulse.Play(gaussian_pulse, d0)
            pulse.call(tmp_sched)

            # We also support:

            # frequency instructions
            pulse.set_frequency(5.0e9, d0)

            # phase instructions
            pulse.shift_phase(0.1, d0)

            # offset contexts
            with pulse.phase_offset(math.pi, d0):
                pulse.play(gaussian_pulse, d0)

The above is just a small taste of what is possible with the builder. See the
rest of the module documentation for more information on its
capabilities.
"""
import collections
import contextvars
import functools
import itertools
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    NewType
)

import numpy as np

from qiskit import circuit
from qiskit.circuit.library import standard_gates as gates
from qiskit.pulse import (
    channels as chans,
    configuration,
    exceptions,
    instructions,
    macros,
    library,
    transforms,
)
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule

#: contextvars.ContextVar[BuilderContext]: active builder
BUILDER_CONTEXTVAR = contextvars.ContextVar("backend")

T = TypeVar('T')  # pylint: disable=invalid-name

StorageLocation = NewType('StorageLocation', Union[chans.MemorySlot, chans.RegisterSlot])


def _compile_lazy_circuit_before(function: Callable[..., T]
                                 ) -> Callable[..., T]:
    """Decorator thats schedules and calls the lazily compiled circuit before
    executing the decorated builder method."""
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        self._compile_lazy_circuit()
        return function(self, *args, **kwargs)
    return wrapper


def _requires_backend(function: Callable[..., T]) -> Callable[..., T]:
    """Decorator a function to raise if it is called without a builder with a
    set backend.
    """
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.backend is None:
            raise exceptions.BackendNotSet(
                'This function requires the builder to '
                'have a "backend" set.')
        return function(self, *args, **kwargs)
    return wrapper


class _PulseBuilder():
    """Builder context class."""

    def __init__(self,
                 backend=None,
                 schedule: Optional[Schedule] = None,
                 name: Optional[str] = None,
                 default_alignment: Union[str, Callable] = 'left',
                 default_transpiler_settings: Mapping = None,
                 default_circuit_scheduler_settings: Mapping = None):
        """Initialize the builder context.

        .. note::
            At some point we may consider incorpating the builder into
            the :class:`~qiskit.pulse.Schedule` class. However, the risk of
            this is tying the user interface to the intermediate
            representation. For now we avoid this at the cost of some code
            duplication.

        Args:
            backend (BaseBackend): Input backend to use in builder. If not set
                certain functionality will be unavailable.
            schedule: Initital schedule to build on. If not supplied
                a schedule will be created.
            name: Name of pulse program to be built. Only used if `schedule`
                is not ``None``.
            default_alignment: Default scheduling alignment policy for the
                builder. One of 'left', 'right', 'sequential', or an alignment
                contextmanager.
            default_transpiler_settings: Default settings for the transpiler.
            default_circuit_scheduler_settings: Default settings for the
                circuit to pulse scheduler.
        """
        #: BaseBackend: Backend instance for context builder.
        self._backend = backend

        #: Union[None, ContextVar]: Token for this ``_PulseBuilder``'s ``ContextVar``.
        self._backend_ctx_token = None

        #: pulse.Schedule: Active schedule of BuilderContext.
        self._context_schedule = None

        #: QuantumCircuit: Lazily constructed quantum circuit
        self._lazy_circuit = None

        # ContextManager: Default alignment context.
        self._default_alignment_context = _align(default_alignment)

        self._transpiler_settings = default_transpiler_settings or {}

        if default_circuit_scheduler_settings is None:
            default_circuit_scheduler_settings = {}
        self._circuit_scheduler_settings = \
            default_circuit_scheduler_settings or {}

        # pulse.Schedule: Root program context-schedule
        self._schedule = schedule or Schedule(name=name)

        self.set_context_schedule(Schedule())

    def __enter__(self) -> Schedule:
        """Enter this builder context and yield either the supplied schedule
        or the schedule created for the user.

        Returns:
            The schedule that the builder will build on.
        """
        self._backend_ctx_token = BUILDER_CONTEXTVAR.set(self)
        self._default_alignment_context.__enter__()
        return self._schedule

    @_compile_lazy_circuit_before
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the builder context and compile the built pulse program."""
        self._default_alignment_context.__exit__(exc_type, exc_val, exc_tb)
        self.compile()
        BUILDER_CONTEXTVAR.reset(self._backend_ctx_token)

    @property
    def backend(self):
        """Returns the builder backend if set.

        Returns:
            Optional[BaseBackend]: The builder's backend.
        """
        return self._backend

    @property
    def context_schedule(self) -> Schedule:
        """Return the current context schedule."""
        return self._context_schedule

    @property
    @_requires_backend
    def num_qubits(self):
        """Get the number of qubits in the backend."""
        return self.backend.configuration().n_qubits

    @property
    def transpiler_settings(self) -> Mapping:
        """The builder's transpiler settings."""
        return self._transpiler_settings

    @transpiler_settings.setter
    @_compile_lazy_circuit_before
    def transpiler_settings(self, settings: Mapping):
        self._compile_lazy_circuit()
        self._transpiler_settings = settings

    @property
    def circuit_scheduler_settings(self) -> Mapping:
        """The builder's circuit to pulse scheduler settings."""
        return self._circuit_scheduler_settings

    @circuit_scheduler_settings.setter
    @_compile_lazy_circuit_before
    def circuit_scheduler_settings(self, settings: Mapping):
        self._compile_lazy_circuit()
        self._circuit_scheduler_settings = settings

    @_compile_lazy_circuit_before
    def compile(self) -> Schedule:
        """Compile and output the built pulse program."""
        # Not much happens because we currently compile as we build.
        # This should be offloaded to a true compilation module
        # once we define a more sophisticated IR.
        program = self._schedule.append(self.context_schedule, inplace=True)
        self.set_context_schedule(Schedule())
        return program

    @_compile_lazy_circuit_before
    def set_context_schedule(self, context_schedule: Schedule):
        """Set the current context's schedule for the builder."""
        self._context_schedule = context_schedule

    @_compile_lazy_circuit_before
    def append_schedule(self, context_schedule: Schedule):
        """Add a :class:`Schedule` to the builder's context schedule.

        Args:
            context_schedule: Schedule to append to the current context schedule.
        """
        self.context_schedule.append(context_schedule, inplace=True)

    @_compile_lazy_circuit_before
    def append_instruction(self, instruction: instructions.Instruction):
        """Add an instruction to the builder's context schedule.

        Args:
            instruction: Instruction to append.
        """
        self.context_schedule.append(instruction, inplace=True)

    def _compile_lazy_circuit(self):
        """Call a QuantumCircuit and append the output pulse schedule
        to the builder's context schedule."""
        if self._lazy_circuit:
            import qiskit.compiler as compiler  # pylint: disable=cyclic-import

            lazy_circuit = self._lazy_circuit
            # reset lazy circuit
            self._lazy_circuit = self.new_circuit()
            transpiled_circuit = compiler.transpile(lazy_circuit,
                                                    self.backend,
                                                    **self.transpiler_settings)
            sched = compiler.schedule(transpiled_circuit,
                                      self.backend,
                                      **self.circuit_scheduler_settings)
            self.call_schedule(sched)

    def call_schedule(self, schedule: Schedule):
        """Call a schedule and append to the builder's context schedule."""
        self.append_schedule(schedule)

    def new_circuit(self):
        """Create a new circuit for lazy circuit scheduling."""
        return circuit.QuantumCircuit(self.num_qubits)

    @_requires_backend
    def call_circuit(self,
                     circ: circuit.QuantumCircuit,
                     lazy: bool = True):
        """Call a circuit in the pulse program.

        The circuit is assumed to be defined on physical qubits.

        If ``lazy == True`` this circuit will extend a lazily constructed
        quantum circuit. When an operation occurs that breaks the underlying
        circuit scheduling assumptions such as adding a pulse instruction or
        changing the alignment context the circuit will be
        transpiled and scheduled into pulses with the current active settings.

        Args:
            circ: Circuit to call.
            lazy: If false the circuit will be transpiled and pulse scheduled
                immediately. Otherwise, it will extend the active lazy circuit
                as defined above.
        """
        if lazy:
            self._call_circuit(circ)
        else:
            self._compile_lazy_circuit()
            self._call_circuit(circ)
            self._compile_lazy_circuit()

    def _call_circuit(self, circ):
        if self._lazy_circuit is None:
            self._lazy_circuit = self.new_circuit()

        for creg in circ.cregs:
            try:
                self._lazy_circuit.add_register(creg)
            except circuit.exceptions.CircuitError:
                pass
        self._lazy_circuit.compose(circ, inplace=True)

    @_requires_backend
    def call_gate(self,
                  gate: circuit.Gate,
                  qubits: Tuple[int, ...],
                  lazy: bool = True):
        """Call the circuit ``gate`` in the pulse program.

        The qubits are assumed to be defined on physical qubits.

        If ``lazy == True`` this circuit will extend a lazily constructed
        quantum circuit. When an operation occurs that breaks the underlying
        circuit scheduling assumptions such as adding a pulse instruction or
        changing the alignment context the circuit will be
        transpiled and scheduled into pulses with the current active settings.

        Args:
            gate: Gate to call.
            qubits: Qubits to call gate on.
            lazy: If false the circuit will be transpiled and pulse scheduled
                immediately. Otherwise, it will extend the active lazy circuit
                as defined above.
        """
        try:
            iter(qubits)
        except TypeError:
            qubits = (qubits,)

        qc = circuit.QuantumCircuit(self.num_qubits)
        qc.append(gate, qargs=qubits)
        self.call_circuit(qc, lazy=lazy)


def build(backend=None,
          schedule: Optional[Schedule] = None,
          name: Optional[str] = None,
          default_alignment: str = 'left',
          default_transpiler_settings: Optional[Dict[str, Any]] = None,
          default_circuit_scheduler_settings: Optional[Dict[str, Any]] = None
          ) -> ContextManager[Schedule]:
    """Create a context manager for launching the imperative pulse builder DSL.

    To enter a building context and starting building a pulse program:

    .. jupyter-execute::

        from qiskit import execute, pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.play(pulse.Constant(100, 0.5), d0)


    While the output program ``pulse_prog`` cannot be executed as we are using
    a mock backend. If a real backend is being used, executing the program is
    done with:

    .. code-block:: python

        qiskit.execute(pulse_prog, backend)

    Args:
        backend (BaseBackend): A Qiskit backend. If not supplied certain
            builder functionality will be unavailable.
        schedule: a *mutable* pulse Schedule in which your pulse program will
            be built.
        name: Name of pulse program to be built. Only used if `schedule`
            is not ``None``.
        default_alignment: Default scheduling alignment for builder.
            One of ``left``, ``right``, ``sequential`` or an alignment
            contextmanager.
        default_transpiler_settings: Default settings for the transpiler.
        default_circuit_scheduler_settings: Default settings for the
            circuit to pulse scheduler.

    Returns:
        A new builder context which has the active builder inititalized.
    """
    return _PulseBuilder(
        backend=backend,
        schedule=schedule,
        name=name,
        default_alignment=default_alignment,
        default_transpiler_settings=default_transpiler_settings,
        default_circuit_scheduler_settings=default_circuit_scheduler_settings)


# Builder Utilities
def _active_builder() -> _PulseBuilder:
    """Get the active builder in the active context.

    Returns:
        The active active builder in this context.

    Raises:
        exceptions.NoActiveBuilder: If a pulse builder function is called
        outside of a builder context.
    """
    try:
        return BUILDER_CONTEXTVAR.get()
    except LookupError:
        raise exceptions.NoActiveBuilder(
            'A Pulse builder function was called outside of '
            'a builder context. Try calling within a builder '
            'context, eg., "with pulse.build() as schedule: ...".')


def active_backend():
    """Get the backend of the currently active builder context.

    Returns:
        BaseBackend: The active backend in the currently active
            builder context.

    Raises:
        exceptions.BackendNotSet: If the builder does not have a backend set.
    """
    builder = _active_builder().backend
    if builder is None:
        raise exceptions.BackendNotSet(
            'This function requires the active builder to '
            'have a "backend" set.')
    return builder


def append_schedule(schedule: Schedule):
    """Call a schedule by appending to the active builder's context schedule."""
    _active_builder().append_schedule(schedule)


def append_instruction(instruction: instructions.Instruction):
    """Append an instruction to the active builder's context schedule.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.builder.append_instruction(pulse.Delay(10, d0))

        print(pulse_prog.instructions)
    """
    _active_builder().append_instruction(instruction)


def num_qubits() -> int:
    """Return number of qubits in the currently active backend.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.num_qubits())

    .. note:: Requires the active builder context to have a backend set.
    """
    return active_backend().configuration().n_qubits


def seconds_to_samples(seconds: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """Obtain the number of samples that will elapse in ``seconds`` on the
    active backend.

    Rounds down.

    Args:
        seconds: Time in seconds to convert to samples.

    Returns:
        The number of samples for the time to elapse
    """
    return int(seconds / active_backend().configuration().dt)


def samples_to_seconds(samples: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Obtain the time in seconds that will elapse for the input number of
    samples on the active backend.

    Args:
        samples: Number of samples to convert to time in seconds.

    Returns:
        The time that elapses in ``samples``.
    """
    return samples * active_backend().configuration().dt


def qubit_channels(qubit: int) -> Set[chans.Channel]:
    """Returns the set of channels associated with a qubit.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.qubit_channels(0))

    .. note:: Requires the active builder context to have a backend set.

    .. note:: A channel may still be associated with another qubit in this list
        such as in the case where significant crosstalk exists.

    """
    return set(active_backend().configuration().get_qubit_channels(qubit))


def _qubits_to_channels(*channels_or_qubits: Union[int, chans.Channel]
                        ) -> Set[chans.Channel]:
    """Returns the unique channels of the input qubits."""
    channels = set()
    for channel_or_qubit in channels_or_qubits:
        if isinstance(channel_or_qubit, int):
            channels |= qubit_channels(channel_or_qubit)
        elif isinstance(channel_or_qubit, chans.Channel):
            channels.add(channel_or_qubit)
        else:
            raise exceptions.PulseError(
                '{} is not a "Channel" or '
                'qubit (integer).'.format(channel_or_qubit))
    return channels


def active_transpiler_settings() -> Dict[str, Any]:
    """Return the current active builder context's transpiler settings.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        transpiler_settings = {'optimization_level': 3}

        with pulse.build(backend,
                         default_transpiler_settings=transpiler_settings):
            print(pulse.active_transpiler_settings())

    """
    return dict(_active_builder().transpiler_settings)


# pylint: disable=invalid-name
def active_circuit_scheduler_settings() -> Dict[str, Any]:
    """Return the current active builder context's circuit scheduler settings.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        circuit_scheduler_settings = {'method': 'alap'}

        with pulse.build(
                backend,
                default_circuit_scheduler_settings=circuit_scheduler_settings):
            print(pulse.active_circuit_scheduler_settings())

    """
    return dict(_active_builder().circuit_scheduler_settings)


# Contexts
def _transform_context(transform: Callable[[Schedule], Schedule],
                       **transform_kwargs: Any
                       ) -> Callable[..., ContextManager[None]]:
    """A tranform context generator, decorator.

    Decorator accepts a transformation function, and then decorates a new
    ContextManager function.

    When the context is entered it creates a new schedule, sets it as the
    builder's context schedule and then yields.

    Finally it will reset the initial builder's context schedule after exiting
    the context and apply the decorated transform function to the
    context Schedule. The output transformed schedule will then be
    appended to the initial builder's context schedule. This effectively builds a stack
    of builder's context schedules that is automatically collapsed upon exiting the context.

    Args:
        transform: Transform to decorate as context.
        transform_kwargs: Additional override keyword arguments for the
            decorated transform.

    Returns:
        A function that generates a new transformation ``ContextManager``.
    """
    def wrap(function):  # pylint: disable=unused-argument
        @functools.wraps(function)
        @contextmanager
        def wrapped_transform(*args, **kwargs):
            builder = _active_builder()
            context_schedule = builder.context_schedule
            transform_schedule = Schedule()
            builder.set_context_schedule(transform_schedule)
            try:
                yield
            finally:
                builder._compile_lazy_circuit()
                transformed_schedule = transform(
                    transform_schedule,
                    *args,
                    **kwargs,
                    **transform_kwargs,
                )
                builder.set_context_schedule(context_schedule)
                builder.append_schedule(transformed_schedule)
        return wrapped_transform

    return wrap


@_transform_context(transforms.align_left)
def align_left() -> ContextManager[None]:
    """Left alignment pulse scheduling context.

    Pulse instructions within this context are scheduled as early as possible
    by shifting them left to the earliest available time.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_left():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will start at t=0
                pulse.play(pulse.Constant(20, 1.0), d1)

        assert pulse_prog.ch_start_time(d0) == pulse_prog.ch_start_time(d1)
    """


@_transform_context(transforms.align_right)
def align_right() -> ContextManager[None]:
    """Right alignment pulse scheduling context.

    Pulse instructions within this context are scheduled as late as possible
    by shifting them right to the latest available time.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_right():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will start at t=80
                pulse.play(pulse.Constant(20, 1.0), d1)

        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_stop_time(d1)
    """


@_transform_context(transforms.align_sequential)
def align_sequential() -> ContextManager[None]:
    """Sequential alignment pulse scheduling context.

    Pulse instructions within this context are scheduled sequentially in time
    such that no two instructions will be played at the same time.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_sequential():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will also start at t=100
                pulse.play(pulse.Constant(20, 1.0), d1)

        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_start_time(d1)
    """


def _align(alignment: str = 'left') -> ContextManager[None]:
    """General alignment context. Used by the :class:`_Builder` to choose the
    default alignment policy.

    Args:
        alignment: Alignment scheduling policy to follow.
            One of "left", "right" or "sequential".

    Returns:
        An alignment context that will schedule the instructions it contains
        according to the selected alignment policy upon exiting the context.

    Raises:
        exceptions.PulseError: If an unsupported alignment context is selected.
    """
    if alignment == 'left':
        return align_left()
    elif alignment == 'right':
        return align_right()
    elif alignment == 'sequential':
        return align_sequential()
    else:
        raise exceptions.PulseError('Alignment "{}" is not '
                                    'supported.'.format(alignment))


@contextmanager
def inline() -> ContextManager[None]:
    """Inline all instructions within this context into the parent context,
    inheriting the scheduling policy of the parent context.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as pulse_prog:
            # will be ignored due to internal grouping
            with pulse.align_left():
                pulse.play(pulse.Constant(10, 1.0), d0)
                with pulse.inline():
                    with pulse.align_right():
                        # this pulse will start at t=0
                        pulse.play(pulse.Constant(100, 1.0), d1)
                        # this pulse will also start at t=0
                        pulse.play(pulse.Constant(20, 1.0), d2)

        assert (pulse_prog.ch_start_time(d0) ==
                pulse_prog.ch_start_time(d1) ==
                pulse_prog.ch_start_time(d2))

    .. warning:: This will cause all scheduling directives within this context
        to be ignored.
    """
    builder = _active_builder()
    context_schedule = builder.context_schedule
    transform_schedule = Schedule()
    builder.set_context_schedule(transform_schedule)
    try:
        yield
    finally:
        builder._compile_lazy_circuit()
        builder.set_context_schedule(context_schedule)
        for _, instruction in transform_schedule.instructions:
            append_instruction(instruction)


@_transform_context(transforms.pad, inplace=True)
def pad(*chs: chans.Channel) -> ContextManager[None]:  # pylint: disable=unused-argument
    """Pad all availale timeslots with delays upon exiting context.

    Args:
        chs: Channels to pad with delays. Defaults to all channels in context
            if none are supplied.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.pad():
                with pulse.align_right():
                    # this pulse will start at t=0
                    pulse.play(pulse.Constant(100, 1.0), d0)
                    # this pulse will start at t=80
                    # a delay will be inserted from t=0 to t=80
                    pulse.play(pulse.Constant(20, 1.0), d1)
        assert pulse_prog.ch_start_time(d0) == pulse_prog.ch_start_time(d1)
        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_stop_time(d1)
    """


@contextmanager
def transpiler_settings(**settings) -> ContextManager[None]:
    """Set the currently active transpiler settings for this context.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.active_transpiler_settings())
            with pulse.transpiler_settings(optimization_level=3):
                print(pulse.active_transpiler_settings())
    """
    builder = _active_builder()
    curr_transpiler_settings = builder.transpiler_settings
    builder.transpiler_settings = collections.ChainMap(
        settings, curr_transpiler_settings)
    try:
        yield
    finally:
        builder.transpiler_settings = curr_transpiler_settings


@contextmanager
def circuit_scheduler_settings(**settings) -> ContextManager[None]:
    """Set the currently active circuit scheduler settings for this context.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.active_circuit_scheduler_settings())
            with pulse.circuit_scheduler_settings(method='alap'):
                print(pulse.active_circuit_scheduler_settings())
    """
    builder = _active_builder()
    curr_circuit_scheduler_settings = builder.circuit_scheduler_settings
    builder.circuit_scheduler_settings = collections.ChainMap(
        settings, curr_circuit_scheduler_settings)
    try:
        yield
    finally:
        builder.circuit_scheduler_settings = curr_circuit_scheduler_settings


@contextmanager
def phase_offset(phase: float,
                 *channels: chans.PulseChannel
                 ) -> ContextManager[None]:
    """Shift the phase of input channels on entry into context and undo on exit.

    Examples:

    .. jupyter-execute::

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            with pulse.phase_offset(math.pi, d0):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 3

    Args:
        phase: Amount of phase offset in radians.
        channels: Channels to offset phase of.

    Yields:
        None
    """
    for channel in channels:
        shift_phase(phase, channel)
    try:
        yield
    finally:
        for channel in channels:
            shift_phase(-phase, channel)


@contextmanager
def frequency_offset(frequency: float,
                     *channels: chans.PulseChannel,
                     compensate_phase: bool = False
                     ) -> ContextManager[None]:
    """Shift the frequency of inputs channels on entry into context and undo on exit.

    Examples:

    .. code-block:: python
        :emphasize-lines: 7, 16

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build(backend) as pulse_prog:
            # shift frequency by 1GHz
            with pulse.frequency_offset(1e9, d0):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 3

        with pulse.build(backend) as pulse_prog:
            # Shift frequency by 1GHz.
            # Undo accumulated phase in the shifted freqeuncy frame
            # when exiting the context.
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 4

    Args:
        frequency: Amount of frequency offset in Hz.
        channels: Channels to offset phase of.
        compensate_phase: Compensate for accumulated phase in accumulated with
            respect to the channels frame at its initial frequency.

    Yields:
        None
    """
    builder = _active_builder()
    t0 = builder.context_schedule.duration

    for channel in channels:
        shift_frequency(frequency, channel)
    try:
        yield
    finally:
        if compensate_phase:
            duration = builder.context_schedule.duration - t0
            dt = active_backend().configuration().dt
            accumulated_phase = duration * dt * frequency % (2*np.pi)
            for channel in channels:
                shift_phase(-accumulated_phase, channel)

        for channel in channels:
            shift_frequency(-frequency, channel)


# Channels
def drive_channel(qubit: int) -> chans.DriveChannel:
    """Return ``DriveChannel`` for ``qubit`` on the active builder backend.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            assert pulse.drive_channel(0) == pulse.DriveChannel(0)

    .. note:: Requires the active builder context to have a backend set.
    """
    return active_backend().configuration().drive(qubit)


def measure_channel(qubit: int) -> chans.MeasureChannel:
    """Return ``MeasureChannel`` for ``qubit`` on the active builder backend.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            assert pulse.measure_channel(0) == pulse.MeasureChannel(0)

    .. note:: Requires the active builder context to have a backend set.
    """
    return active_backend().configuration().measure(qubit)


def acquire_channel(qubit: int) -> chans.AcquireChannel:
    """Return ``AcquireChannel`` for ``qubit`` on the active builder backend.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            assert pulse.acquire_channel(0) == pulse.AcquireChannel(0)

    .. note:: Requires the active builder context to have a backend set.
    """
    return active_backend().configuration().acquire(qubit)


def control_channels(*qubits: Iterable[int]) -> List[chans.ControlChannel]:
    """Return ``AcquireChannel`` for ``qubit`` on the active builder backend.

    Return the secondary drive channel for the given qubit -- typically
    utilized for controlling multi-qubit interactions.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()
        with pulse.build(backend):
            assert pulse.control_channels(0, 1) == [pulse.ControlChannel(0)]

    .. note:: Requires the active builder context to have a backend set.

    Args:
      qubits: Tuple or list of ordered qubits of the form
        `(control_qubit, target_qubit)`.

    Returns:
        List of control channels associated with the supplied ordered list
        of qubits.
    """
    return active_backend().configuration().control(qubits=qubits)


# Base Instructions
def delay(duration: int,
          channel: chans.Channel):
    """Delay on a ``channel`` for a ``duration``.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.delay(10, d0)

    Args:
        duration: Number of cycles to delay for on ``channel``.
        channel: Channel to delay on.
    """
    append_instruction(instructions.Delay(duration, channel))


def play(pulse: Union[library.Pulse, np.ndarray],
         channel: chans.PulseChannel):
    """Play a ``pulse`` on a ``channel``.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.play(pulse.Constant(10, 1.0), d0)

    Args:
        pulse: Pulse to play.
        channel: Channel to play pulse on.
    """
    if not isinstance(pulse, library.Pulse):
        pulse = library.Waveform(pulse)

    append_instruction(instructions.Play(pulse, channel))


def acquire(duration: int,
            qubit_or_channel: Union[int, chans.AcquireChannel],
            register: StorageLocation,
            **metadata: Union[configuration.Kernel,
                              configuration.Discriminator]):
    """Acquire for a ``duration`` on a ``channel`` and store the result
    in a ``register``.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.MeasureChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as pulse_prog:
            pulse.acquire(100, d0, mem0)

            # measurement metadata
            kernel = pulse.configuration.Kernel('linear_discriminator')
            pulse.acquire(100, d0, mem0, kernel=kernel)

    .. note:: The type of data acquire will depend on the execution
        ``meas_level``.

    Args:
        duration: Duration to acquire data for
        qubit_or_channel: Either the qubit to acquire data for or the specific
            :class:`~qiskit.pulse.channels.AcquireChannel` to acquire on.
        register: Location to store measured result.
        metadata: Additional metadata for measurement. See
            :class:`~qiskit.pulse.instructions.Acquire` for more information.

    Raises:
        exceptions.PulseError: If the register type is not supported.
    """
    if isinstance(qubit_or_channel, int):
        qubit_or_channel = chans.AcquireChannel(qubit_or_channel)

    if isinstance(register, chans.MemorySlot):
        append_instruction(instructions.Acquire(
            duration, qubit_or_channel, mem_slot=register, **metadata))
    elif isinstance(register, chans.RegisterSlot):
        append_instruction(instructions.Acquire(
            duration, qubit_or_channel, reg_slot=register, **metadata))
    else:
        raise exceptions.PulseError(
            'Register of type: "{}" is not supported'.format(type(register)))


def set_frequency(frequency: float,
                  channel: chans.PulseChannel):
    """Set the ``frequency`` of a pulse ``channel``.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.set_frequency(1e9, d0)

    Args:
        frequency: Frequency in Hz to set channel to.
        channel: Channel to set frequency of.
    """
    append_instruction(instructions.SetFrequency(frequency, channel))


def shift_frequency(frequency: float,
                    channel: chans.PulseChannel):
    """Shift the ``frequency`` of a pulse ``channel``.

    Examples:

    .. code-block:: python
        :emphasize-lines: 6

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.shift_frequency(1e9, d0)

    Args:
        frequency: Frequency in Hz to shift channel frequency by.
        channel: Channel to shift frequency of.
    """
    append_instruction(instructions.ShiftFrequency(frequency, channel))


def set_phase(phase: float,
              channel: chans.PulseChannel):
    """Set the ``phase`` of a pulse ``channel``.

    Examples:

    .. code-block:: python
        :emphasize-lines: 8

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.set_phase(math.pi, d0)

    Args:
        phase: Phase in radians to set channel carrier signal to.
        channel: Channel to set phase of.
    """
    append_instruction(instructions.SetPhase(phase, channel))


def shift_phase(phase: float,
                channel: chans.PulseChannel):
    """Shift the ``phase`` of a pulse ``channel``.

    Examples:

    .. jupyter-execute::

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.shift_phase(math.pi, d0)

    Args:
        phase: Phase in radians to shift channel carrier signal by.
        channel: Channel to shift phase of.
    """
    append_instruction(instructions.ShiftPhase(phase, channel))


def snapshot(label: str,
             snapshot_type: str = 'statevector'):
    """Simulator snapshot.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse

        with pulse.build() as pulse_prog:
            pulse.snapshot('first', 'statevector')

    Args:
        label: Label for snapshot.
        snapshot_type: Type of snapshot.
    """
    append_instruction(
        instructions.Snapshot(label, snapshot_type=snapshot_type))


def call_schedule(schedule: Schedule):
    """Call a pulse ``schedule`` in the builder context.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.pulse import builder

        d0 = pulse.DriveChannel(0)

        sched = pulse.Schedule()
        sched += pulse.Play(pulse.Constant(10, 1.0), d0)

        with pulse.build() as pulse_prog:
            builder.call_schedule(sched)

        assert pulse_prog == sched

    Args:
        Schedule to call.
    """
    _active_builder().call_schedule(schedule)


def call_circuit(circ: circuit.QuantumCircuit):
    """Call a quantum ``circuit`` within the active builder context.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. jupyter-execute::

        from qiskit import circuit, pulse, schedule, transpile
        from qiskit.pulse import builder
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        d0 = pulse.DriveChannel(0)

        qc = circuit.QuantumCircuit(2)
        qc.cx(0, 1)
        qc_transpiled = transpile(qc, optimization_level=3)
        sched = schedule(qc_transpiled, backend)

        with pulse.build(backend) as pulse_prog:
            # with default settings
            builder.call_circuit(qc)

        with pulse.build(backend) as pulse_prog:
            with pulse.transpiler_settings(optimization_level=3):
                builder.call_circuit(qc)

        assert pulse_prog == sched

    .. note:: Requires the active builder context to have a backend set.

    Args:
        Circuit to call.
    """
    _active_builder().call_circuit(circ, lazy=True)


def call(target: Union[circuit.QuantumCircuit, Schedule]):
    """Call the ``target`` within the currently active builder context.

    Examples:

    .. jupyter-execute::

        from qiskit import circuit, pulse, schedule, transpile
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        qc = circuit.QuantumCircuit(2)
        qc.cx(0, 1)
        qc_transpiled = transpile(qc, optimization_level=3)
        sched = schedule(qc_transpiled, backend)

        with pulse.build(backend) as pulse_prog:
                pulse.call(sched)
                pulse.call(qc)

    Args:
        target: Target circuit or pulse schedule to call.

    Raises:
        exceptions.PulseError: If the input ``target`` type is not supported.
    """
    if isinstance(target, circuit.QuantumCircuit):
        call_circuit(target)
    elif isinstance(target, Schedule):
        call_schedule(target)
    else:
        raise exceptions.PulseError(
            'Target of type "{}" is not supported.'.format(type(target)))


# Directives
def barrier(*channels_or_qubits: Union[chans.Channel, int]):
    """Barrier directive for a set of channels and qubits.

    This directive prevents the compiler from moving instructions across
    the barrier. Consider the case where we want to enforce that one pulse
    happens after another on separate channels, this can be done with:

    .. jupyter-kernel:: python3
        :id: barrier

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build(backend) as barrier_pulse_prog:
            pulse.play(pulse.Constant(10, 1.0), d0)
            pulse.barrier(d0, d1)
            pulse.play(pulse.Constant(10, 1.0), d1)

    Of course this could have been accomplished with:

    .. jupyter-execute::

        from qiskit.pulse import transforms

        with pulse.build(backend) as aligned_pulse_prog:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(10, 1.0), d0)
                pulse.play(pulse.Constant(10, 1.0), d1)

        barrier_pulse_prog = transforms.remove_directives(barrier_pulse_prog)
        assert barrier_pulse_prog == aligned_pulse_prog

    The barrier allows the pulse compiler to take care of more advanced
    scheduling aligment operations across channels. For example
    in the case where we are calling an outside circuit or schedule and
    want to align a pulse at the end of one call:

    .. jupyter-execute::

        import math

        d0 = pulse.DriveChannel(0)

        with pulse.build(backend) as pulse_prog:
            with pulse.align_right():
                pulse.x(1)
                # Barrier qubit 1 and d0.
                pulse.barrier(1, d0)
                # Due to barrier this will play before the gate on qubit 1.
                pulse.play(pulse.Constant(10, 1.0), d0)
                # This will end at the same time as the pulse above due to
                # the barrier.
                pulse.x(1)

    .. note:: Requires the active builder context to have a backend set if
        qubits are barriered on.

    Args:
        channels_or_qubits: Channels or qubits to barrier.
    """
    channels = _qubits_to_channels(*channels_or_qubits)
    if len(channels) > 1:
        append_instruction(directives.RelativeBarrier(*channels))


# Macros
def measure(qubits: Union[List[int], int],
            registers: Union[List[StorageLocation], StorageLocation] = None,
            ) -> Union[List[StorageLocation], StorageLocation]:
    """Measure a qubit within the currently active builder context.

    At the pulse level a measurement is composed of both a stimulus pulse and
    an acquisition instruction which tells the systems measurement unit to
    acquire data and process it. We provide this measurement macro to automate
    the process for you, but if desired full control is still available with
    :func:`acquire` and :func:`play`.

    To use the measurement it is as simple as specifying the qubit you wish to
    measure:

    .. jupyter-kernel:: python3
        :id: measure

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        qubit = 0

        with pulse.build(backend) as pulse_prog:
            # Do something to the qubit.
            qubit_drive_chan = pulse.drive_channel(0)
            pulse.play(pulse.Constant(100, 1.0), qubit_drive_chan)
            # Measure the qubit.
            reg = pulse.measure(qubit)

    For now it is not possible to do much with the handle to ``reg`` but in the
    future we will support using this handle to a result register to build
    up ones program. It is also possible to supply this register:

    .. jupyter-execute::

        with pulse.build(backend) as pulse_prog:
            pulse.play(pulse.Constant(100, 1.0), qubit_drive_chan)
            # Measure the qubit.
            mem0 = pulse.MemorySlot(0)
            reg = pulse.measure(qubit, mem0)

        assert reg == mem0

    .. note:: Requires the active builder context to have a backend set.

    Args:
        qubits: Physical qubit to measure.
        registers: Register to store result in. If not selected the current
            behaviour is to return the :class:`MemorySlot` with the same
            index as ``qubit``. This register will be returned.
    Returns:
        The ``register`` the qubit measurement result will be stored in.
    """
    backend = active_backend()

    try:
        qubits = list(qubits)
    except TypeError:
        qubits = [qubits]

    if registers is None:
        registers = [chans.MemorySlot(qubit) for qubit in qubits]
    else:
        try:
            registers = list(registers)
        except TypeError:
            registers = [registers]

    measure_sched = macros.measure(
        qubits=qubits,
        inst_map=backend.defaults().instruction_schedule_map,
        meas_map=backend.configuration().meas_map,
        qubit_mem_slots={qubit: register.index for qubit, register in zip(qubits, registers)})
    call_schedule(measure_sched)

    if len(qubits) == 1:
        return registers[0]
    else:
        return registers


def measure_all() -> List[chans.MemorySlot]:
    r"""Measure all qubits within the currently active builder context.

    A simple macro function to measure all of the qubits in the device at the
    same time. This is useful for handling device ``meas_map`` and single
    measurement constraints.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            # Measure all qubits and return associated registers.
            regs = pulse.measure_all()

    .. note::
        Requires the active builder context to have a backend set.

    Returns:
        The ``register``\s the qubit measurement results will be stored in.
    """
    backend = active_backend()
    qubits = range(num_qubits())
    registers = [chans.MemorySlot(qubit) for qubit in qubits]
    measure_sched = macros.measure(
        qubits=qubits,
        inst_map=backend.defaults().instruction_schedule_map,
        meas_map=backend.configuration().meas_map,
        qubit_mem_slots={qubit: qubit for qubit in qubits})
    call_schedule(measure_sched)

    return registers


def delay_qubits(duration: int,
                 *qubits: Union[int, Iterable[int]]):
    r"""Insert delays on all of the :class:`channels.Channel`\s that correspond
    to the input ``qubits`` at the same time.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse3Q

        backend = FakeOpenPulse3Q()

        with pulse.build(backend) as pulse_prog:
            # Delay for 100 cycles on qubits 0, 1 and 2.
            regs = pulse.delay_qubits(100, 0, 1, 2)

    .. note:: Requires the active builder context to have a backend set.

    Args:
        duration: Duration to delay for.
        qubits: Physical qubits to delay on. Delays will be inserted based on
            the channels returned by :func:`pulse.qubit_channels`.
    """
    qubit_chans = set(itertools.chain.from_iterable(
        qubit_channels(qubit) for qubit in qubits))
    with align_left():
        for chan in qubit_chans:
            delay(duration, chan)


# Gate instructions
def call_gate(gate: circuit.Gate, qubits: Tuple[int, ...], lazy: bool = True):
    """Call a gate and lazily schedule it to its corresponding
    pulse instruction.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    .. jupyter-kernel:: python3
        :id: call_gate

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.pulse import builder
        from qiskit.circuit.library import standard_gates as gates
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            builder.call_gate(gates.CXGate(), (0, 1))

    We can see the role of the transpiler in scheduling gates by optimizing
    away two consecutive CNOT gates:

    .. jupyter-execute::

        with pulse.build(backend) as pulse_prog:
            with pulse.transpiler_settings(optimization_level=3):
                builder.call_gate(gates.CXGate(), (0, 1))
                builder.call_gate(gates.CXGate(), (0, 1))

        assert pulse_prog == pulse.Schedule()

    .. note:: If multiple gates are called in a row they may be optimized by
        the transpiler, depending on the
        :func:`pulse.active_transpiler_settings``.

    .. note:: Requires the active builder context to have a backend set.

    Args:
        gate: Circuit gate instance to call.
        qubits: Qubits to call gate on.
        lazy: If ``false`` the gate will be compiled immediately, otherwise
            it will be added onto a lazily evaluated quantum circuit to be
            compiled when the builder is forced to by a circuit assumption
            being broken, such as the inclusion of a pulse instruction or
            new alignment context.
    """
    _active_builder().call_gate(gate, qubits, lazy=lazy)


def cx(control: int, target: int):
    """Call a :class:`~qiskit.circuit.library.standard_gates.CXGate` on the
    input physical qubits.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.cx(0, 1)

    """
    call_gate(gates.CXGate(), (control, target))


def u1(theta: float, qubit: int):
    """Call a :class:`~qiskit.circuit.library.standard_gates.U1Gate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. jupyter-execute::

        import math

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.u1(math.pi, 1)

    """
    call_gate(gates.U1Gate(theta), qubit)


def u2(phi: float, lam: float, qubit: int):
    """Call a :class:`~qiskit.circuit.library.standard_gates.U2Gate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. jupyter-execute::

        import math

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.u2(0, math.pi, 1)

    """
    call_gate(gates.U2Gate(phi, lam), qubit)


def u3(theta: float, phi: float, lam: float, qubit: int):
    """Call a :class:`~qiskit.circuit.library.standard_gates.U3Gate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. jupyter-execute::

        import math

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.u3(math.pi, 0, math.pi, 1)

    """
    call_gate(gates.U3Gate(theta, phi, lam), qubit)


def x(qubit: int):
    """Call a :class:`~qiskit.circuit.library.standard_gates.XGate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favour of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.test.mock import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.x(0)

    """
    call_gate(gates.XGate(), qubit)
