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

r"""

.. _pulse_builder:

=============
Pulse Builder
=============

..
    We actually want people to think of these functions as being defined within the ``qiskit.pulse``
    namespace, not the submodule ``qiskit.pulse.builder``.

.. currentmodule: qiskit.pulse

Use the pulse builder DSL to write pulse programs with an imperative syntax.

.. warning::
    The pulse builder interface is still in active development. It may have
    breaking API changes without deprecation warnings in future releases until
    otherwise indicated.


The pulse builder provides an imperative API for writing pulse programs
with less difficulty than the :class:`~qiskit.pulse.Schedule` API.
It contextually constructs a pulse schedule and then emits the schedule for
execution. For example, to play a series of pulses on channels is as simple as:


.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import pulse

   dc = pulse.DriveChannel
   d0, d1, d2, d3, d4 = dc(0), dc(1), dc(2), dc(3), dc(4)

   with pulse.build(name='pulse_programming_in') as pulse_prog:
       pulse.play([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], d0)
       pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], d1)
       pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], d2)
       pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], d3)
       pulse.play([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], d4)

   pulse_prog.draw()

To begin pulse programming we must first initialize our program builder
context with :func:`build`, after which we can begin adding program
statements. For example, below we write a simple program that :func:`play`\s
a pulse:

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import pulse

   d0 = pulse.DriveChannel(0)

   with pulse.build() as pulse_prog:
       pulse.play(pulse.Constant(100, 1.0), d0)

   pulse_prog.draw()

The builder initializes a :class:`.pulse.Schedule`, ``pulse_prog``
and then begins to construct the program within the context. The output pulse
schedule will survive after the context is exited and can be used like a
normal Qiskit schedule.

Pulse programming has a simple imperative style. This leaves the programmer
to worry about the raw experimental physics of pulse programming and not
constructing cumbersome data structures.

We can optionally pass a :class:`~qiskit.providers.Backend` to
:func:`build` to enable enhanced functionality. Below, we prepare a Bell state
by automatically compiling the required pulses from their gate-level
representations, while simultaneously applying a long decoupling pulse to a
neighboring qubit. We terminate the experiment with a measurement to observe the
state we prepared. This program which mixes circuits and pulses will be
automatically lowered to be run as a pulse program.

With the pulse builder we are able to blend programming on qubits and channels.
While the pulse schedule is based on instructions that operate on
channels, the pulse builder automatically handles the mapping from qubits to
channels for you.

.. autofunction:: build


Channels
========

Methods to return the correct channels for the respective qubit indices.

.. plot::
   :include-source:
   :nofigs:

    from qiskit import pulse
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=2, calibrate_instructions=True)

    with pulse.build(backend) as drive_sched:
        d0 = pulse.drive_channel(0)
        print(d0)

.. code-block:: text

    DriveChannel(0)

.. autofunction:: acquire_channel
.. autofunction:: control_channels
.. autofunction:: drive_channel
.. autofunction:: measure_channel


Instructions
============

Pulse instructions are available within the builder interface. Here's an example:

.. plot::
   :alt: Output from the previous code.
   :include-source:

    from qiskit import pulse
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=2, calibrate_instructions=True)

    with pulse.build(backend) as drive_sched:
        d0 = pulse.drive_channel(0)
        a0 = pulse.acquire_channel(0)

        pulse.play(pulse.Constant(10, 1.0), d0)
        pulse.delay(20, d0)
        pulse.shift_phase(3.14/2, d0)
        pulse.set_phase(3.14, d0)
        pulse.shift_frequency(1e7, d0)
        pulse.set_frequency(5e9, d0)

        with pulse.build() as temp_sched:
            pulse.play(pulse.Gaussian(20, 1.0, 3.0), d0)
            pulse.play(pulse.Gaussian(20, -1.0, 3.0), d0)

        pulse.call(temp_sched)
        pulse.acquire(30, a0, pulse.MemorySlot(0))

    drive_sched.draw()

.. autofunction:: acquire
.. autofunction:: barrier
.. autofunction:: call
.. autofunction:: delay
.. autofunction:: play
.. autofunction:: reference
.. autofunction:: set_frequency
.. autofunction:: set_phase
.. autofunction:: shift_frequency
.. autofunction:: shift_phase
.. autofunction:: snapshot


Contexts
========

Builder aware contexts that modify the construction of a pulse program. For
example an alignment context like :func:`align_right` may
be used to align all pulses as late as possible in a pulse program.

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import pulse

   d0 = pulse.DriveChannel(0)
   d1 = pulse.DriveChannel(1)

   with pulse.build() as pulse_prog:
       with pulse.align_right():
           # this pulse will start at t=0
           pulse.play(pulse.Constant(100, 1.0), d0)
           # this pulse will start at t=80
           pulse.play(pulse.Constant(20, 1.0), d1)

   pulse_prog.draw()

.. autofunction:: align_equispaced
.. autofunction:: align_func
.. autofunction:: align_left
.. autofunction:: align_right
.. autofunction:: align_sequential
.. autofunction:: frequency_offset
.. autofunction:: phase_offset


Macros
======

Macros help you add more complex functionality to your pulse program.

.. plot::
   :include-source:
   :nofigs:

    from qiskit import pulse
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=2, calibrate_instructions=True)

    with pulse.build(backend) as measure_sched:
        mem_slot = pulse.measure(0)
        print(mem_slot)

.. code-block:: text

    MemorySlot(0)

.. autofunction:: measure
.. autofunction:: measure_all
.. autofunction:: delay_qubits


Utilities
=========

The utility functions can be used to gather attributes about the backend and modify
how the program is built.

.. plot::
   :include-source:
   :nofigs:

    from qiskit import pulse

    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=2, calibrate_instructions=True)

    with pulse.build(backend) as u3_sched:
        print('Number of qubits in backend: {}'.format(pulse.num_qubits()))

        samples = 160
        print('There are {} samples in {} seconds'.format(
            samples, pulse.samples_to_seconds(160)))

        seconds = 1e-6
        print('There are {} seconds in {} samples.'.format(
            seconds, pulse.seconds_to_samples(1e-6)))

.. code-block:: text

    Number of qubits in backend: 1
    There are 160 samples in 3.5555555555555554e-08 seconds
    There are 1e-06 seconds in 4500 samples.

.. autofunction:: active_backend
.. autofunction:: num_qubits
.. autofunction:: qubit_channels
.. autofunction:: samples_to_seconds
.. autofunction:: seconds_to_samples
"""
from __future__ import annotations
import contextvars
import functools
import itertools
import sys
import uuid
import warnings
from collections.abc import Generator, Callable, Iterable
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import TypeVar, ContextManager, TypedDict, Union, Optional, Dict

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
    channels as chans,
    configuration,
    exceptions,
    instructions,
    macros,
    library,
    transforms,
)
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    from typing_extensions import Unpack

#: contextvars.ContextVar[BuilderContext]: active builder
BUILDER_CONTEXTVAR: contextvars.ContextVar["_PulseBuilder"] = contextvars.ContextVar("backend")

T = TypeVar("T")

StorageLocation = Union[chans.MemorySlot, chans.RegisterSlot]


def _requires_backend(function: Callable[..., T]) -> Callable[..., T]:
    """Decorator a function to raise if it is called without a builder with a
    set backend.
    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.backend is None:
            raise exceptions.BackendNotSet(
                'This function requires the builder to have a "backend" set.'
            )
        return function(self, *args, **kwargs)

    return wrapper


class _PulseBuilder:
    """Builder context class."""

    __alignment_kinds__ = {
        "left": transforms.AlignLeft(),
        "right": transforms.AlignRight(),
        "sequential": transforms.AlignSequential(),
    }

    def __init__(
        self,
        backend=None,
        block: ScheduleBlock | None = None,
        name: str | None = None,
        default_alignment: str | AlignmentKind = "left",
    ):
        """Initialize the builder context.

        .. note::
            At some point we may consider incorporating the builder into
            the :class:`~qiskit.pulse.Schedule` class. However, the risk of
            this is tying the user interface to the intermediate
            representation. For now we avoid this at the cost of some code
            duplication.

        Args:
            backend (Backend): Input backend to use in
                builder. If not set certain functionality will be unavailable.
            block: Initital ``ScheduleBlock`` to build on.
            name: Name of pulse program to be built.
            default_alignment: Default scheduling alignment for builder.
                One of ``left``, ``right``, ``sequential`` or an instance of
                :class:`~qiskit.pulse.transforms.alignments.AlignmentKind` subclass.

        Raises:
            PulseError: When invalid ``default_alignment`` or `block` is specified.
        """
        #: Backend: Backend instance for context builder.
        self._backend = backend

        # Token for this ``_PulseBuilder``'s ``ContextVar``.
        self._backend_ctx_token: contextvars.Token[_PulseBuilder] | None = None

        # Stack of context.
        self._context_stack: list[ScheduleBlock] = []

        #: str: Name of the output program
        self._name = name

        # Add root block if provided. Schedule will be built on top of this.
        if block is not None:
            if isinstance(block, ScheduleBlock):
                root_block = block
            elif isinstance(block, Schedule):
                root_block = self._naive_typecast_schedule(block)
            else:
                raise exceptions.PulseError(
                    f"Input `block` type {block.__class__.__name__} is "
                    "not a valid format. Specify a pulse program."
                )
            self._context_stack.append(root_block)

        # Set default alignment context
        if isinstance(default_alignment, AlignmentKind):  # AlignmentKind instance
            alignment = default_alignment
        else:  # str identifier
            alignment = _PulseBuilder.__alignment_kinds__.get(default_alignment, default_alignment)
        if not isinstance(alignment, AlignmentKind):
            raise exceptions.PulseError(
                f"Given `default_alignment` {repr(default_alignment)} is "
                "not a valid transformation. Set one of "
                f'{", ".join(_PulseBuilder.__alignment_kinds__.keys())}, '
                "or set an instance of `AlignmentKind` subclass."
            )
        self.push_context(alignment)

    def __enter__(self) -> ScheduleBlock:
        """Enter this builder context and yield either the supplied schedule
        or the schedule created for the user.

        Returns:
            The schedule that the builder will build on.
        """
        self._backend_ctx_token = BUILDER_CONTEXTVAR.set(self)
        output = self._context_stack[0]
        output._name = self._name or output.name

        return output

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the builder context and compile the built pulse program."""
        self.compile()
        BUILDER_CONTEXTVAR.reset(self._backend_ctx_token)

    @property
    def backend(self):
        """Returns the builder backend if set.

        Returns:
            Optional[Backend]: The builder's backend.
        """
        return self._backend

    def push_context(self, alignment: AlignmentKind):
        """Push new context to the stack."""
        self._context_stack.append(ScheduleBlock(alignment_context=alignment))

    def pop_context(self) -> ScheduleBlock:
        """Pop the last context from the stack."""
        if len(self._context_stack) == 1:
            raise exceptions.PulseError("The root context cannot be popped out.")

        return self._context_stack.pop()

    def get_context(self) -> ScheduleBlock:
        """Get current context.

        Notes:
            New instruction can be added by `.append_subroutine` or `.append_instruction` method.
            Use above methods rather than directly accessing to the current context.
        """
        return self._context_stack[-1]

    @property
    @_requires_backend
    def num_qubits(self):
        """Get the number of qubits in the backend."""
        # backendV2
        if isinstance(self.backend, BackendV2):
            return self.backend.num_qubits
        return self.backend.configuration().n_qubits

    def compile(self) -> ScheduleBlock:
        """Compile and output the built pulse program."""
        # Not much happens because we currently compile as we build.
        # This should be offloaded to a true compilation module
        # once we define a more sophisticated IR.

        while len(self._context_stack) > 1:
            current = self.pop_context()
            self.append_subroutine(current)

        return self._context_stack[0]

    def append_instruction(self, instruction: instructions.Instruction):
        """Add an instruction to the builder's context schedule.

        Args:
            instruction: Instruction to append.
        """
        self._context_stack[-1].append(instruction)

    def append_reference(self, name: str, *extra_keys: str):
        """Add external program as a :class:`~qiskit.pulse.instructions.Reference` instruction.

        Args:
            name: Name of subroutine.
            extra_keys: Assistance keys to uniquely specify the subroutine.
        """
        inst = instructions.Reference(name, *extra_keys)
        self.append_instruction(inst)

    def append_subroutine(self, subroutine: Schedule | ScheduleBlock):
        """Append a :class:`ScheduleBlock` to the builder's context schedule.

        This operation doesn't create a reference. Subroutine is directly
        appended to current context schedule.

        Args:
            subroutine: ScheduleBlock to append to the current context block.

        Raises:
            PulseError: When subroutine is not Schedule nor ScheduleBlock.
        """
        if not isinstance(subroutine, (ScheduleBlock, Schedule)):
            raise exceptions.PulseError(
                f"'{subroutine.__class__.__name__}' is not valid data format in the builder. "
                "'Schedule' and 'ScheduleBlock' can be appended to the builder context."
            )

        if len(subroutine) == 0:
            return
        if isinstance(subroutine, Schedule):
            subroutine = self._naive_typecast_schedule(subroutine)
        self._context_stack[-1].append(subroutine)

    @singledispatchmethod
    def call_subroutine(
        self,
        subroutine: Schedule | ScheduleBlock,
        name: str | None = None,
        value_dict: dict[ParameterExpression, ParameterValueType] | None = None,
        **kw_params: ParameterValueType,
    ):
        """Call a schedule or circuit defined outside of the current scope.

        The ``subroutine`` is appended to the context schedule as a call instruction.
        This logic just generates a convenient program representation in the compiler.
        Thus, this doesn't affect execution of inline subroutines.
        See :class:`~pulse.instructions.Call` for more details.

        Args:
            subroutine: Target schedule or circuit to append to the current context.
            name: Name of subroutine if defined.
            value_dict: Parameter object and assigned value mapping. This is more precise way to
                identify a parameter since mapping is managed with unique object id rather than
                name. Especially there is any name collision in a parameter table.
            kw_params: Parameter values to bind to the target subroutine
                with string parameter names. If there are parameter name overlapping,
                these parameters are updated with the same assigned value.

        Raises:
            PulseError:
                - When input subroutine is not valid data format.
        """
        raise exceptions.PulseError(
            f"Subroutine type {subroutine.__class__.__name__} is "
            "not valid data format. Call "
            "Schedule, or ScheduleBlock."
        )

    @call_subroutine.register
    def _(
        self,
        target_block: ScheduleBlock,
        name: Optional[str] = None,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        **kw_params: ParameterValueType,
    ):
        if len(target_block) == 0:
            return

        # Create local parameter assignment
        local_assignment = {}
        for param_name, value in kw_params.items():
            params = target_block.get_parameters(param_name)
            if not params:
                raise exceptions.PulseError(
                    f"Parameter {param_name} is not defined in the target subroutine. "
                    f'{", ".join(map(str, target_block.parameters))} can be specified.'
                )
            for param in params:
                local_assignment[param] = value

        if value_dict:
            if local_assignment.keys() & value_dict.keys():
                warnings.warn(
                    "Some parameters provided by 'value_dict' conflict with one through "
                    "keyword arguments. Parameter values in the keyword arguments "
                    "are overridden by the dictionary values.",
                    UserWarning,
                )
            local_assignment.update(value_dict)

        if local_assignment:
            target_block = target_block.assign_parameters(local_assignment, inplace=False)

        if name is None:
            # Add unique string, not to accidentally override existing reference entry.
            keys: tuple[str, ...] = (target_block.name, uuid.uuid4().hex)
        else:
            keys = (name,)

        self.append_reference(*keys)
        self.get_context().assign_references({keys: target_block}, inplace=True)

    @call_subroutine.register
    def _(
        self,
        target_schedule: Schedule,
        name: Optional[str] = None,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        **kw_params: ParameterValueType,
    ):
        if len(target_schedule) == 0:
            return

        self.call_subroutine(
            self._naive_typecast_schedule(target_schedule),
            name=name,
            value_dict=value_dict,
            **kw_params,
        )

    @staticmethod
    def _naive_typecast_schedule(schedule: Schedule):
        # Naively convert into ScheduleBlock
        from qiskit.pulse.transforms import inline_subroutines, flatten, pad

        preprocessed_schedule = inline_subroutines(flatten(schedule))
        pad(preprocessed_schedule, inplace=True, pad_with=instructions.TimeBlockade)

        # default to left alignment, namely ASAP scheduling
        target_block = ScheduleBlock(name=schedule.name)
        for _, inst in preprocessed_schedule.instructions:
            target_block.append(inst, inplace=True)

        return target_block

    def get_dt(self):
        """Retrieve dt differently based on the type of Backend"""
        if isinstance(self.backend, BackendV2):
            return self.backend.dt
        return self.backend.configuration().dt


@deprecate_pulse_func
def build(
    backend=None,
    schedule: ScheduleBlock | None = None,
    name: str | None = None,
    default_alignment: str | AlignmentKind | None = "left",
) -> ContextManager[ScheduleBlock]:
    """Create a context manager for launching the imperative pulse builder DSL.

    Args:
        backend (Backend): A Qiskit backend. If not supplied certain
            builder functionality will be unavailable.
        schedule: A pulse ``ScheduleBlock`` in which your pulse program will be built.
        name: Name of pulse program to be built.
        default_alignment: Default scheduling alignment for builder.
            One of ``left``, ``right``, ``sequential`` or an alignment context.

    Returns:
        A new builder context which has the active builder initialized.
    """
    return _PulseBuilder(
        backend=backend,
        block=schedule,
        name=name,
        default_alignment=default_alignment,
    )


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
    except LookupError as ex:
        raise exceptions.NoActiveBuilder(
            "A Pulse builder function was called outside of "
            "a builder context. Try calling within a builder "
            'context, eg., "with pulse.build() as schedule: ...".'
        ) from ex


@deprecate_pulse_func
def active_backend():
    """Get the backend of the currently active builder context.

    Returns:
        Backend: The active backend in the currently active
            builder context.

    Raises:
        exceptions.BackendNotSet: If the builder does not have a backend set.
    """
    builder = _active_builder().backend
    if builder is None:
        raise exceptions.BackendNotSet(
            'This function requires the active builder to have a "backend" set.'
        )
    return builder


def append_schedule(schedule: Schedule | ScheduleBlock):
    """Call a schedule by appending to the active builder's context block.

    Args:
        schedule: Schedule or ScheduleBlock to append.
    """
    _active_builder().append_subroutine(schedule)


def append_instruction(instruction: instructions.Instruction):
    """Append an instruction to the active builder's context schedule.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.builder.append_instruction(pulse.Delay(10, d0))

        print(pulse_prog.instructions)

    .. code-block:: text

        ((0, Delay(10, DriveChannel(0))),)
    """
    _active_builder().append_instruction(instruction)


@deprecate_pulse_func
def num_qubits() -> int:
    """Return number of qubits in the currently active backend.

    .. note:: Requires the active builder context to have a backend set.
    """
    if isinstance(active_backend(), BackendV2):
        return active_backend().num_qubits
    return active_backend().configuration().n_qubits


@deprecate_pulse_func
def seconds_to_samples(seconds: float | np.ndarray) -> int | np.ndarray:
    """Obtain the number of samples that will elapse in ``seconds`` on the
    active backend.

    Rounds down.

    Args:
        seconds: Time in seconds to convert to samples.

    Returns:
        The number of samples for the time to elapse
    """
    dt = _active_builder().get_dt()
    if isinstance(seconds, np.ndarray):
        return (seconds / dt).astype(int)
    return int(seconds / dt)


@deprecate_pulse_func
def samples_to_seconds(samples: int | np.ndarray) -> float | np.ndarray:
    """Obtain the time in seconds that will elapse for the input number of
    samples on the active backend.

    Args:
        samples: Number of samples to convert to time in seconds.

    Returns:
        The time that elapses in ``samples``.
    """
    return samples * _active_builder().get_dt()


@deprecate_pulse_func
def qubit_channels(qubit: int) -> set[chans.Channel]:
    """Returns the set of channels associated with a qubit.

    .. note:: Requires the active builder context to have a backend set.

    .. note:: A channel may still be associated with another qubit in this list
        such as in the case where significant crosstalk exists.

    """

    # implement as the inner function to avoid API change for a patch release in 0.24.2.
    def get_qubit_channels_v2(backend: BackendV2, qubit: int):
        r"""Return a list of channels which operate on the given ``qubit``.
        Returns:
            List of ``Channel``\s operated on my the given ``qubit``.
        """
        channels = []

        # add multi-qubit channels
        for node_qubits in backend.coupling_map:
            if qubit in node_qubits:
                control_channel = backend.control_channel(node_qubits)
                if control_channel:
                    channels.extend(control_channel)

        # add single qubit channels
        channels.append(backend.drive_channel(qubit))
        channels.append(backend.measure_channel(qubit))
        channels.append(backend.acquire_channel(qubit))
        return channels

    # backendV2
    if isinstance(active_backend(), BackendV2):
        return set(get_qubit_channels_v2(active_backend(), qubit))
    return set(active_backend().configuration().get_qubit_channels(qubit))


def _qubits_to_channels(*channels_or_qubits: int | chans.Channel) -> set[chans.Channel]:
    """Returns the unique channels of the input qubits."""
    channels = set()
    for channel_or_qubit in channels_or_qubits:
        if isinstance(channel_or_qubit, int):
            channels |= qubit_channels(channel_or_qubit)
        elif isinstance(channel_or_qubit, chans.Channel):
            channels.add(channel_or_qubit)
        else:
            raise exceptions.PulseError(
                f'{channel_or_qubit} is not a "Channel" or qubit (integer).'
            )
    return channels


# Contexts


@contextmanager
@deprecate_pulse_func
def align_left() -> Generator[None, None, None]:
    """Left alignment pulse scheduling context.

    Pulse instructions within this context are scheduled as early as possible
    by shifting them left to the earliest available time.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_left():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will start at t=0
                pulse.play(pulse.Constant(20, 1.0), d1)
        pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

        assert pulse_prog.ch_start_time(d0) == pulse_prog.ch_start_time(d1)

    Yields:
        None
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignLeft())
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
@deprecate_pulse_func
def align_right() -> Generator[None, None, None]:
    """Right alignment pulse scheduling context.

    Pulse instructions within this context are scheduled as late as possible
    by shifting them right to the latest available time.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_right():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will start at t=80
                pulse.play(pulse.Constant(20, 1.0), d1)
        pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_stop_time(d1)

    Yields:
        None
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignRight())
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
@deprecate_pulse_func
def align_sequential() -> Generator[None, None, None]:
    """Sequential alignment pulse scheduling context.

    Pulse instructions within this context are scheduled sequentially in time
    such that no two instructions will be played at the same time.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_sequential():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will also start at t=100
                pulse.play(pulse.Constant(20, 1.0), d1)
        pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_start_time(d1)

    Yields:
        None
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignSequential())
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
@deprecate_pulse_func
def align_equispaced(duration: int | ParameterExpression) -> Generator[None, None, None]:
    """Equispaced alignment pulse scheduling context.

    Pulse instructions within this context are scheduled with the same interval spacing such that
    the total length of the context block is ``duration``.
    If the total free ``duration`` cannot be evenly divided by the number of instructions
    within the context, the modulo is split and then prepended and appended to
    the returned schedule. Delay instructions are automatically inserted in between pulses.

    This context is convenient to write a schedule for periodical dynamic decoupling or
    the Hahn echo sequence.

    Examples:

    .. plot::
       :alt: Output from the previous code.
       :include-source:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        x90 = pulse.Gaussian(10, 0.1, 3)
        x180 = pulse.Gaussian(10, 0.2, 3)

        with pulse.build() as hahn_echo:
            with pulse.align_equispaced(duration=100):
                pulse.play(x90, d0)
                pulse.play(x180, d0)
                pulse.play(x90, d0)

        hahn_echo.draw()

    Args:
        duration: Duration of this context. This should be larger than the schedule duration.

    Yields:
        None

    Notes:
        The scheduling is performed for sub-schedules within the context rather than
        channel-wise. If you want to apply the equispaced context for each channel,
        you should use the context independently for channels.
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignEquispaced(duration=duration))
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
@deprecate_pulse_func
def align_func(
    duration: int | ParameterExpression, func: Callable[[int], float]
) -> Generator[None, None, None]:
    """Callback defined alignment pulse scheduling context.

    Pulse instructions within this context are scheduled at the location specified by
    arbitrary callback function `position` that takes integer index and returns
    the associated fractional location within [0, 1].
    Delay instruction is automatically inserted in between pulses.

    This context may be convenient to write a schedule of arbitrary dynamical decoupling
    sequences such as Uhrig dynamical decoupling.

    Examples:

    .. plot::
       :alt: Output from the previous code.
       :include-source:

        import numpy as np
        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        x90 = pulse.Gaussian(10, 0.1, 3)
        x180 = pulse.Gaussian(10, 0.2, 3)

        def udd10_pos(j):
            return np.sin(np.pi*j/(2*10 + 2))**2

        with pulse.build() as udd_sched:
            pulse.play(x90, d0)
            with pulse.align_func(duration=300, func=udd10_pos):
                for _ in range(10):
                    pulse.play(x180, d0)
            pulse.play(x90, d0)

        udd_sched.draw()

    Args:
        duration: Duration of context. This should be larger than the schedule duration.
        func: A function that takes an index of sub-schedule and returns the
            fractional coordinate of of that sub-schedule.
            The returned value should be defined within [0, 1].
            The pulse index starts from 1.

    Yields:
        None

    Notes:
        The scheduling is performed for sub-schedules within the context rather than
        channel-wise. If you want to apply the numerical context for each channel,
        you need to apply the context independently to channels.
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignFunc(duration=duration, func=func))
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def general_transforms(alignment_context: AlignmentKind) -> Generator[None, None, None]:
    """Arbitrary alignment transformation defined by a subclass instance of
    :class:`~qiskit.pulse.transforms.alignments.AlignmentKind`.

    Args:
        alignment_context: Alignment context instance that defines schedule transformation.

    Yields:
        None

    Raises:
        PulseError: When input ``alignment_context`` is not ``AlignmentKind`` subclasses.
    """
    if not isinstance(alignment_context, AlignmentKind):
        raise exceptions.PulseError("Input alignment context is not `AlignmentKind` subclass.")

    builder = _active_builder()
    builder.push_context(alignment_context)
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
@deprecate_pulse_func
def phase_offset(phase: float, *channels: chans.PulseChannel) -> Generator[None, None, None]:
    """Shift the phase of input channels on entry into context and undo on exit.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

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
@deprecate_pulse_func
def frequency_offset(
    frequency: float, *channels: chans.PulseChannel, compensate_phase: bool = False
) -> Generator[None, None, None]:
    """Shift the frequency of inputs channels on entry into context and undo on exit.

    Args:
        frequency: Amount of frequency offset in Hz.
        channels: Channels to offset frequency of.
        compensate_phase: Compensate for accumulated phase accumulated with
            respect to the channels' frame at its initial frequency.

    Yields:
        None
    """
    builder = _active_builder()
    # TODO: Need proper implementation of compensation. t0 may depend on the parent context.
    #  For example, the instruction position within the equispaced context depends on
    #  the current total number of instructions, thus adding more instruction after
    #  offset context may change the t0 when the parent context is transformed.
    t0 = builder.get_context().duration

    for channel in channels:
        shift_frequency(frequency, channel)
    try:
        yield
    finally:
        if compensate_phase:
            duration = builder.get_context().duration - t0

            accumulated_phase = 2 * np.pi * ((duration * builder.get_dt() * frequency) % 1)
            for channel in channels:
                shift_phase(-accumulated_phase, channel)

        for channel in channels:
            shift_frequency(-frequency, channel)


# Channels
@deprecate_pulse_func
def drive_channel(qubit: int) -> chans.DriveChannel:
    """Return ``DriveChannel`` for ``qubit`` on the active builder backend.

    .. note:: Requires the active builder context to have a backend set.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().drive_channel(qubit)
    return active_backend().configuration().drive(qubit)


@deprecate_pulse_func
def measure_channel(qubit: int) -> chans.MeasureChannel:
    """Return ``MeasureChannel`` for ``qubit`` on the active builder backend.

    .. note:: Requires the active builder context to have a backend set.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().measure_channel(qubit)
    return active_backend().configuration().measure(qubit)


@deprecate_pulse_func
def acquire_channel(qubit: int) -> chans.AcquireChannel:
    """Return ``AcquireChannel`` for ``qubit`` on the active builder backend.

    .. note:: Requires the active builder context to have a backend set.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().acquire_channel(qubit)
    return active_backend().configuration().acquire(qubit)


@deprecate_pulse_func
def control_channels(*qubits: Iterable[int]) -> list[chans.ControlChannel]:
    """Return ``ControlChannel`` for ``qubit`` on the active builder backend.

    Return the secondary drive channel for the given qubit -- typically
    utilized for controlling multi-qubit interactions.

    .. note:: Requires the active builder context to have a backend set.

    Args:
      qubits: Tuple or list of ordered qubits of the form
        `(control_qubit, target_qubit)`.

    Returns:
        List of control channels associated with the supplied ordered list
        of qubits.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().control_channel(qubits)
    return active_backend().configuration().control(qubits=qubits)


# Base Instructions
@deprecate_pulse_func
def delay(duration: int, channel: chans.Channel, name: str | None = None):
    """Delay on a ``channel`` for a ``duration``.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.delay(10, d0)

    Args:
        duration: Number of cycles to delay for on ``channel``.
        channel: Channel to delay on.
        name: Name of the instruction.
    """
    append_instruction(instructions.Delay(duration, channel, name=name))


@deprecate_pulse_func
def play(pulse: library.Pulse | np.ndarray, channel: chans.PulseChannel, name: str | None = None):
    """Play a ``pulse`` on a ``channel``.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.play(pulse.Constant(10, 1.0), d0)

    Args:
        pulse: Pulse to play.
        channel: Channel to play pulse on.
        name: Name of the pulse.
    """
    if not isinstance(pulse, library.Pulse):
        pulse = library.Waveform(pulse)

    append_instruction(instructions.Play(pulse, channel, name=name))


class _MetaDataType(TypedDict, total=False):
    kernel: configuration.Kernel
    discriminator: configuration.Discriminator
    mem_slot: chans.MemorySlot
    reg_slot: chans.RegisterSlot
    name: str


@deprecate_pulse_func
def acquire(
    duration: int,
    qubit_or_channel: int | chans.AcquireChannel,
    register: StorageLocation,
    **metadata: Unpack[_MetaDataType],
):
    """Acquire for a ``duration`` on a ``channel`` and store the result
    in a ``register``.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        acq0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as pulse_prog:
            pulse.acquire(100, acq0, mem0)

            # measurement metadata
            kernel = pulse.configuration.Kernel('linear_discriminator')
            pulse.acquire(100, acq0, mem0, kernel=kernel)

    .. note:: The type of data acquire will depend on the execution ``meas_level``.

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
        append_instruction(
            instructions.Acquire(duration, qubit_or_channel, mem_slot=register, **metadata)
        )
    elif isinstance(register, chans.RegisterSlot):
        append_instruction(
            instructions.Acquire(duration, qubit_or_channel, reg_slot=register, **metadata)
        )
    else:
        raise exceptions.PulseError(f'Register of type: "{type(register)}" is not supported')


@deprecate_pulse_func
def set_frequency(frequency: float, channel: chans.PulseChannel, name: str | None = None):
    """Set the ``frequency`` of a pulse ``channel``.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.set_frequency(1e9, d0)

    Args:
        frequency: Frequency in Hz to set channel to.
        channel: Channel to set frequency of.
        name: Name of the instruction.
    """
    append_instruction(instructions.SetFrequency(frequency, channel, name=name))


@deprecate_pulse_func
def shift_frequency(frequency: float, channel: chans.PulseChannel, name: str | None = None):
    """Shift the ``frequency`` of a pulse ``channel``.

    Examples:

    .. plot::
        :include-source:
        :nofigs:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.shift_frequency(1e9, d0)

    Args:
        frequency: Frequency in Hz to shift channel frequency by.
        channel: Channel to shift frequency of.
        name: Name of the instruction.
    """
    append_instruction(instructions.ShiftFrequency(frequency, channel, name=name))


@deprecate_pulse_func
def set_phase(phase: float, channel: chans.PulseChannel, name: str | None = None):
    """Set the ``phase`` of a pulse ``channel``.

    Examples:

    .. plot::
        :include-source:
        :nofigs:

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.set_phase(math.pi, d0)

    Args:
        phase: Phase in radians to set channel carrier signal to.
        channel: Channel to set phase of.
        name: Name of the instruction.
    """
    append_instruction(instructions.SetPhase(phase, channel, name=name))


@deprecate_pulse_func
def shift_phase(phase: float, channel: chans.PulseChannel, name: str | None = None):
    """Shift the ``phase`` of a pulse ``channel``.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.shift_phase(math.pi, d0)

    Args:
        phase: Phase in radians to shift channel carrier signal by.
        channel: Channel to shift phase of.
        name: Name of the instruction.
    """
    append_instruction(instructions.ShiftPhase(phase, channel, name))


@deprecate_pulse_func
def snapshot(label: str, snapshot_type: str = "statevector"):
    """Simulator snapshot.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        with pulse.build() as pulse_prog:
            pulse.snapshot('first', 'statevector')

    Args:
        label: Label for snapshot.
        snapshot_type: Type of snapshot.
    """
    append_instruction(instructions.Snapshot(label, snapshot_type=snapshot_type))


@deprecate_pulse_func
def call(
    target: Schedule | ScheduleBlock | None,
    name: str | None = None,
    value_dict: dict[ParameterValueType, ParameterValueType] | None = None,
    **kw_params: ParameterValueType,
):
    """Call the subroutine within the currently active builder context with arbitrary
    parameters which will be assigned to the target program.

    .. note::

        If the ``target`` program is a :class:`.ScheduleBlock`, then a :class:`.Reference`
        instruction will be created and appended to the current context.
        The ``target`` program will be immediately assigned to the current scope as a subroutine.
        If the ``target`` program is :class:`.Schedule`, it will be wrapped by the
        :class:`.Call` instruction and appended to the current context to avoid
        a mixed representation of :class:`.ScheduleBlock` and :class:`.Schedule`.
        If the ``target`` program is a :class:`.QuantumCircuit` it will be scheduled
        and the new :class:`.Schedule` will be added as a :class:`.Call` instruction.

    Examples:

        1. Calling a schedule block (recommended)

        .. plot::
           :include-source:
           :nofigs:
           :context: reset

            from qiskit import circuit, pulse
            from qiskit.providers.fake_provider import GenericBackendV2

            backend = GenericBackendV2(num_qubits=5, calibrate_instructions=True)

            with pulse.build() as x_sched:
                pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

            with pulse.build() as pulse_prog:
                pulse.call(x_sched)

            print(pulse_prog)

        .. code-block:: text

            ScheduleBlock(
                ScheduleBlock(
                    Play(
                        Gaussian(duration=160, amp=(0.1+0j), sigma=40),
                        DriveChannel(0)
                    ),
                    name="block0",
                    transform=AlignLeft()
                ),
                name="block1",
                transform=AlignLeft()
            )

        The actual program is stored in the reference table attached to the schedule.

        .. plot::
           :include-source:
           :nofigs:
           :context:

            print(pulse_prog.references)

        .. code-block:: text

            ReferenceManager:
              - ('block0', '634b3b50bd684e26a673af1fbd2d6c81'): ScheduleBlock(Play(Gaussian(...

        In addition, you can call a parameterized target program with parameter assignment.

        .. plot::
           :include-source:
           :nofigs:
           :context:

            amp = circuit.Parameter("amp")

            with pulse.build() as subroutine:
                pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

            with pulse.build() as pulse_prog:
                pulse.call(subroutine, amp=0.1)
                pulse.call(subroutine, amp=0.3)

            print(pulse_prog)

        .. code-block:: text

            ScheduleBlock(
                ScheduleBlock(
                    Play(
                        Gaussian(duration=160, amp=(0.1+0j), sigma=40),
                        DriveChannel(0)
                    ),
                    name="block2",
                    transform=AlignLeft()
                ),
                ScheduleBlock(
                    Play(
                        Gaussian(duration=160, amp=(0.3+0j), sigma=40),
                        DriveChannel(0)
                    ),
                    name="block2",
                    transform=AlignLeft()
                ),
                name="block3",
                transform=AlignLeft()
            )

        If there is a name collision between parameters, you can distinguish them by specifying
        each parameter object in a python dictionary. For example,

        .. plot::
           :include-source:
           :nofigs:
           :context:

            amp1 = circuit.Parameter('amp')
            amp2 = circuit.Parameter('amp')

            with pulse.build() as subroutine:
                pulse.play(pulse.Gaussian(160, amp1, 40), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(160, amp2, 40), pulse.DriveChannel(1))

            with pulse.build() as pulse_prog:
                pulse.call(subroutine, value_dict={amp1: 0.1, amp2: 0.3})

            print(pulse_prog)

        .. code-block:: text

            ScheduleBlock(
                ScheduleBlock(
                    Play(Gaussian(duration=160, amp=(0.1+0j), sigma=40), DriveChannel(0)),
                    Play(Gaussian(duration=160, amp=(0.3+0j), sigma=40), DriveChannel(1)),
                    name="block4",
                    transform=AlignLeft()
                ),
                name="block5",
                transform=AlignLeft()
            )

        2. Calling a schedule

        .. plot::
           :include-source:
           :nofigs:
           :context:

            x_sched = backend.instruction_schedule_map.get("x", (0,))

            with pulse.build(backend) as pulse_prog:
                pulse.call(x_sched)

            print(pulse_prog)

        .. code-block:: text

            ScheduleBlock(
                Call(
                    Schedule(
                        (
                            0,
                            Play(
                                Drag(
                                    duration=160,
                                    amp=(0.18989731546729305+0j),
                                    sigma=40,
                                    beta=-1.201258305015517,
                                    name='drag_86a8'
                                ),
                                DriveChannel(0),
                                name='drag_86a8'
                            )
                        ),
                        name="x"
                    ),
                    name='x'
                ),
                name="block6",
                transform=AlignLeft()
            )

        Currently, the backend calibrated gates are provided in the form of :class:`~.Schedule`.
        The parameter assignment mechanism is available also for schedules.
        However, the called schedule is not treated as a reference.


    Args:
        target: Target circuit or pulse schedule to call.
        name: Optional. A unique name of subroutine if defined. When the name is explicitly
            provided, one cannot call different schedule blocks with the same name.
        value_dict: Optional. Parameters assigned to the ``target`` program.
            If this dictionary is provided, the ``target`` program is copied and
            then stored in the main built schedule and its parameters are assigned to the given values.
            This dictionary is keyed on :class:`~.Parameter` objects,
            allowing parameter name collision to be avoided.
        kw_params: Alternative way to provide parameters.
            Since this is keyed on the string parameter name,
            the parameters having the same name are all updated together.
            If you want to avoid name collision, use ``value_dict`` with :class:`~.Parameter`
            objects instead.
    """
    _active_builder().call_subroutine(target, name, value_dict, **kw_params)


@deprecate_pulse_func
def reference(name: str, *extra_keys: str):
    """Refer to undefined subroutine by string keys.

    A :class:`~qiskit.pulse.instructions.Reference` instruction is implicitly created
    and a schedule can be separately registered to the reference at a later stage.

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import pulse

        with pulse.build() as main_prog:
            pulse.reference("x_gate", "q0")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

        main_prog.assign_references(subroutine_dict={("x_gate", "q0"): subroutine})

    Args:
        name: Name of subroutine.
        extra_keys: Helper keys to uniquely specify the subroutine.
    """
    _active_builder().append_reference(name, *extra_keys)


# Directives
@deprecate_pulse_func
def barrier(*channels_or_qubits: chans.Channel | int, name: str | None = None):
    """Barrier directive for a set of channels and qubits.

    This directive prevents the compiler from moving instructions across
    the barrier. The barrier allows the pulse compiler to take care of more advanced
    scheduling alignment operations across channels.

    .. note:: Requires the active builder context to have a backend set if
        qubits are barriered on.

    Args:
        channels_or_qubits: Channels or qubits to barrier.
        name: Name for the barrier
    """
    channels = _qubits_to_channels(*channels_or_qubits)
    if len(channels) > 1:
        append_instruction(directives.RelativeBarrier(*channels, name=name))


# Macros
def macro(func: Callable):
    """Wrap a Python function and activate the parent builder context at calling time.

    This enables embedding Python functions as builder macros. This generates a new
    :class:`pulse.Schedule` that is embedded in the parent builder context with
    every call of the decorated macro function. The decorated macro function will
    behave as if the function code was embedded inline in the parent builder context
    after parameter substitution.

    Args:
        func: The Python function to enable as a builder macro. There are no
            requirements on the signature of the function, any calls to pulse
            builder methods will be added to builder context the wrapped function
            is called from.

    Returns:
        Callable: The wrapped ``func``.
    """
    func_name = getattr(func, "__name__", repr(func))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _builder = _active_builder()
        # activate the pulse builder before calling the function
        with build(backend=_builder.backend, name=func_name) as built:
            output = func(*args, **kwargs)

        _builder.call_subroutine(built)
        return output

    return wrapper


@deprecate_pulse_func
def measure(
    qubits: list[int] | int,
    registers: list[StorageLocation] | StorageLocation = None,
) -> list[StorageLocation] | StorageLocation:
    """Measure a qubit within the currently active builder context.

    At the pulse level a measurement is composed of both a stimulus pulse and
    an acquisition instruction which tells the systems measurement unit to
    acquire data and process it. We provide this measurement macro to automate
    the process for you, but if desired full control is still available with
    :func:`acquire` and :func:`play`.

    For now it is not possible to do much with the handle to ``reg`` but in the
    future we will support using this handle to a result register to build
    up ones program.

    .. note:: Requires the active builder context to have a backend set.

    Args:
        qubits: Physical qubit to measure.
        registers: Register to store result in. If not selected the current
            behavior is to return the :class:`MemorySlot` with the same
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
        backend=backend,
        qubit_mem_slots={qubit: register.index for qubit, register in zip(qubits, registers)},
    )

    # note this is not a subroutine.
    # just a macro to automate combination of stimulus and acquisition.
    # prepare unique reference name based on qubit and memory slot index.
    qubits_repr = "&".join(map(str, qubits))
    mslots_repr = "&".join((str(r.index) for r in registers))
    _active_builder().call_subroutine(measure_sched, name=f"measure_{qubits_repr}..{mslots_repr}")

    if len(qubits) == 1:
        return registers[0]
    else:
        return registers


@deprecate_pulse_func
def measure_all() -> list[chans.MemorySlot]:
    r"""Measure all qubits within the currently active builder context.

    A simple macro function to measure all of the qubits in the device at the
    same time. This is useful for handling device ``meas_map`` and single
    measurement constraints.

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
        backend=backend,
        qubit_mem_slots={qubit: qubit for qubit in qubits},
    )

    # note this is not a subroutine.
    # just a macro to automate combination of stimulus and acquisition.
    _active_builder().call_subroutine(measure_sched, name="measure_all")

    return registers


@deprecate_pulse_func
def delay_qubits(duration: int, *qubits: int):
    r"""Insert delays on all the :class:`channels.Channel`\s that correspond
    to the input ``qubits`` at the same time.

    .. note:: Requires the active builder context to have a backend set.

    Args:
        duration: Duration to delay for.
        qubits: Physical qubits to delay on. Delays will be inserted based on
            the channels returned by :func:`pulse.qubit_channels`.
    """
    qubit_chans = set(itertools.chain.from_iterable(qubit_channels(qubit) for qubit in qubits))
    with align_left():
        for chan in qubit_chans:
            delay(duration, chan)
