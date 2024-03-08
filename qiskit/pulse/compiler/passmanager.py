# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager for pulse programs."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Any

from qiskit.passmanager import BasePassManager
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse.schedule import ScheduleBlock


PulseProgramT = Any
"""Type alias representing whatever pulse programs."""


class BasePulsePassManager(BasePassManager, ABC):
    """A pulse compiler base class.

    The pulse pass manager takes :class:`.ScheduleBlock` as an input and schedules
    instructions within the block context according to the alignment specification
    as a part of lowering.

    Since pulse sequence is a lower-end representation of quantum programs,
    the compiler may require detailed description of the
    target control electronics to generate functional output programs.
    Qiskit :class:`~qiskit.provider.Backend` object may inject vendor specific
    plugin passes that may consume such hardware specification
    that the vendor may also provide as a custom :class:`.Target` model.

    Qiskit pulse pass manager relies on the :class:`.IrBlock` as an intermediate
    representation on which all compiler passes are applied.
    A developer must define a subclass of the ``BasePulsePassManager`` for
    each desired compiler backend representation along with the logic to
    interface with our intermediate representation.
    """

    def _passmanager_frontend(
        self,
        input_program: ScheduleBlock,
        **kwargs,
    ) -> SequenceIR:

        def _wrap_recursive(_prog):
            _ret = SequenceIR(alignment=_prog.alignment_context)
            for _elm in _prog.blocks:
                if isinstance(_elm, ScheduleBlock):
                    _ret.append(_wrap_recursive(_elm))
                else:
                    _ret.append(_elm)
            return _ret

        return _wrap_recursive(input_program)

    # pylint: disable=arguments-differ
    def run(
        self,
        pulse_programs: ScheduleBlock | list[ScheduleBlock],
        callback: Callable | None = None,
        num_processes: int | None = None,
    ) -> PulseProgramT | list[PulseProgramT]:
        """Run all the passes on the input pulse programs.

        Args:
            pulse_programs: Input pulse programs to transform via all the registered passes.
                When a list of schedules are passed, the transform is performed in parallel
                for each input schedule with multiprocessing.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    task (GenericPass): the pass being run
                    passmanager_ir (Any): depending on pass manager subclass
                    property_set (PropertySet): the property set
                    running_time (float): the time to execute the pass
                    count (int): the index for the pass execution

                The exact arguments pass expose the internals of the pass
                manager and are subject to change as the pass manager internals
                change. If you intend to reuse a callback function over
                multiple releases be sure to check that the arguments being
                passed are the same.

                To use the callback feature you define a function that will
                take in kwargs dict and access the variables. For example::

                    def callback_func(**kwargs):
                        task = kwargs['task']
                        passmanager_ir = kwargs['passmanager_ir']
                        property_set = kwargs['property_set']
                        running_time = kwargs['running_time']
                        count = kwargs['count']
                        ...
            num_processes: The maximum number of parallel processes to launch if parallel
                execution is enabled. This argument overrides ``num_processes`` in the user
                configuration file, and the ``QISKIT_NUM_PROCS`` environment variable. If set
                to ``None`` the system default or local user configuration will be used.

        Returns:
            The transformed program(s).
        """
        return super().run(
            in_programs=pulse_programs,
            callback=callback,
            num_processes=num_processes,
        )


class BlockToIrCompiler(BasePulsePassManager):
    """A specialized pulse compiler for IR backend.

    This compiler outputs :class:`.SequenceIR`, which is an intermediate representation
    of the pulse program in Qiskit.
    """

    def _passmanager_backend(
        self,
        passmanager_ir: SequenceIR,
        in_program: ScheduleBlock,
        **kwargs,
    ) -> SequenceIR:
        return passmanager_ir


class BlockTranspiler(BasePulsePassManager):
    """A specialized pulse compiler for ScheduleBlock backend.

    This compiler (transpiler) outputs :class:`.ScheduleBlock`, which
    is an identical data format to the input program.
    """

    def _passmanager_backend(
        self,
        passmanager_ir: SequenceIR,
        in_program: ScheduleBlock,
        **kwargs,
    ) -> ScheduleBlock:

        def _unwrap_recursive(_prog):
            _ret = ScheduleBlock(alignment_context=_prog.alignment)
            for _elm in _prog.elements():
                if isinstance(_elm, SequenceIR):
                    _ret.append(_unwrap_recursive(_elm), inplace=True)
                else:
                    _ret.append(_elm, inplace=True)
            return _ret

        out_block = _unwrap_recursive(passmanager_ir)
        out_block.metadata = in_program.metadata.copy()

        return out_block
