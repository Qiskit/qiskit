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

"""Pass manager for pulse schedules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiskit.passmanager import BasePassManager
from qiskit.pulse.compiler.converters import schedule_to_ir, ir_to_schedule
from qiskit.pulse.schedule import ScheduleBlock

# TODO replace with actual type
PulseIR = object


class PulsePassManager(BasePassManager):
    """A pass manager for compiling Qiskit ScheduleBlock programs."""

    def _passmanager_frontend(
        self,
        input_program: ScheduleBlock,
        **kwargs,
    ) -> PulseIR:
        return schedule_to_ir(input_program)

    def _passmanager_backend(
        self,
        passmanager_ir: PulseIR,
        in_program: ScheduleBlock,
        **kwargs,
    ) -> Any:
        output = kwargs.get("output", "schedule_block")

        if output == "pulse_ir":
            return passmanager_ir
        if output == "schedule_block":
            return ir_to_schedule(passmanager_ir)

        raise ValueError(f"Specified target format '{output}' is not supported.")

    # pylint: disable=arguments-differ
    def run(
        self,
        schedules: ScheduleBlock | list[ScheduleBlock],
        output: str = "schedule_block",
        callback: Callable | None = None,
        num_processes: int | None = None,
    ) -> Any:
        """Run all the passes on the input pulse schedules.

        Args:
            schedules: Input pulse programs to transform via all the registered passes.
                When a list of schedules are passed, the transform is performed in parallel
                for each input schedule with multiprocessing.
            output: Format of the output program::

                    schedule_block: Return in :class:`.ScheduleBlock` format.
                    pulse_ir: Return in :class:`.PulseIR` format.

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
            The transformed program(s) in specified program format.
        """
        return super().run(
            in_programs=schedules,
            callback=callback,
            num_processes=num_processes,
            output=output,
        )
