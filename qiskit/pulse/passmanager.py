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

"""Passmanager module for pulse schedules."""

import contextlib
import logging
import time
from typing import Union, List

from qiskit import pulse
from qiskit.pulse.basepasses import BasePass
from qiskit.pulse import exceptions
from qiskit.pulse.states import State

logger = logging.getLogger(__name__)


class PassManager:
    """Manager for a set of Passes and their scheduling during compilation."""

    def __init__(
            self,
            passes: Union[BasePass, List[BasePass]] = None,
            name="",
    ):
        """Initialize an empty `PassManager` object (with no passes scheduled).

        Args:
            passes: A list of passes to add to passmanger.
            name: Name of this passmanager.
        """
        # the pass manager's schedule of passes.
        # Populated via PassManager.append().

        self._passes = []
        if passes is not None:
            try:
                iter(passes)
            except TypeError:
                passes = [passes]

            for pass_ in passes:
                self.append(pass_)

        self.name = name

        self.state = State()
        # passes already run that have not been invalidated
        self.valid_passes = set()

    @property
    def passes(self) -> List[BasePass]:
        """Return this PassManager's passes."""
        return self._passes

    def append(
            self,
            pass_: BasePass,
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            pass_: A pass to be added to the passmanager
        Raises:
            CompilerError: if a pass in passes is not a proper pass.

        """
        self._passes.append(pass_)

    def replace(
            self,
            index: int,
            pass_: BasePass,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            pass_: The pass to replace with.

        Raises:
            CompilerError: if a pass in passes is not a proper pass.
        """
        try:
            self._passes[index] = pass_
        except IndexError:
            raise exceptions.CompilerError(
                'Index to replace %s does not exists' % index)

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._passes)

    def __getitem__(self, index):
        new_passmanager = PassManager()
        _passes = self._passes[index]
        new_passmanager._passes = _passes
        return new_passmanager

    def __add__(self, other):
        if isinstance(other, PassManager):
            new_passmanager = PassManager()
            new_passmanager._passes = self._passes + other._passes
            return new_passmanager
        else:
            try:
                new_passmanager = PassManager()
                new_passmanager._passes += self._passes
                new_passmanager.append(other)
                return new_passmanager
            except exceptions.CompilerError:
                raise TypeError('unsupported operand type + for {} and {}'.format(
                    self.__class__, other.__class__))

    def run(
            self,
            program: pulse.Program,
    ) -> pulse.Program:
        """Run all the passes on the supplied pulse program.

        Args:
            program: Program to compile.

        Returns:
            The transformed program.
        """
        for pass_ in self._passes:
            program = self._do_pass(pass_, program)

        return program

    def _do_pass(
        self,
        pass_: BasePass,
        program: pulse.Program,
    ) -> pulse.Program:
        """Do a pass and its "requires"."""
        # First, do the requires of pass_
        for required_pass in pass_.requires:
            program = self._do_pass(required_pass, program)

        # Run the pass itself, if not already run
        if pass_ not in self.valid_passes:
            program = self._run_this_pass(pass_, program)

            # update the valid_passes property
            self._update_valid_passes(pass_)

        return program

    def _run_this_pass(self, pass_, program):
        pass_.state = self.state
        if pass_.is_analysis_pass:
            with self._log_pass(pass_.name):
                pass_.run(program)
                self.state = pass_.state

        if pass_.is_validation_pass:
            # Measure time if we have a callback or logging set
            with self._log_pass(pass_.name):
                pass_.run(program)
        else:
            # Measure time if we have a callback or logging set
            with self._log_pass(pass_.name):
                new_program = pass_.run(program)
                self.state = pass_.state
                self.state.program = new_program

            if not isinstance(new_program, pulse.Program):
                raise exceptions.CompilerError(
                    "Transformation passes should return a transformed Program."
                    "The pass {} is returning a {}".format(
                        type(pass_).__name__, type(new_program)))
            program = new_program
        return program

    @contextlib.contextmanager
    def _log_pass(self, name):
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            log_msg = "Pass: %s - %.5f (ms)" % (
                name, (end_time - start_time) * 1000)
            logger.info(log_msg)

    def _update_valid_passes(self, pass_):
        self.valid_passes.add(pass_)
        if not pass_.is_analysis_pass:  # Analysis passes preserve all
            self.valid_passes.intersection_update(set(pass_.preserves))

    def run_schedules(
        self,
        schedules: Union[pulse.Schedule, List[pulse.Schedule]]
    ) -> List[pulse.Schedule]:
        """Run all the passes on the supplied ``schedules``.

        Args:
            schedules: List of schedules to run the passes on.

        Returns:
            List of modified pulse schedules
        """
        try:
            iter(schedules)
        except TypeError:
            schedules = [schedules]

        return self.run(pulse.Program(schedules)).schedules
