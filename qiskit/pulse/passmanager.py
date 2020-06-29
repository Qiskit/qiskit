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
from typing import Union, List

from qiskit import pulse
from qiskit.pulse.basepasses import BasePass
from qiskit.pulse.exceptions import CompilerError


class PassManager:
    """Manager for a set of Passes and their scheduling during compilation."""

    def __init__(
            self,
            passes: Union[BasePass, List[BasePass]] = None,
    ):
        """Initialize an empty `PassManager` object (with no passes scheduled).

        Args:
            passes: A list of passes to add to passmanger.
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

        self.property_set = None

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
            raise CompilerError('Index to replace %s does not exists' % index)

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
            except CompilerError:
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
            pass_.property_set = self.property_set
            program = pass_.run(program)

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
