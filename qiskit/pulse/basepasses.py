# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base pulse compiler passes."""

from abc import abstractmethod
from collections.abc import Hashable
from inspect import signature
from typing import Any

from qiskit import pulse
from qiskit.pulse.states import Analysis, State


class MetaPass(type):
    """Metaclass for pulse compilation passes.

    Enforces the creation of some fields in the pass while allowing passes to
    override ``__init__``.
    """

    def __call__(cls, *args, **kwargs):
        pass_instance = type.__call__(cls, *args, **kwargs)
        pass_instance._hash = hash(MetaPass._freeze_init_parameters(cls, args, kwargs))
        return pass_instance

    @staticmethod
    def _freeze_init_parameters(class_, args, kwargs):
        self_guard = object()
        init_signature = signature(class_.__init__)
        bound_signature = init_signature.bind(self_guard, *args, **kwargs)
        arguments = [('class_.__name__', class_.__name__)]
        for name, value in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)


class BasePass(metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        self.requires = []  # List of passes that requires
        self.preserves = []  # List of passes that preserves
        self.state = State()
        self._hash = None

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def name(self):
        """Return the name of the pass."""
        return self.__class__.__name__

    @property
    def analysis(self) -> Analysis:
        """Return the current analysis information available to this pass."""
        return self.state.analysis

    @abstractmethod
    def run(self, program: pulse.Program) -> pulse.Program:
        """Run a pass on the pulse program. This is implemented by the pass developer.

        Args:
            program: The program on which the pass is run.
        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        raise NotImplementedError

    @property
    def is_transformation_pass(self):
        """Check if the pass is a transformation pass."""
        return isinstance(self, TransformationPass)

    @property
    def is_analysis_pass(self):
        """Check if the pass is an analysis pass."""
        return isinstance(self, AnalysisPass)

    @property
    def is_lowering_pass(self):
        """Check if the pass is a lowering pass."""
        return isinstance(self, LoweringPass)

    @property
    def is_validation_pass(self):
        """Check if the pass is a validation pass."""
        return isinstance(self, ValidationPass)


class AnalysisPass(BasePass):  # pylint: disable=abstract-method
    """An analysis pass: changes the property set and not the pulse schedule."""

    def run(self, program: pulse.Program) -> pulse.Program:
        """Run the ``analyze`` method."""
        self.analyze(program)
        return program

    @abstractmethod
    def analyze(self, program: pulse.Program):
        """Perform program analysis."""
        raise NotImplementedError


class TransformationPass(BasePass):  # pylint: disable=abstract-method
    """An analysis pass: changes the pulse schedule and not the property set."""

    def run(self, program: pulse.Program) -> pulse.Program:
        """Run the ``analyze`` method."""
        return self.transform(program)

    @abstractmethod
    def transform(self, program: pulse.Program) -> pulse.Program:
        """Perform program analysis."""
        raise NotImplementedError


class LoweringPass(BasePass):
    """A lowering pass: Emits a ``lowering`` field in the compiler ``State`` it
    should not modify the input program."""

    def run(self, program: pulse.Program) -> pulse.Program:
        """Run a pass on the pulse program. This is implemented by the pass developer.

        Args:
            program: The program on which the pass is run.
        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        self.state.lowered = self.lower(program)
        return program

    @abstractmethod
    def lower(self, program: pulse.Program) -> Any:
        """Lower the pulse program returning the lowered program."""


class ValidationPass(BasePass):  # pylint: disable=abstract-method
    """Validates a property of the pulse program. Does not modify the attributes
    or program."""

    @abstractmethod
    def run(self, program: pulse.Program) -> pulse.Program:
        """Validate a property of the supplied program.

        Raises:
            CompilerError: If the program fails validation.
        """
        self.validate(program)
        return program

    @abstractmethod
    def validate(self, program: pulse.Program) -> pulse.Program:
        """Validate the program. Raise :class:`qiskit.pulse.exceptions.CompilerError`
        if validation fails."""
        raise NotImplementedError
