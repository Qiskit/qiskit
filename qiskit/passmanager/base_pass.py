# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Metaclass of Qiskit pass manager pass."""

from abc import abstractmethod
from collections.abc import Hashable
from inspect import signature
from typing import Any

from .propertyset import PropertySet


class MetaPass(type):
    """Metaclass for transpiler passes.

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
        arguments = [("class_.__name__", class_.__name__)]
        for name, value in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)


class GenericPass(metaclass=MetaPass):
    """Generic pass class with IR-agnostic minimal functionality."""

    def __init__(self):
        self.requires = []  # List of passes that requires
        self.preserves = []  # List of passes that preserves
        self.property_set = PropertySet()  # This pass's pointer to the pass manager's property set.
        self._hash = hash(None)

    def __hash__(self):
        return self._hash

    def __iter__(self):
        # To avoid normalization overhead.
        # A pass must be scheduled in iterable format in the pass manager,
        # which causes serious normalization overhead, since passes can be nested.
        # This iterator allows the pass manager to treat BasePass and FlowController equally.
        yield self

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            if len(other) == 1:
                # Reference might be List[BasePass] which was conventional representation
                # of a single base pass in the pass manager.
                other = next(iter(other))
        except TypeError:
            return False
        return hash(self) == hash(other)

    def name(self):
        """Return the name of the pass."""
        return self.__class__.__name__

    @abstractmethod
    def run(self, passmanager_ir: Any):
        """Run a pass on the pass manager IR. This is implemented by the pass developer.

        Args:
            passmanager_ir: the dag on which the pass is run.

        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        raise NotImplementedError
