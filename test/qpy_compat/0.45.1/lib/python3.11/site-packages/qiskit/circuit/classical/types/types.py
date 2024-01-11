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

"""Type-system definition for the expression tree."""

# Given the nature of the tree representation and that there are helper functions associated with
# many of the classes whose arguments naturally share names with themselves, it's inconvenient to
# use synonyms everywhere.  This goes for the builtin 'type' as well.
# pylint: disable=redefined-builtin,redefined-outer-name

from __future__ import annotations

__all__ = [
    "Type",
    "Bool",
    "Uint",
]

import typing


class _Singleton(type):
    """Metaclass to make the child, which should take zero initialisation arguments, a singleton
    object."""

    def _get_singleton_instance(cls):
        return cls._INSTANCE

    @classmethod
    def __prepare__(mcs, name, bases):  # pylint: disable=unused-argument
        return {"__new__": mcs._get_singleton_instance}

    @staticmethod
    def __new__(cls, name, bases, namespace):
        out = super().__new__(cls, name, bases, namespace)
        out._INSTANCE = object.__new__(out)  # pylint: disable=invalid-name
        return out


class Type:
    """Root base class of all nodes in the type tree.  The base case should never be instantiated
    directly.

    This must not be subclassed by users; subclasses form the internal data of the representation of
    expressions, and it does not make sense to add more outside of Qiskit library code."""

    __slots__ = ()

    @property
    def kind(self):
        """Get the kind of this type.  This is exactly equal to the Python type object that defines
        this type, that is ``t.kind is type(t)``, but is exposed like this to make it clear that
        this a hashable enum-like discriminator you can rely on."""
        return self.__class__

    # Enforcement of immutability.  The constructor methods need to manually skip this.

    def __setattr__(self, _key, _value):
        raise AttributeError(f"'{self.kind.__name__}' instances are immutable")

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __setstate__(self, state):
        _dict, slots = state
        for slot, value in slots.items():
            # We need to overcome the type's enforcement of immutability post initialisation.
            super().__setattr__(slot, value)


@typing.final
class Bool(Type, metaclass=_Singleton):
    """The Boolean type.  This has exactly two values: ``True`` and ``False``."""

    __slots__ = ()

    def __repr__(self):
        return "Bool()"

    def __eq__(self, other):
        return isinstance(other, Bool)


@typing.final
class Uint(Type):
    """An unsigned integer of fixed bit width."""

    __slots__ = ("width",)

    def __init__(self, width: int):
        if isinstance(width, int) and width <= 0:
            raise ValueError("uint width must be greater than zero")
        super(Type, self).__setattr__("width", width)

    def __repr__(self):
        return f"Uint({self.width})"

    def __eq__(self, other):
        return isinstance(other, Uint) and self.width == other.width
