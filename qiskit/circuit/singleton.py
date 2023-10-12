# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Definition of the base singleton apparatus."""

import operator

from .instruction import Instruction
from .gate import Gate

# Summary
# =======
#
# The machinery in this file is for defining subclasses of `Instruction` and `Gate` that
# preferentially return a shared immutable singleton instance when instantiated.  Taking the example
# of `XGate`, the final user-facing result is that:
#
# * There is a regular class called `XGate`, which derives from `Gate`.
#
# * Doing something like `XGate(label="my_gate")` produces an object whose type is exactly `XGate`,
#   and all the mutability works completely as expected; all the methods resolve to exactly those
#   defined by `XGate`, `Gate`, or parents.
#
# * Doing `XGate()` produces a singleton object whose type is a synthetic `_SingletonXGate` class,
#   which derives `XGate` but overrides `__setattr__` to make itself immutable.  The object itself
#   has precisely the same instance attributes as `XGate()` would have if there was no singleton
#   handling.  This object will return itself under copy, deepcopy and roundtrip through pickle.
#
# The same is true for, for example, `Measure`, except that it's a subclass of `Instruction` only,
# and not `Gate`.
#
# From a library-author perspective, all that's needed to enhance a `Gate` or `Instruction` with
# this behaviour is to inherit from `SingletonGate` (`SingletonInstruction`) instead of `Gate`
# (`Instruction`), and for the `__init__` method to have defaults for all of its arguments (these
# will be the state of the singleton instance).  For example:
#
#     class XGate(SingletonGate):
#         def __init__(self, label=None):
#             super().__init__("x", 1, [], label=label)
#
#
# Implementation
# ==============
#
# There are several moving parts to tackle here.  The behaviour of having `XGate()` return some
# singleton object that is an (inexact) instance of `XGate` but _without_ calling `__init__`
# requires us to override `type.__call__`.  This means that `XGate` must have a metaclass that
# defines `__call__` to return the singleton instance.
#
# Next, we need to ensure that there _is_ a singleton instance for `XGate()` to return.  This can be
# done dynamically on each call (i.e. check if the instance exists and create it if not), but since
# we also want that instance to be very special, it's easier to hook in and create it during the
# definition of the `XGate` type object.  This also has the advantage that we do not need to make
# the singleton object pickleable; we only need to specify where to retrieve it from during the
# unpickle, because the creation of the base type object will recreate the singleton.
#
# We want the singleton instance to:
#
# * be immutable; it should reject all attempts to mutate itself.
# * have exactly the same state as an `XGate()` would have had if there was no singleton handling.
#
# We do this in a three-step procedure:
#
# 1. Before creating any singletons, we separately define the overrides needed to make an
#    `Instruction` and a `Gate` immutable.  This is `_SingletonInstructionOverrides` and
#    `_SingletonGateOverrides`.
#
# 2. While we are creating the `XGate` type object, we dynamically _also_ create a subclass of it
#    that has the immutable overrides in its method-resolution order in the correct place. These
#    override the standard methods / properties that are defined on the mutable gate (we do not
#    attempt to override any cases where the type object we are creating has extra inplace methods).
#
# 3. We can't instantiate this new subclass, because when it calls `XGate.__init__`, it will attempt
#    to set some attributes, and these will be rejected by immutability.  Instead, we first create a
#    completely regular `XGate` instance, and then we dynamically change its type to the singleton
#    class, freezing it.
#
# We could do this entirely within the metaclass machinery, but that would require `XGate` to be
# defined as something like
#
#   class XGate(Gate, metaclass=_SingletonMeta, overrides=_SingletonGateOverrides): ...
#
# which is super inconvenient (or we'd have to have `_SingletonMeta` do a bunch of fragile
# introspection).  Instead, we use the `abc.ABC`/`abc.ABCMeta` pattern of defining a concrete middle
# class (`SingletonGate` in the `XGate` case) that sets the metaclass, selects the overrides to be
# applied, and has an `__init_subclass__` that applies the singleton-subclass-creation steps above.
# The overrides are in separate classes so that _mutable_ `XGate` instances do not have them in
# their own MROs; doing this is easier to implement, but requires all the setters and checkers to
# dance around at runtime trying to validate whether mutating the instance is allowed.


def _impl_new(cls, *_args, **_kwargs):
    # __new__ for the singleton instances.
    raise TypeError(f"cannot create '{cls.__name__}' instances")


def _impl_init_subclass(base, overrides):
    # __init_subclass__ for the classes that make their children singletons (e.g. `SingletonGate`)

    def __init_subclass__(cls, *, create_singleton=True, **kwargs):
        super(base, cls).__init_subclass__(**kwargs)
        if not create_singleton:
            return

        # We need to make a new type object that pokes in the overrides into the correct
        # place in the method-resolution order.
        singleton_class = _SingletonMeta.__new__(
            _SingletonMeta,
            f"_Singleton{cls.__name__}",
            (cls, overrides),
            # This is a dynamically generated class so it's got no module.  The slot layout of the
            # singleton class needs to match any layout in the base.
            {"__module__": None, "__slots__": (), "__new__": _impl_new, "_base_class": cls},
            create_singleton=False,
        )

        # Make a mutable instance, fully instantiate all lazy properties, then freeze it.
        cls._singleton_instance = cls(_force_mutable=True)
        cls._singleton_instance._define()
        cls._singleton_instance.__class__ = singleton_class

    return classmethod(__init_subclass__)


class _SingletonMeta(type(Instruction)):
    # The inheritance above is to ensure metaclass compatibility with `Instruction`, though pylint
    # doesn't understand that this is a subclass of `type`, so uses metaclass `self` conventions.

    # pylint: disable=bad-classmethod-argument,no-self-argument

    __slots__ = ()

    def __new__(mcs, name, bases, namespace, *, overrides=None, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if overrides is not None:
            # The `__init_subclass__` logic is shared between `SingletonInstruction` and
            # `SingletonGate`, but we can't do `__init_subclass__ = _impl_init_subclass(...)`
            # inside the class definition bodies because it couldn't make `super()` calls.
            cls.__init_subclass__ = _impl_init_subclass(cls, overrides)
        return cls

    def __call__(cls, *args, _force_mutable=False, **kwargs):
        if not _force_mutable and not args and not kwargs:
            # This class attribute is created by the singleton-creation base classes'
            # `__init_subclass__` methods; see `_impl_init_subclass`.
            return cls._singleton_instance
        return super().__call__(*args, **kwargs)


class _SingletonInstructionOverrides(Instruction):
    """Overrides for all the mutable methods and properties of `Instruction` to make it
    immutable."""

    __slots__ = ()

    def c_if(self, classical, val):
        return self.to_mutable().c_if(classical, val)

    @property
    def base_class(self):
        # `type(self)` will actually be the dynamic `_SingletonXGate` (e.g.) created by
        # `SingletonGate.__init_subclass__` during the instantiation of `XGate`, since this class
        # is never the concrete type of a class.
        return type(self)._base_class

    @property
    def mutable(self):
        return False

    def to_mutable(self):
        return self.base_class(_force_mutable=True)

    def copy(self, name=None):
        if name is None:
            return self
        out = self.to_mutable()
        out.name = name
        return out

    def __setattr__(self, key, value):
        raise NotImplementedError(
            f"This '{self.base_class.__name__}' object is immutable."
            " You can get a mutable version by calling 'to_mutable()'."
        )

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo=None):
        return self

    def __reduce__(self):
        return (operator.attrgetter("_singleton_instance"), (self.base_class,))


class SingletonInstruction(
    Instruction, metaclass=_SingletonMeta, overrides=_SingletonInstructionOverrides
):
    """A base class to use for :class:`~.circuit.Instruction` objects that by default are singleton
    instances.

    This class should be used for instruction classes that have fixed definitions and do not contain
    any unique state. The canonical example of something like this is :class:`.Measure` which has an
    immutable definition and any instance of :class:`.Measure` is the same. Using singleton
    instructions as a base class for these types of gate classes provides a large advantage in the
    memory footprint of multiple instructions.

    The exception to be aware of with this class though are the :class:`~.circuit.Instruction`
    attributes :attr:`.label`, :attr:`.condition`, :attr:`.duration`, and :attr:`.unit` which can be
    set differently for specific instances of gates.  For :class:`~.SingletonGate` usage to be sound
    setting these attributes is not available and they can only be set at creation time, or on an
    object that has been specifically made mutable using :meth:`to_mutable`. If any of these
    attributes are used during creation, then instead of using a single shared global instance of
    the same gate a new separate instance will be created."""

    __slots__ = ()


class _SingletonGateOverrides(_SingletonInstructionOverrides, Gate):
    """Overrides for all the mutable methods and properties of `Gate` to make it immutable.

    This class just exists for the principle; there's no additional overrides required compared
    to :class:`~.circuit.Instruction`."""

    __slots__ = ()


class SingletonGate(Gate, metaclass=_SingletonMeta, overrides=_SingletonGateOverrides):
    """A base class to use for :class:`.Gate` objects that by default are singleton instances.

    This class should be used for gate classes that have fixed definitions and do not contain any
    unique state. The canonical example of something like this is :class:`~.HGate` which has an
    immutable definition and any instance of :class:`~.HGate` is the same. Using singleton gates as
    a base class for these types of gate classes provides a large advantage in the memory footprint
    of multiple gates.

    This class is very similar to :class:`SingletonInstruction`, except implies unitary
    :class:`.Gate` semantics as well.  The same caveats around setting attributes in that class
    apply here as well."""

    __slots__ = ()
