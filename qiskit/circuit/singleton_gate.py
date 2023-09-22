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
"""
Singleton gate classes.
"""
import copy

from qiskit.circuit.gate import Gate
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.exceptions import CircuitError


SINGLETONGATE_ATTR_SET = frozenset(
    (
        "definition",
        "unit",
        "duration",
        "condition",
        "label",
        "_label",
        "_condition",
        "_duration",
        "_unit",
        "_definition",
        "_name",
        "_num_qubits",
        "_num_clbits",
        "_params",
        "params",
    )
)


class SingletonGate(Gate):
    """A base class to use for Gate objects that by default are singleton instances

    This class should be used for gate classes that have fixed definitions and
    do not contain any unique state. The canonical example of something like
    this is :class:`~.HGate` which has an immutable definition and any
    instance of :class:`~.HGate` is the same. Using singleton gates
    as a base class for these types of gate classes provides a large
    advantage in the memory footprint of multiple gates.

    The exception to be aware of with this class though are the :class:`~.Gate`
    attributes :attr:`.label`, :attr:`.condition`, :attr:`.duration`, and
    :attr:`.unit` which can be set differently for specific instances of gates.
    For :class:`~.SingletonGate` usage to be sound setting these attributes
    is not available and they can only be set at creation time. If any of these
    attributes are used, then instead of using a single shared global instance
    of the same gate a new separate instance will be created.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if args or (  # pylint: disable=too-many-boolean-expressions
            kwargs
            and (
                "label" in kwargs
                or "_condition" in kwargs
                or "duration" in kwargs
                or "unit" in kwargs
            )
        ):
            return super().__new__(cls)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, _condition=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._condition = _condition

    def __getnewargs_ex__(self):
        if not self.mutable:
            return ((), {})
        return ((self.label, self._condition, self.duration, self.unit), {})

    def c_if(self, classical, val):
        if not isinstance(classical, (ClassicalRegister, Clbit)):
            raise CircuitError("c_if must be used with a classical register or classical bit")
        if val < 0:
            raise CircuitError("condition value should be non-negative")
        if isinstance(classical, Clbit):
            # Casting the conditional value as Boolean when
            # the classical condition is on a classical bit.
            val = bool(val)
        instance = type(self)(
            label=self.label, _condition=(classical, val), duration=self.duration, unit=self.unit
        )
        return instance

    @property
    def mutable(self) -> bool:
        return self is not self._instance

    def to_mutable(self):
        if not self.mutable:
            instance = super().__new__(type(self))
            # Coming from a shared singleton none of the arguments to
            # __init__ can be set, so this is the correct behavior for
            # initializing a new mutable instance
            instance.__init__()
            return instance
        else:
            return copy.deepcopy(self)

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str):
        if self is self._instance:
            raise NotImplementedError(
                f"This gate class {type(self)} does not support manually setting a "
                "label on an instance. Instead you must set the label when instantiating a new object."
            )
        self._label = label

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, condition):
        if self is self._instance:
            raise NotImplementedError(
                f"This gate class {type(self)} does not support manually setting a "
                "condition on an instance. Instead you must set the label when instantiating a new "
                "object or via the .c_if() method"
            )
        self._condition = condition

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        if self is self._instance:
            raise NotImplementedError(
                f"This gate class {type(self)} does not support manually setting a "
                "duration on an instance. Instead you must set the duration when instantiating a "
                "new object."
            )
        self._duration = duration

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        if self is self._instance:
            raise NotImplementedError(
                f"This gate class {type(self)} does not support manually setting a "
                "unit on an instance. Instead you must set the unit when instantiating a "
                "new object."
            )
        self._unit = unit

    def __deepcopy__(self, _memo=None):
        if not self.mutable:
            return self
        else:
            return type(self)(
                label=self.label, _condition=self.condition, duration=self.duration, unit=self.unit
            )

    def __setattr__(self, name, value):
        if self.mutable:
            super().__setattr__(name, value)
        else:
            if name not in SINGLETONGATE_ATTR_SET:
                raise NotImplementedError(
                    "Setting custom attributes is not allowed on a singleton gate"
                )
            super().__setattr__(name, value)

    def copy(self, name=None):
        if not self.mutable and name is None:
            return self
        return super().copy(name=name)
