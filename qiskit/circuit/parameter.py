# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Parameter Class for variable parameters.
"""

from uuid import uuid4

from qiskit.circuit.exceptions import CircuitError
from qiskit.utils import optionals as _optionals

from .parameterexpression import ParameterExpression


class Parameter(ParameterExpression):
    """Parameter Class for variable parameters.

    A parameter is a variable value that is not required to be fixed
    at circuit definition.

    Examples:

        Construct a variable-rotation X gate using circuit parameters.

        .. plot::
           :include-source:

           from qiskit.circuit import QuantumCircuit, Parameter

           # create the parameter
           phi = Parameter('phi')
           qc = QuantumCircuit(1)

           # parameterize the rotation
           qc.rx(phi, 0)
           qc.draw('mpl')

           # bind the parameters after circuit to create a bound circuit
           bc = qc.bind_parameters({phi: 3.14})
           bc.measure_all()
           bc.draw('mpl')
    """

    __slots__ = ("_name", "_uuid", "_hash")

    def __new__(cls, name, uuid=None):  # pylint: disable=unused-argument
        # Parameter relies on self._uuid being set prior to other attributes
        # (e.g. symbol_map) which may depend on self._uuid for Parameter's hash
        # or __eq__ functions.
        obj = object.__new__(cls)

        if uuid is None:
            obj._uuid = uuid4()
        else:
            obj._uuid = uuid

        obj._hash = hash(obj._uuid)
        return obj

    def __getnewargs__(self):
        # Unpickling won't in general call __init__ but will always call
        # __new__. Specify arguments to be passed to __new__ when unpickling.

        return (self.name, self._uuid)

    def __init__(self, name: str):
        """Create a new named :class:`Parameter`.

        Args:
            name: name of the ``Parameter``, used for visual representation. This can
                be any unicode string, e.g. "ϕ".
        """
        self._name = name
        if not _optionals.HAS_SYMENGINE:
            from sympy import Symbol

            symbol = Symbol(name)
        else:
            import symengine

            symbol = symengine.Symbol(name)
        super().__init__(symbol_map={self: symbol}, expr=symbol)

    def subs(self, parameter_map: dict, allow_unknown_parameters: bool = False):
        """Substitute self with the corresponding parameter in ``parameter_map``."""
        if self in parameter_map:
            return parameter_map[self]
        if allow_unknown_parameters:
            return self
        raise CircuitError(
            "Cannot bind Parameters ({}) not present in "
            "expression.".format([str(p) for p in parameter_map])
        )

    @property
    def name(self):
        """Returns the name of the :class:`Parameter`."""
        return self._name

    def __str__(self):
        return self.name

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self._uuid == other._uuid
        elif isinstance(other, ParameterExpression):
            return super().__eq__(other)
        else:
            return False

    def __hash__(self):
        return self._hash

    def __getstate__(self):
        return {"name": self._name}

    def __setstate__(self, state):
        self._name = state["name"]
        if not _optionals.HAS_SYMENGINE:
            from sympy import Symbol

            symbol = Symbol(self._name)
        else:
            import symengine

            symbol = symengine.Symbol(self._name)
        super().__init__(symbol_map={self: symbol}, expr=symbol)
