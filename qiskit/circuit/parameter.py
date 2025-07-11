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

from __future__ import annotations

from uuid import uuid4, UUID

import qiskit._accelerate.circuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.optionals import HAS_SYMPY

from .parameterexpression import ParameterExpression

ParameterExpressionBase = qiskit._accelerate.circuit.ParameterExpression


class Parameter(ParameterExpression):
    """A compile-time symbolic parameter.

    The value of a :class:`Parameter` must be entirely determined before a circuit begins execution.
    Typically this will mean that you should supply values for all :class:`Parameter`\\ s in a
    circuit using :meth:`.QuantumCircuit.assign_parameters`, though certain hardware vendors may
    allow you to give them a circuit in terms of these parameters, provided you also pass the values
    separately.

    This is the atom of :class:`.ParameterExpression`, and is itself an expression.  The numeric
    value of a parameter need not be fixed while the circuit is being defined.

    Examples:

        Construct a variable-rotation X gate using circuit parameters.

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit.circuit import QuantumCircuit, Parameter

           # create the parameter
           phi = Parameter('phi')
           qc = QuantumCircuit(1)

           # parameterize the rotation
           qc.rx(phi, 0)
           qc.draw('mpl')

           # bind the parameters after circuit to create a bound circuit
           bc = qc.assign_parameters({phi: 3.14})
           bc.measure_all()
           bc.draw('mpl')
    """

    __slots__ = "_hash"

    def __new__(
        cls, name: str | None = None, uuid: UUID | None = None
    ):  # pylint: disable=super-init-not-called
        """
        Args:
            name: name of the ``Parameter``, used for visual representation. This can
                be any Unicode string, e.g. "ϕ".
            uuid: For advanced usage only.  Override the UUID of this parameter, in order to make it
                compare equal to some other parameter object.  By default, two parameters with the
                same name do not compare equal to help catch shadowing bugs when two circuits
                containing the same named parameters are spurious combined.  Setting the ``uuid``
                field when creating two parameters to the same thing (along with the same name)
                allows them to be equal.  This is useful during serialization and deserialization.
        """
        if uuid != None:
            uuid = int(uuid)

        if name == None:
            self = super().__new__(cls, symbol_map=None, expr=None)
        else:
            self = super().__new__(
                cls, symbol_map=None, expr=ParameterExpressionBase.Symbol(name, uuid)
            )

        self._hash = None
        self._parameter_symbols = {self}
        return self

    def assign(self, parameter, value):
        if parameter != self:
            # Corresponds to superclass calls to `subs` and `bind` that would implicitly set
            # `allow_unknown_parameters=False`.
            raise CircuitError(
                f"Cannot bind Parameters ({[str(parameter)]}) not present in expression."
            )
        if isinstance(value, ParameterExpression):
            # This is the `super().subs` case.
            return value
        # This is the `super().bind` case, where we're required to return a `ParameterExpression`,
        # so we need to lift the given value to a symbolic expression.
        return ParameterExpressionBase.Value(value)

    def subs(self, parameter_map: dict, allow_unknown_parameters: bool = False):
        """Substitute self with the corresponding parameter in ``parameter_map``."""
        if self in parameter_map:
            return parameter_map[self]
        if allow_unknown_parameters:
            return self
        raise CircuitError(
            f"Cannot bind Parameters ({[str(p) for p in parameter_map]}) not present in "
            "expression."
        )

    @property
    def uuid(self) -> UUID:
        """Returns the :class:`~uuid.UUID` of the :class:`Parameter`.

        In advanced use cases, this property can be passed to the
        :class:`Parameter` constructor to produce an instance that compares
        equal to another instance.
        """
        return UUID(int=self.get_uuid())

    def _hash_key(self):
        # `ParameterExpression` needs to be able to hash all its contained `Parameter` instances in
        # its hash as part of the equality comparison but has its own more complete symbolic
        # expression, so its full hash key is split into `(parameter_keys, symbolic_expression)`.
        # This method lets containing expressions get only the bits they need for equality checks in
        # the first value, without wasting time re-hashing individual symbols.
        return super().__hash__()

    def __hash__(self):
        # This is precached for performance, since it's used a lot and we are immutable.
        if self._hash == None:
            self._hash = super().__hash__()
        return self._hash

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @property
    def _uuid(self) -> UUID:
        """Returns UUID property of the :class:`Parameter`"""
        return UUID(int=self.get_uuid())

    @HAS_SYMPY.require_in_call
    def sympify(self):
        import sympy

        return sympy.Symbol(self.name)
