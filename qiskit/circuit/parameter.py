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
           bc = qc.assign_parameters({phi: 3.14})
           bc.measure_all()
           bc.draw('mpl')
    """

    __slots__ = ("_name", "_uuid", "_hash")

    # This `__init__` does not call the super init, because we can't construct the
    # `_parameter_symbols` dictionary we need to pass to it before we're entirely initialised
    # anyway, because `ParameterExpression` depends heavily on the structure of `Parameter`.

    def __init__(
        self, name: str, *, uuid: UUID | None = None
    ):  # pylint: disable=super-init-not-called
        """Create a new named :class:`Parameter`.

        Args:
            name: name of the ``Parameter``, used for visual representation. This can
                be any unicode string, e.g. "ϕ".
            uuid: For advanced usage only.  Override the UUID of this parameter, in order to make it
                compare equal to some other parameter object.  By default, two parameters with the
                same name do not compare equal to help catch shadowing bugs when two circuits
                containing the same named parameters are spurious combined.  Setting the ``uuid``
                field when creating two parameters to the same thing (along with the same name)
                allows them to be equal.  This is useful during serialization and deserialization.
        """
        self._name = name
        self._uuid = uuid4() if uuid is None else uuid
        if not _optionals.HAS_SYMENGINE:
            from sympy import Symbol

            symbol = Symbol(name)
        else:
            import symengine

            symbol = symengine.Symbol(name)

        self._symbol_expr = symbol
        self._parameter_keys = frozenset((self._hash_key(),))
        self._hash = hash((self._parameter_keys, self._symbol_expr))
        self._parameter_symbols = {self: symbol}
        self._name_map = None

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
        if _optionals.HAS_SYMENGINE:
            from symengine import sympify
        else:
            from sympy import sympify
        return ParameterExpression({}, sympify(value))

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

    def _hash_key(self):
        # `ParameterExpression` needs to be able to hash all its contained `Parameter` instances in
        # its hash as part of the equality comparison but has its own more complete symbolic
        # expression, so its full hash key is split into `(parameter_keys, symbolic_expression)`.
        # This method lets containing expressions get only the bits they need for equality checks in
        # the first value, without wasting time re-hashing individual Sympy/Symengine symbols.
        return (self._name, self._uuid)

    def __hash__(self):
        # This is precached for performance, since it's used a lot and we are immutable.
        return self._hash

    # We have to manually control the pickling so that the hash is computable before the unpickling
    # operation attempts to put this parameter into a hashmap.

    def __getstate__(self):
        return (self._name, self._uuid, self._symbol_expr)

    def __setstate__(self, state):
        self._name, self._uuid, self._symbol_expr = state
        self._parameter_keys = frozenset((self._hash_key(),))
        self._hash = hash((self._parameter_keys, self._symbol_expr))
        self._parameter_symbols = {self: self._symbol_expr}
        self._name_map = None
