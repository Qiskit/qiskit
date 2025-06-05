# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Sampler Pub class
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Tuple, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

from .bindings_array import BindingsArray, BindingsArrayLike
from .shape import ShapedMixin

# Public API classes
__all__ = ["SamplerPubLike"]


class SamplerPub(ShapedMixin):
    """Primitive Unified Bloc (pub) for a sampler.

    This is the basic computational unit of a sampler. A sampler accepts one or more pubs when being
    run. If the pub's circuit is parametric then it can also specify many parameter value sets to
    bind the circuit against at execution time in an array-like format, and the results of the
    sampler are correspondingly shaped.

    If a ``shots`` value is provided to a sampler pub, then this value takes precedence
    over any value provided to the :meth:`~.Sampler.run` method.

    The value of a sampler pub's :attr:`~.shape` is typically taken from the ``parameter_values``
    shape at construction time. However, it can also be specified manually as a constructor
    argument, in which case it can be chosen to exceed the shape of the ``parameter_values`` so long
    as the two shape tuples are still broadcastable. This can be used to inject non-trivial axes
    into the execution. For example, if the provided bindings array object that specifies the
    parameter values has shape ``(2, 3)``, and if the shape tuple ``(8, 2, 3)`` is provided, then
    the shape of the sampler pub will be ``(8, 2, 3)`` because of the implicit leading ``(1,)`` on
    the binding arrays via standard broadcasting rules.

    Args:
        circuit: A quantum circuit.
        parameter_values: A bindings array of parameter values valid for the circuit. This is
            optional if the circuit is non-parametric.
        shots: The number of shots to sample for each parameter value. This value takes
            precedence over any value owned by or supplied to a sampler.
        shape: The shape of the pub, or ``None`` to infer it from the ``parameter_values``.
        validate: If ``True``, the input data is validated during initialization.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray | None = None,
        shots: int | None = None,
        shape: Tuple[int, ...] | None = None,
        validate: bool = True,
    ):
        super().__init__()
        self._circuit = circuit
        self._parameter_values = parameter_values or BindingsArray()
        self._shots = shots
        self._shape = shape or self._parameter_values.shape

        if validate:
            self.validate()

    @property
    def circuit(self) -> QuantumCircuit:
        """A quantum circuit."""
        return self._circuit

    @property
    def parameter_values(self) -> BindingsArray:
        """A bindings array."""
        return self._parameter_values

    @property
    def shots(self) -> int | None:
        """An specific number of shots to run with (optional).

        This value takes precedence over any value owed by or supplied to a sampler.
        """
        return self._shots

    @classmethod
    def coerce(cls, pub: SamplerPubLike, shots: int | None = None) -> SamplerPub:
        """Coerce a :class:`~.SamplerPubLike` object into a :class:`~.SamplerPub` instance.

        Args:
            pub: An object to coerce.
            shots: An optional default number of shots to use if not
                   already specified by the pub-like object.

        Returns:
            A coerced sampler pub.
        """
        # Validate shots kwarg if provided
        if shots is not None:
            if not isinstance(shots, Integral) or isinstance(shots, bool):
                raise TypeError("shots must be an integer")
            if shots <= 0:
                raise ValueError("shots must be positive")

        if isinstance(pub, SamplerPub):
            if pub.shots is None and shots is not None:
                return cls(
                    circuit=pub.circuit,
                    parameter_values=pub.parameter_values,
                    shots=shots,
                    shape=pub.shape,
                    validate=False,  # Assume Pub is already validated
                )
            return pub

        if isinstance(pub, QuantumCircuit):
            return cls(circuit=pub, shots=shots, validate=True)

        if isinstance(pub, CircuitInstruction):
            raise ValueError(
                f"An invalid Sampler pub-like was given ({type(pub)}). "
                "If you want to run a single circuit, "
                "you need to wrap it with `[]` like `sampler.run([circuit])` "
                "instead of `sampler.run(circuit)`."
            )

        if len(pub) not in [1, 2, 3, 4]:
            raise ValueError(
                f"The length of pub must be 1, 2, 3 or 4, but length {len(pub)} is given."
            )
        circuit = pub[0]

        if len(pub) > 1 and pub[1] is not None:
            values = pub[1]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)
        else:
            parameter_values = None

        if len(pub) > 2 and pub[2] is not None:
            shots = pub[2]

        return cls(
            circuit=circuit,
            parameter_values=parameter_values,
            shots=shots,
            shape=(pub[3] if len(pub) > 3 and pub[3] is not None else None),
            validate=True,
        )

    def validate(self):
        """Validate the pub."""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.parameter_values.validate()

        if self.shots is not None:
            if not isinstance(self.shots, Integral) or isinstance(self.shots, bool):
                raise TypeError("shots must be an integer")
            if self.shots <= 0:
                raise ValueError("shots must be positive")

        # Cross validate circuits and parameter values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            message = (
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )
            if num_parameters == 0:
                message += (
                    " Note that if you want to run a single pub, you need to wrap it with `[]` like "
                    "`sampler.run([(circuit, param_values)])` instead of "
                    "`sampler.run((circuit, param_values))`."
                )
            raise ValueError(message)

        # Validate shape consistency
        if not isinstance(self.shape, tuple) or not all(isinstance(idx, int) for idx in self.shape):
            raise ValueError(f"The shape must be a tuple of integers, found {self.shape} instead.")

        try:
            broadcast_shape = np.broadcast_shapes(self.shape, self.parameter_values.shape)
        except ValueError as exc:
            raise ValueError(
                f"The shape of the parameter values, {self.parameter_values.shape}, "
                f"is not compatible with the shape of the pub, {self.shape}."
            ) from exc

        if broadcast_shape != self.shape:
            raise ValueError(
                f"The shape of the parameter values, {self.parameter_values.shape}, "
                f"exceeds the shape of the pub, {self.shape}, on some axis."
            )


SamplerPubLike = Union[
    SamplerPub,
    QuantumCircuit,
    Tuple[QuantumCircuit],
    Tuple[QuantumCircuit, BindingsArrayLike],
    Tuple[QuantumCircuit, BindingsArrayLike, Union[Integral, None]],
    Tuple[QuantumCircuit, BindingsArrayLike, Union[Integral, None], Tuple[int, ...]],
]
"""Types coercible into a :class:`~.SamplerPub`.

This can either be a :class:`~.SamplerPub` itself, 
a single non-parametric :class:`~.QuantumCircuit`, or a tuple with entries
``(circuit, parameter_values, shots, shape)`` where the last three values
can optionally be absent or set to ``None``.
The following formats are valid:

 * :class:`~.QuantumCircuit`
 * (:class:`~.QuantumCircuit`,)
* (:class:`~.QuantumCircuit`, :class:`~.BindingsArrayLike`\\)
 * (:class:`~.QuantumCircuit`, :class:`~.BindingsArrayLike`, 
   :class:`int`\\)
 * (:class:`~.QuantumCircuit`, :class:`~.BindingsArrayLike`, 
   :class:`int`, :class:`tuple[int, ...]`\\)

If ``parameter_values`` are not provided, the circuit must have no 
:attr:`~.QuantumCircuit.parameters`.
If ``shots`` is not provided, the sampler will supply a value.
If ``shape`` is not provided, it will be inferred from the ``parameter_values``. 
"""
