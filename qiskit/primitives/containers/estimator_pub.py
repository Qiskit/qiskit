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
Estimator Pub class
"""

from __future__ import annotations

from numbers import Real
from collections.abc import Mapping
from typing import Tuple, Union

import numpy as np

from qiskit import QuantumCircuit

from .bindings_array import BindingsArray, BindingsArrayLike
from .observables_array import ObservablesArray, ObservablesArrayLike
from .shape import ShapedMixin

# Public API classes
__all__ = ["EstimatorPubLike"]


class EstimatorPub(ShapedMixin):
    """Primitive Unified Bloc (pub) for an estimator.

    This is the basic computational unit of an estimator. An estimator accepts one or more pubs when
    being run. If the pub's circuit is parametric then it can also specify many parameter value sets
    and observables to bind the circuit against at execution time in array-like formats, and the
    results of the estimator are correspondingly shaped.

    If a ``precision`` value is provided to an estimator pub, then this value takes precedence over
    any value provided to the :meth:`~.Estimator.run` method.

    The value of an estimator pub's :attr:`~.shape` is typically taken as the shape of the
    ``observables`` array broadcasted with the shape of the ``parameter_values`` shape at
    construction time. However, it can also be specified manually as a constructor argument, in
    which case it can be chosen to exceed the shape of either or both of these arrays so long as all
    shapes are still broadcastable. This can be used to inject non-trivial axes into the execution.
    For example, if the provided bindings array object that specifies the parameter values has shape
    ``(2, 1)``, if the observables have shape ``(1, 2, 3)`` and if the shape tuple ``(8, 2, 3)`` is
    provided, then the shape of the estimator pub will be ``(8, 2, 3)`` via standard broadcasting
    rules.

    Args:
        circuit: A quantum circuit.
        observables: An observables array.
        parameter_values: A bindings array, if the circuit is parametric.
        precision: An optional target precision for expectation value estimates.
        shape: The shape of the pub, or ``None`` to infer by taking the broadcasted shape between
            ``observables`` and ``parameter_values``.
        validate: Whether to validate arguments during initialization.

    Raises:
        ValueError: If the ``observables`` and ``parameter_values`` are not broadcastable, that
            is, if their shapes, when right-aligned, do not agree or equal 1.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: ObservablesArray,
        parameter_values: BindingsArray | None = None,
        precision: float | None = None,
        shape: Tuple[int, ...] | None = None,
        validate: bool = True,
    ):
        super().__init__()
        self._circuit = circuit
        self._observables = observables
        self._parameter_values = parameter_values or BindingsArray()
        self._precision = precision

        if shape is None:
            try:
                # _shape has to be defined to properly be Shaped, so we can't put it in validation
                self._shape = np.broadcast_shapes(
                    self.observables.shape, self.parameter_values.shape
                )
            except ValueError as ex:
                raise ValueError(
                    f"The observables shape {self.observables.shape} and the parameter values "
                    f"shape {self.parameter_values.shape} are not broadcastable."
                ) from ex
        else:
            self._shape = shape

        if validate:
            self.validate()

    @property
    def circuit(self) -> QuantumCircuit:
        """A quantum circuit."""
        return self._circuit

    @property
    def observables(self) -> ObservablesArray:
        """An observables array."""
        return self._observables

    @property
    def parameter_values(self) -> BindingsArray:
        """A bindings array."""
        return self._parameter_values

    @property
    def precision(self) -> float | None:
        """The target precision for expectation value estimates (optional)."""
        return self._precision

    @classmethod
    def coerce(cls, pub: EstimatorPubLike, precision: float | None = None) -> EstimatorPub:
        """Coerce :class:`~.EstimatorPubLike` into :class:`~.EstimatorPub`.

        Args:
            pub: A compatible object for coercion.
            precision: an optional default precision to use if not
                       already specified by the pub-like object.

        Returns:
            An estimator pub.
        """
        # Validate precision kwarg if provided
        if precision is not None:
            if not isinstance(precision, Real):
                raise TypeError(f"precision must be a real number, not {type(precision)}.")
            if precision < 0:
                raise ValueError("precision must be non-negative")
        if isinstance(pub, EstimatorPub):
            if pub.precision is None and precision is not None:
                return cls(
                    circuit=pub.circuit,
                    observables=pub.observables,
                    parameter_values=pub.parameter_values,
                    precision=precision,
                    shape=pub.shape,
                    validate=False,  # Assume Pub is already validated
                )
            return pub

        if isinstance(pub, QuantumCircuit):
            raise ValueError(
                f"An invalid Estimator pub-like was given ({type(pub)}). "
                "If you want to run a single pub, you need to wrap it with `[]` like "
                "`estimator.run([(circuit, observables, param_values)])` "
                "instead of `estimator.run((circuit, observables, param_values))`."
            )

        if len(pub) not in [2, 3, 4, 5]:
            raise ValueError(
                f"The length of pub must be 2, 3, 4, or 5 but length {len(pub)} is given."
            )
        circuit = pub[0]
        observables = ObservablesArray.coerce(pub[1])

        if len(pub) > 2 and pub[2] is not None:
            values = pub[2]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)
        else:
            parameter_values = None

        if len(pub) > 3 and pub[3] is not None:
            precision = pub[3]

        shape = None
        if len(pub) > 4 and pub[4] is not None:
            shape = pub[4]

        return cls(
            circuit=circuit,
            observables=observables,
            parameter_values=parameter_values,
            precision=precision,
            shape=shape,
            validate=True,
        )

    def validate(self):
        """Validate the pub."""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.observables.validate()
        self.parameter_values.validate()

        if self.precision is not None:
            if not isinstance(self.precision, Real):
                raise TypeError(f"precision must be a real number, not {type(self.precision)}.")
            if self.precision < 0:
                raise ValueError("precision must be non-negative.")

        # Validate shape consistency
        if not isinstance(self.shape, tuple) or not all(isinstance(idx, int) for idx in self.shape):
            raise ValueError(f"The shape must be a tuple of integers, found {self.shape} instead.")

        try:
            broadcast_shape = np.broadcast_shapes(
                self.shape, self.parameter_values.shape, self.observables.shape
            )
        except ValueError as exc:
            raise ValueError(
                f"The shape of the parameter values, {self.parameter_values.shape}, "
                f"is not compatible with the shape of the pub, {self.shape}."
            ) from exc

        if broadcast_shape != self.shape:
            raise ValueError(
                f"The shape of the observables, {self.observables.shape}, or the shape of the "
                f"parameter values, {self.parameter_values.shape}, exceeds the shape of the pub, "
                f"{self.shape}, on some axis."
            )

        # Cross validate circuits and observables
        for i, observable in np.ndenumerate(self.observables):
            num_qubits = len(next(iter(observable)))
            if self.circuit.num_qubits != num_qubits:
                raise ValueError(
                    f"The number of qubits of the circuit ({self.circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable ({num_qubits})."
                )

        # Cross validate circuits and parameter_values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            raise ValueError(
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )


EstimatorPubLike = Union[
    EstimatorPub,
    Tuple[QuantumCircuit, ObservablesArrayLike],
    Tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike],
    Tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike, Real],
    Tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike, Real, Tuple[int, ...]],
]
"""Types coercible into an :class:`~.EstimatorPub`.
 
This can either be an :class:`~.EstimatorPub` itself, or a tuple with entries
``(circuit, observables, parameter_values, precision, shape)`` where the last three values 
can optionally be absent or set to ``None``.
The following formats are valid:

 * (:class:`~.QuantumCircuit`, :class:`~.ObservablesArrayLike`\\)
 * (:class:`~.QuantumCircuit`, :class:`~.ObservablesArrayLike`, :class:`~.BindingsArrayLike`\\)
 * (:class:`~.QuantumCircuit`, :class:`~.ObservablesArrayLike`, :class:`~.BindingsArrayLike`, 
   :class:`float`\\)
 * (:class:`~.QuantumCircuit`, :class:`~.ObservablesArrayLike`, :class:`~.BindingsArrayLike`, 
   :class:`float`, :class:`tuple[int, ...]`\\)

If ``parameter_values`` are not provided, the circuit must have no 
:attr:`~.QuantumCircuit.parameters`.
If ``precision`` is not provided, the estimator will supply a value.
If ``shape`` is not provided, it will be inferred by broadcasting the shapes of 
``observables`` and ``parameter_values``. 
"""
