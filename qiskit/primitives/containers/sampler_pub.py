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

from typing import Tuple, Union

from qiskit import QuantumCircuit

from .bindings_array import BindingsArray, BindingsArrayLike
from .shape import ShapedMixin


class SamplerPub(ShapedMixin):
    """Pub (Primitive Unified Bloc) for Sampler.

    Pub is composed of double (circuit, parameter_values).
    """

    __slots__ = ("_circuit", "_parameter_values")

    def __init__(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray | None = None,
        validate: bool = False,
    ):
        """Initialize a sampler pub.

        Args:
            circuit: a quantum circuit.
            parameter_values: a bindings array.
            validate: if True, the input data is validated during initialization.
        """
        self._circuit = circuit
        self._parameter_values = parameter_values or BindingsArray()
        self._shape = self._parameter_values.shape
        if validate:
            self.validate()

    @property
    def parameter_values(self) -> BindingsArray:
        """A bindings array"""
        return self._parameter_values

    @classmethod
    def coerce(cls, pub: SamplerPubLike) -> SamplerPub:
        """Coerce SamplerPubLike into SamplerPub.

        Args:
            pub: an object to be Sampler pub.

        Returns:
            A coerced sampler pub.
        """
        if isinstance(pub, SamplerPub):
            return pub
        if isinstance(pub, QuantumCircuit):
            return cls(circuit=pub)
        if len(pub) not in [1, 2]:
            raise ValueError(f"The length of pub must be 1 or 2, but length {len(pub)} is given.")
        circuit = pub[0]
        if len(pub) == 1:
            return cls(circuit=circuit)
        parameter_values = BindingsArray.coerce(pub[1])
        return cls(circuit=circuit, parameter_values=parameter_values)

    def validate(self):
        """Validate the pub."""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.parameter_values.validate()

        # Cross validate circuits and parameter values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            raise ValueError(
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )


SamplerPubLike = Union[
    SamplerPub, QuantumCircuit, Tuple[QuantumCircuit], Tuple[QuantumCircuit, BindingsArrayLike]
]
