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


"""
Sampler Pub class
"""

from __future__ import annotations

from typing import Tuple, Union

from qiskit import QuantumCircuit

from .base_pub import BasePub
from .bindings_array import BindingsArray, BindingsArrayLike
from .dataclasses import frozen_dataclass
from .shape import ShapedMixin


@frozen_dataclass
class SamplerPub(BasePub, ShapedMixin):
    """Pub (Primitive Unified Bloc) for Sampler.

    Pub is composed of double (circuit, parameter_values).
    """

    parameter_values: BindingsArray = BindingsArray(shape=())
    _shape: Tuple[int, ...] = ()

    def __post_init__(self):
        self._shape = self.parameter_values.shape

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
            return cls(circuit=pub)
        parameter_values = BindingsArray.coerce(pub[1])
        return cls(circuit=circuit, parameter_values=parameter_values)

    def validate(self):
        """Validate the pub."""
        super(SamplerPub, self).validate()  # pylint: disable=super-with-arguments
        # I'm not sure why these arguments for super are needed. But if no args, tests are failed
        # for Python >=3.10. Seems to be some bug, but I can't fix.
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
