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
Container class for an Estimator observable.
"""
from __future__ import annotations

from typing import Union, Iterable, Mapping as MappingType
from collections.abc import Mapping
from collections import defaultdict
from numbers import Complex

from qiskit.quantum_info import Pauli, SparsePauliOp

ObservableLike = Union[
    str,
    Pauli,
    SparsePauliOp,
    MappingType[Union[str, Pauli], complex],
    Iterable[Union[str, Pauli, SparsePauliOp]],
]
"""Types that can be natively used to construct a :const:`BasisObservable`."""


class Observable(Mapping):
    """A sparse container for a Hermitian observable for an :class:`.Estimator` primitive."""

    __slots__ = ("_data", "_num_qubits", "_terms")

    def __init__(
        self,
        data: Mapping[str, complex],
        validate: bool = True,
    ):
        """Initialize an observables array.

        Args:
            data: The observable data.
            validate: If ``True``, the input data is validated during initialization.

        Raises:
            ValueError: If ``validate=True`` and the input observable-like is not valid.
        """
        self._data = data
        self._num_qubits: int = len(next(iter(data)))
        self._terms: str = ""
        if validate:
            self.validate()

    def __repr__(self):
        return f"{type(self).__name__}({self._data})"

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def terms(self) -> str:
        """Return a string containing all unique basis terms used in the observable"""
        if not self._terms:
            # QUESTION: Should terms be `tuple[str, ...]` instead
            # to allow for basis identification using more than 1 character?
            self._terms = "".join(set().union(*self._data))
        return self._terms

    @property
    def num_qubits(self) -> int:
        """The number of qubits in the observable"""
        return self._num_qubits

    def validate(self):
        """Validate the consistency in observables array."""
        if not isinstance(self._data, Mapping):
            raise TypeError(f"Observable data type {type(self._data)} is not a Mapping.")
        for key, value in self._data.items():
            if not isinstance(key, str):
                raise TypeError(f"Item {key} is not a str")
            if not isinstance(value, Complex):
                raise TypeError(f"Value {value} is not a complex number")

    @classmethod
    def coerce(cls, observable: ObservableLike) -> Observable:
        """Coerce an observable-like object into an :class:`.Observable`.

        Args:
            observable: The observable-like input.

        Returns:
            A coerced observables array.

        Raises:
            TypeError: If the input cannot be formatted because its type is not valid.
            ValueError: If the input observable is invalid.
        """

        # Pauli-type conversions
        if isinstance(observable, SparsePauliOp):
            # Call simplify to combine duplicate keys before converting to a mapping
            data = dict(observable.simplify(atol=0).to_list())
            return cls.coerce(data)

        if isinstance(observable, Pauli):
            label, phase = observable[:].to_label(), observable.phase
            data = {label: 1} if phase == 0 else {label: (-1j) ** phase}
            return cls.coerce(data)

        # String conversion
        if isinstance(observable, str):
            return cls.coerce({observable: 1})

        # Mapping conversion (with possible Pauli keys)
        if isinstance(observable, Mapping):
            # NOTE: This assumes length of keys is number of qubits
            #       this might not be very robust
            # We also compute terms while iterating to save doing it again later
            terms = set()
            num_qubits = len(next(iter(observable)))
            unique = defaultdict(complex)
            for basis, coeff in observable.items():
                if isinstance(basis, Pauli):
                    basis, phase = basis[:].to_label(), basis.phase
                    if phase != 0:
                        coeff = coeff * (-1j) ** phase
                # Validate basis
                if len(basis) != num_qubits:
                    raise ValueError(
                        "Number of qubits must be the same for all observable basis elements."
                    )
                terms = terms.union(basis)
                unique[basis] += coeff
            obs = Observable(dict(unique))
            # Manually set terms so a second iteration over data is not needed
            obs._data = "".join(terms)
            return obs

        raise TypeError(f"Invalid observable type: {type(observable)}")
