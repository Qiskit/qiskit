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

import re
from collections.abc import Mapping, Iterable
from collections import defaultdict
from functools import lru_cache
from typing import Union
from numbers import Complex

from qiskit.quantum_info import Pauli, SparsePauliOp

ObservableLike = Union[
    str,
    Pauli,
    SparsePauliOp,
    Mapping[Union[str, Pauli], complex],
    Iterable[Union[str, Pauli, SparsePauliOp]],
]
"""Types that can be natively used to construct a :const:`BasisObservable`."""


class Observable(Mapping):
    """A sparse container for a Hermitian observable for an :class:`.Estimator` primitive."""

    ALLOWED_BASIS: str = "IXYZ01+-lr"
    """The allowed characters in :class:`.Observable` strings."""

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
        self._num_qubits = len(next(iter(data)))
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
    def num_qubits(self) -> int:
        """The number of qubits in the observable"""
        return self._num_qubits

    def validate(self):
        """Validate the consistency in observables array."""
        if not isinstance(self._data, Mapping):
            raise TypeError(f"Observable data type {type(self._data)} is not a Mapping.")
        for key, value in self._data.items():
            try:
                self._validate_basis(key)
                self._validate_coeff(value)
            except TypeError as ex:
                raise TypeError(f"Invalid type for item ({key}, {value})") from ex
            except Exception as ex:  # pylint: disable = broad-except
                raise ValueError(f"Invalid value for item ({key}, {value})") from ex

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
            cls._validate_basis(label)
            data = {label: 1} if phase == 0 else {label: (-1j) ** phase}
            return Observable(data)

        # String conversion
        if isinstance(observable, str):
            cls._validate_basis(observable)
            return cls.coerce({observable: 1})

        # Mapping conversion (with possible Pauli keys)
        if isinstance(observable, Mapping):
            # NOTE: This assumes length of keys is number of qubits
            #       this might not be very robust
            num_qubits = len(next(iter(observable)))
            unique = defaultdict(complex)
            for basis, coeff in observable.items():
                if isinstance(basis, Pauli):
                    basis, phase = basis[:].to_label(), basis.phase
                    if phase != 0:
                        coeff = coeff * (-1j) ** phase
                # Validate basis
                cls._validate_basis(basis)
                if len(basis) != num_qubits:
                    raise ValueError(
                        "Number of qubits must be the same for all observable basis elements."
                    )
                unique[basis] += coeff
            return Observable(dict(unique))

        raise TypeError(f"Invalid observable type: {type(observable)}")

    @classmethod
    def _validate_basis(cls, basis: any) -> None:
        """Validate a basis string.

        Args:
            basis: a basis object to validate.

        Raises:
            TypeError: If the input basis is not a string
            ValueError: If basis string contains invalid characters
        """
        if not isinstance(basis, str):
            raise TypeError(f"basis {basis} is not a string")

        # NOTE: the allowed basis characters can be overridden by modifying the class
        # attribute ALLOWED_BASIS
        allowed_pattern = _regex_match(cls.ALLOWED_BASIS)
        if not allowed_pattern.match(basis):
            invalid_pattern = _regex_invalid(cls.ALLOWED_BASIS)
            invalid_chars = list(set(invalid_pattern.findall(basis)))
            raise ValueError(
                f"Observable basis string '{basis}' contains invalid characters {invalid_chars},"
                f" allowed characters are {list(cls.ALLOWED_BASIS)}.",
            )

    @classmethod
    def _validate_coeff(cls, coeff: any):
        """Validate the consistency in observables array."""
        if not isinstance(coeff, Complex):
            raise TypeError(f"Value {coeff} is not a complex number")


@lru_cache(1)
def _regex_match(allowed_chars: str) -> re.Pattern:
    """Return pattern for matching if a string contains only the allowed characters."""
    return re.compile(f"^[{re.escape(allowed_chars)}]*$")


@lru_cache(1)
def _regex_invalid(allowed_chars: str) -> re.Pattern:
    """Return pattern for selecting invalid strings"""
    return re.compile(f"[^{re.escape(allowed_chars)}]")
