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
ND-Array container class for Estimator observables.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping as _Mapping
from functools import lru_cache
from typing import Union, Mapping, overload

import numpy as np
from numpy.typing import ArrayLike

from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp, SparseObservable

from .object_array import object_array
from .shape import ShapedMixin, shape_tuple

# Public API classes
__all__ = ["ObservableLike", "ObservablesArrayLike"]

ObservableLike = Union[
    str,
    Pauli,
    SparsePauliOp,
    Mapping[Union[str, Pauli], float],
]
"""Types that can be natively used to construct a Hermitian Estimator observable."""


ObservablesArrayLike = Union[ObservableLike, ArrayLike]
"""Types that can be natively converted to an array of Hermitian Estimator observables."""


class ObservablesArray(ShapedMixin):
    """An ND-array of Hermitian observables for an :class:`.Estimator` primitive."""

    __slots__ = ("_array", "_shape")

    def __init__(
        self,
        observables: ObservablesArrayLike,
        num_qubits: int | None = None,
        copy: bool = True,
        validate: bool = True,
    ):
        """Initialize an observables array.

        Args:
            observables: An array-like of basis observable compatible objects.
            copy: Specify the ``copy`` kwarg of the :func:`.object_array` function
                when initializing observables.
            num_qubits: The number of qubits of the observables. If not specified, the number of
                qubits will be inferred from the observables. If specified, then the specified
                number of qubits must match the number of qubits in the observables.
            validate: If true, coerce entries into the internal format and validate them. If false,
                the input should already be an array-like.

        Raises:
            ValueError: If ``validate=True`` and the input observables array is not valid.
        """
        super().__init__()
        if isinstance(observables, ObservablesArray):
            observables = observables._array

        self._array = object_array(observables, copy=copy, list_types=(PauliList,))
        self._shape = self._array.shape
        self._num_qubits = num_qubits

        if validate:
            for ndi, obs in np.ndenumerate(self._array):
                basis_obs = self.coerce_observable(obs)
                if self._num_qubits is None:
                    self._num_qubits = basis_obs.num_qubits
                elif self._num_qubits != basis_obs.num_qubits:
                    raise ValueError(
                        "The number of qubits must be the same for all observables in the "
                        "observables array."
                    )
                self._array[ndi] = basis_obs
        elif self._num_qubits is None and self._array.size > 0:
            self._num_qubits = self._array.reshape(-1)[0].num_qubits

        # can happen for empty arrays
        if self._num_qubits is None:
            self._num_qubits = 0

    @staticmethod
    def _obs_to_dict(obs: SparseObservable) -> Mapping[str, float]:
        """Convert a sparse observable to a mapping from Pauli strings to coefficients"""
        result = {}
        for sparse_pauli_str, pauli_qubits, coeff in obs.to_sparse_list():

            if len(sparse_pauli_str) == 0:
                full_pauli_str = "I" * obs.num_qubits
            else:
                sorted_lists = sorted(zip(pauli_qubits, sparse_pauli_str))
                string_fragments = []
                prev_qubit = -1
                for qubit, pauli in sorted_lists:
                    string_fragments.append("I" * (qubit - prev_qubit - 1) + pauli)
                    prev_qubit = qubit

                string_fragments.append("I" * (obs.num_qubits - max(pauli_qubits) - 1))
                full_pauli_str = "".join(string_fragments)[::-1]

            # We know that the dictionary doesn't contain yet full_pauli_str as a key
            # because the observable is guaranteed to be simplified
            result[full_pauli_str] = np.real(coeff)

        return result

    def __repr__(self):
        prefix = f"{type(self).__name__}("
        suffix = f", shape={self.shape})"
        array = np.array2string(self.__array__(), prefix=prefix, suffix=suffix, threshold=50)
        return prefix + array + suffix

    def tolist(self) -> list | ObservableLike:
        """Convert to a nested list.

        Similar to Numpy's ``tolist`` method, the level of nesting
        depends on the dimension of the observables array. In the
        case of dimension 0 the method returns a single observable
        (``dict`` in the case of a weighted sum of Paulis) instead of a list.

        Examples::
            Return values for a one-element list vs one element:

                >>> from qiskit.primitives.containers.observables_array import ObservablesArray
                >>> oa = ObservablesArray.coerce(["Z"])
                >>> print(type(oa.tolist()))
                <class 'list'>
                >>> oa = ObservablesArray.coerce("Z")
                >>> print(type(oa.tolist()))
                <class 'dict'>
        """
        return self.__array__().tolist()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # pylint: disable=unused-argument
        """Convert to a Numpy.ndarray"""
        if dtype is None or dtype == object:
            tmp_result = self.__getitem__(tuple(slice(None) for _ in self._array.shape))
            if len(self._array.shape) == 0:
                result = np.ndarray(shape=self._array.shape, dtype=dict)
                result[()] = tmp_result
            else:
                result = np.ndarray(tmp_result.shape, dtype=dict)
                for ndi, obs in np.ndenumerate(tmp_result._array):
                    result[ndi] = self._obs_to_dict(obs)
            return result
        raise ValueError("Type must be 'None' or 'object'")

    @overload
    def __getitem__(self, args: int | tuple[int, ...]) -> Mapping[str, float]: ...

    @overload
    def __getitem__(self, args: slice | tuple[slice, ...]) -> ObservablesArray: ...

    def __getitem__(self, args):
        item = self._array[args]
        if not isinstance(item, np.ndarray):
            return self._obs_to_dict(item)

        return ObservablesArray(item, copy=False, validate=False)

    def reshape(self, *shape: int | Iterable[int]) -> ObservablesArray:
        """Return a new array with a different shape.

        This results in a new view of the same arrays.

        Args:
            shape: The shape of the returned array.

        Returns:
            A new array.
        """
        shape = shape_tuple(*shape)
        return ObservablesArray(self._array.reshape(shape), copy=False, validate=False)

    def ravel(self) -> ObservablesArray:
        """Return a new array with one dimension.

        The returned array has a :attr:`shape` given by ``(size, )``, where
        the size is the :attr:`~size` of this array.

        Returns:
            A new flattened array.
        """
        return self.reshape(self.size)

    @property
    def num_qubits(self) -> int:
        """Return the observable array's number of qubits"""
        return self._num_qubits

    @classmethod
    def coerce_observable(cls, observable: ObservableLike) -> SparseObservable:
        """Format an observable-like object into the internal format.

        Args:
            observable: The observable-like to format.

        Returns:
            The coerced observable.

        Raises:
            TypeError: If the input cannot be formatted because its type is not valid.
            ValueError: If the input observable is invalid.
        """
        # Pauli-type conversions
        if isinstance(observable, SparsePauliOp):
            observable = SparseObservable.from_sparse_pauli_op(observable)
        elif isinstance(observable, Pauli):
            observable = SparseObservable.from_pauli(observable)
        elif isinstance(observable, str):
            observable = SparseObservable.from_label(observable)
        elif isinstance(observable, _Mapping):
            term_list = []
            for basis, coeff in observable.items():
                if isinstance(basis, str):
                    term_list.append((basis, coeff))
                elif isinstance(basis, Pauli):
                    unphased_basis, phase = basis[:].to_label(), basis.phase
                    term_list.append((unphased_basis, complex(0, 1) ** phase * coeff))
                else:
                    raise TypeError(f"Invalid observable basis type: {type(basis)}")
            observable = SparseObservable.from_list(term_list)

        if isinstance(observable, SparseObservable):
            # Check that the operator has real coeffs
            coeffs = np.real_if_close(observable.coeffs)
            if np.iscomplexobj(coeffs):
                raise ValueError(
                    "Non-Hermitian input observable: the input SparsePauliOp has non-zero"
                    " imaginary part in its coefficients."
                )

            return SparseObservable.from_raw_parts(
                observable.num_qubits,
                coeffs,
                observable.bit_terms,
                observable.indices,
                observable.boundaries,
            ).simplify(tol=0)

        raise TypeError(f"Invalid observable type: {type(observable)}")

    @classmethod
    def coerce(cls, observables: ObservablesArrayLike) -> ObservablesArray:
        """Coerce ObservablesArrayLike into ObservableArray.

        Args:
            observables: an object to be observables array.

        Returns:
            A coerced observables array.
        """
        if isinstance(observables, ObservablesArray):
            return observables
        return cls(observables)

    def validate(self):
        """Validate the consistency in observables array."""
        for obs in self._array.reshape(-1):
            if obs.num_qubits != self.num_qubits:
                raise ValueError(
                    "An observable was detected, whose number of qubits"
                    " does not match the array's number of qubits"
                )


@lru_cache(1)
def _regex_match(allowed_chars: str) -> re.Pattern:
    """Return pattern for matching if a string contains only the allowed characters."""
    return re.compile(f"^[{re.escape(allowed_chars)}]*$")


@lru_cache(1)
def _regex_invalid(allowed_chars: str) -> re.Pattern:
    """Return pattern for selecting invalid strings"""
    return re.compile(f"[^{re.escape(allowed_chars)}]")
