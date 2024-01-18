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

from typing import Iterable, Mapping, Union, overload

import numpy as np
from numpy.typing import ArrayLike

from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from .object_array import object_array
from .shape import ShapedMixin
from .observable import Observable, ObservableLike


class ObservablesArray(ShapedMixin):
    r"""An ND-array of :class:`.Observable`\s for an :class:`.Estimator` primitive."""

    __slots__ = ("_array", "_shape")

    def __init__(
        self,
        observables: ArrayLike | ObservableLike,
        copy: bool = True,
        validate: bool = True,
    ):
        """Initialize an observables array.

        Args:
            observables: An array-like of basis observable compatible objects.
            copy: Specify the ``copy`` kwarg of the :func:`.object_array` function
                when initializing observables.
            validate: If True, convert :const:`.BasisObservableLike` input objects
                to :const:`.BasisObservable` objects and validate. If False the
                input should already be an array-like of valid
                :const:`.BasisObservble` objects.

        Raises:
            ValueError: If ``validate=True`` and the input observables is not valid.
        """
        super().__init__()
        if isinstance(observables, ObservablesArray):
            observables = observables._array
        self._array = object_array(observables, copy=copy, list_types=(PauliList,))
        self._shape = self._array.shape
        if validate:
            # Convert array items to Observable objects
            # and validate they are on the same number of qubits
            num_qubits = None
            for ndi, obs in np.ndenumerate(self._array):
                basis_obs = Observable.coerce(obs)
                basis_num_qubits = len(next(iter(basis_obs)))
                if num_qubits is None:
                    num_qubits = basis_num_qubits
                elif basis_num_qubits != num_qubits:
                    raise ValueError(
                        "The number of qubits must be the same for all observables in the "
                        "observables array."
                    )
                self._array[ndi] = basis_obs

    def __repr__(self):
        prefix = f"{type(self).__name__}("
        suffix = f", shape={self.shape})"
        array = np.array2string(self._array, prefix=prefix, suffix=suffix, threshold=50)
        return prefix + array + suffix

    def tolist(self) -> list:
        """Convert to a nested list"""
        return self._array.tolist()

    def __array__(self, dtype=None):
        """Convert to an Numpy.ndarray"""
        if dtype is None or dtype == object:
            return self._array
        raise ValueError("Type must be 'None' or 'object'")

    @overload
    def __getitem__(self, args: int | tuple[int, ...]) -> Observable:
        ...

    @overload
    def __getitem__(self, args: slice) -> ObservablesArray:
        ...

    def __getitem__(self, args):
        item = self._array[args]
        if not isinstance(item, np.ndarray):
            return item
        return ObservablesArray(item, copy=False, validate=False)

    def reshape(self, shape: int | Iterable[int]) -> ObservablesArray:
        """Return a new array with a different shape.

        This results in a new view of the same arrays.

        Args:
            shape: The shape of the returned array.

        Returns:
            A new array.
        """
        return ObservablesArray(self._array.reshape(shape), copy=False, validate=False)

    def ravel(self) -> ObservablesArray:
        """Return a new array with one dimension.

        The returned array has a :attr:`shape` given by ``(size, )``, where
        the size is the :attr:`~size` of this array.

        Returns:
            A new flattened array.
        """
        return self.reshape(self.size)

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

        if isinstance(observables, (str, SparsePauliOp, Pauli, Mapping)):
            observables = [observables]

        # Convert array items to Observable objects and validate they are on the
        # same number of qubis.  Note that we copy some of the validation method
        # here to avoid double iteration of the array
        data = object_array(observables, copy=True, list_types=(PauliList,))
        num_qubits = None
        for ndi, obs in np.ndenumerate(data):
            basis_obs = Observable.coerce(obs)
            if num_qubits is None:
                num_qubits = basis_obs.num_qubits
            elif basis_obs.num_qubits != num_qubits:
                raise ValueError(
                    "The number of qubits must be the same for all observables in the "
                    "observables array."
                )
            data[ndi] = basis_obs

        return cls(data, validate=False)

    def validate(self):
        """Validate the consistency in observables array."""
        # Convert array items to Observable objects
        # and validate they are on the same number of qubits
        if not isinstance(self._array, np.ndarray) or self._array.dtype != object:
            raise TypeError("Data should be an object ndarray")

        num_qubits = None
        for ndi, obs in np.ndenumerate(self._array):
            if not isinstance(obs, Observable):
                raise TypeError(f"item at index {ndi} is a {type(obs)}, not an Observable.")
            obs.validate()
            if num_qubits is None:
                num_qubits = obs.num_qubits
            elif obs.num_qubits != num_qubits:
                raise ValueError(
                    "The number of qubits must be the same for all observables in the "
                    "observables array."
                )


ObservablesArrayLike = Union[ObservablesArray, ArrayLike, ObservableLike]
"""Types that can be natively converted to an ObservablesArray"""
