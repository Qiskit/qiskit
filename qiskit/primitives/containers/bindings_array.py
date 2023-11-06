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
Bindings array class
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain, product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType

from .shape import ShapedMixin, ShapeInput, shape_tuple

ParameterLike = Union[Parameter, str]


class BindingsArray(ShapedMixin):
    r"""Stores many possible parameter binding values for a :class:`qiskit.QuantumCircuit`.

    Similar to a ``inspect.BoundArguments`` instance, which stores arguments that can be bound to a
    compatible Python function, this class stores both values without names, so that their ordering
    is important, as well as values attached to ``qiskit.circuit.Parameters``. However, a dense
    rectangular array of possible values is stored for each parameter, so that this class is akin to
    an object-array of ``inspect.BoundArguments``.

    The storage format is a list of arrays, ``[vals0, vals1, ...]``, as well as a dictionary of
    arrays attached to parameters, ``{params0: kwvals0, ...}``. Crucially, the last dimension of
    each array indexes one or more parameters. For example, if the last dimension of ``vals1`` is
    25, then it represents an array of possible binding values for 25 distinct parameters, where its
    leading shape is the array :attr:`~.shape` of its binding array. This implies a degeneracy of the
    storage format: ``[vals, vals1[..., :10], vals1[..., 10:], ...]`` is exactly equivalent to
    ``[vals0, vals1, ...]`` in the bindings it specifies. This complication has been included to
    satisfy two competing constraints:

        * Arrays with different dtypes cannot be concatenated into a single array, so that multiple
          arrays are required for generality.
        * It is extremely convenient to put everything into a small number of big arrays, when
          possible.

    .. code-block:: python

        # 0-d array (i.e. only one binding)
        BindingsArray([1, 2, 3], {"a": 4, ("b", "c"): [5, 6]})

        # single array, last index is parameters
        BindingsArray(np.empty((10, 10, 100)))

        # multiple arrays, where each last index is parameters. notice that it's smart enough to
        # figure out that a missing last dimension corresponds to a single parameter.
        BindingsArray(
            [np.empty((10, 10, 100)), np.empty((10, 10)), np.empty((10, 10, 20), dtype=complex)],
            {("c", "a"): np.empty((10, 10, 2)), "b": np.empty((10, 10))}
        )
    """
    __slots__ = ("_vals", "_kwvals")

    def __init__(
        self,
        vals: Union[None, ArrayLike, Iterable[ArrayLike]] = None,
        kwvals: Union[None, Mapping[ParameterLike, Iterable[ParameterValueType]], ArrayLike] = None,
        shape: Optional[ShapeInput] = None,
    ):
        """
        The ``shape`` argument does not need to be provided whenever it can unambiguously
        be inferred from the provided arrays. Ambiguity arises because an array provided to the
        constructor might represent values for either a single parameter, with an implicit missing
        last dimension of size ``1``, or for many parameters, where the size of the last dimension
        is the number of parameters it is providing values to. This ambiguity can be broken in the
        following common ways:

            * Only a single array is provided to ``vals``, and no arrays to ``kwvals``, in which case
              it is assumed that the last dimension is over many parameters.
            * Multiple arrays are given whose shapes differ only in the last dimension size.
            * Some array is given in ``kwvals`` where the key contains multiple
              :class:`~.Parameter` s, whose length the last dimension of the array must therefore match.

        Args:
            vals: One or more arrays, where the last index of each corresponds to
                distinct parameters. If their dtypes allow it, concatenating these
                arrays over the last axis is equivalent to providing them separately.
            kwvals: A mapping from one or more parameters to arrays of values to bind
                them to, where the last axis is over parameters.
            shape: The leading shape of every array in these bindings.

        Raises:
            ValueError: If all inputs are ``None``.
            ValueError: If the shape cannot be automatically inferred from the arrays, or if there
                is some inconsistency in the shape of the given arrays.
        """
        super().__init__()

        if vals is None and kwvals is None and shape is None:
            raise ValueError("Must specify a shape if no values are present")
        if vals is None:
            vals = []
        if kwvals is None:
            kwvals = {}

        vals = [vals] if isinstance(vals, np.ndarray) else [np.array(v, copy=False) for v in vals]
        # TODO str will be used for internal data (_kwvals) instead of Parameter.
        # This requires https://github.com/Qiskit/qiskit/issues/7107
        kwvals = {
            (p,) if isinstance(p, Parameter) else tuple(p): np.array(val, copy=False)
            for p, val in kwvals.items()
        }

        if shape is None:
            # jump through hoops to find out user's intended shape
            shape = _infer_shape(vals, kwvals)

        # shape checking, and normalization so that each last index must be over parameters
        self._shape = shape_tuple(shape)
        for idx, val in enumerate(vals):
            vals[idx] = _standardize_shape(val, self._shape)

        self._vals = vals
        self._kwvals = kwvals

        self.validate()

    def __getitem__(self, args) -> BindingsArray:
        # because the parameters live on the last axis, we don't need to do anything special to
        # accomodate them because there will always be an implicit slice(None, None, None)
        # on all unspecified trailing dimensions
        # separately, we choose to not disallow args which touch the last dimension, even though it
        # would not be a particularly friendly way to chop parameters
        vals = [val[args] for val in self._vals]
        kwvals = {params: val[args] for params, val in self._kwvals.items()}
        try:
            shape = next(chain(vals, kwvals.values())).shape[:-1]
        except StopIteration:
            shape = ()
        return BindingsArray(vals, kwvals, shape)

    @property
    def kwvals(self) -> Dict[Tuple[str, ...], np.ndarray]:
        """The keyword values of this array."""
        return {_format_key(k): v for k, v in self._kwvals.items()}

    @property
    def num_parameters(self) -> int:
        """The total number of parameters."""
        return sum(val.shape[-1] for val in chain(self.vals, self._kwvals.values()))

    @property
    def vals(self) -> List[np.ndarray]:
        """The non-keyword values of this array."""
        return self._vals

    def bind_at_idx(self, circuit: QuantumCircuit, idx: Tuple[int, ...]) -> QuantumCircuit:
        """Return the circuit bound to the values at the provided index.

        Args:
            circuit: The circuit to bind.
            idx: A tuple of indices, on for each dimension of this array.

        Returns:
            The bound circuit.

        Raises:
            ValueError: If the index doesn't have the right number of values.
        """
        if len(idx) != self.ndim:
            raise ValueError(f"Expected {idx} to index all dimensions of {self.shape}")

        flat_vals = (val for vals in self.vals for val in vals[idx])

        if not self._kwvals:
            # special case to avoid constructing a dictionary input
            return circuit.assign_parameters(list(flat_vals))

        parameters = dict(zip(circuit.parameters, flat_vals))
        parameters.update(
            (param, val)
            for params, vals in self._kwvals.items()
            for param, val in zip(params, vals[idx])
        )
        return circuit.assign_parameters(parameters)

    def bind_flat(self, circuit: QuantumCircuit) -> Iterable[QuantumCircuit]:
        """Yield a bound circuit for every array index in flattened order.

        Args:
            circuit: The circuit to bind.

        Yields:
            Bound circuits, in flattened array order.
        """
        for idx in product(*map(range, self.shape)):
            yield self.bind_at_idx(circuit, idx)

    def bind_all(self, circuit: QuantumCircuit) -> np.ndarray:
        """Return an object array of bound circuits with the same shape.

        Args:
            circuit: The circuit to bind.

        Returns:
            An object array of the same shape containing all bound circuits.
        """
        arr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            arr[idx] = self.bind_at_idx(circuit, idx)
        return arr

    def ravel(self) -> BindingsArray:
        """Return a new :class:`~BindingsArray` with one dimension.

        The returned bindings array has a :attr:`shape` given by ``(size, )``, where the size is the
        :attr:`~size` of this bindings array.

        Returns:
            A new bindings array.
        """
        return self.reshape(self.size)

    def reshape(self, shape: Union[int, Iterable[int]]) -> BindingsArray:
        """Return a new :class:`~BindingsArray` with a different shape.

        This results in a new view of the same arrays.

        Args:
            shape: The shape of the returned bindings array.

        Returns:
            A new bindings array.

        Raises:
            ValueError: If the provided shape has a different product than the current size.
        """
        shape = (shape, -1) if isinstance(shape, int) else (*shape, -1)
        if np.prod(shape[:-1]).astype(int) != self.size:
            raise ValueError("Reshaping cannot change the total number of elements.")
        vals = [val.reshape(shape) for val in self._vals]
        kwvals = {params: val.reshape(shape) for params, val in self._kwvals.items()}
        return BindingsArray(vals, kwvals, shape[:-1])

    @classmethod
    def coerce(cls, bindings_array: BindingsArrayLike) -> BindingsArray:
        """Coerce BindingsArrayLike into BindingsArray

        Args:
            bindings_array: an object to be bindings array.

        Returns:
            A coerced bindings array.
        """
        if isinstance(bindings_array, Sequence):
            bindings_array = np.array(bindings_array)
        if bindings_array is None:
            bindings_array = cls([], shape=(1,))
        elif isinstance(bindings_array, np.ndarray):
            if bindings_array.ndim == 1:
                bindings_array = bindings_array.reshape((1, -1))
            bindings_array = cls(bindings_array)
        elif isinstance(bindings_array, Mapping):
            bindings_array = cls(kwvals=bindings_array)
        else:
            raise TypeError(f"Unsupported type {type(bindings_array)} is given.")
        return bindings_array

    def validate(self):
        """Validate the consistency in bindings_array."""
        for parameters, val in self._kwvals.items():
            val = self._kwvals[parameters] = _standardize_shape(val, self._shape)
            if len(parameters) != val.shape[-1]:
                raise ValueError(
                    f"Length of {parameters} inconsistent with last dimension of {val}"
                )


def _standardize_shape(val: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Return ``val`` or ``val[..., None]``.

    Args:
        val: The array whose shape to standardize.
        shape: The shape to standardize to.

    Returns:
        An array with one more dimension than ``len(shape)``, and whose leading dimensions match
        ``shape``.

    Raises:
        ValueError: If the leading shape of ``val`` does not match the ``shape``.
    """
    if val.shape == shape:
        val = val[..., None]
    elif val.ndim - 1 != len(shape) or val.shape[:-1] != shape:
        raise ValueError(f"Array with shape {val.shape} inconsistent with {shape}")
    return val


def _infer_shape(
    vals: List[np.ndarray], kwvals: Dict[Tuple[Parameter, ...], np.ndarray]
) -> Tuple[int, ...]:
    """Return a shape tuple that consistently defines the leading dimensions of all arrays.

    Args:
        vals: A list of arrays.
        kwvals: A mapping from tuples to arrays, where the length of each tuple should match the
            last dimension of the corresponding array.

    Returns:
        A shape tuple that matches the leading dimension of every array.

    Raises:
        ValueError: If this cannot be done unambiguously.
    """
    only_possible_shapes = None

    def examine_array(*possible_shapes):
        nonlocal only_possible_shapes
        if only_possible_shapes is None:
            only_possible_shapes = set(possible_shapes)
        else:
            only_possible_shapes.intersection_update(possible_shapes)

    for parameters, val in kwvals.items():
        if len(parameters) > 1:
            # here, the last dimension _has_  to be over parameters
            examine_array(val.shape[:-1])
        elif val.shape[-1] != 1:
            # here, if the last dimension is not 1 then the shape is the shape
            examine_array(val.shape)
        else:
            # here, the last dimension could be over parameters or not
            examine_array(val.shape, val.shape[:-1])

    if len(vals) == 1 and len(kwvals) == 0:
        examine_array(vals[0].shape[:-1])
    else:
        for val in vals:
            # here, the last dimension could be over parameters or not
            examine_array(val.shape, val.shape[:-1])

    if len(only_possible_shapes) == 1:
        return next(iter(only_possible_shapes))
    elif len(only_possible_shapes) == 0:
        raise ValueError("Could not find any consistent shape.")
    raise ValueError("Could not unambiguously determine the intended shape; specify shape manually")


def _format_key(key: tuple[Parameter | str, ...]):
    return tuple(k.name if isinstance(k, Parameter) else k for k in key)


BindingsArrayLike = Union[
    BindingsArray,
    NDArray,
    "Mapping[Parameter, NDArray]",
    "Sequence[NDArray]",
    None,
]
