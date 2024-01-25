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
Bindings array class
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain, islice
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType

from .shape import ShapedMixin, ShapeInput, shape_tuple

ParameterLike = Union[Parameter, str]


class BindingsArray(ShapedMixin):
    r"""Stores parameter binding value sets for a :class:`qiskit.QuantumCircuit`.

    A single parameter binding set provides numeric values to bind to a circuit with free
    :class:`qiskit.circuit.Parameter`\s. An instance of this class stores an array-valued
    collection of such sets. The simplest example is a 0-d array consisting of a single
    parameter binding set, whereas an n-d array of parameter binding sets represents an
    n-d sweep over values.

    The storage format is a list of arrays, ``[vals0, vals1, ...]``, as well as a dictionary of
    arrays attached to parameters, ``{params0: kwvals0, ...}``. A convention is used
    where the last dimension of each array indexes (a subset of) circuit parameters. For
    example, if the last dimension of ``vals1`` is 25, then it represents an array of
    possible binding values for 25 distinct parameters, where its leading shape is the
    array :attr:`~.shape` of its binding array. This implies a degeneracy of the storage
    format: ``[vals, vals1[..., :10], vals1[..., 10:], ...]`` is exactly equivalent to
    ``[vals0, vals1, ...]`` in the bindings it specifies. This allows flexibility about whether
    values for different parameters are stored in one big array, or across several smaller
    arrays. It also allows different parameters to use different dtypes.

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
        vals: ArrayLike | Iterable[ArrayLike] | None = None,
        kwvals: Mapping[ParameterLike, Iterable[ParameterValueType]] | ArrayLike | None = None,
        shape: ShapeInput | None = None,
    ):
        r"""
        Initialize a ``BindingsArray``. It can take parameter vectors and dictionaries.

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
              :class:`~.Parameter`\s, whose length the last dimension of the array must therefore match.

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

        if vals is None:
            vals = []
        if kwvals is None:
            kwvals = {}

        vals = [vals] if isinstance(vals, np.ndarray) else [np.array(v, copy=False) for v in vals]
        kwvals = {
            _format_key((p,))
            if isinstance(p, Parameter)
            else _format_key(p): np.array(val, copy=False)
            for p, val in kwvals.items()
        }

        if shape is None:
            # jump through hoops to find out user's intended shape
            shape = _infer_shape(vals, kwvals)

        # shape checking, and normalization so that each last index must be over parameters
        self._shape = shape_tuple(shape)
        for idx, val in enumerate(vals):
            vals[idx] = _standardize_shape(val, self._shape)

        self._vals: list[np.ndarray] = vals
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

    def __repr__(self):
        descriptions = [f"shape={self.shape}", f"num_parameters={self.num_parameters}"]
        if num_kwval_params := sum(val.shape[-1] for val in self._kwvals.values()):
            names = list(islice(map(repr, chain.from_iterable(map(_format_key, self._kwvals))), 5))
            if len(names) < num_kwval_params:
                names.append("...")
            descriptions.append(f"parameters=[{', '.join(names)}]")
        return f"{type(self).__name__}(<{', '.join(descriptions)}>)"

    @property
    def kwvals(self) -> dict[tuple[str, ...], np.ndarray]:
        """The keyword values of this array."""
        return self._kwvals

    @property
    def num_parameters(self) -> int:
        """The total number of parameters."""
        return sum(val.shape[-1] for val in chain(self.vals, self._kwvals.values()))

    @property
    def vals(self) -> list[np.ndarray]:
        """The non-keyword values of this array."""
        return self._vals

    def bind(self, circuit: QuantumCircuit, loc: tuple[int, ...]) -> QuantumCircuit:
        """Return a new circuit bound to the values at the provided index.

        Args:
            circuit: The circuit to bind.
            loc: A tuple of indices, on for each dimension of this array.

        Returns:
            The bound circuit.

        Raises:
            ValueError: If the index doesn't have the right number of values.
        """
        if len(loc) != self.ndim:
            raise ValueError(f"Expected {loc} to index all dimensions of {self.shape}")

        flat_vals = (val for vals in self.vals for val in vals[loc])

        if not self._kwvals:
            # special case to avoid constructing a dictionary input
            return circuit.assign_parameters(list(flat_vals))

        parameters = dict(zip(circuit.parameters, flat_vals))
        parameters.update(
            (param, val)
            for params, vals in self._kwvals.items()
            for param, val in zip(params, vals[loc])
        )
        return circuit.assign_parameters(parameters)

    def bind_all(self, circuit: QuantumCircuit) -> np.ndarray:
        """Return an object array of bound circuits with the same shape.

        Args:
            circuit: The circuit to bind.

        Returns:
            An object array of the same shape containing all bound circuits.
        """
        arr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            arr[idx] = self.bind(circuit, idx)
        return arr

    def ravel(self) -> BindingsArray:
        """Return a new :class:`~BindingsArray` with one dimension.

        The returned bindings array has a :attr:`shape` given by ``(size, )``, where the size is the
        :attr:`~size` of this bindings array.

        Returns:
            A new bindings array.
        """
        return self.reshape(self.size)

    def reshape(self, *shape: int | Iterable[int]) -> BindingsArray:
        """Return a new :class:`~BindingsArray` with a different shape.

        This results in a new view of the same arrays.

        Args:
            shape: The shape of the returned bindings array.

        Returns:
            A new bindings array.

        Raises:
            ValueError: If the provided shape has a different product than the current size.
        """
        shape = shape_tuple(shape)
        if any(dim < 0 for dim in shape):
            # to reliably catch the ValueError, we need to manually deal with negative values
            positive_size = np.prod([dim for dim in shape if dim >= 0], dtype=int)
            missing_dim = self.size // positive_size
            shape = tuple(dim if dim >= 0 else missing_dim for dim in shape)

        if np.prod(shape, dtype=int) != self.size:
            raise ValueError("Reshaping cannot change the total number of elements.")

        vals = [val.reshape(shape + val.shape[-1:]) for val in self._vals]
        kwvals = {ps: val.reshape(shape + val.shape[-1:]) for ps, val in self._kwvals.items()}
        return BindingsArray(vals, kwvals, shape=shape)

    @classmethod
    def coerce(cls, bindings_array: BindingsArrayLike) -> BindingsArray:
        """Coerce an input that is :class:`~BindingsArrayLike` into a new :class:`~BindingsArray`.

        Args:
            bindings_array: An object to be bindings array.

        Returns:
            A new bindings array.
        """
        if isinstance(bindings_array, Sequence):
            bindings_array = np.array(bindings_array)
        if bindings_array is None:
            bindings_array = cls()
        elif isinstance(bindings_array, np.ndarray):
            bindings_array = cls(bindings_array)
        elif isinstance(bindings_array, Mapping):
            bindings_array = cls(kwvals=bindings_array)
        elif isinstance(bindings_array, BindingsArray):
            return bindings_array
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


def _standardize_shape(val: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
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
    vals: list[np.ndarray], kwvals: dict[tuple[Parameter, ...], np.ndarray]
) -> tuple[int, ...]:
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
        elif val.shape == () or val.shape == (1,) or val.shape[-1] != 1:
            # here, if the last dimension is not 1 or shape is () or (1,) then the shape is the shape
            examine_array(val.shape)
        else:
            # here, the last dimension could be over parameters or not
            examine_array(val.shape, val.shape[:-1])

    if len(vals) == 1 and len(kwvals) == 0:
        examine_array(vals[0].shape[:-1])
    elif len(vals) == 0 and len(kwvals) == 0:
        examine_array(())
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
    return tuple(map(_param_name, key))


def _param_name(param: Parameter | str) -> str:
    return param.name if isinstance(param, Parameter) else param


BindingsArrayLike = Union[
    BindingsArray,
    ArrayLike,
    "Mapping[Parameter, ArrayLike]",
    "Sequence[ArrayLike]",
    None,
]
