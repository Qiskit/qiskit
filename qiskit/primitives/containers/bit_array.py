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
BitArray
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.result import Counts, sampled_expectation_value

from .shape import ShapedMixin, ShapeInput, shape_tuple

# this lookup table tells you how many bits are 1 in each uint8 value
_WEIGHT_LOOKUP = np.unpackbits(np.arange(256, dtype=np.uint8).reshape(-1, 1), axis=1).sum(axis=1)


def _min_num_bytes(num_bits: int) -> int:
    """Return the minimum number of bytes needed to store ``num_bits``."""
    return num_bits // 8 + (num_bits % 8 > 0)


def _unpack(bitarray: BitArray) -> NDArray[np.uint8]:
    arr = np.unpackbits(bitarray.array, axis=-1, bitorder="big")
    arr = arr[..., -1 : -bitarray.num_bits - 1 : -1]
    return arr


def _pack(arr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    arr = arr[..., ::-1]
    num_bits = arr.shape[-1]
    pad_size = -num_bits % 8
    if pad_size > 0:
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(pad_size, 0)]
        arr = np.pad(arr, pad_width, constant_values=0)
    arr = np.packbits(arr, axis=-1, bitorder="big")
    return arr, num_bits


class BitArray(ShapedMixin):
    """Stores an array of bit values.

    This object contains a single, contiguous block of data that represents an array of bitstrings.
    The last axis is over packed bits, the second last axis is over shots, and the preceding axes
    correspond to the shape of the pub that was executed to sample these bits.
    """

    def __init__(self, array: NDArray[np.uint8], num_bits: int):
        """
        Args:
            array: The ``uint8`` data array.
            num_bits: How many bit are in each outcome.

        Raises:
            TypeError: If the input is not a NumPy array with type ``numpy.uint8``.
            ValueError: If the input array has fewer than two axes, or the size of the last axis
                is not the smallest number of bytes that can contain ``num_bits``.
        """
        super().__init__()

        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray not {type(array)}")
        if array.dtype != np.uint8:
            raise TypeError(f"Input array must have dtype uint8, not {array.dtype}.")
        if array.ndim < 2:
            raise ValueError("The input array must have at least two axes.")
        if array.shape[-1] != (expected := _min_num_bytes(num_bits)):
            raise ValueError(f"The input array is expected to have {expected} bytes per shot.")

        self._array = array
        self._num_bits = num_bits
        # second last dimension is shots, last dimension is packed bits
        self._shape = self._array.shape[:-2]

    def _prepare_broadcastable(self, other: "BitArray") -> tuple[NDArray[np.uint8], ...]:
        """Validation and broadcasting of two bit arrays before element-wise binary operation."""
        if self.num_bits != other.num_bits:
            raise ValueError(f"'num_bits' must match in {self} and {other}.")
        self_shape = self.shape + (self.num_shots,)
        other_shape = other.shape + (other.num_shots,)
        try:
            shape = np.broadcast_shapes(self_shape, other_shape) + (self._array.shape[-1],)
        except ValueError as ex:
            raise ValueError(f"{self} and {other} are not compatible for this operation.") from ex
        return np.broadcast_to(self.array, shape), np.broadcast_to(other.array, shape)

    def __and__(self, other: "BitArray") -> "BitArray":
        return BitArray(np.bitwise_and(*self._prepare_broadcastable(other)), self.num_bits)

    def __eq__(self, other: "BitArray") -> bool:
        if (n := self.num_bits) != other.num_bits:
            return False
        arrs = [self._array, other._array]
        if n % 8 > 0:
            # ignore straggling bits on the left
            mask = np.array([255 >> ((-n) % 8)] + [255] * (n // 8), dtype=np.uint8)
            arrs = [np.bitwise_and(arr, mask) for arr in arrs]
        return np.array_equal(*arrs, equal_nan=False)

    def __invert__(self) -> "BitArray":
        return BitArray(np.bitwise_not(self._array), self.num_bits)

    def __or__(self, other: "BitArray") -> "BitArray":
        return BitArray(np.bitwise_or(*self._prepare_broadcastable(other)), self.num_bits)

    def __xor__(self, other: "BitArray") -> "BitArray":
        return BitArray(np.bitwise_xor(*self._prepare_broadcastable(other)), self.num_bits)

    def __repr__(self):
        desc = f"<shape={self.shape}, num_shots={self.num_shots}, num_bits={self.num_bits}>"
        return f"BitArray({desc})"

    def __getitem__(self, indices):
        if isinstance(indices, tuple) and len(indices) >= self.ndim + 2:
            raise ValueError("BitArrays cannot be sliced along the bits axis, see marginalize() instead.")
        return BitArray(self._array[indices], self.num_bits)

    @property
    def array(self) -> NDArray[np.uint8]:
        """The raw NumPy array of data."""
        return self._array

    @property
    def num_bits(self) -> int:
        """The number of bits in the register that this array stores data for.

        For example, a ``ClassicalRegister(5, "meas")`` would result in ``num_bits=5``.
        """
        return self._num_bits

    @property
    def num_shots(self) -> int:
        """The number of shots sampled from the register in each configuration.

        More precisely, the length of the second last axis of :attr:`~.array`.
        """
        return self._array.shape[-2]

    @staticmethod
    def _bytes_to_bitstring(data: bytes, num_bits: int, mask: int) -> str:
        val = int.from_bytes(data, "big") & mask
        return bin(val)[2:].zfill(num_bits)

    @staticmethod
    def _bytes_to_int(data: bytes, mask: int) -> int:
        return int.from_bytes(data, "big") & mask

    def _get_counts(
        self, *, loc: int | tuple[int, ...] | None, converter: Callable[[bytes], str | int]
    ) -> dict[str, int] | dict[int, int]:
        arr = self._array.reshape(-1, self._array.shape[-1]) if loc is None else self._array[loc]

        counts = defaultdict(int)
        for shot_row in arr:
            counts[converter(shot_row.tobytes())] += 1
        return dict(counts)

    def bitcount(self) -> NDArray[np.uint64]:
        """Compute the number of ones appearing in the binary representation of each shot.

        Returns:
            A ``numpy.uint64``-array with shape ``(*shape, num_shots)``.
        """
        return _WEIGHT_LOOKUP[self._array].sum(axis=-1)

    @staticmethod
    def from_bool_array(
        array: NDArray[np.bool_], order: Literal["big", "little"] = "big"
    ) -> "BitArray":
        """Construct a new bit array from an array of bools.

        Args:
            array: The array to convert, with "bitstrings" along the last axis.
            order: One of ``"big"`` or ``"little"``, indicating whether ``array[..., 0]``
                correspond to the most significant bits or the least significant bits of each
                bitstring, respectively.

        Returns:
            A new bit array.
        """
        array = np.asarray(array, dtype=bool)

        if array.ndim < 2:
            raise ValueError("Expecting at least two dimensions.")

        if order == "little":
            # np.unpackbits assumes "big"
            array = array[..., ::-1]

        num_bits = array.shape[-1]
        if remainder := (-num_bits) % 8:
            # unpackbits pads with zeros on the wrong side with respect to what we want, so
            # we manually pad to the nearest byte
            pad = np.zeros(shape_tuple(array.shape[:-1], remainder), dtype=bool)
            array = np.concatenate([pad, array], axis=-1)

        return BitArray(np.packbits(array, axis=-1), num_bits=num_bits)

    @staticmethod
    def from_counts(
        counts: Mapping[str | int, int] | Iterable[Mapping[str | int, int]],
        num_bits: int | None = None,
    ) -> "BitArray":
        """Construct a new bit array from one or more ``Counts``-like objects.

        The ``counts`` can have keys that are (uniformly) integers, hexstrings, or bitstrings.
        Their values represent numbers of occurrences of that value.

        Args:
            counts: One or more counts-like mappings with the same number of shots.
            num_bits: The desired number of bits per shot. If unset, the biggest value found sets
                this value.

        Returns:
            A new bit array with shape ``()`` for single input counts, or ``(N,)`` for an iterable
            of :math:`N` counts.

        Raises:
            ValueError: If different mappings have different numbers of shots.
            ValueError: If no counts dictionaries are supplied.
        """
        if singleton := isinstance(counts, Mapping):
            counts = [counts]
        else:
            counts = list(counts)
            if not counts:
                raise ValueError("At least one counts mapping expected.")

        counts = [
            mapping.int_outcomes() if isinstance(mapping, Counts) else mapping for mapping in counts
        ]

        data = (v for mapping in counts for vs, count in mapping.items() for v in repeat(vs, count))

        bit_array = BitArray.from_samples(data, num_bits)
        if not singleton:
            if bit_array.num_shots % len(counts) > 0:
                raise ValueError("All of your mappings need to have the same number of shots.")
            bit_array = bit_array.reshape(len(counts), bit_array.num_shots // len(counts))
        return bit_array

    @staticmethod
    def from_samples(
        samples: Iterable[str] | Iterable[int], num_bits: int | None = None
    ) -> "BitArray":
        """Construct a new bit array from an iterable of bitstrings, hexstrings, or integers.

        All samples are assumed to be integers if the first one is. Strings are all assumed to be
        bitstrings whenever the first string doesn't start with ``"0x"``.

        Consider pairing this method with :meth:`~reshape` if your samples represent nested data.

        Args:
            samples: A list of bitstrings, a list of integers, or a list of hexstrings.
            num_bits: The desired number of bits per sample. If unset, the biggest sample provided
                is used to determine this value.

        Returns:
            A new bit array.

        Raises:
            ValueError: If no strings are given.
        """
        samples = iter(samples)
        try:
            first_sample = next(samples)
        except StopIteration as ex:
            raise ValueError("At least one sample is required.") from ex

        ints = chain([first_sample], samples)
        if isinstance(first_sample, str):
            base = 16 if first_sample.startswith("0x") else 2
            ints = (int(val, base=base) for val in ints)

        if num_bits is None:
            # we are forced to prematurely look at every iterand in this case
            ints = list(ints)
            num_bits = max(map(int.bit_length, ints))

        num_bytes = _min_num_bytes(num_bits)
        data = b"".join(val.to_bytes(num_bytes, "big") for val in ints)
        array = np.frombuffer(data, dtype=np.uint8, count=len(data))
        return BitArray(array.reshape(-1, num_bytes), num_bits)

    def get_counts(self, loc: int | tuple[int, ...] | None = None) -> dict[str, int]:
        """Return a counts dictionary with bitstring keys.

        Args:
            loc: Which entry of this array to return a dictionary for. If ``None``, counts from
                all positions in this array are unioned together.

        Returns:
            A dictionary mapping bitstrings to the number of occurrences of that bitstring.
        """
        mask = 2**self.num_bits - 1
        converter = partial(self._bytes_to_bitstring, num_bits=self.num_bits, mask=mask)
        return self._get_counts(loc=loc, converter=converter)

    def get_int_counts(self, loc: int | tuple[int, ...] | None = None) -> dict[int, int]:
        r"""Return a counts dictionary, where bitstrings are stored as ``int``\s.

        Args:
            loc: Which entry of this array to return a dictionary for. If ``None``, counts from
                all positions in this array are unioned together.

        Returns:
            A dictionary mapping ``ints`` to the number of occurrences of that ``int``.

        """
        converter = partial(self._bytes_to_int, mask=2**self.num_bits - 1)
        return self._get_counts(loc=loc, converter=converter)

    def get_bitstrings(self, loc: int | tuple[int, ...] | None = None) -> list[str]:
        """Return a list of bitstrings.

        Args:
            loc: Which entry of this array to return a dictionary for. If ``None``, counts from
                all positions in this array are unioned together.

        Returns:
            A list of bitstrings.
        """
        mask = 2**self.num_bits - 1
        converter = partial(self._bytes_to_bitstring, num_bits=self.num_bits, mask=mask)
        arr = self._array.reshape(-1, self._array.shape[-1]) if loc is None else self._array[loc]
        return [converter(shot_row.tobytes()) for shot_row in arr]

    def reshape(self, *shape: ShapeInput) -> "BitArray":
        """Return a new reshaped bit array.

        The :attr:`~num_shots` axis is either included or excluded from the reshaping procedure
        depending on which picture the new shape is compatible with. For example, for a bit array
        with shape ``(20, 5)`` and ``64`` shots, a reshape to ``(100,)`` would leave the
        number of shots intact, whereas a reshape to ``(200, 32)`` would change the number of
        shots to ``32``.

        Args:
            *shape: The new desired shape.

        Returns:
            A new bit array.

        Raises:
            ValueError: If the size corresponding to your new shape is not equal to either
                :attr:`~size`, or the product of :attr:`~size` and :attr:`~num_shots`.
        """
        shape = shape_tuple(shape)
        if (size := np.prod(shape, dtype=int)) == self.size:
            shape = shape_tuple(shape, self._array.shape[-2:])
        elif size == self.size * self.num_shots:
            shape = shape_tuple(shape, self._array.shape[-1:])
        else:
            raise ValueError("Cannot change the size of the array.")
        return BitArray(self._array.reshape(shape), self.num_bits)

    def transpose(self, axes: tuple[int, ...] | list[int] | None = None) -> "BitArray":
        """Return a bit array with axes transposed.

        Args:
            axes: Tuple or list of ints, optional. See
                `numpy.transpose
                <https://numpy.org/doc/stable/reference/generated/numpy.transpose.html>`_
                for the details.

        Returns:
            BitArray: A bit array with axes permuted.

        Raises:
            ValueError: If ``axes`` don't match this bit array.
            ValueError: If ``axes`` includes any indices that are out of bounds.
        """
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        if len(axes) != self.ndim:
            raise ValueError("axes don't match bit array")
        for i in axes:
            if i >= self.ndim or self.ndim + i < 0:
                raise ValueError(
                    f"axis {i} is out of bounds for bit array of dimension {self.ndim}."
                )
        axes = tuple(i if i >= 0 else self.ndim + i for i in axes) + (-2, -1)
        return BitArray(self._array.transpose(axes), self.num_bits)

    def marginalize(self, indices: int | Sequence[int]) -> "BitArray":
        """Return a bit array marginalized over some indices of interest.

        .. note::
            The convention used by this method is that the index ``0`` corresponds to the least-significant 
            bit in the :attr:`~array`, or equivalently the right-most bitstring entry as returned by 
            :meth:`~get_counts` or :meth:`~get_bitstrings`, etc.

            If this bit array was produced by a sampler, then an index ``i`` corresponds to the   
            :class:`~.ClassicalRegister` location ``creg[i]]`.
        Args:
            indices: The bit positions of interest to marginalize over.

        Returns:
            A bit array marginalized to bits of interests.

        Raises:
            ValueError: If there are any invalid indices to marginalize over.
        """
        if isinstance(indices, int):
            indices = (indices,)
        for index in indices:
            if index < 0 or index >= self.num_bits:
                raise ValueError(
                    f"index {index} is out of bounds for the number of bits {self.num_bits}."
                )
        # This implementation introduces a temporary 8x memory overhead due to bit 
        # unpacking. This could be fixed using bitwise functions, at the expense of a
        # more complicated implementation.
        arr = _unpack(self)
        arr = arr[..., indices]
        arr, num_bits = _pack(arr)
        return BitArray(arr, num_bits)

    def expectation_value(
        self, operator: str | Pauli | SparsePauliOp, loc: int | tuple[int, ...] | None = None
    ) -> float:
        """Compute the expectation value of an operator.

        .. note:
            This method returns the real part of the expectation value even if
            the operator has complex coefficients due to the specification of
            :func:~.sampled_expval_complex.

        Args:
            operator: The operator for the expectation value
            loc: Which entry of this array to compute the expectation value.
                If ``None``, counts from all positions in this array are unioned together.

        Returns:
            The expectation value.
        """
        counts = self.get_counts(loc)
        return sampled_expectation_value(counts, operator)

    @staticmethod
    def concatenate(bitarrays: Sequence[BitArray], axis: int = 0) -> BitArray:
        """Join a sequence of bit arrays along an existing axis.

        Args:
            bitarrays: The bit arrays must have (1) the same number of bits,
                (2) the same number of shots, and
                (3) the same shape, except in the dimension corresponding to axis
                (the first, by default).
            axis: The axis along which the arrays will be joined. Default is 0.

        Returns:
            BitArray: The concatenated bit array.

        Raises:
            ValueError: if the sequence of bit arrays is empty.
            ValueError: if any bit arrays has a different number of bits.
            ValueError: if any bit arrays has a different number of shots.
            ValueError: if any bit arrays has a different number of dimensions.
        """
        if len(bitarrays) == 0:
            raise ValueError("Need at least one bit array to concatenate")
        num_bits = bitarrays[0].num_bits
        num_shots = bitarrays[0].num_shots
        ndim = bitarrays[0].ndim
        for i, ba in enumerate(bitarrays):
            if ba.num_bits != num_bits:
                raise ValueError(
                    "All bit arrays must have same number of bits, "
                    f"but the bit array at index 0 has {num_bits} bits "
                    f"and the bit array at index {i} has {ba.num_bits} bits."
                )
            if ba.num_shots != num_shots:
                raise ValueError(
                    "All bit arrays must have same number of shots, "
                    f"but the bit array at index 0 has {num_shots} shots "
                    f"and the bit array at index {i} has {ba.num_shots} shots."
                )
            if ba.ndim != ndim:
                raise ValueError(
                    "All bit arrays must have same number of dimensions, "
                    f"but the bit array at index 0 has {ndim} dimension(s) "
                    f"and the bit array at index {i} has {ba.ndim} dimension(s)."
                )
        if axis is None:
            raise ValueError(f"axis {axis} must be a non-negative integer.")
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {axis} is out of bounds for bit array of dimension {ndim}.")
        data = np.concatenate([ba.array for ba in bitarrays], axis=axis)
        return BitArray(data, num_bits)

    @staticmethod
    def stack_shots(bitarrays: Sequence[BitArray]) -> BitArray:
        """Join a sequence of bit arrays along shots.

        Args:
            bitarrays: The bit arrays must have (1) the same number of bits,
                and (2) the same shape.

        Returns:
            BitArray: The stacked bit array.

        Raises:
            ValueError: if the sequence of bit arrays is empty.
            ValueError: if any bit arrays has a different number of bits.
            ValueError: if any bit arrays has a different shape.
        """
        if len(bitarrays) == 0:
            raise ValueError("Need at least one bit array to stack")
        num_bits = bitarrays[0].num_bits
        shape = bitarrays[0].shape
        for i, ba in enumerate(bitarrays):
            if ba.num_bits != num_bits:
                raise ValueError(
                    "All bit arrays must have same number of bits, "
                    f"but the bit array at index 0 has {num_bits} bits "
                    f"and the bit array at index {i} has {ba.num_bits} bits."
                )
            if ba.shape != shape:
                raise ValueError(
                    "All bit arrays must have same shape, "
                    f"but the bit array at index 0 has shape {shape} "
                    f"and the bit array at index {i} has shape {ba.shape}."
                )
        data = np.concatenate([ba.array for ba in bitarrays], axis=-2)
        return BitArray(data, num_bits)

    @staticmethod
    def stack_bits(bitarrays: Sequence[BitArray]) -> BitArray:
        """Join a sequence of bit arrays along bits.

        Args:
            bitarrays: The bit arrays must have (1) the same number of shots,
                and (2) the same shape.

        Returns:
            BitArray: The stacked bit array.

        Raises:
            ValueError: if the sequence of bit arrays is empty.
            ValueError: if any bit arrays has a different number of shots.
            ValueError: if any bit arrays has a different shape.
        """
        if len(bitarrays) == 0:
            raise ValueError("Need at least one bit array to stack")
        num_shots = bitarrays[0].num_shots
        shape = bitarrays[0].shape
        for i, ba in enumerate(bitarrays):
            if ba.num_shots != num_shots:
                raise ValueError(
                    "All bit arrays must have same number of shots, "
                    f"but the bit array at index 0 has {num_shots} shots "
                    f"and the bit array at index {i} has {ba.num_shots} shots."
                )
            if ba.shape != shape:
                raise ValueError(
                    "All bit arrays must have same shape, "
                    f"but the bit array at index 0 has shape {shape} "
                    f"and the bit array at index {i} has shape {ba.shape}."
                )
        data = np.concatenate([_unpack(ba) for ba in bitarrays], axis=-1)
        data, num_bits = _pack(data)
        return BitArray(data, num_bits)
