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
BitArray
"""

from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .shape import ShapedMixin


class BitArray(ShapedMixin):
    """Stores bit outcomes.

    This object is somewhat analagous to an object array of ``memory=True`` results.
    However, unlike an object array, all of the data is contiguous, stored in one big array.
    The last axis is over packed bits, the second last axis is over samples, and the preceding
    axes correspond to the shape of the task that was executed.
    We use the word "samples" in reference to the fact that this is a primary data type returned by
    the Sampler whose job is to supply samples, and whose name we can't easily change.
    This is certainly confusing because this library uses the word "samples" to also refer to random
    circuit instances.
    """

    def __init__(self, array: NDArray[np.uint8], num_bits: int):
        """
        Args:
            array: The data, where the last axis is over packed bits, the second last axis is over
                shots, and the preceding axes correspond to the shape of the experiment. The byte
                order is big endian.
            num_bits: How many bit are in each outcome.

        Raises:
            ValueError: If the input array has fewer than two axes, or the size of the last axis
                is not the smallest number of bytes that can contain ``num_bits``.
        """
        super().__init__()
        self._array = np.array(array, copy=False, dtype=np.uint8)
        self._num_bits = num_bits
        # second last dimension is shots/samples, last dimension is packed bits
        self._shape = self._array.shape[:-2]

        if self._array.ndim < 2:
            raise ValueError("The input array must have at least two axes.")
        if self._array.shape[-1] != (expected := num_bits // 8 + (num_bits % 8 > 0)):
            raise ValueError(f"The input array is expected to have {expected} bytes per sample.")

    def __repr__(self):
        desc = f"<num_samples={self.num_samples}, num_bits={self.num_bits}, shape={self.shape}>"
        return f"BitArray({desc})"

    @property
    def array(self) -> NDArray[np.uint8]:
        """The raw NumPy array of data."""
        return self._array

    @property
    def num_bits(self) -> int:
        """The number of bits in the register this array stores data for.

        For example, a ``ClassicalRegister(5, "meas")`` would have ``num_bits=5``.
        """
        return self._num_bits

    @property
    def num_samples(self) -> int:
        """The number of samples sampled from the register in each configuration.

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

    def _get_counts(self, *, loc: Optional[Tuple[int, ...]], converter: Callable) -> Dict[str, int]:
        if loc is None and self.size == 1:
            loc = (0,) * self.ndim

        elif loc is None:
            raise ValueError(
                f"Your BitArray has shape {self.shape}, meaning that it actually represents "
                f"{self.size} different count dictionaries. You need to use the `loc` argument of "
                "this function to pick one of them."
            )

        counts = defaultdict(int)
        for shot_row in self._array[loc]:
            counts[converter(shot_row.tobytes())] += 1
        return dict(counts)

    def get_counts(self, loc: Optional[Tuple[int, ...]] = None) -> Dict[str, int]:
        """Return a counts dictionary.

        Args:
            loc: Which entry of this array to return a dictionary for.

        Returns:
            A dictionary mapping bitstrings to the number of occurrences of that bitstring.

        Raises:
            ValueError: If this array has a non-trivial size and no ``loc`` is provided.
        """
        mask = 2**self.num_bits - 1
        converter = partial(self._bytes_to_bitstring, num_bits=self.num_bits, mask=mask)
        return self._get_counts(loc=loc, converter=converter)

    def get_int_counts(self, loc: Optional[Tuple[int, ...]] = None) -> Dict[int, int]:
        r"""Return a counts dictionary, where bitstrings are stored as ``int``\s.

        Args:
            loc: Which entry of this array to return a dictionary for.

        Returns:
            A dictionary mapping ``ints`` to the number of occurrences of that ``int``.

        Raises:
            ValueError: If this array has a non-trivial size and no ``loc`` is provided.
        """
        converter = partial(self._bytes_to_int, mask=2**self.num_bits - 1)
        return self._get_counts(loc=loc, converter=converter)
