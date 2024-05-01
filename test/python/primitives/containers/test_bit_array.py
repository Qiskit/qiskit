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

"""Unit tests for BitArray."""

from itertools import product

import ddt
import numpy as np

from qiskit.primitives.containers import BitArray
from qiskit.result import Counts
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def u_8(arr):
    """Convenience function to instantiate a ``numpy.uint8`` array with few characters."""
    return np.array(arr, dtype=np.uint8)


@ddt.ddt
class BitArrayTestCase(QiskitTestCase):
    """Test the DataBin class."""

    @ddt.idata(product([(), (3, 4, 5)], [1, 6], [8, 13, 26]))
    @ddt.unpack
    def test_container(self, shape, num_shots, num_bits):
        """Test the constructor and basic attributes."""
        num_bytes = num_bits // 8 + (num_bits % 8 > 0)
        size = np.prod(shape).astype(int) * num_shots * num_bytes
        arr = np.arange(size, dtype=np.uint8).reshape(shape + (num_shots, num_bytes))

        bit_array = BitArray(arr, num_bits)
        self.assertEqual(bit_array.shape, shape)
        self.assertEqual(bit_array.size, np.prod(shape).astype(int))
        self.assertEqual(bit_array.ndim, len(shape))
        self.assertEqual(bit_array.num_shots, num_shots)
        self.assertEqual(bit_array.num_bits, num_bits)
        self.assertTrue(np.all(bit_array.array == arr))
        self.assertEqual(bit_array.array.shape[-1], num_bytes)

    def test_constructor_exceptions(self):
        """Test the constructor raises exceptions properly."""
        with self.assertRaisesRegex(TypeError, "must be a numpy.ndarray"):
            BitArray([], 1)

        with self.assertRaisesRegex(TypeError, "must have dtype uint8"):
            BitArray(np.zeros(1, dtype=float), 1)

        with self.assertRaisesRegex(ValueError, "at least two axes"):
            BitArray(np.zeros((1,), dtype=np.uint8), 1)

        with self.assertRaisesRegex(ValueError, "3 bytes per shot"):
            BitArray(np.empty((2, 3, 4, 5), dtype=np.uint8), 23)

    def test_get_counts(self):
        """Test conversion to counts."""
        # note that [234, 100] requires 16 bits, not 15; we are testing that get_counts ignores the
        # junk columns
        bit_array = BitArray(u_8([[3, 5], [3, 5], [234, 100]]), num_bits=15)
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_counts(), {bs1: 2, bs2: 1})

        bit_array = BitArray(
            u_8([[[3, 5], [3, 5], [234, 100]], [[0, 1], [1, 0], [1, 0]]]),
            num_bits=15,
        )
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_counts(0), {bs1: 2, bs2: 1})
        bs3 = "0000000" + "00000001"  # 0, 1
        bs4 = "0000001" + "00000000"  # 1, 0
        self.assertEqual(bit_array.get_counts(1), {bs3: 1, bs4: 2})

        # test that providing no location takes the union over all shots
        self.assertEqual(bit_array.get_counts(), {bs1: 2, bs2: 1, bs3: 1, bs4: 2})

        bit_array = BitArray(np.zeros([1024, 2], dtype=np.uint8), num_bits=16)
        bs5 = "00000000" + "00000000"
        self.assertEqual(bit_array.get_counts(), {bs5: 1024})
        self.assertEqual(bit_array.get_counts(1), {bs5: 2})
        self.assertEqual(bit_array.get_counts((0, 1)), {bs5: 1})

        bit_array = BitArray(np.zeros([1024, 1], dtype=np.uint8), num_bits=8)
        bs6 = "00000000"
        self.assertEqual(bit_array.get_counts(), {bs6: 1024})
        self.assertEqual(bit_array.get_counts(1), {bs6: 1})
        with self.assertRaises(IndexError):
            bit_array.get_counts((0, 1))

        # test with no classical register
        bit_array = BitArray(np.zeros([1024, 0], dtype=np.uint8), num_bits=0)
        self.assertEqual(bit_array.get_counts(), {"": 1024})
        self.assertEqual(bit_array.get_counts(1), {})
        with self.assertRaises(IndexError):
            bit_array.get_counts((0, 1))

    def test_get_int_counts(self):
        """Test conversion to int counts."""
        # note that [234, 100] requires 16 bits, not 15; we are testing that get_counts ignores the
        # junk columns
        bit_array = BitArray(u_8([[3, 5], [3, 5], [234, 100]]), num_bits=15)
        val1 = (3 << 8) + 5
        val2 = ((234 & 127) << 8) + 100
        self.assertEqual(bit_array.get_int_counts(), {val1: 2, val2: 1})

        bit_array = BitArray(
            u_8([[[3, 5], [3, 5], [234, 100]], [[0, 1], [1, 0], [1, 0]]]),
            num_bits=15,
        )
        val1 = (3 << 8) + 5
        val2 = ((234 & 127) << 8) + 100
        self.assertEqual(bit_array.get_int_counts(0), {val1: 2, val2: 1})
        val3 = 1
        val4 = 1 << 8
        self.assertEqual(bit_array.get_int_counts(1), {val3: 1, val4: 2})

        # test that providing no location takes the union over all shots
        self.assertEqual(bit_array.get_int_counts(), {val1: 2, val2: 1, val3: 1, val4: 2})

        bit_array = BitArray(np.zeros([1024, 2], dtype=np.uint8), num_bits=16)
        val5 = 0
        self.assertEqual(bit_array.get_int_counts(), {val5: 1024})
        self.assertEqual(bit_array.get_int_counts(1), {val5: 2})
        self.assertEqual(bit_array.get_int_counts((0, 1)), {val5: 1})

        bit_array = BitArray(np.zeros([1024, 1], dtype=np.uint8), num_bits=8)
        self.assertEqual(bit_array.get_int_counts(), {0: 1024})
        self.assertEqual(bit_array.get_int_counts(1), {0: 1})
        with self.assertRaises(IndexError):
            bit_array.get_int_counts((0, 1))

        # test with no classical register
        bit_array = BitArray(np.zeros([1024, 0], dtype=np.uint8), num_bits=0)
        self.assertEqual(bit_array.get_int_counts(), {0: 1024})
        self.assertEqual(bit_array.get_int_counts(1), {})
        with self.assertRaises(IndexError):
            bit_array.get_int_counts((0, 1))

    def test_get_bitstrings(self):
        """Test conversion to bitstrings."""
        # note that [234, 100] requires 16 bits, not 15; we are testing that get_counts ignores the
        # junk columns
        bit_array = BitArray(u_8([[3, 5], [3, 5], [234, 100]]), num_bits=15)
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_bitstrings(), [bs1, bs1, bs2])

        bit_array = BitArray(
            u_8([[[3, 5], [3, 5], [234, 100]], [[0, 1], [1, 0], [1, 0]]]),
            num_bits=15,
        )
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_bitstrings(0), [bs1, bs1, bs2])
        bs3 = "0000000" + "00000001"  # 0, 1
        bs4 = "0000001" + "00000000"  # 1, 0
        self.assertEqual(bit_array.get_bitstrings(1), [bs3, bs4, bs4])

        # test that providing no location takes the union over all shots
        self.assertEqual(bit_array.get_bitstrings(), [bs1, bs1, bs2, bs3, bs4, bs4])

        bit_array = BitArray(np.zeros([16, 1], dtype=np.uint8), num_bits=8)
        bs5 = ["00000000"] * 16
        self.assertEqual(bit_array.get_bitstrings(), bs5)
        self.assertEqual(bit_array.get_bitstrings(1), ["00000000"])
        with self.assertRaises(IndexError):
            bit_array.get_bitstrings((0, 1))

        bit_array = BitArray(np.zeros([16, 2], dtype=np.uint8), num_bits=16)
        bs6 = ["0000000000000000"] * 16
        self.assertEqual(bit_array.get_bitstrings(), bs6)
        self.assertEqual(bit_array.get_bitstrings(1), ["0000000000000000", "0000000000000000"])
        self.assertEqual(bit_array.get_bitstrings((0, 1)), ["0000000000000000"])

        # test with no classical register
        bit_array = BitArray(np.zeros([16, 0], dtype=np.uint8), num_bits=0)
        bs7 = [""] * 16
        self.assertEqual(bit_array.get_bitstrings(), bs7)
        self.assertEqual(bit_array.get_bitstrings(0), [""])
        with self.assertRaises(IndexError):
            bit_array.get_bitstrings((0, 1))

    def test_equality(self):
        """Test the equality operator"""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba3 = BitArray.from_bool_array([[1, 1, 0], [1, 1, 0]])
        ba4 = BitArray.from_bool_array([[[1, 0, 0], [1, 1, 0]]])
        self.assertEqual(ba1, ba1)
        self.assertEqual(ba1, ba2)
        self.assertNotEqual(ba1, ba3)
        self.assertNotEqual(ba1, ba4)

        ba5 = BitArray(u_8([[4, 200], [255, 10]]), num_bits=13)
        ba6 = BitArray(u_8([[4, 200], [255, 10]]), num_bits=12)
        ba7 = BitArray(u_8([[4, 200], [31, 10]]), num_bits=13)
        self.assertNotEqual(ba5, ba6)
        self.assertEqual(ba5, ba7)  # test masking

    def test_logical_and(self):
        """Test the logical AND operator."""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 1], [0, 1, 1]])
        self.assertEqual(ba1 & ba2, BitArray.from_bool_array([[1, 0, 0], [0, 1, 0]]))

        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 0]])
        self.assertEqual(ba1 & ba2, BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]]))

    def test_logical_or(self):
        """Test the logical OR operator."""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 0, 1], [0, 1, 0]])
        self.assertEqual(ba1 | ba2, BitArray.from_bool_array([[1, 0, 1], [1, 1, 0]]))

        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 0]])
        self.assertEqual(ba1 | ba2, BitArray.from_bool_array([[1, 1, 0], [1, 1, 0]]))

    def test_logical_not(self):
        """Test the logical OR operator."""
        ba = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        self.assertEqual(~ba, BitArray.from_bool_array([[0, 1, 1], [0, 0, 1]]))

    def test_logical_xor(self):
        """Test the logical XOR operator."""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 0, 1], [0, 1, 0]])
        self.assertEqual(ba1 ^ ba2, BitArray.from_bool_array([[0, 0, 1], [1, 0, 0]]))

        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 0]])
        self.assertEqual(ba1 ^ ba2, BitArray.from_bool_array([[0, 1, 0], [0, 0, 0]]))

    def test_from_bool_array(self):
        """Test the from_bool_array static_constructor."""

        bit_array = BitArray.from_bool_array(
            [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]]
        )
        self.assertEqual(bit_array, BitArray(u_8([[[10], [3]], [[8], [1]]]), 4))

        bit_array = BitArray.from_bool_array(
            [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]], order="little"
        )
        self.assertEqual(bit_array, BitArray(u_8([[[5], [12]], [[1], [8]]]), 4))

        bit_array = BitArray.from_bool_array(
            [[0, 0, 1, 1, 1] + [0, 0, 0, 0, 0, 0, 1, 1] + [0, 0, 0, 0, 0, 0, 0, 1]]
        )
        self.assertEqual(bit_array, BitArray(u_8([[7, 3, 1]]), 21))

        bit_array = BitArray.from_bool_array(
            [[1, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0]], order="little"
        )
        self.assertEqual(bit_array, BitArray(u_8([[7, 3, 1]]), 21))

    @ddt.data("counts", "int", "hex", "bit")
    def test_from_counts(self, counts_type):
        """Test the from_counts static constructor."""

        def convert(counts: Counts):
            if counts_type == "int":
                return counts.int_outcomes()
            if counts_type == "hex":
                return counts.hex_outcomes()
            if counts_type == "bit":
                return {bin(val): count for val, count in counts.int_outcomes().items()}
            return counts

        counts1 = convert(Counts({"0b101010": 2, "0b1": 3, "0x010203": 4}))
        counts2 = convert(Counts({1: 3, 2: 6}))

        bit_array = BitArray.from_counts(counts1)
        expected = BitArray(u_8([[0, 0, 42]] * 2 + [[0, 0, 1]] * 3 + [[1, 2, 3]] * 4), 17)
        self.assertEqual(bit_array, expected)

        bit_array = BitArray.from_counts(iter([counts1]))
        expected = BitArray(u_8([[[0, 0, 42]] * 2 + [[0, 0, 1]] * 3 + [[1, 2, 3]] * 4]), 17)
        self.assertEqual(bit_array, expected)

        bit_array = BitArray.from_counts(iter([counts1, counts2]))
        expected = [
            [[0, 0, 42]] * 2 + [[0, 0, 1]] * 3 + [[1, 2, 3]] * 4,
            [[0, 0, 1]] * 3 + [[0, 0, 2]] * 6,
        ]
        self.assertEqual(bit_array, BitArray(u_8(expected), 17))

    def test_from_samples_bitstring(self):
        """Test the from_samples static constructor."""
        bit_array = BitArray.from_samples(["110", "1", "1111111111"])
        self.assertEqual(bit_array, BitArray(u_8([[0, 6], [0, 1], [3, 255]]), 10))

        bit_array = BitArray.from_samples(["110", "1", "1111111111"], 20)
        self.assertEqual(bit_array, BitArray(u_8([[0, 0, 6], [0, 0, 1], [0, 3, 255]]), 20))

    def test_from_samples_hex(self):
        """Test the from_samples static constructor."""
        bit_array = BitArray.from_samples(["0x01", "0x0a12", "0x0105"])
        self.assertEqual(bit_array, BitArray(u_8([[0, 1], [10, 18], [1, 5]]), 12))

        bit_array = BitArray.from_samples(["0x01", "0x0a12", "0x0105"], 20)
        self.assertEqual(bit_array, BitArray(u_8([[0, 0, 1], [0, 10, 18], [0, 1, 5]]), 20))

    def test_from_samples_int(self):
        """Test the from_samples static constructor."""
        bit_array = BitArray.from_samples([1, 2578, 261])
        self.assertEqual(bit_array, BitArray(u_8([[0, 1], [10, 18], [1, 5]]), 12))

        bit_array = BitArray.from_samples([1, 2578, 261], 20)
        self.assertEqual(bit_array, BitArray(u_8([[0, 0, 1], [0, 10, 18], [0, 1, 5]]), 20))

    def test_reshape(self):
        """Test the reshape method."""
        # this creates incrementing bitstrings from 0 to 360 * 32 - 1
        data = np.frombuffer(np.arange(360 * 32, dtype=np.uint64).tobytes(), dtype=np.uint8)
        data = data.reshape(-1, 32, 8)[..., 1::-1]
        ba = BitArray(data, 15)

        self.assertEqual(ba.reshape(120, 3).shape, (120, 3))
        self.assertEqual(ba.reshape(120, 3).num_shots, 32)
        self.assertEqual(ba.reshape(120, 3).num_bits, 15)
        self.assertTrue(
            np.array_equal(ba.reshape(60, 6).array[2, 3], data.reshape(60, 6, 32, 2)[2, 3])
        )

        self.assertEqual(ba.reshape(360 * 32).shape, ())
        self.assertEqual(ba.reshape(360 * 32).num_shots, 360 * 32)
        self.assertEqual(ba.reshape(360 * 32).num_bits, 15)

        self.assertEqual(ba.reshape(360 * 2, 16).shape, (720,))
        self.assertEqual(ba.reshape(360 * 2, 16).num_shots, 16)
        self.assertEqual(ba.reshape(360 * 2, 16).num_bits, 15)
