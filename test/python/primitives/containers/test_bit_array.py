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
from test import QiskitTestCase

import ddt
import numpy as np

from qiskit.primitives.containers import BitArray
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.result import Counts


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

    def test_transpose(self):
        """Test the transpose method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        # Since the input dtype is uint16, bit array requires at least two u8.
        # Thus, 9 is the minimum number of qubits, i.e., 8 + 1.
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("default arg"):
            ba2 = ba.transpose()
            self.assertEqual(ba2.shape, (3, 2, 1))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((k, j, i)))

        with self.subTest("tuple 1"):
            ba2 = ba.transpose((2, 1, 0))
            self.assertEqual(ba2.shape, (3, 2, 1))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((k, j, i)))

        with self.subTest("tuple 2"):
            ba2 = ba.transpose((0, 1, 2))
            self.assertEqual(ba2.shape, (1, 2, 3))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("tuple 3"):
            ba2 = ba.transpose((0, 2, 1))
            self.assertEqual(ba2.shape, (1, 3, 2))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, k, j)))

        with self.subTest("tuple, negative indices"):
            ba2 = ba.transpose((0, -1, -2))
            self.assertEqual(ba2.shape, (1, 3, 2))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, k, j)))

        with self.subTest("ints"):
            ba2 = ba.transpose(2, 1, 0)
            self.assertEqual(ba2.shape, (3, 2, 1))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((k, j, i)))

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "axes don't match bit array"):
                _ = ba.transpose((0, 1))
            with self.assertRaisesRegex(ValueError, "axes don't match bit array"):
                _ = ba.transpose((0, 1, 2, 3))
            with self.assertRaisesRegex(ValueError, "axis [0-9]+ is out of bounds for bit array"):
                _ = ba.transpose((0, 1, 4))
            with self.assertRaisesRegex(ValueError, "axis -[0-9]+ is out of bounds for bit array"):
                _ = ba.transpose((0, 1, -4))
            with self.assertRaisesRegex(ValueError, "repeated axis in transpose"):
                _ = ba.transpose((0, 1, 1))

    def test_concatenate(self):
        """Test the concatenate function."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        concatenate = BitArray.concatenate

        with self.subTest("2 arrays, default"):
            ba2 = concatenate([ba, ba])
            self.assertEqual(ba2.shape, (2, 2, 3))
            for j, k in product(range(2), range(3)):
                self.assertEqual(ba2.get_counts((0, j, k)), ba2.get_counts((1, j, k)))

        with self.subTest("2 arrays, axis"):
            ba2 = concatenate([ba, ba], axis=1)
            self.assertEqual(ba2.shape, (1, 4, 3))
            for j, k in product(range(2), range(3)):
                self.assertEqual(ba2.get_counts((0, j, k)), ba2.get_counts((0, j + 2, k)))

        with self.subTest("3 arrays"):
            ba2 = concatenate([ba, ba, ba])
            self.assertEqual(ba2.shape, (3, 2, 3))
            for j, k in product(range(2), range(3)):
                self.assertEqual(ba2.get_counts((0, j, k)), ba2.get_counts((1, j, k)))
                self.assertEqual(ba2.get_counts((1, j, k)), ba2.get_counts((2, j, k)))

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "Need at least one bit array to concatenate"):
                _ = concatenate([])
            with self.assertRaisesRegex(ValueError, "axis -1 is out of bounds"):
                _ = concatenate([ba, ba], -1)
            with self.assertRaisesRegex(ValueError, "axis 100 is out of bounds"):
                _ = concatenate([ba, ba], 100)

            ba2 = BitArray(data, 10)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same number of bits"):
                _ = concatenate([ba, ba2])

            data2 = np.frombuffer(np.arange(30, dtype=np.uint16).tobytes(), dtype=np.uint8)
            data2 = data2.reshape(1, 2, 3, 5, 2)[..., ::-1]
            ba2 = BitArray(data2, 9)
            with self.assertRaisesRegex(
                ValueError, "All bit arrays must have same number of shots"
            ):
                _ = concatenate([ba, ba2])

            ba2 = ba.reshape(2, 3)
            with self.assertRaisesRegex(
                ValueError, "All bit arrays must have same number of dimensions"
            ):
                _ = concatenate([ba, ba2])

    def test_concatenate_shots(self):
        """Test the concatenate_shots function."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        concatenate_shots = BitArray.concatenate_shots

        with self.subTest("2 arrays"):
            ba2 = concatenate_shots([ba, ba])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 9)
            self.assertEqual(ba2.num_shots, 2 * ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                expected = {key: val * 2 for key, val in ba.get_counts((i, j, k)).items()}
                counts2 = ba2.get_counts((i, j, k))
                self.assertEqual(counts2, expected)

        with self.subTest("3 arrays"):
            ba2 = concatenate_shots([ba, ba, ba])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 9)
            self.assertEqual(ba2.num_shots, 3 * ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                expected = {key: val * 3 for key, val in ba.get_counts((i, j, k)).items()}
                counts2 = ba2.get_counts((i, j, k))
                self.assertEqual(counts2, expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "Need at least one bit array to stack"):
                _ = concatenate_shots([])

            ba2 = BitArray(data, 10)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same number of bits"):
                _ = concatenate_shots([ba, ba2])

            ba2 = ba.reshape(2, 3)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same shape"):
                _ = concatenate_shots([ba, ba2])

    def test_concatenate_bits(self):
        """Test the concatenate_bits function."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        concatenate_bits = BitArray.concatenate_bits

        with self.subTest("2 arrays"):
            ba_01 = ba.slice_bits([0, 1])
            ba2 = concatenate_bits([ba, ba_01])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 11)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                bs = ba.get_bitstrings((i, j, k))
                bs_01 = ba_01.get_bitstrings((i, j, k))
                expected = [s1 + s2 for s1, s2 in zip(bs_01, bs)]
                bs2 = ba2.get_bitstrings((i, j, k))
                self.assertEqual(bs2, expected)

        with self.subTest("3 arrays"):
            ba_01 = ba.slice_bits([0, 1])
            ba2 = concatenate_bits([ba, ba_01, ba_01])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 13)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                bs = ba.get_bitstrings((i, j, k))
                bs_01 = ba_01.get_bitstrings((i, j, k))
                expected = [s1 + s1 + s2 for s1, s2 in zip(bs_01, bs)]
                bs2 = ba2.get_bitstrings((i, j, k))
                self.assertEqual(bs2, expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "Need at least one bit array to stack"):
                _ = concatenate_bits([])

            data2 = np.frombuffer(np.arange(30, dtype=np.uint16).tobytes(), dtype=np.uint8)
            data2 = data2.reshape(1, 2, 3, 5, 2)[..., ::-1]
            ba2 = BitArray(data2, 9)
            with self.assertRaisesRegex(
                ValueError, "All bit arrays must have same number of shots"
            ):
                _ = concatenate_bits([ba, ba2])

            ba2 = ba.reshape(2, 3)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same shape"):
                _ = concatenate_bits([ba, ba2])

    def test_getitem(self):
        """Test the __getitem__ method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("all"):
            ba2 = ba[:]
            self.assertEqual(ba2.shape, (1, 2, 3))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("no slice"):
            ba2 = ba[0, 1, 2]
            self.assertEqual(ba2.shape, ())
            self.assertEqual(ba.get_counts((0, 1, 2)), ba2.get_counts())

        with self.subTest("slice"):
            ba2 = ba[0, :, 2]
            self.assertEqual(ba2.shape, (2,))
            for j in range(2):
                self.assertEqual(ba.get_counts((0, j, 2)), ba2.get_counts(j))

    def test_slice_bits(self):
        """Test the slice_bits method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("all"):
            ba2 = ba.slice_bits(range(ba.num_bits))
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("1 bit, int"):
            ba2 = ba.slice_bits(0)
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_counts((i, j, k)), {"0": 5, "1": 5})

        with self.subTest("1 bit, list"):
            ba2 = ba.slice_bits([0])
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_counts((i, j, k)), {"0": 5, "1": 5})

        with self.subTest("2 bits"):
            ba2 = ba.slice_bits([0, 1])
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, 2)
            even = {"00": 3, "01": 3, "10": 2, "11": 2}
            odd = {"10": 3, "11": 3, "00": 2, "01": 2}
            for count, (i, j, k) in enumerate(product(range(1), range(2), range(3))):
                expect = even if count % 2 == 0 else odd
                self.assertEqual(ba2.get_counts((i, j, k)), expect)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "index -1 is out of bounds"):
                _ = ba.slice_bits(-1)
            with self.assertRaisesRegex(ValueError, "index 9 is out of bounds"):
                _ = ba.slice_bits(9)

    def test_slice_shots(self):
        """Test the slice_shots method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("all"):
            ba2 = ba.slice_shots(range(ba.num_shots))
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("1 shot, int"):
            ba2 = ba.slice_shots(0)
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_bitstrings((i, j, k)), [ba.get_bitstrings((i, j, k))[0]])

        with self.subTest("1 shot, list"):
            ba2 = ba.slice_shots([0])
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_bitstrings((i, j, k)), [ba.get_bitstrings((i, j, k))[0]])

        with self.subTest("multiple shots"):
            indices = [1, 2, 3, 5, 8]
            ba2 = ba.slice_shots(indices)
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, len(indices))
            for i, j, k in product(range(1), range(2), range(3)):
                expected = [
                    bs for ind, bs in enumerate(ba.get_bitstrings((i, j, k))) if ind in indices
                ]
                self.assertEqual(ba2.get_bitstrings((i, j, k)), expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "index -1 is out of bounds"):
                _ = ba.slice_shots(-1)
            with self.assertRaisesRegex(ValueError, "index 10 is out of bounds"):
                _ = ba.slice_shots(10)

    def test_expectation_values(self):
        """Test the expectation_values method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        op = "I" * 8 + "Z"
        op2 = "I" * 8 + "0"
        op3 = "I" * 8 + "1"
        pauli = Pauli(op)
        sp_op = SparsePauliOp(op)
        sp_op2 = SparsePauliOp.from_sparse_list([("Z", [6], 1)], num_qubits=9)

        with self.subTest("str"):
            expval = ba.expectation_values(op)
            # both 0 and 1 appear 5 times
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.zeros((ba.shape)))

            expval = ba.expectation_values(op2)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.full((ba.shape), 0.5))

            expval = ba.expectation_values(op3)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.full((ba.shape), 0.5))

            ba2 = ba.slice_bits(6)
            # 6th bit are all 0
            expval = ba2.expectation_values("Z")
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.ones(ba.shape))

            ba3 = ba.slice_bits(5)
            # 5th bit distributes as follows.
            # (0, 0, 0) {'0': 10}
            # (0, 0, 1) {'0': 10}
            # (0, 0, 2) {'0': 10}
            # (0, 1, 0) {'0': 2, '1': 8}
            # (0, 1, 1) {'1': 10}
            # (0, 1, 2) {'1': 10}
            expval = ba3.expectation_values("0")
            expected = np.array([[[1, 1, 1], [0.2, 0, 0]]])
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, expected)

        with self.subTest("Pauli"):
            expval = ba.expectation_values(pauli)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.zeros((ba.shape)))

        with self.subTest("SparsePauliOp"):
            expval = ba.expectation_values(sp_op)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.zeros((ba.shape)))

            expval = ba.expectation_values(sp_op2)
            # 6th bit are all 0
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.ones((ba.shape)))

        with self.subTest("ObservableArray"):
            obs = ["Z", "0", "1"]
            ba2 = ba.slice_bits(5)
            expval = ba2.expectation_values(obs)
            expected = np.array([[[1, 1, 0], [-0.6, 0, 1]]])
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, expected)

            ba4 = BitArray.from_counts([{0: 1}, {1: 1}]).reshape(2, 1)
            expval = ba4.expectation_values(obs)
            expected = np.array([[1, 1, 0], [-1, 0, 1]])
            self.assertEqual(expval.shape, (2, 3))
            np.testing.assert_allclose(expval, expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                _ = ba.expectation_values([op, op2])
            with self.assertRaisesRegex(ValueError, "One or more operators not same length"):
                _ = ba.expectation_values("Z")
            with self.assertRaisesRegex(ValueError, "is not diagonal"):
                _ = ba.expectation_values("X" * ba.num_bits)

    def test_postselection(self):
        """Test the postselection method. (Commented code was used to generate test data)."""
        # import random
        # import numpy as np
        # from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        # from qiskit_aer import AerSimulator
        # from qiskit_ibm_runtime import SamplerV2
        # from qiskit.primitives.containers.bit_array import BitArray
        # from qiskit.circuit.library import IGate, XGate
        # from qiskit_aer.noise import QuantumError
        # error = QuantumError(noise_ops=[(IGate(), 0.5),(XGate(),0.5)])
        # num_shots = 97
        # num_cregs = 3
        # bits_per_creg = 17
        # depth = 19 # < num_bits/2

        # seed = 0
        # num_bits = num_cregs * bits_per_creg
        # cregs = [ClassicalRegister(bits_per_creg) for _ in range(num_cregs)]
        # qc = QuantumCircuit(QuantumRegister(1), *cregs)
        # bitpairs = []
        # shuffled_bits = np.arange(num_bits)
        # np.random.seed(seed)
        # np.random.shuffle(shuffled_bits)
        # for i in range(depth):
        #     qc.append(error,[0])
        #     bit1 = shuffled_bits[2*i]
        #     bit2 = shuffled_bits[2*i+1]
        #     qc.measure(0,bit1)
        #     qc.measure(0,bit2)
        #     bitpairs.append([bit1,bit2])
        # bitpairs = np.array(bitpairs)

        # print(qc.draw(fold=-1))

        # backend = AerSimulator(seed_simulator=seed)
        # sampler = SamplerV2(mode=backend)

        # result = sampler.run([qc], shots=num_shots).result()
        # data = result[0].data
        # barry = BitArray.concatenate_bits(list(data.values()))

        num_bits = 51
        barry = BitArray(
            np.array(
                [
                    [0, 66, 98, 212, 66, 84, 20],
                    [5, 14, 46, 10, 23, 161, 160],
                    [1, 0, 32, 240, 6, 8, 4],
                    [1, 79, 2, 36, 69, 204, 18],
                    [4, 79, 34, 230, 83, 204, 50],
                    [0, 8, 68, 240, 65, 28, 132],
                    [1, 4, 104, 56, 6, 185, 4],
                    [0, 103, 7, 38, 0, 233, 178],
                    [4, 109, 97, 6, 83, 245, 50],
                    [1, 102, 35, 36, 70, 204, 16],
                    [1, 101, 101, 36, 6, 249, 146],
                    [4, 73, 40, 26, 83, 101, 38],
                    [1, 72, 0, 214, 69, 68, 52],
                    [1, 41, 77, 252, 69, 61, 150],
                    [0, 36, 109, 60, 66, 156, 148],
                    [4, 78, 66, 52, 81, 220, 20],
                    [4, 6, 106, 26, 82, 148, 36],
                    [1, 32, 65, 38, 68, 28, 48],
                    [0, 34, 11, 216, 64, 4, 4],
                    [5, 66, 106, 232, 22, 121, 0],
                    [1, 98, 79, 248, 4, 121, 132],
                    [1, 41, 69, 210, 69, 20, 166],
                    [1, 103, 35, 242, 70, 204, 38],
                    [0, 73, 64, 16, 1, 80, 6],
                    [1, 65, 8, 236, 4, 105, 18],
                    [0, 72, 36, 228, 3, 72, 144],
                    [5, 71, 102, 34, 86, 253, 162],
                    [5, 41, 101, 50, 23, 57, 166],
                    [1, 38, 39, 214, 70, 132, 180],
                    [1, 71, 70, 32, 68, 220, 130],
                    [5, 5, 72, 26, 84, 181, 38],
                    [4, 69, 32, 198, 18, 225, 50],
                    [5, 44, 37, 38, 23, 169, 176],
                    [5, 108, 105, 24, 23, 241, 4],
                    [0, 108, 1, 36, 1, 200, 16],
                    [4, 14, 74, 248, 17, 185, 4],
                    [1, 77, 32, 54, 7, 200, 54],
                    [5, 102, 79, 46, 84, 220, 176],
                    [0, 70, 106, 62, 2, 216, 52],
                    [1, 15, 74, 24, 69, 181, 6],
                    [4, 74, 102, 2, 83, 117, 160],
                    [5, 12, 32, 36, 23, 136, 16],
                    [1, 109, 109, 56, 7, 249, 134],
                    [4, 72, 100, 4, 19, 113, 144],
                    [1, 13, 12, 56, 69, 173, 134],
                    [1, 45, 37, 38, 7, 136, 178],
                    [5, 11, 98, 244, 87, 28, 22],
                    [5, 76, 100, 48, 87, 220, 132],
                    [5, 70, 46, 10, 86, 196, 160],
                    [1, 67, 74, 216, 68, 117, 6],
                    [1, 109, 37, 214, 7, 225, 182],
                    [5, 14, 78, 58, 21, 185, 164],
                    [5, 45, 97, 210, 87, 181, 38],
                    [4, 38, 111, 24, 82, 181, 132],
                    [0, 76, 4, 224, 65, 204, 128],
                    [5, 104, 97, 194, 23, 80, 32],
                    [5, 8, 68, 2, 21, 16, 160],
                    [0, 40, 45, 62, 3, 8, 180],
                    [4, 65, 0, 20, 16, 97, 22],
                    [4, 42, 43, 222, 83, 4, 52],
                    [1, 8, 0, 212, 69, 4, 20],
                    [0, 73, 36, 50, 67, 109, 166],
                    [4, 40, 41, 220, 19, 0, 20],
                    [0, 110, 103, 196, 67, 245, 144],
                    [0, 99, 71, 4, 64, 117, 146],
                    [1, 79, 34, 34, 7, 200, 34],
                    [1, 72, 32, 2, 71, 101, 32],
                    [1, 106, 111, 42, 7, 121, 160],
                    [1, 70, 102, 20, 6, 208, 148],
                    [0, 32, 33, 228, 66, 45, 16],
                    [5, 68, 4, 208, 84, 229, 132],
                    [0, 102, 71, 194, 64, 245, 160],
                    [4, 0, 76, 234, 80, 28, 160],
                    [0, 111, 43, 232, 67, 237, 2],
                    [1, 41, 105, 234, 7, 57, 34],
                    [0, 46, 99, 50, 3, 185, 36],
                    [1, 110, 79, 58, 5, 249, 164],
                    [1, 104, 97, 2, 7, 113, 32],
                    [5, 69, 108, 62, 86, 253, 182],
                    [0, 44, 97, 194, 3, 144, 32],
                    [1, 69, 100, 212, 70, 212, 150],
                    [5, 79, 14, 238, 85, 204, 178],
                    [1, 8, 100, 6, 71, 20, 176],
                    [1, 74, 98, 198, 71, 117, 48],
                    [4, 77, 68, 242, 17, 216, 166],
                    [0, 76, 76, 58, 65, 253, 164],
                    [4, 35, 107, 248, 82, 61, 6],
                    [1, 98, 43, 46, 70, 109, 48],
                    [0, 98, 107, 44, 2, 88, 16],
                    [0, 7, 6, 54, 0, 169, 182],
                    [4, 8, 100, 52, 19, 24, 148],
                    [1, 106, 99, 34, 71, 125, 32],
                    [5, 32, 101, 22, 86, 20, 180],
                    [5, 5, 40, 10, 86, 132, 34],
                    [0, 10, 2, 192, 65, 37, 0],
                    [5, 12, 76, 56, 85, 189, 132],
                    [5, 1, 32, 214, 22, 0, 54],
                ],
                dtype="uint8",
            ),
            num_bits=num_bits,
        )

        bitpairs = np.array(
            [
                [16, 43],
                [27, 35],
                [17, 37],
                [10, 22],
                [31, 30],
                [12, 38],
                [20, 50],
                [33, 41],
                [14, 46],
                [29, 11],
                [8, 13],
                [26, 4],
                [2, 28],
                [34, 7],
                [45, 32],
                [40, 1],
                [18, 48],
                [42, 15],
                [25, 5],
            ],
            dtype=int,
        )

        np.random.seed(0)

        iden = Pauli("I" * num_bits)

        expectations_ps0 = []
        expectations_ps1 = []
        for _ in range(50):
            np.random.shuffle(bitpairs)
            num_bitpairs = np.random.randint(1, 10)
            bitpair_subset = bitpairs[:num_bitpairs]
            print(f"{num_bitpairs = }")

            obs = iden.copy()
            obs[bitpair_subset[:, 1]] = "Z"

            selection = np.random.randint(0, 2, size=num_bitpairs, dtype=bool)

            barry_ps0 = barry.postselect(indices=bitpair_subset[:, 0], selection=selection)
            if barry_ps0.num_shots > 0:
                expt = barry_ps0.expectation_values(obs)
                thy = (-1) ** np.sum(selection)
                expectations_ps0.append([thy, expt])

            barry_ps1 = barry.postselect(
                indices=bitpair_subset[:, 0], selection=np.logical_not(selection)
            )
            if barry_ps1.num_shots > 0:
                expt = barry_ps1.expectation_values(obs)
                thy = (-1) ** np.sum(np.logical_not(selection))
                expectations_ps1.append([thy, expt])

        expectations_ps0 = np.array(expectations_ps0, dtype=float)
        expectations_ps1 = np.array(expectations_ps1, dtype=float)

        np.testing.assert_allclose(expectations_ps0[:, 0], expectations_ps0[:, 1])
        np.testing.assert_allclose(expectations_ps1[:, 0], expectations_ps1[:, 1])
