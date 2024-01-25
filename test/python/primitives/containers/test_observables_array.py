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

"""Test ObservablesArray"""

import itertools as it

import ddt
import numpy as np

import qiskit.quantum_info as qi
from qiskit.primitives import ObservablesArray
from qiskit.test import QiskitTestCase


@ddt.ddt
class ObservablesArrayTestCase(QiskitTestCase):
    """Test the ObservablesArray class"""

    @ddt.data(0, 1, 2)
    def test_format_observable_str(self, num_qubits):
        """Test format_observable for allowed basis str input"""
        for chars in it.permutations(ObservablesArray.ALLOWED_BASIS, num_qubits):
            label = "".join(chars)
            obs = ObservablesArray.format_observable(label)
            self.assertEqual(obs, {label: 1})

    def test_format_observable_custom_basis(self):
        """Test format_observable for custom allowed basis"""

        class PauliArray(ObservablesArray):
            """Custom array allowing only Paulis, not projectors"""

            ALLOWED_BASIS = "IXYZ"

        with self.assertRaises(ValueError):
            PauliArray.format_observable("0101")
        for p in qi.pauli_basis(1):
            obs = PauliArray.format_observable(p)
            self.assertEqual(obs, {p.to_label(): 1})

    @ddt.data("iXX", "012", "+/-")
    def test_format_observable_invalid_str(self, basis):
        """Test format_observable for Pauli input"""
        with self.assertRaises(ValueError):
            ObservablesArray.format_observable(basis)

    @ddt.data(1, 2, 3)
    def test_format_observable_pauli(self, num_qubits):
        """Test format_observable for Pauli input"""
        for p in qi.pauli_basis(num_qubits):
            obs = ObservablesArray.format_observable(p)
            self.assertEqual(obs, {p.to_label(): 1})

    @ddt.data(0, 1, 2, 3)
    def test_format_observable_phased_pauli(self, phase):
        """Test format_observable for phased Pauli input"""
        pauli = qi.Pauli("IXYZ")
        pauli.phase = phase
        coeff = (-1j) ** phase
        obs = ObservablesArray.format_observable(pauli)
        self.assertIsInstance(obs, dict)
        self.assertEqual(list(obs.keys()), ["IXYZ"])
        np.testing.assert_allclose(
            list(obs.values()), [coeff], err_msg=f"Wrong value for Pauli {pauli}"
        )

    @ddt.data("+IXYZ", "-IXYZ", "iIXYZ", "+iIXYZ", "-IXYZ")
    def test_format_observable_phased_pauli_str(self, pauli):
        """Test format_observable for phased Pauli input"""
        pauli = qi.Pauli(pauli)
        coeff = (-1j) ** pauli.phase
        obs = ObservablesArray.format_observable(pauli)
        self.assertIsInstance(obs, dict)
        self.assertEqual(list(obs.keys()), ["IXYZ"])
        np.testing.assert_allclose(
            list(obs.values()), [coeff], err_msg=f"Wrong value for Pauli {pauli}"
        )

    def test_format_observable_phased_sparse_pauli_op(self):
        """Test format_observable for SparsePauliOp input with phase paulis"""
        op = qi.SparsePauliOp(["+I", "-X", "iY", "-iZ"], [1, 2, 3, 4])
        obs = ObservablesArray.format_observable(op)
        self.assertIsInstance(obs, dict)
        self.assertEqual(len(obs), 4)
        self.assertEqual(sorted(obs.keys()), sorted(["I", "X", "Y", "Z"]))
        np.testing.assert_allclose([obs[i] for i in ["I", "X", "Y", "Z"]], [1, -2, 3j, -4j])

    def test_format_observable_zero_sparse_pauli_op(self):
        """Test format_observable for SparsePauliOp input with zero val coeffs"""
        op = qi.SparsePauliOp(["I", "X", "Y", "Z"], [0, 0, 0, 1])
        obs = ObservablesArray.format_observable(op)
        self.assertIsInstance(obs, dict)
        self.assertEqual(len(obs), 1)
        self.assertEqual(sorted(obs.keys()), ["Z"])
        self.assertEqual(obs["Z"], 1)

    def test_format_observable_duplicate_sparse_pauli_op(self):
        """Test format_observable for SparsePauliOp wiht duplicate paulis"""
        op = qi.SparsePauliOp(["XX", "-XX", "iXX", "-iXX"], [2, 1, 3, 2])
        obs = ObservablesArray.format_observable(op)
        self.assertIsInstance(obs, dict)
        self.assertEqual(len(obs), 1)
        self.assertEqual(list(obs.keys()), ["XX"])
        self.assertEqual(obs["XX"], 1 + 1j)

    def test_format_observable_pauli_mapping(self):
        """Test format_observable for pauli-keyed Mapping input"""
        mapping = dict(zip(qi.pauli_basis(1), range(1, 5)))
        obs = ObservablesArray.format_observable(mapping)
        target = {key.to_label(): val for key, val in mapping.items()}
        self.assertEqual(obs, target)

    def test_format_invalid_mapping_qubits(self):
        """Test an error is raised when different qubits in mapping keys"""
        mapping = {"IX": 1, "XXX": 2}
        with self.assertRaises(ValueError):
            ObservablesArray.format_observable(mapping)

    def test_format_invalid_mapping_basis(self):
        """Test an error is raised when keys contain invalid characters"""
        mapping = {"XX": 1, "0Z": 2, "02": 3}
        with self.assertRaises(ValueError):
            ObservablesArray.format_observable(mapping)

    def test_init_nested_list_str(self):
        """Test init with nested lists of str"""
        obj = [["X", "Y", "Z"], ["0", "1", "+"]]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 6)
        self.assertEqual(obs.shape, (2, 3))

    def test_init_nested_list_sparse_pauli_op(self):
        """Test init with nested lists of SparsePauliOp"""
        obj = [[qi.SparsePauliOp(qi.random_pauli_list(2, 3)) for _ in range(3)] for _ in range(5)]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 15)
        self.assertEqual(obs.shape, (5, 3))

    def test_init_single_sparse_pauli_op(self):
        """Test init with single SparsePauliOps"""
        obj = qi.SparsePauliOp(qi.random_pauli_list(2, 3))
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 1)
        self.assertEqual(obs.shape, ())

    def test_init_pauli_list(self):
        """Test init with PauliList"""
        obs = ObservablesArray(qi.pauli_basis(2))
        self.assertEqual(obs.size, 16)
        self.assertEqual(obs.shape, (16,))

    def test_init_nested_pauli_list(self):
        """Test init with nested PauliList"""
        obj = [qi.random_pauli_list(2, 3) for _ in range(5)]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 15)
        self.assertEqual(obs.shape, (5, 3))

    def test_init_ragged_array(self):
        """Test init with ragged input"""
        obj = [["X", "Y"], ["X", "Y", "Z"]]
        with self.assertRaises(ValueError):
            ObservablesArray(obj)

    def test_init_validate_false(self):
        """Test init validate kwarg"""
        obj = [["A", "B", "C"], ["D", "E", "F"]]
        obs = ObservablesArray(obj, validate=False)
        self.assertEqual(obs.shape, (2, 3))
        self.assertEqual(obs.size, 6)
        for i in range(2):
            for j in range(3):
                self.assertEqual(obs[i, j], obj[i][j])

    def test_init_validate_true(self):
        """Test init validate kwarg"""
        obj = [["A", "B", "C"], ["D", "E", "F"]]
        with self.assertRaises(ValueError):
            ObservablesArray(obj, validate=True)

    @ddt.data(0, 1, 2, 3)
    def test_size_and_shape_single(self, ndim):
        """Test size and shape method for size=1 array"""
        obs = {"XX": 1}
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs, validate=False)
        self.assertEqual(arr.size, 1, msg="Incorrect ObservablesArray.size")
        self.assertEqual(arr.shape, (1,) * ndim, msg="Incorrect ObservablesArray.shape")

    @ddt.data(0, 1, 2, 3)
    def test_tolist_single(self, ndim):
        """Test tolist method for size=1 array"""
        obs = {"XX": 1}
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs, validate=False)
        ls = arr.tolist()
        self.assertEqual(ls, obs)

    @ddt.data(0, 1, 2, 3)
    def test_array_single(self, ndim):
        """Test __array__ method for size=1 array"""
        obs = {"XX": 1}
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs, validate=False)
        nparr = np.array(arr)
        self.assertEqual(nparr.dtype, object)
        self.assertEqual(nparr.shape, arr.shape)
        self.assertEqual(nparr.size, arr.size)
        self.assertTrue(np.all(nparr == np.array(obs)))

    @ddt.data(0, 1, 2, 3)
    def test_getitem_single(self, ndim):
        """Test __getitem__ method for size=1 array"""
        base_obs = {"XX": 1}
        obs = base_obs
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs, validate=False)
        idx = ndim * (0,)
        item = arr[idx]
        self.assertEqual(item, base_obs)

    def test_tolist_1d(self):
        """Test tolist method"""
        obj = ["A", "B", "C", "D"]
        obs = ObservablesArray(obj, validate=False)
        self.assertEqual(obs.tolist(), obj)

    def test_tolist_2d(self):
        """Test tolist method"""
        obj = [["A", "B", "C"], ["D", "E", "F"]]
        obs = ObservablesArray(obj, validate=False)
        self.assertEqual(obs.tolist(), obj)

    def test_array_1d(self):
        """Test __array__ dunder method"""
        obj = np.array(["A", "B", "C", "D"], dtype=object)
        obs = ObservablesArray(obj, validate=False)
        self.assertTrue(np.all(np.array(obs) == obj))

    def test_array_2d(self):
        """Test __array__ dunder method"""
        obj = np.array([["A", "B", "C"], ["D", "E", "F"]], dtype=object)
        obs = ObservablesArray(obj, validate=False)
        self.assertTrue(np.all(np.array(obs) == obj))

    def test_getitem_1d(self):
        """Test __getitem__ for 1D array"""
        obj = np.array(["A", "B", "C", "D"], dtype=object)
        obs = ObservablesArray(obj, validate=False)
        for i in range(obj.size):
            self.assertEqual(obs[i], obj[i])

    def test_getitem_2d(self):
        """Test __getitem__ for 2D array"""
        obj = np.array([["A", "B", "C"], ["D", "E", "F"]], dtype=object)
        obs = ObservablesArray(obj, validate=False)
        for i in range(obj.shape[0]):
            row = obs[i]
            self.assertIsInstance(row, ObservablesArray)
            self.assertEqual(row.shape, (3,))
            self.assertTrue(np.all(np.array(row) == obj[i]))

    def test_ravel(self):
        """Test ravel method"""
        bases_flat = qi.pauli_basis(2).to_labels()
        bases = [bases_flat[4 * i : 4 * (i + 1)] for i in range(4)]
        obs = ObservablesArray(bases)
        flat = obs.ravel()
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.shape, (16,))
        self.assertEqual(flat.size, 16)
        for (
            i,
            label,
        ) in enumerate(bases_flat):
            self.assertEqual(flat[i], {label: 1})

    def test_reshape(self):
        """Test reshape method"""
        bases = qi.pauli_basis(2)
        labels = np.array(bases.to_labels(), dtype=object)
        obs = ObservablesArray(qi.pauli_basis(2))

        def various_formats(shape):
            # call reshape with a single argument
            yield [shape]
            yield [(-1,) + shape[1:]]
            yield [np.array(shape)]
            yield [list(shape)]
            yield [list(map(np.int64, shape))]
            yield [tuple(map(np.int64, shape))]

            # call reshape with multiple arguments
            yield shape
            yield np.array(shape)
            yield list(shape)
            yield list(map(np.int64, shape))
            yield tuple(map(np.int64, shape))

        for shape in [(16,), (4, 4), (2, 4, 2), (2, 2, 2, 2), (1, 8, 1, 2)]:
            with self.subTest(shape):
                for input_shape in various_formats(shape):
                    obs_rs = obs.reshape(*input_shape)
                    self.assertEqual(obs_rs.shape, shape)
                    labels_rs = labels.reshape(shape)
                    for idx in np.ndindex(shape):
                        self.assertEqual(
                            obs_rs[idx],
                            {labels_rs[idx]: 1},
                            msg=f"failed for shape {shape} with input format {input_shape}",
                        )

    def test_validate(self):
        """Test the validate method"""
        ObservablesArray({"XX": 1}).validate()
        ObservablesArray([{"XX": 1}] * 5).validate()
        ObservablesArray([{"XX": 1}] * 15).reshape((3, 5)).validate()

        obs = ObservablesArray([{"XX": 1}, {"XYZ": 1}], validate=False)
        with self.assertRaisesRegex(ValueError, "number of qubits must be the same"):
            obs.validate()
