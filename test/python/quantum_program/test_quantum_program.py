# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the public surface of qiskit.quantum_program."""

import ast
import inspect
import pathlib
from collections import OrderedDict, namedtuple

import numpy as np

import qiskit.quantum_program
from qiskit.quantum_program import DataTree
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestTypeStub(QiskitTestCase):
    """Tests that pyi stubs don't drift from reality."""

    @staticmethod
    def declarations() -> dict[str, type]:
        """The members the stub declares."""
        source = pathlib.Path(inspect.getfile(qiskit.quantum_program)).with_suffix(".pyi")
        classes = (
            node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef)
        )
        return {
            cls.name: {
                node.name if isinstance(node, ast.FunctionDef) else node.target.id
                for node in cls.body
                if isinstance(node, (ast.FunctionDef, ast.AnnAssign))
            }
            for cls in classes
        }

    def test_stub_covers_public_module(self):
        """Test that everything the module exports is declared in the stub."""
        self.assertEqual(set(self.declarations()), set(qiskit.quantum_program.__all__))

    def test_stub_declares_class_members(self):
        """Test that the stub declares exactly the members the class has."""
        # Written out rather than filtered, so that a new member is a deliberate edit here. These
        # are the names pyo3 adds of its own accord, alongside those the class declares.
        generated = {"__doc__", "__ge__", "__gt__", "__le__", "__lt__", "__module__", "__ne__"}
        # A stub spells the constructor `__init__` where a pyclass exposes `__new__`.
        declared = self.declarations()["DataTree"] - {"__init__"} | {"__new__"}
        self.assertEqual(declared, set(vars(DataTree)) - generated)


class TestDataTree(QiskitTestCase):
    """Tests for DataTree, the container a program's inputs and outputs travel in."""

    def test_list_parses_to_unnamed_branch(self):
        """Test that a list becomes a branch addressed only by position."""
        tree = DataTree([1, 2])
        self.assertEqual(len(tree), 2)
        self.assertEqual([tree[0], tree[1]], [1, 2])

    def test_tuple_parses_like_a_list(self):
        """Test that a tuple parses the same way a list does."""
        self.assertEqual(DataTree((1, 2)), DataTree([1, 2]))

    def test_dict_parses_to_named_branch(self):
        """Test that a dict becomes a branch with its keys as names."""
        tree = DataTree({"counts": 3, "shots": 100})
        self.assertEqual(tree["counts"], 3)
        self.assertEqual(tree["shots"], 100)

    def test_dict_keeps_insertion_order(self):
        """Test that names appear in the order the dict was written, not sorted."""
        self.assertEqual(DataTree({"z": 1, "a": 2, "m": 3}).keys(), ["z", "a", "m"])
        self.assertEqual(DataTree(OrderedDict([("z", 1), ("a", 2)])).keys(), ["z", "a"])

    def test_namedtuple_parses_to_named_branch(self):
        """Test that a namedtuple is named by its fields, not caught by the tuple case."""
        point = namedtuple("Point", ["x", "y"])
        self.assertEqual(DataTree(point(1, 2)), DataTree({"x": 1, "y": 2}))

    def test_malformed_namedtuple_refused(self):
        """Test that fields failing to name each item exactly once are refused."""

        class TooFew(tuple):
            """A tuple naming fewer fields than it holds items."""

            _fields = ("x",)

        class Repeated(tuple):
            """A tuple naming one field twice."""

            _fields = ("x", "x")

        for malformed in (TooFew((1, 2)), Repeated((1, 2))):
            with self.assertRaisesRegex(ValueError, "does not name each of the 2 items"):
                DataTree(malformed)

    def test_nesting_parses_all_the_way_down(self):
        """Test that containers inside containers each become a branch."""
        tree = DataTree({"a": [1, {"b": 2}]})
        self.assertEqual(tree["a"][0], 1)
        self.assertEqual(tree["a"][1]["b"], 2)

    def test_other_objects_become_leaves(self):
        """Test that a leaf holds whatever object it was given, of whatever type."""
        array = np.zeros(2)
        tree = DataTree({"array": array, "text": "abc", "none": None})
        self.assertIs(tree["array"], array)
        self.assertEqual(tree["text"], "abc")
        self.assertIsNone(tree["none"])

    def test_parsing_a_tree_is_idempotent(self):
        """Test that a tree parses to an equal tree."""
        tree = DataTree({"a": [1, 2]})
        self.assertEqual(DataTree(tree), tree)

    def test_hook_decomposes_an_object(self):
        """Test that __datatree__ is used in place of treating the object as a leaf."""

        class Port:
            """An object that describes itself as a branch."""

            def __datatree__(self):
                return DataTree({"c0": 1, "c1": 2})

        self.assertEqual(DataTree(Port()), DataTree({"c0": 1, "c1": 2}))
        self.assertEqual(DataTree({"port": Port()}), DataTree({"port": {"c0": 1, "c1": 2}}))

    def test_hook_must_return_a_tree(self):
        """Test that a hook returning anything but a DataTree is an error."""

        class Sloppy:
            """An object whose hook returns a dict."""

            def __datatree__(self):
                return {"a": 1}

        with self.assertRaisesRegex(TypeError, "__datatree__ returned dict"):
            DataTree(Sloppy())

    def test_hook_looked_up_on_the_type(self):
        """Test that the hook is found on the type rather than on the instance."""

        class Proxy:
            """An object that answers any attribute."""

            def __getattr__(self, name):
                return 7

        class Port:
            """An object that describes itself as a branch."""

            def __datatree__(self):
                return DataTree({"a": 1})

        self.assertTrue(DataTree(Proxy()).is_leaf)
        self.assertTrue(DataTree(Port).is_leaf)

    def test_child_addressed_by_name_or_position(self):
        """Test that the same child answers to its name and to where it sits."""
        tree = DataTree({"a": 1, "b": 2})
        self.assertEqual(tree["b"], tree[1])
        self.assertEqual(tree[-1], 2)

    def test_reading_a_child_unwraps_a_leaf(self):
        """Test that reading a branch gives a tree and reading a leaf gives its value."""
        tree = DataTree({"counts": [3, 4]})
        self.assertEqual(tree["counts"], DataTree([3, 4]))
        self.assertEqual(tree["counts"][0], 3)

    def test_value_addressed_by_dotted_path(self):
        """Test that a path of names and positions reaches a value below a child."""
        tree = DataTree({"counts": [3, {"creg": 7}], "ev": 0.1})
        self.assertEqual(tree["counts.0"], 3)
        self.assertEqual(tree["counts.1.creg"], 7)
        self.assertEqual(tree["0.1.creg"], 7)
        self.assertEqual(tree["counts.1"], DataTree({"creg": 7}))
        self.assertIn("counts.1.creg", tree)
        self.assertNotIn("counts.2", tree)

    def test_bad_address_raises(self):
        """Test that a missing name, an out-of-range position and an unusable key each raise."""
        tree = DataTree({"a": 1})
        with self.assertRaises(KeyError):
            _ = tree["b"]
        with self.assertRaises(IndexError):
            _ = tree[1]
        with self.assertRaisesRegex(TypeError, "not by float"):
            _ = tree[1.5]
        with self.assertRaises(KeyError):
            _ = tree[""]

    def test_contains_asks_about_addresses(self):
        """Test that `in` reads its argument the way indexing does."""
        tree = DataTree({"a": np.zeros(2)})
        self.assertIn("a", tree)
        self.assertIn(0, tree)
        self.assertNotIn("b", tree)
        self.assertNotIn(1, tree)

    def test_length_and_iteration(self):
        """Test that length and iteration cover the children, in order."""
        tree = DataTree({"a": 1, "b": [2, 3]})
        self.assertEqual(len(tree), 2)
        self.assertEqual(list(tree), [1, DataTree([2, 3])])

    def test_leaf_has_no_children(self):
        """Test that a leaf holds no children and its value is read through `leaf`."""
        tree = DataTree(5)
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf, 5)
        self.assertEqual(len(tree), 0)
        self.assertEqual(list(tree), [])

    def test_branch_has_no_leaf_value(self):
        """Test that a branch raises when asked for a leaf value."""
        tree = DataTree([1])
        self.assertFalse(tree.is_leaf)
        with self.assertRaisesRegex(TypeError, "no leaf value"):
            _ = tree.leaf

    def test_named_branch_is_a_mapping(self):
        """Test that a branch naming its children converts to a dict."""
        tree = DataTree({"a": 1, "b": 2})
        self.assertTrue(tree.is_mapping)
        self.assertEqual(tree.keys(), ["a", "b"])
        self.assertEqual(tree.values(), [1, 2])
        self.assertEqual(tree.items(), [("a", 1), ("b", 2)])
        self.assertEqual(dict(tree), {"a": 1, "b": 2})
        self.assertEqual(dict(**tree), {"a": 1, "b": 2})

    def test_unnamed_branch_is_not_a_mapping(self):
        """Test that a branch naming none of its children has no mapping form."""
        tree = DataTree([1, 2])
        self.assertFalse(tree.is_mapping)
        for access in (tree.keys, tree.values, tree.items):
            with self.assertRaisesRegex(TypeError, "unnamed children"):
                access()
        with self.assertRaises(TypeError):
            dict(tree)

    def test_leaf_is_not_a_mapping(self):
        """Test that a leaf has no children to name, and so no mapping form."""
        self.assertFalse(DataTree(5).is_mapping)
        with self.assertRaisesRegex(TypeError, "a leaf of a data tree has no mapping form"):
            dict(DataTree(5))

    def test_empty_branch_is_an_empty_mapping(self):
        """Test that an empty branch converts to an empty dict."""
        self.assertTrue(DataTree([]).is_mapping)
        self.assertEqual(dict(DataTree([])), {})

    def test_unaddressable_name_refused(self):
        """Test that a name with a dot, one of all digits, and a non-string are refused."""
        with self.assertRaisesRegex(ValueError, r'contains "\."'):
            DataTree({"a.b": 1})
        with self.assertRaisesRegex(ValueError, "only of digits"):
            DataTree({"12": 1})
        with self.assertRaisesRegex(TypeError, "not with int"):
            DataTree({1: 2})

    def test_equality_compares_structure_and_leaves(self):
        """Test that trees are equal when built the same way with equal leaves."""
        self.assertEqual(DataTree({"a": [1, 2]}), DataTree({"a": [1, 2]}))
        self.assertNotEqual(DataTree({"a": [1, 2]}), DataTree({"a": [1, 3]}))
        self.assertNotEqual(DataTree({"a": [1, 2]}), DataTree({"a": [1, 2, 3]}))
        self.assertNotEqual(DataTree([1]), 1)

    def test_naming_affects_equality(self):
        """Test that trees differing only in which children are named are unequal."""
        self.assertNotEqual(DataTree({"a": 1}), DataTree([1]))
        self.assertNotEqual(DataTree({"a": 1}), DataTree({"b": 1}))
        self.assertNotEqual(DataTree(1), DataTree([1]))

    def test_identical_leaves_skip_comparison(self):
        """Test that a tree whose leaves are arrays can be compared against itself."""
        array = np.zeros(2)
        self.assertEqual(DataTree({"a": array}), DataTree({"a": array}))

    def test_tree_is_unhashable(self):
        """Test that a tree is unhashable, since its leaves need not be."""
        with self.assertRaises(TypeError):
            hash(DataTree([1]))

    def test_repr(self):
        """Test that repr renders a leaf by its own repr and a branch in brackets."""
        self.assertEqual(repr(DataTree(5)), "DataTree(5)")
        self.assertEqual(
            repr(DataTree({"counts": [3, 4], "shots": 100})),
            "DataTree([counts: [3, 4], shots: 100])",
        )
