# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the methods in ``utils.deprecation``."""


from qiskit import QiskitError
from qiskit.utils.deprecation import deprecate_function, deprecate_arguments
from qiskit.test import QiskitTestCase


class DummyClass:
    """This is short description. Let's make it
    multiline"""

    def __init__(self, arg1: int = None, arg2: [int] = None):
        self.arg1 = arg1
        self.arg2 = arg2

    @deprecate_function(
        "The DummyClass.foo() method is being deprecated. " "Use the DummyClass.some_othermethod()",
        docstring_version="1.2.3",
    )
    def foo_deprecated(self, index_arg2: int):
        """A multi-line
        docstring.

        Here are more details.

        Args:
            index_arg2: `index_arg2` description

        Returns:
            int: returns `arg2[index_arg2]`

        Raises:
            QiskitError: if `len(self.arg2) < index_arg2`
        """
        if len(self.arg2) < index_arg2:
            raise QiskitError("there is an error")
        return self.arg2[index_arg2]

    @deprecate_arguments({"if_arg1": "other_if_arg1"}, docstring_version="1.2.3")
    def bar_with_deprecated_arg(
        self, if_arg1: int = None, index_arg2: int = None, other_if_arg1: int = None
    ):
        """
        A multi-line short
        docstring.

        This is the long description

        Args:
            if_arg1: `if_arg1` description with
               multi-line
            index_arg2: `index_arg2` description
            other_if_arg1: `other_if_arg1` description

        Returns:
            int or None: if `if_arg1 == self.arg1`, returns `arg2[index_arg2]`
        """
        if other_if_arg1 == self.arg1 or if_arg1 == self.arg1:
            return self.arg2[index_arg2]
        return None


class TestDeprecation(QiskitTestCase):
    """Test deprecation decorators."""

    def assertDeprecationWarning(self, warn, expected_msg):
        """Checks there only one exception and `expected_msg` is the message"""
        self.assertEqual(len(warn.warnings), 1)
        self.assertEqual(len(warn.warnings[0].message.args), 1)
        self.assertEqual(warn.warnings[0].message.args[0], expected_msg)

    def test_raise_deprecate_function(self):
        """Test deprecate_function raises."""

        dummy = DummyClass(arg2=[1, 2, 3])
        with self.assertWarns(DeprecationWarning) as warn:
            output = dummy.foo_deprecated(0)
        self.assertDeprecationWarning(
            warn,
            "The DummyClass.foo() method is being deprecated. Use the DummyClass.some_othermethod()",
        )
        self.assertEqual(output, 1)

    def test_raise_deprecate_arguments(self):
        """Test deprecate_arguments raises."""

        dummy = DummyClass(arg1=3, arg2=[1, 2, 3])
        with self.assertWarns(DeprecationWarning) as warn:
            output = dummy.bar_with_deprecated_arg(if_arg1=3, index_arg2=0)
        self.assertDeprecationWarning(
            warn,
            "bar_with_deprecated_arg keyword argument if_arg1 is deprecated and replaced"
            " with other_if_arg1.",
        )
        self.assertEqual(output, 1)

    def test_docstring_deprecate_function(self):
        """Test deprecate_function docstring."""

        dummy = DummyClass()
        deprecated_docstring = dummy.foo_deprecated.__doc__
        expected = """A multi-line
        docstring.

        .. deprecated:: 1.2.3
          The DummyClass.foo() method is being deprecated. Use the DummyClass.some_othermethod()

        Here are more details.

        Args:
            index_arg2: `index_arg2` description

        Returns:
            int: returns `arg2[index_arg2]`

        Raises:
            QiskitError: if `len(self.arg2) < index_arg2`
        """
        self.assertEqual(deprecated_docstring, expected)

    def test_docstring_deprecate_arguments(self):
        """Test deprecate_arguments docstring."""

        dummy = DummyClass()
        deprecated_docstring = dummy.bar_with_deprecated_arg.__doc__
        expected = """
        A multi-line short
        docstring.

        This is the long description

        Args:
            if_arg1:
                .. deprecated:: 1.2.3
                    The keyword argument ``if_arg1`` is deprecated.
                    Please, use ``other_if_arg1`` instead.

            index_arg2: `index_arg2` description
            other_if_arg1: `other_if_arg1` description

        Returns:
            int or None: if `if_arg1 == self.arg1`, returns `arg2[index_arg2]`
        """
        self.assertEqual(deprecated_docstring, expected)
