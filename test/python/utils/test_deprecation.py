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

"""Tests for the functions in ``utils.deprecation``."""

from textwrap import dedent

from qiskit.test import QiskitTestCase
from qiskit.utils.deprecation import (
    add_deprecation_to_docstring,
    deprecate_function,
    deprecate_arguments,
)


class TestDeprecationDecorators(QiskitTestCase):
    """Test that the decorators in ``utils.deprecation`` correctly log warnings and get added to
    docstring."""

    def test_deprecate_arguments_message(self) -> None:
        """Test that `@deprecate_arguments` adds the correct message to the docstring."""

        @deprecate_arguments(
            {"old_arg1": "new_arg1", "old_arg2": None},
            category=PendingDeprecationWarning,
            since="9.99",
        )
        def my_func() -> None:
            pass

        self.assertEqual(
            my_func.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99_pending
                  {my_func.__qualname__} keyword argument old_arg1 is deprecated and replaced with \
new_arg1.

                .. deprecated:: 9.99_pending
                  {my_func.__qualname__} keyword argument old_arg2 is deprecated and will in the \
future be removed.
                """
            ),
        )

    def test_deprecate_function_docstring(self) -> None:
        """Test that `@deprecate_function` adds the correct message to the docstring."""

        @deprecate_function("Stop using my_func!", since="9.99")
        def my_func() -> None:
            pass

        self.assertEqual(
            my_func.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99
                  Stop using my_func!
                """
            ),
        )

    def test_deprecate_arguments_runtime_warning(self) -> None:
        """Test that `@deprecate_arguments` warns whenever the arguments are used.

        Also check that old arguments are passed in as their new alias.
        """

        @deprecate_arguments({"arg1": None, "arg2": "new_arg2"}, since="9.99")
        def my_func(*, arg1: str = "a", new_arg2: str) -> None:
            del arg1
            self.assertEqual(new_arg2, "z")

        my_func(new_arg2="z")  # No warnings if no deprecated args used.
        with self.assertWarnsRegex(DeprecationWarning, "arg1"):
            my_func(arg1="a", new_arg2="z")
        with self.assertWarnsRegex(DeprecationWarning, "arg2"):
            # `arg2` should be converted into `new_arg2`.
            my_func(arg2="z")  # pylint: disable=missing-kwoa

    def test_deprecate_function_runtime_warning(self) -> None:
        """Test that `@deprecate_function` warns whenever the function is used."""

        @deprecate_function("Stop using my_func!", since="9.99")
        def my_func() -> None:
            pass

        with self.assertWarnsRegex(DeprecationWarning, "Stop using my_func!"):
            my_func()


class AddDeprecationDocstringTest(QiskitTestCase):
    """Test that we correctly insert the deprecation directive at the right location.

    When determining the ``expected`` output, manually modify the docstring of a function
    (in any Qiskit repo) to have the same structure. Then, build the docs to make sure that it
    renders correctly.
    """

    def test_add_deprecation_docstring_no_meta_lines(self) -> None:
        """When no metadata lines like Args, the directive should be added to the end."""

        def func1():
            pass

        add_deprecation_to_docstring(func1, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func1.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99
                  Deprecated!
                """
            ),
        )

        def func2():
            """Docstring."""

        add_deprecation_to_docstring(func2, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func2.__doc__,
            dedent(
                """\
                Docstring.

                .. deprecated:: 9.99
                  Deprecated!
                """
            ),
        )

        indent = "            "

        def func3():
            """Docstring extending
            to a new line."""

        add_deprecation_to_docstring(func3, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func3.__doc__,
            (
                f"""Docstring extending
            to a new line.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
            ),
        )

        def func4():
            """
            Docstring starting on a new line.
            """

        add_deprecation_to_docstring(func4, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func4.__doc__,
            (
                f"""\

            Docstring starting on a new line.
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
            ),
        )

        def func5():
            """
            Paragraph 1, line 1.
            Line 2.

            Paragraph 2.

            """

        add_deprecation_to_docstring(func5, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func5.__doc__,
            (
                f"""\

            Paragraph 1, line 1.
            Line 2.

            Paragraph 2.

{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
            ),
        )

        def func6():
            """Blah.

            A list.
              * element 1
              * element 2
                continued
            """

        add_deprecation_to_docstring(func6, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func6.__doc__,
            (
                f"""Blah.

            A list.
              * element 1
              * element 2
                continued
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
            ),
        )

    def test_add_deprecation_docstring_meta_lines(self) -> None:
        """When there are metadata lines like Args, the directive should be inserted in-between the
        summary and those lines."""
        indent = "            "

        def func1():
            """
            Returns:
                Content.

            Raises:
                SomeError
            """

        add_deprecation_to_docstring(func1, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func1.__doc__,
            (
                f"""\
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Returns:
                Content.

            Raises:
                SomeError
{indent}"""
            ),
        )

        def func2():
            """Docstring.

            Returns:
                Content.
            """

        add_deprecation_to_docstring(func2, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func2.__doc__,
            (
                f"""Docstring.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Returns:
                Content.
{indent}"""
            ),
        )

        def func3():
            """
            Docstring starting on a new line.

            Paragraph 2.

            Examples:
                Content.
            """

        add_deprecation_to_docstring(func3, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func3.__doc__,
            (
                f"""\

            Docstring starting on a new line.

            Paragraph 2.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Examples:
                Content.
{indent}"""
            ),
        )

    def test_add_deprecation_docstring_multiple_entries(self) -> None:
        """Multiple entries are appended correctly."""

        def func1():
            pass

        add_deprecation_to_docstring(func1, msg="Deprecated #1!", since="9.99", pending=False)
        add_deprecation_to_docstring(func1, msg="Deprecated #2!", since="9.99", pending=False)
        self.assertEqual(
            func1.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99
                  Deprecated #1!

                .. deprecated:: 9.99
                  Deprecated #2!
                """
            ),
        )

        indent = "            "

        def func2():
            """
            Docstring starting on a new line.
            """

        add_deprecation_to_docstring(func2, msg="Deprecated #1!", since="9.99", pending=False)
        add_deprecation_to_docstring(func2, msg="Deprecated #2!", since="9.99", pending=False)
        self.assertEqual(
            func2.__doc__,
            (
                f"""\

            Docstring starting on a new line.
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated #1!
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated #2!
{indent}"""
            ),
        )

        def func3():
            """Docstring.

            Yields:
                Content.
            """

        add_deprecation_to_docstring(func3, msg="Deprecated #1!", since="9.99", pending=False)
        add_deprecation_to_docstring(func3, msg="Deprecated #2!", since="9.99", pending=False)
        self.assertEqual(
            func3.__doc__,
            (
                f"""Docstring.
{indent}
            .. deprecated:: 9.99
              Deprecated #1!
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated #2!
{indent}

            Yields:
                Content.
{indent}"""
            ),
        )

    def test_add_deprecation_docstring_pending(self) -> None:
        """The version string should end in `_pending` when pending."""

        def func():
            pass

        add_deprecation_to_docstring(func, msg="Deprecated!", since="9.99", pending=True)
        self.assertEqual(
            func.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99_pending
                  Deprecated!
                """
            ),
        )

    def test_add_deprecation_docstring_since_not_set(self) -> None:
        """The version string should be `unknown` when ``None``."""

        def func():
            pass

        add_deprecation_to_docstring(func, msg="Deprecated!", since=None, pending=False)
        self.assertEqual(
            func.__doc__,
            dedent(
                """\

                .. deprecated:: unknown
                  Deprecated!
                """
            ),
        )

    def test_add_deprecation_docstring_newline_msg_banned(self) -> None:
        """Test that `\n` is banned in the deprecation message, as it breaks Sphinx rendering."""

        def func():
            pass

        with self.assertRaises(ValueError):
            add_deprecation_to_docstring(func, msg="line1\nline2", since="9.99", pending=False)

    def test_add_deprecation_docstring_initial_metadata_line_banned(self) -> None:
        """Test that the docstring cannot start with e.g. `Args:`."""

        def func():
            """Args:
            Foo.
            """

        with self.assertRaises(ValueError):
            add_deprecation_to_docstring(func, msg="Deprecated!", since="9.99", pending=False)
