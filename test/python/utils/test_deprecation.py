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
from typing import Callable

from qiskit.test import QiskitTestCase
from qiskit.utils.deprecation import (
    _add_deprecation_to_docstring,
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
                """\

                .. deprecated:: 9.99_pending
                  TestDeprecationDecorators.test_deprecate_arguments_message.<locals>.my_func \
keyword argument old_arg1 is deprecated and replaced with new_arg1.

                .. deprecated:: 9.99_pending
                  TestDeprecationDecorators.test_deprecate_arguments_message.<locals>.my_func \
keyword argument old_arg2 is deprecated and will in the future be removed.
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
            assert new_arg2 == "z"

        my_func(new_arg2="z")  # No warnings if no deprecated args used.
        with self.assertWarns(DeprecationWarning):
            my_func(arg1="a", new_arg2="z")
        with self.assertWarns(DeprecationWarning):
            # `arg2` should be converted into `new_arg2`.
            my_func(arg2="z")  # pylint: disable=missing-kwoa

    def test_deprecate_function_runtime_warning(self) -> None:
        """Test that `@deprecate_function` warns whenever the function is used."""

        @deprecate_function("Stop using my_func!", since="9.99")
        def my_func() -> None:
            pass

        with self.assertWarns(DeprecationWarning):
            my_func()


class DeprecationExtensionTest(QiskitTestCase):
    """Test that we correctly insert the deprecation directive at the right location.

    These test cases were manually created by changing the docstring of a function in
    ``qiskit-ibmq-provider`` and building the docs to ensure that everything rendered correctly.
    (Any Qiskit repo will do.) To get the ``expected`` lines, you can add this extension to
    ``conf.py``:

        def print_docstring(app, what, name, obj, options, lines):
            if name != "FILL ME IN WITH QISKIT.MODULE.FUNCTION_NAME":
                return
            print("HERE")
            print(obj.__doc__)

        def setup(app):
            app.connect('autodoc-process-docstring', print_docstring)

    Then, choose a function and modify it's docstring. Update the extension with that function's
    name. Build the docs and look at the output.
    """

    def assert_docstring(
        self,
        func: Callable,
        *,
        expected: str,
        msg: str = "Deprecated!",
        pending: bool = False,
        since_is_set: bool = True,
        num_deprecations: int = 1,
    ) -> None:
        """Add docstring to ``func`` and check that it worked as expected."""
        for _ in range(num_deprecations):
            _add_deprecation_to_docstring(
                func, msg=msg, since="9.99" if since_is_set else None, pending=pending
            )
        self.assertEqual(func.__doc__, expected)

    def test_deprecation_docstring_added_to_end_when_no_meta_lines(self) -> None:
        """When no metadata lines like Args, the directive should be added to the end."""

        def func1():
            pass

        self.assert_docstring(
            func1,
            expected=dedent(
                """\

                .. deprecated:: 9.99
                  Deprecated!
                """
            ),
        )

        def func2():
            """Docstring."""

        self.assert_docstring(
            func2,
            expected=dedent(
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

        self.assert_docstring(
            func3,
            expected=(
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

        self.assert_docstring(
            func4,
            expected=(
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

        self.assert_docstring(
            func5,
            expected=(
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

        self.assert_docstring(
            func6,
            expected=(
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

    def test_deprecation_docstring_added_before_meta_lines(self) -> None:
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

        self.assert_docstring(
            func1,
            expected=(
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

        self.assert_docstring(
            func2,
            expected=(
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

        self.assert_docstring(
            func3,
            expected=(
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

    def test_deprecation_docstring_multiple_entries(self) -> None:
        """Multiple entries are appended correctly."""

        def func1():
            pass

        self.assert_docstring(
            func1,
            num_deprecations=2,
            expected=dedent(
                """\

                .. deprecated:: 9.99
                  Deprecated!

                .. deprecated:: 9.99
                  Deprecated!
                """
            ),
        )

        indent = "            "

        def func2():
            """
            Docstring starting on a new line.
            """

        self.assert_docstring(
            func2,
            num_deprecations=2,
            expected=(
                f"""\

            Docstring starting on a new line.
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
            ),
        )

        def func3():
            """Docstring.

            Yields:
                Content.
            """

        self.assert_docstring(
            func3,
            num_deprecations=2,
            expected=(
                f"""Docstring.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Yields:
                Content.
{indent}"""
            ),
        )

    def test_deprecation_docstring_pending(self) -> None:
        """The version string should end in `_pending` when pending."""

        def func():
            pass

        self.assert_docstring(
            func,
            pending=True,
            expected=dedent(
                """\

                .. deprecated:: 9.99_pending
                  Deprecated!
                """
            ),
        )

    def test_deprecation_docstring_since_not_set(self) -> None:
        """The version string should be `unknown` when ``None``."""

        def func():
            pass

        self.assert_docstring(
            func,
            since_is_set=False,
            expected=dedent(
                """\

                .. deprecated:: unknown
                  Deprecated!
                """
            ),
        )

    def test_deprecation_docstring_newline_msg_banned(self) -> None:
        """Test that `\n` is banned in the deprecation message, as it breaks Sphinx rendering."""

        def func():
            pass

        with self.assertRaises(ValueError):
            self.assert_docstring(func, msg="line1\nline2", expected="doesnt matter")
