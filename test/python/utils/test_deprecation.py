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
    deprecate_arg,
    deprecate_arguments,
    deprecate_func,
    deprecate_function,
)


@deprecate_func(
    since="9.99",
    additional_msg="Instead, use new_func().",
    removal_timeline="in 2 releases",
)
def _deprecated_func():
    pass


class _Foo:
    @deprecate_func(since="9.99", pending=True)
    def __init__(self):
        super().__init__()

    @property
    @deprecate_func(since="9.99", is_property=True)
    def my_property(self):
        """Property."""
        return 0

    @my_property.setter
    @deprecate_func(since="9.99")
    def my_property(self, value):
        pass

    @deprecate_func(since="9.99", additional_msg="Stop using this!")
    def my_method(self):
        """Method."""

    def normal_method(self):
        """Method."""


class TestDeprecationDecorators(QiskitTestCase):
    """Test that the decorators in ``utils.deprecation`` correctly log warnings and get added to
    docstring."""

    def test_deprecate_func_docstring(self) -> None:
        """Test that `@deprecate_func` adds the correct message to the docstring."""

        self.assertEqual(
            _deprecated_func.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99
                  The function ``{__name__}._deprecated_func()`` is deprecated as of qiskit-terra \
9.99. It will be removed in 2 releases. Instead, use new_func().
                """
            ),
        )
        self.assertEqual(
            _Foo.__init__.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99_pending
                  The class ``{__name__}._Foo`` is pending deprecation as of qiskit-terra 9.99. It \
will be marked deprecated in a future release, and then removed no earlier than 3 months after \
the release date.
                """
            ),
        )
        self.assertEqual(
            _Foo.my_method.__doc__,
            dedent(
                f"""\
                Method.

                .. deprecated:: 9.99
                  The method ``{__name__}._Foo.my_method()`` is deprecated as of qiskit-terra \
9.99. It will be removed no earlier than 3 months after the release date. Stop using this!
                """
            ),
        )
        self.assertEqual(
            _Foo.my_property.__doc__,
            dedent(
                f"""\
                Property.

                .. deprecated:: 9.99
                  The property ``{__name__}._Foo.my_property`` is deprecated as of qiskit-terra \
9.99. It will be removed no earlier than 3 months after the release date.
                """
            ),
        )

    def test_deprecate_arg_docstring(self) -> None:
        """Test that `@deprecate_arg` adds the correct message to the docstring."""

        @deprecate_arg("arg1", since="9.99", removal_timeline="in 2 releases")
        @deprecate_arg("arg2", pending=True, since="9.99")
        @deprecate_arg(
            "arg3",
            since="9.99",
            deprecation_description="Using the argument arg3",
            new_alias="new_arg3",
        )
        @deprecate_arg(
            "arg4",
            since="9.99",
            additional_msg="Instead, use foo.",
            # This predicate always fails, but it should not impact storing the deprecation
            # metadata. That ensures the deprecation still shows up in our docs.
            predicate=lambda arg4: False,
        )
        def my_func() -> None:
            pass

        self.assertEqual(
            my_func.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99
                  ``{__name__}.{my_func.__qualname__}()``'s argument ``arg4`` is deprecated as of \
qiskit-terra 9.99. It will be removed no earlier than 3 months after the release date. Instead, \
use foo.

                .. deprecated:: 9.99
                  Using the argument arg3 is deprecated as of qiskit-terra 9.99. It will be \
removed no earlier than 3 months after the release date. Instead, use the argument ``new_arg3``, \
which behaves identically.

                .. deprecated:: 9.99_pending
                  ``{__name__}.{my_func.__qualname__}()``'s argument ``arg2`` is pending \
deprecation as of qiskit-terra 9.99. It will be marked deprecated in a future release, and then \
removed no earlier than 3 months after the release date.

                .. deprecated:: 9.99
                  ``{__name__}.{my_func.__qualname__}()``'s argument ``arg1`` is deprecated as of \
qiskit-terra 9.99. It will be removed in 2 releases.
                """
            ),
        )

    def test_deprecate_arguments_docstring(self) -> None:
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

    def test_deprecate_func_runtime_warning(self) -> None:
        """Test that `@deprecate_func` warns whenever the function is used."""

        with self.assertWarns(DeprecationWarning):
            _deprecated_func()
        with self.assertWarns(PendingDeprecationWarning):
            instance = _Foo()
        with self.assertWarns(DeprecationWarning):
            instance.my_method()
        with self.assertWarns(DeprecationWarning):
            _ = instance.my_property
        with self.assertWarns(DeprecationWarning):
            instance.my_property = 1
        instance.normal_method()

    def test_deprecate_arg_runtime_warning(self) -> None:
        """Test that `@deprecate_arg` warns whenever the arguments are used.

        Also check these edge cases:
        * If `new_alias` is set, pass the old argument as the new alias.
        * If `predicate` is set, only warn if the predicate is True.
        """

        @deprecate_arg("arg1", since="9.99")
        @deprecate_arg("arg2", new_alias="new_arg2", since="9.99")
        @deprecate_arg("arg3", predicate=lambda arg3: arg3 == "deprecated value", since="9.99")
        def my_func(*, arg1: str = "a", new_arg2: str, arg3: str = "a") -> None:
            del arg1
            del arg3
            assert new_arg2 == "z"

        my_func(new_arg2="z")  # No warnings if no deprecated args used.
        with self.assertWarnsRegex(DeprecationWarning, "arg1"):
            my_func(arg1="a", new_arg2="z")
        with self.assertWarnsRegex(DeprecationWarning, "arg2"):
            # `arg2` should be converted into `new_arg2`.
            my_func(arg2="z")  # pylint: disable=missing-kwoa

        # Test the `predicate` functionality.
        my_func(new_arg2="z", arg3="okay value")
        with self.assertWarnsRegex(DeprecationWarning, "arg3"):
            my_func(new_arg2="z", arg3="deprecated value")

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
