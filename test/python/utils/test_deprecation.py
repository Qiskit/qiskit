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

from qiskit.test import QiskitTestCase
from qiskit.utils.deprecation import (
    _DeprecationMetadataEntry,
    deprecate_function,
    deprecate_argument,
    deprecate_arguments,
)


class TestDeprecations(QiskitTestCase):
    """Test functions in ``utils.deprecation``."""

    def test_deprecate_argument_metadata(self) -> None:
        """Test that `@deprecate_argument` stores the correct metadata."""

        @deprecate_argument("arg1", since="9.99")
        @deprecate_argument("arg2", pending=True, since="9.99")
        @deprecate_argument(
            "arg3",
            since="9.99",
            deprecation_description="Using the argument arg3",
            new_alias="new_arg3",
        )
        @deprecate_argument(
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
            getattr(my_func, _DeprecationMetadataEntry.dunder_name),
            [
                _DeprecationMetadataEntry(
                    (
                        "TestDeprecations.test_deprecate_argument_metadata.<locals>.my_func()'s "
                        "argument `arg4` is deprecated as of Qiskit Terra 9.99. It will be removed "
                        "no earlier than 3 months after the release date. Instead, use foo."
                    ),
                    since="9.99",
                    pending=False,
                ),
                _DeprecationMetadataEntry(
                    (
                        "Using the argument arg3 is deprecated as of Qiskit Terra 9.99. It will "
                        "be removed no earlier than 3 months after the release date. Instead, "
                        "use the argument `new_arg3`, which behaves identically."
                    ),
                    since="9.99",
                    pending=False,
                ),
                _DeprecationMetadataEntry(
                    (
                        "TestDeprecations.test_deprecate_argument_metadata.<locals>.my_func()'s "
                        "argument `arg2` is pending deprecation as of Qiskit Terra 9.99. "
                        "It will be marked deprecated in a future release, and then removed in a "
                        "future release."
                    ),
                    since="9.99",
                    pending=True,
                ),
                _DeprecationMetadataEntry(
                    (
                        "TestDeprecations.test_deprecate_argument_metadata.<locals>.my_func()'s "
                        "argument `arg1` is deprecated as of Qiskit Terra 9.99. It will "
                        "be removed no earlier than 3 months after the release date."
                    ),
                    since="9.99",
                    pending=False,
                ),
            ],
        )

    def test_deprecate_arguments_metadata(self) -> None:
        """Test that `@deprecate_arguments` stores the correct metadata."""

        @deprecate_arguments(
            {"old_arg1": "new_arg1", "old_arg2": None},
            category=PendingDeprecationWarning,
            since="9.99",
        )
        def my_func() -> None:
            pass

        self.assertEqual(
            getattr(my_func, _DeprecationMetadataEntry.dunder_name),
            [
                _DeprecationMetadataEntry(
                    "TestDeprecations.test_deprecate_arguments_metadata.<locals>.my_func keyword "
                    "argument old_arg1 is deprecated and replaced with new_arg1.",
                    since="9.99",
                    pending=True,
                ),
                _DeprecationMetadataEntry(
                    "TestDeprecations.test_deprecate_arguments_metadata.<locals>.my_func keyword "
                    "argument old_arg2 is deprecated and will in the future be removed.",
                    since="9.99",
                    pending=True,
                ),
            ],
        )

    def test_deprecate_function_metadata(self) -> None:
        """Test that `@deprecate_function` stores the correct metadata."""

        @deprecate_function("Stop using my_func!", since="9.99")
        def my_func() -> None:
            pass

        self.assertEqual(
            getattr(my_func, _DeprecationMetadataEntry.dunder_name),
            [_DeprecationMetadataEntry("Stop using my_func!", since="9.99", pending=False)],
        )

    def test_deprecate_argument_runtime_warning(self) -> None:
        """Test that `@deprecate_argument` warns whenever the arguments are used.

        Also check these edge cases:
        * If `new_alias` is set, pass the old argument as the new alias.
        * If `predicate` is set, only warn if the predicate is True.
        """

        @deprecate_argument("arg1", since="9.99")
        @deprecate_argument("arg2", new_alias="new_arg2", since="9.99")
        @deprecate_argument("arg3", predicate=lambda arg3: arg3 == "deprecated value", since="9.99")
        def my_func(*, arg1: str = "a", new_arg2: str, arg3: str = "a") -> None:
            del arg1
            del arg3
            assert new_arg2 == "z"

        my_func(new_arg2="z")  # No warnings if no deprecated args used.
        with self.assertWarns(DeprecationWarning):
            my_func(arg1="a", new_arg2="z")
        with self.assertWarns(DeprecationWarning):
            # `arg2` should be converted into `new_arg2`.
            my_func(arg2="z")  # pylint: disable=missing-kwoa

        # Test the `predicate` functionality.
        my_func(new_arg2="z", arg3="okay value")
        with self.assertWarns(DeprecationWarning):
            my_func(new_arg2="z", arg3="deprecated value")

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
