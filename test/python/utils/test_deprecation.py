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
    deprecate_arguments,
)


class TestDeprecations(QiskitTestCase):
    """Test functions in ``utils.deprecation``."""

    def test_deprecations_store_metadata(self) -> None:
        """Test that our deprecation decorators store the metadata in __qiskit_deprecations__.

        This should support multiple deprecations on the same function.
        """

        @deprecate_function("Stop using my_func!")
        @deprecate_arguments({"old_arg": "new_arg"}, category=PendingDeprecationWarning)
        def my_func(old_arg: int, new_arg: int) -> None:
            del old_arg
            del new_arg

        self.assertEqual(
            getattr(my_func, _DeprecationMetadataEntry.dunder_name),
            [
                _DeprecationMetadataEntry(
                    "my_func keyword argument old_arg is deprecated and replaced with new_arg.",
                    since="TODO",
                    pending=True,
                ),
                _DeprecationMetadataEntry("Stop using my_func!", since="TODO", pending=False),
            ],
        )
