# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager test cases."""

import io
import re
from itertools import zip_longest
from logging import StreamHandler, getLogger

from qiskit.passmanager import PassManagerState, WorkflowStatus, PropertySet
from qiskit.test import QiskitTestCase


class PassManagerTestCase(QiskitTestCase):
    """Test case for the pass manager module."""

    def setUp(self):
        super().setUp()

        self.output = io.StringIO()
        self.state = PassManagerState(
            workflow_status=WorkflowStatus(),
            property_set=PropertySet(),
        )

        logger = getLogger()
        self.addCleanup(logger.setLevel, logger.level)

        logger.setLevel("DEBUG")
        logger.addHandler(StreamHandler(self.output))

    def assertLogEqual(self, func, expected_lines, *args, exception_type=None):
        """Execute provided function and verify logger.

        Args:
            func (Callable): Function to test.
            expected_lines (List[str]): Expected log output.
            args (Any): Arguments to the function.
            exception_type (Type): Optional. Expected exception for error handling.

        Returns:
            Any: Output values from the function.
        """
        if exception_type:
            self.assertRaises(exception_type, func, *args)
            out = None
        else:
            out = func(*args)

        self.output.seek(0)
        recorded_lines = [line.rstrip() for line in self.output.readlines()]
        for i, (expected, recorded) in enumerate(zip_longest(expected_lines, recorded_lines)):
            expected = expected or ""
            recorded = recorded or ""
            if not re.fullmatch(expected, recorded):
                raise AssertionError(
                    f"Log didn't match. Mismatch found at line #{i}.\n\n"
                    f"Expected:\n{self._format_log(expected_lines)}\n"
                    f"Recorded:\n{self._format_log(recorded_lines)}"
                )
        return out

    def _format_log(self, lines):
        out = ""
        for i, line in enumerate(lines):
            out += f"#{i:02d}: {line}\n"
        return out
