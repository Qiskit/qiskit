# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Error pass"""

import unittest

from qiskit.transpiler.passes import Error
from qiskit.test import QiskitTestCase
from qiskit.transpiler.exceptions import TranspilerError


class TestErrorPass(QiskitTestCase):
    """Tests the Error pass"""

    def test_default(self):
        """Raise error with a message (default)"""
        pass_ = Error(msg="a message")
        with self.assertRaises(TranspilerError) as excep:
            pass_.run(None)
        self.assertEqual(excep.exception.message, "a message")

    def test_raise(self):
        """Raise error with a message with variables"""
        pass_ = Error(msg="a {variable}", action="raise")
        pass_.property_set["variable"] = "message"
        with self.assertRaises(TranspilerError) as excep:
            pass_.run(None)
        self.assertEqual(excep.exception.message, "a message")

    def test_warning(self):
        """Warning error (message with variable)"""
        pass_ = Error(msg="a {variable}", action="warn")
        pass_.property_set["variable"] = "message"
        with self.assertWarns(Warning) as warn:
            pass_.run(None)
        self.assertEqual(warn.warning.args[-1], "a message")

    def test_logger(self):
        """Logger error (message with variable)"""
        pass_ = Error(msg="a {variable}", action="log")
        pass_.property_set["variable"] = "message"
        with self.assertLogs("qiskit.transpiler.passes.utils.error", level="INFO") as log:
            pass_.run(None)
        self.assertEqual(log.output, ["INFO:qiskit.transpiler.passes.utils.error:a message"])


if __name__ == "__main__":
    unittest.main()
