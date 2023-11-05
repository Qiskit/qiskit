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
# pylint: disable=missing-class-docstring

"""Pass manager test cases."""

from test.python.passmanager import PassManagerTestCase

from qiskit.passmanager import GenericPass, BasePassManager
from qiskit.passmanager.flow_controllers import DoWhileController, ConditionalController


class RemoveFive(GenericPass):
    def run(self, passmanager_ir):
        return passmanager_ir.replace("5", "")


class AddDigit(GenericPass):
    def run(self, passmanager_ir):
        return passmanager_ir + "0"


class CountDigits(GenericPass):
    def run(self, passmanager_ir):
        self.property_set["ndigits"] = len(passmanager_ir)


class ToyPassManager(BasePassManager):
    def _passmanager_frontend(self, input_program, **kwargs):
        return str(input_program)

    def _passmanager_backend(self, passmanager_ir, in_program, **kwargs):
        return int(passmanager_ir)


class TestPassManager(PassManagerTestCase):
    def test_single_task(self):
        """Test case: Pass manager with a single task."""

        task = RemoveFive()
        data = 12345
        pm = ToyPassManager(task)
        expected = [r"Pass: RemoveFive - (\d*\.)?\d+ \(ms\)"]
        with self.assertLogContains(expected):
            out = pm.run(data)
        self.assertEqual(out, 1234)

    def test_property_set(self):
        """Test case: Pass manager can access property set."""

        task = CountDigits()
        data = 12345
        pm = ToyPassManager(task)
        pm.run(data)
        self.assertDictEqual(pm.property_set, {"ndigits": 5})

    def test_do_while_controller(self):
        """Test case: Do while controller that repeats tasks until the condition is met."""

        def _condition(property_set):
            return property_set["ndigits"] < 7

        controller = DoWhileController([AddDigit(), CountDigits()], do_while=_condition)
        data = 12345
        pm = ToyPassManager(controller)
        pm.property_set["ndigits"] = 5
        expected = [
            r"Pass: AddDigit - (\d*\.)?\d+ \(ms\)",
            r"Pass: CountDigits - (\d*\.)?\d+ \(ms\)",
            r"Pass: AddDigit - (\d*\.)?\d+ \(ms\)",
            r"Pass: CountDigits - (\d*\.)?\d+ \(ms\)",
        ]
        with self.assertLogContains(expected):
            out = pm.run(data)
        self.assertEqual(out, 1234500)

    def test_conditional_controller(self):
        """Test case: Conditional controller that run task when the condition is met."""

        def _condition(property_set):
            return property_set["ndigits"] > 6

        controller = ConditionalController([RemoveFive()], condition=_condition)
        data = [123456789, 45654, 36785554]
        pm = ToyPassManager([CountDigits(), controller])
        out = pm.run(data)
        self.assertListEqual(out, [12346789, 45654, 36784])

    def test_string_input(self):
        """Test case: Running tasks once for a single string input.

        Details:
            When the pass manager receives a sequence of input values,
            it duplicates itself and run the tasks on each input element in parallel.
            If the input is string, this can be accidentally recognized as a sequence.
        """

        class StringPassManager(BasePassManager):
            def _passmanager_frontend(self, input_program, **kwargs):
                return input_program

            def _passmanager_backend(self, passmanager_ir, in_program, **kwargs):
                return passmanager_ir

        class Task(GenericPass):
            def run(self, passmanager_ir):
                return passmanager_ir

        task = Task()
        data = "12345"
        pm = StringPassManager(task)

        # Should be run only one time
        expected = [r"Pass: Task - (\d*\.)?\d+ \(ms\)"]
        with self.assertLogContains(expected):
            out = pm.run(data)
        self.assertEqual(out, data)
