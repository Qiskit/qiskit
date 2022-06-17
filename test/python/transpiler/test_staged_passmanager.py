# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring,missing-class-docstring

"""Test the staged passmanager logic"""

from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import Optimize1qGates, Unroller, Depth
from qiskit.test import QiskitTestCase


class TestStagedPassManager(QiskitTestCase):
    def test_default_stages(self):
        spm = StagedPassManager()
        self.assertEqual(
            spm.stages, ["init", "layout", "routing", "translation", "optimization", "scheduling"]
        )
        spm = StagedPassManager(
            init=PassManager([Optimize1qGates()]),
            routing=PassManager([Unroller(["u", "cx"])]),
            scheduling=PassManager([Depth()]),
        )
        self.assertEqual(
            [x.__class__.__name__ for passes in spm.passes() for x in passes["passes"]],
            ["Optimize1qGates", "Unroller", "Depth"],
        )

    def test_inplace_edit(self):
        spm = StagedPassManager(stages=["single_stage"])
        spm.single_stage = PassManager([Optimize1qGates(), Depth()])
        self.assertEqual(
            [x.__class__.__name__ for passes in spm.passes() for x in passes["passes"]],
            ["Optimize1qGates", "Depth"],
        )
        spm.single_stage.append(Unroller(["u"]))
        spm.single_stage.append(Depth())
        self.assertEqual(
            [x.__class__.__name__ for passes in spm.passes() for x in passes["passes"]],
            ["Optimize1qGates", "Depth", "Unroller", "Depth"],
        )

    def test_invalid_stage(self):
        with self.assertRaises(AttributeError):
            StagedPassManager(stages=["init"], translation=PassManager())

    def test_pre_phase_is_valid_stage(self):
        spm = StagedPassManager(stages=["init"], pre_init=PassManager([Depth()]))
        self.assertEqual(
            [x.__class__.__name__ for passes in spm.passes() for x in passes["passes"]],
            ["Depth"],
        )

    def test_append_extend_not_implemented(self):
        spm = StagedPassManager()
        with self.assertRaises(NotImplementedError):
            spm.append(Depth())
        with self.assertRaises(NotImplementedError):
            spm += PassManager()

    def test_invalid_stages(self):
        invalid_stages = [
            "two words",
            "two-words",
            "two+words",
            "two&words",
            "[two_words]",
            "<two_words>",
            "{two_words}",
            "(two_words)",
            "two^words",
            "two_words!",
            "^two_words",
            "@two_words",
            "two~words",
            r"two\words",
            "two/words",
        ]
        all_stages = invalid_stages + ["two_words", "init"]

        with self.assertRaises(ValueError) as err:
            StagedPassManager(all_stages)
        message = str(err.exception)
        for stage in invalid_stages:
            self.assertIn(stage, message)
