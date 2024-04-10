# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""PassManager pretty-print."""

import unittest.mock
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestPresetPassManager(QiskitTestCase):
    """Pretty-print of preset StagedPassManager"""

    def test_preset0(self):
        """Pretty-print of preset level 0 passmanager."""
        level_0 = generate_preset_pass_manager(0).pprint()
        expected = "\n".join(
            [
                "pre_init",
                "  [0]",
                "    - ContainsInstruction",
                "  [1] ConditionalController",
                "    - Error",
                "translation",
                "  [2]",
                "    - UnitarySynthesis",
                "  [3]",
                "    - HighLevelSynthesis",
                "  [4]",
                "    - BasisTranslator",
            ]
        )
        self.assertEqual(expected, level_0)

    def test_preset1(self):
        """Pretty-print of preset level 1 passmanager."""
        level_1 = generate_preset_pass_manager(1).pprint()
        expected = "\n".join(
            [
                "pre_init",
                "  [0]",
                "    - ContainsInstruction",
                "  [1] ConditionalController",
                "    - Error",
                "init",
                "  [2]",
                "    - InverseCancellation",
                "translation",
                "  [3]",
                "    - UnitarySynthesis",
                "  [4]",
                "    - HighLevelSynthesis",
                "  [5]",
                "    - BasisTranslator",
                "optimization",
                "  [6]",
                "    - Depth",
                "  [7]",
                "    - FixedPoint",
                "  [8]",
                "    - Size",
                "  [9]",
                "    - FixedPoint",
                "  [10] DoWhileController",
                "    - Optimize1qGatesDecomposition",
                "    - InverseCancellation",
                "    - GatesInBasis",
                "    - Nested flow controller",
                "    - Depth",
                "    - FixedPoint",
                "    - Size",
                "    - FixedPoint",
            ]
        )
        self.assertEqual(expected, level_1)

    def test_preset2(self):
        """Pretty-print of preset level 2 passmanager."""
        level_2 = generate_preset_pass_manager(2).pprint()
        expected = "\n".join(
            [
                "pre_init",
                "  [0]",
                "    - ContainsInstruction",
                "  [1] ConditionalController",
                "    - Error",
                "init",
                "  [2]",
                "    - InverseCancellation",
                "  [3]",
                "    - CommutativeCancellation",
                "translation",
                "  [4]",
                "    - UnitarySynthesis",
                "  [5]",
                "    - HighLevelSynthesis",
                "  [6]",
                "    - BasisTranslator",
                "optimization",
                "  [7]",
                "    - Depth",
                "  [8]",
                "    - FixedPoint",
                "  [9]",
                "    - Size",
                "  [10]",
                "    - FixedPoint",
                "  [11] DoWhileController",
                "    - Optimize1qGatesDecomposition",
                "    - CommutativeCancellation",
                "    - GatesInBasis",
                "    - Nested flow controller",
                "    - Depth",
                "    - FixedPoint",
                "    - Size",
                "    - FixedPoint",
            ]
        )
        self.assertEqual(expected, level_2)

    def test_preset3(self):
        """Pretty-print of preset level 3 passmanager."""
        level_3 = generate_preset_pass_manager(3).pprint()
        expected = "\n".join(
            [
                "pre_init",
                "  [0]",
                "    - ContainsInstruction",
                "  [1] ConditionalController",
                "    - Error",
                "init",
                "  [2]",
                "    - UnitarySynthesis",
                "  [3]",
                "    - HighLevelSynthesis",
                "  [4]",
                "    - Unroll3qOrMore",
                "  [5]",
                "    - OptimizeSwapBeforeMeasure",
                "  [6]",
                "    - RemoveDiagonalGatesBeforeMeasure",
                "  [7]",
                "    - InverseCancellation",
                "  [8]",
                "    - CommutativeCancellation",
                "translation",
                "  [9]",
                "    - UnitarySynthesis",
                "  [10]",
                "    - HighLevelSynthesis",
                "  [11]",
                "    - BasisTranslator",
                "optimization",
                "  [12]",
                "    - Depth",
                "  [13]",
                "    - Size",
                "  [14]",
                "    - MinimumPoint",
                "  [15] DoWhileController",
                "    - Collect2qBlocks",
                "    - ConsolidateBlocks",
                "    - UnitarySynthesis",
                "    - Optimize1qGatesDecomposition",
                "    - CommutativeCancellation",
                "    - GatesInBasis",
                "    - Nested flow controller",
                "    - Depth",
                "    - Size",
                "    - MinimumPoint",
            ]
        )
        self.assertEqual(expected, level_3)


if __name__ == "__main__":
    unittest.main()
