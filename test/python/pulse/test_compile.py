# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A test cases for pulse compilation."""

from qiskit.pulse.compiler import BlockTranspiler

from test import QiskitTestCase  # pylint: disable=wrong-import-order
from . import _dummy_programs as schedule_lib


class TestCompiler(QiskitTestCase):
    """Test case for pulse compiler."""

    def test_roundtrip_simple(self):
        """Test just returns the input program."""
        # Just convert an input to PulseIR and convert it back to ScheduleBlock.
        pm = BlockTranspiler()

        in_prog = schedule_lib.play_gaussian()
        out_prog = pm.run(pulse_programs=in_prog)
        self.assertEqual(in_prog, out_prog)

    def test_roundtrip_nested(self):
        """Test just returns the input program."""
        # Just convert an input to PulseIR and convert it back to ScheduleBlock.
        pm = BlockTranspiler()

        in_prog = schedule_lib.play_and_inner_block_sequential()
        out_prog = pm.run(pulse_programs=in_prog)
        self.assertEqual(in_prog, out_prog)
