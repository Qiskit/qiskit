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
from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir.ir import IrBlock
from qiskit.providers.fake_provider import GenericBackendV2

from test import QiskitTestCase  # pylint: disable=wrong-import-order
from . import _dummy_programs as schedule_lib


class _DummyPass(TransformationPass):
    """A test pass that doesn't perform any transformation."""

    def run(self, passmanager_ir: IrBlock) -> IrBlock:
        return passmanager_ir

    def __hash__(self) -> int:
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name == other.__class__.name__


class TestCompiler(QiskitTestCase):
    """Test case for pulse compiler."""

    def test_roundtrip_simple(self):
        """Test just returns the input program."""
        # Just convert an input to PulseIR and convert it back to ScheduleBlock.
        backend = GenericBackendV2(2)
        pm = BlockTranspiler([_DummyPass(backend.target)])

        in_prog = schedule_lib.play_gaussian()
        out_prog = pm.run(pulse_programs=in_prog)
        self.assertEqual(in_prog, out_prog)

    def test_roundtrip_nested(self):
        """Test just returns the input program."""
        # Just convert an input to PulseIR and convert it back to ScheduleBlock.
        backend = GenericBackendV2(2)
        pm = BlockTranspiler([_DummyPass(backend.target)])

        in_prog = schedule_lib.play_and_inner_block_sequential()
        out_prog = pm.run(pulse_programs=in_prog)
        self.assertEqual(in_prog, out_prog)
