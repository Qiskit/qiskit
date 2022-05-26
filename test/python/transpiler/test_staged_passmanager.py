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

"""Test the passmanager logic"""

import copy

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import U2Gate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager, PropertySet, StagedPassManager
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import Optimize1qGates, Unroller, Depth
from qiskit.test import QiskitTestCase


class TestStagedPassManager(QiskitTestCase):
    def test_default_stages(self):
        spm = StagedPassManager()
        self.assertEqual(
            spm.phases, ["init", "layout", "routing", "translation", "optimization", "scheduling"]
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
        spm = StagedPassManager(phases=["single_stage"])
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
            StagedPassManager(phases=["init"], translation=PassManager())

    def test_pre_phase_is_valid_stage(self):
        spm = StagedPassManager(phases=["init"], pre_init=PassManager([Depth()]))
        self.assertEqual(
            [x.__class__.__name__ for passes in spm.passes() for x in passes["passes"]],
            ["Depth"],
        )

    def test_append_extend_not_implemented(self):
        spm = StagedPassManager()
        with self.assertRaises(NotImplementedError):
            spm.append(Depth())
        with self.assertRaises(NotImplementedError):
            spm + PassManager()
