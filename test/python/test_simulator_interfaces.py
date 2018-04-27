# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,no-value-for-parameter,broad-except

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Tests for checking qiskit interfaces to simulators."""

import unittest
import qiskit as qk
import qiskit.extensions.qasm_simulator_cpp
from qiskit.tools.qi.qi import state_fidelity
from qiskit.wrapper import register, available_backends, get_backend, execute
from .common import requires_qe_access, QiskitTestCase

class TestSimulatorExtensions(QiskitTestCase):
    """Test instruction extensions for simulators:
    save, load, noise, snapshot, wait
    """
    _desired_fidelity = 0.99

    def test_save_load(self):
        """save |+>|0>, do some stuff, then load"""
        q = qk.QuantumRegister(2)
        c = qk.ClassicalRegister(2)
        circ = qk.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.save(1)
        circ.cx(q[0], q[1])
        circ.cx(q[1], q[0])
        circ.h(q[1])
        circ.load(1)

        sim = 'local_statevector_simulator_cpp'
        result = execute(circ, sim)
        statevector = result.get_statevector()
        target = [0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j]
        fidelity = state_fidelity(statevector, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "save-load statevector has low fidelity{0:.2g}.".format(fidelity))

    def test_snapshot(self):
        """save |+>|0>, do some stuff, then load"""
        q = qk.QuantumRegister(2)
        c = qk.ClassicalRegister(2)
        circ = qk.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.snapshot(3)
        circ.cx(q[0], q[1])
        circ.h(q[1])
        circ.load(1)

        sim = 'local_statevector_simulator_cpp'
        result = execute(circ, sim)
        snapshot = result.get_snapshots()[3]
        statevector = snapshot['quantum_state']
        target = [0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]
        fidelity = state_fidelity(statevector, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "snapshot statevector has low fidelity{0:.2g}.".format(fidelity))

if __name__ == '__main__':
    unittest.main(verbosity=2)
