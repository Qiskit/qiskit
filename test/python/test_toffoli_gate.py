# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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

"""Test of Quantum Program with Toffoli gates"""

import unittest

from qiskit import QuantumProgram

from .common import QiskitTestCase


class TestToffoliGate(QiskitTestCase):
    def test_ccx(self):
        """Checks a CCNOT gate.

        Based on https://github.com/QISKit/qiskit-sdk-py/pull/172.
        """
        Q_program = QuantumProgram()
        q = Q_program.create_quantum_register('q', 3)
        c = Q_program.create_classical_register('c', 3)
        pqm = Q_program.create_circuit('pqm', [q], [c])

        # Toffoli gate.
        pqm.ccx(q[0], q[1], q[2])

        # Measurement.
        for k in range(3):
            pqm.measure(q[k], c[k])

        # Prepare run.
        circuits = ['pqm']
        backend = 'local_qasm_simulator'
        shots = 1024  # the number of shots in the experiment

        # Run.
        result = Q_program.execute(circuits, backend=backend, shots=shots,
                                   max_credits=3, wait=10, timeout=240)

        self.assertEqual({'000': 1024}, result.get_counts('pqm'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
