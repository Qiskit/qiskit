# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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

import unittest

from qiskit import QuantumProgram

from .common import QiskitTestCase


class CompileSkipTranslationTest(QiskitTestCase):
    """Test compilaton with skip translation."""

    def test_simple_compile(self):
        """
        Compares with and without skip_translation
        """
        name = 'test_simple'
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit(name, [qr], [cr])
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.measure(qr, cr)

        rtrue = qp.compile([name], backend='local_qasm_simulator', shots=1024,
                           skip_translation=True)
        rfalse = qp.compile([name], backend='local_qasm_simulator', shots=1024,
                            skip_translation=False)
        self.assertEqual(rtrue['config'], rfalse['config'])
        self.assertEqual(rtrue['circuits'], rfalse['circuits'])

    def test_simple_execute(self):
        name = 'test_simple'
        seed = 42
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit(name, [qr], [cr])
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.measure(qr, cr)

        rtrue = qp.execute(name, seed=seed, skip_translation=True)
        rfalse = qp.execute(name, seed=seed, skip_translation=False)
        self.assertEqual(rtrue.get_counts(), rfalse.get_counts())


if __name__ == '__main__':
    unittest.main()
