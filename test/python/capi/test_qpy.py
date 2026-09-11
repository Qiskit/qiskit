# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
from pathlib import Path
import tempfile

from qiskit import QuantumCircuit, capi, qpy
from test import QiskitTestCase


class TestQpyCAPI(QiskitTestCase):
    def test_dump_python_circuit_data(self):
        """The Python ctypes binding can dump Python-owned CircuitData as QPY."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        circuit_ptr = capi.qk_circuit_borrow_from_python(circuit._data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "circuit.qpy"
            result = capi.qk_qpy_dump_file(circuit_ptr, os.fsencode(filename), 18)

            self.assertEqual(result, capi.QkExitCode.Success.value.value)
            self.assertTrue(filename.is_file())
            with filename.open("rb") as qpy_file:
                loaded = qpy.load(qpy_file)

        self.assertEqual(loaded, [circuit])
