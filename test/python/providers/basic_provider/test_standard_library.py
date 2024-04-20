# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring, missing-module-docstring

import unittest

from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
import qiskit.circuit.library.standard_gates as lib
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestStandard1Q(QiskitTestCase):
    """Standard Extension Test. Gates with a single Qubit"""

    def setUp(self):
        super().setUp()
        self.seed = 43
        self.shots = 1
        self.circuit = QuantumCircuit(4)

    def test_barrier(self):
        self.circuit.barrier(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_barrier_none(self):
        self.circuit.barrier()
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ccx(self):
        self.circuit.ccx(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ccz(self):
        self.circuit.ccz(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ch(self):
        self.circuit.ch(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cp(self):
        self.circuit.cp(0, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_crx(self):
        self.circuit.crx(1, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cry(self):
        self.circuit.cry(1, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_crz(self):
        self.circuit.crz(1, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cswap(self):
        self.circuit.cswap(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cu1(self):
        self.circuit.append(lib.CU1Gate(1), [1, 2])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cu3(self):
        self.circuit.append(lib.CU3Gate(1, 2, 3), [1, 2])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cx(self):
        self.circuit.cx(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ecr(self):
        self.circuit.ecr(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cy(self):
        self.circuit.cy(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cz(self):
        self.circuit.cz(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_h(self):
        self.circuit.h(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_iden(self):
        self.circuit.id(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rx(self):
        self.circuit.rx(1, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ry(self):
        self.circuit.ry(1, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rz(self):
        self.circuit.rz(1, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rxx(self):
        self.circuit.rxx(1, 1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ryy(self):
        self.circuit.ryy(1, 1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rzz(self):
        self.circuit.rzz(1, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_s(self):
        self.circuit.s(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_sdg(self):
        self.circuit.sdg(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_sx(self):
        self.circuit.sx(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_sxdg(self):
        self.circuit.sxdg(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_swap(self):
        self.circuit.swap(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_iswap(self):
        self.circuit.iswap(1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_r(self):
        self.circuit.r(0.5, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_t(self):
        self.circuit.t(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_tdg(self):
        self.circuit.tdg(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u1(self):
        self.circuit.append(lib.U1Gate(1), [1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u2(self):
        self.circuit.append(lib.U2Gate(1, 2), [1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u3(self):
        self.circuit.append(lib.U3Gate(1, 2, 3), [1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_x(self):
        self.circuit.x(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_y(self):
        self.circuit.y(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_z(self):
        self.circuit.z(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cs(self):
        self.circuit.cs(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_csdg(self):
        self.circuit.csdg(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_csx(self):
        self.circuit.csx(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cu(self):
        self.circuit.cu(0.5, 0.5, 0.5, 0.5, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_dcx(self):
        self.circuit.dcx(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_global_phase(self):
        qc = self.circuit
        qc.append(lib.GlobalPhaseGate(0.1), [])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
