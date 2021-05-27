# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the NoiseAdaptiveLayout pass"""

from datetime import datetime
import unittest
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Nduv, Gate


def make_qubit_with_error(readout_error):
    """Create a qubit for BackendProperties"""
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    return [
        Nduv(name="T1", date=calib_time, unit="µs", value=100.0),
        Nduv(name="T2", date=calib_time, unit="µs", value=100.0),
        Nduv(name="frequency", date=calib_time, unit="GHz", value=5.0),
        Nduv(name="readout_error", date=calib_time, unit="", value=readout_error),
    ]


class TestNoiseAdaptiveLayout(QiskitTestCase):
    """Tests the NoiseAdaptiveLayout pass."""

    def test_on_linear_topology(self):
        """
        Test that the mapper identifies the correct gate in a linear topology
        """
        calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
        qr = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        qubit_list = []
        ro_errors = [0.01, 0.01, 0.01]
        for ro_error in ro_errors:
            qubit_list.append(make_qubit_with_error(ro_error))
        p01 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.9)]
        g01 = Gate(name="CX0_1", gate="cx", parameters=p01, qubits=[0, 1])
        p12 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.1)]
        g12 = Gate(name="CX1_2", gate="cx", parameters=p12, qubits=[1, 2])
        gate_list = [g01, g12]
        bprop = BackendProperties(
            last_update_date=calib_time,
            backend_name="test_backend",
            qubits=qubit_list,
            backend_version="1.0.0",
            gates=gate_list,
            general=[],
        )
        nalayout = NoiseAdaptiveLayout(bprop)
        nalayout.run(dag)
        initial_layout = nalayout.property_set["layout"]
        self.assertNotEqual(initial_layout[qr[0]], 0)
        self.assertNotEqual(initial_layout[qr[1]], 0)

    def test_bad_readout(self):
        """Test that the mapper avoids bad readout unit"""
        calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
        qr = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        qubit_list = []
        ro_errors = [0.01, 0.01, 0.8]
        for ro_error in ro_errors:
            qubit_list.append(make_qubit_with_error(ro_error))
        p01 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.1)]
        g01 = Gate(name="CX0_1", gate="cx", parameters=p01, qubits=[0, 1])
        p12 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.1)]
        g12 = Gate(name="CX1_2", gate="cx", parameters=p12, qubits=[1, 2])
        gate_list = [g01, g12]
        bprop = BackendProperties(
            last_update_date=calib_time,
            backend_name="test_backend",
            qubits=qubit_list,
            backend_version="1.0.0",
            gates=gate_list,
            general=[],
        )
        nalayout = NoiseAdaptiveLayout(bprop)
        nalayout.run(dag)
        initial_layout = nalayout.property_set["layout"]
        self.assertNotEqual(initial_layout[qr[0]], 2)
        self.assertNotEqual(initial_layout[qr[1]], 2)

    def test_grid_layout(self):
        """
        Test that the mapper identifies best location for a star-like program graph
        Machine row1: (0, 1, 2)
        Machine row2: (3, 4, 5)
        """
        calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
        qr = QuantumRegister(4, name="q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[1], qr[3])
        circuit.cx(qr[2], qr[3])
        dag = circuit_to_dag(circuit)
        qubit_list = []
        ro_errors = [0.01] * 6
        for ro_error in ro_errors:
            qubit_list.append(make_qubit_with_error(ro_error))
        p01 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.3)]
        p03 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.3)]
        p12 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.3)]
        p14 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.1)]
        p34 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.1)]
        p45 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.1)]
        p25 = [Nduv(date=calib_time, name="gate_error", unit="", value=0.3)]
        g01 = Gate(name="CX0_1", gate="cx", parameters=p01, qubits=[0, 1])
        g03 = Gate(name="CX0_3", gate="cx", parameters=p03, qubits=[0, 3])
        g12 = Gate(name="CX1_2", gate="cx", parameters=p12, qubits=[1, 2])
        g14 = Gate(name="CX1_4", gate="cx", parameters=p14, qubits=[1, 4])
        g34 = Gate(name="CX3_4", gate="cx", parameters=p34, qubits=[3, 4])
        g45 = Gate(name="CX4_5", gate="cx", parameters=p45, qubits=[4, 5])
        g25 = Gate(name="CX2_5", gate="cx", parameters=p25, qubits=[2, 5])
        gate_list = [g01, g03, g12, g14, g34, g45, g25]
        bprop = BackendProperties(
            last_update_date=calib_time,
            backend_name="test_backend",
            qubits=qubit_list,
            backend_version="1.0.0",
            gates=gate_list,
            general=[],
        )
        nalayout = NoiseAdaptiveLayout(bprop)
        nalayout.run(dag)
        initial_layout = nalayout.property_set["layout"]
        for qid in range(4):
            for qloc in [0, 2]:
                self.assertNotEqual(initial_layout[qr[qid]], qloc)


if __name__ == "__main__":
    unittest.main()
