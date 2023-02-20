# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for the CrosstalkAdaptiveSchedule transpiler pass.
"""

import unittest
from datetime import datetime
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.passes.optimization import CrosstalkAdaptiveSchedule
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Nduv, Gate
from qiskit.utils import optionals


def make_noisy_qubit(t_1=50.0, t_2=50.0):
    """Create a qubit for BackendProperties"""
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    return [
        Nduv(name="T1", date=calib_time, unit="µs", value=t_1),
        Nduv(name="T2", date=calib_time, unit="µs", value=t_2),
        Nduv(name="frequency", date=calib_time, unit="GHz", value=5.0),
        Nduv(name="readout_error", date=calib_time, unit="", value=0.01),
    ]


def create_fake_machine():
    """Create a 6 qubit machine to test crosstalk adaptive schedules"""
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    qubit_list = []
    for _ in range(6):
        qubit_list.append(make_noisy_qubit())
    cx01 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.05),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=500.0),
    ]
    cx12 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.05),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=501.0),
    ]
    cx23 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.05),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=502.0),
    ]
    cx34 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.05),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=503.0),
    ]
    cx45 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.05),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=504.0),
    ]
    gcx01 = Gate(name="CX0_1", gate="cx", parameters=cx01, qubits=[0, 1])
    gcx12 = Gate(name="CX1_2", gate="cx", parameters=cx12, qubits=[1, 2])
    gcx23 = Gate(name="CX2_3", gate="cx", parameters=cx23, qubits=[2, 3])
    gcx34 = Gate(name="CX3_4", gate="cx", parameters=cx34, qubits=[3, 4])
    gcx45 = Gate(name="CX4_5", gate="cx", parameters=cx45, qubits=[4, 5])
    u_1 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.001),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=100.0),
    ]
    gu10 = Gate(name="u1_0", gate="u1", parameters=u_1, qubits=[0])
    gu11 = Gate(name="u1_1", gate="u1", parameters=u_1, qubits=[1])
    gu12 = Gate(name="u1_2", gate="u1", parameters=u_1, qubits=[2])
    gu13 = Gate(name="u1_3", gate="u1", parameters=u_1, qubits=[3])
    gu14 = Gate(name="u1_4", gate="u1", parameters=u_1, qubits=[4])
    gu15 = Gate(name="u1_4", gate="u1", parameters=u_1, qubits=[5])
    u_2 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.001),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=100.0),
    ]
    gu20 = Gate(name="u2_0", gate="u2", parameters=u_2, qubits=[0])
    gu21 = Gate(name="u2_1", gate="u2", parameters=u_2, qubits=[1])
    gu22 = Gate(name="u2_2", gate="u2", parameters=u_2, qubits=[2])
    gu23 = Gate(name="u2_3", gate="u2", parameters=u_2, qubits=[3])
    gu24 = Gate(name="u2_4", gate="u2", parameters=u_2, qubits=[4])
    gu25 = Gate(name="u2_4", gate="u2", parameters=u_2, qubits=[5])
    u_3 = [
        Nduv(date=calib_time, name="gate_error", unit="", value=0.001),
        Nduv(date=calib_time, name="gate_length", unit="ns", value=100.0),
    ]
    gu30 = Gate(name="u3_0", gate="u3", parameters=u_3, qubits=[0])
    gu31 = Gate(name="u3_1", gate="u3", parameters=u_3, qubits=[1])
    gu32 = Gate(name="u3_2", gate="u3", parameters=u_3, qubits=[2])
    gu33 = Gate(name="u3_3", gate="u3", parameters=u_3, qubits=[3])
    gu34 = Gate(name="u3_4", gate="u3", parameters=u_3, qubits=[4])
    gu35 = Gate(name="u3_5", gate="u3", parameters=u_3, qubits=[5])

    gate_list = [
        gcx01,
        gcx12,
        gcx23,
        gcx34,
        gcx45,
        gu10,
        gu11,
        gu12,
        gu13,
        gu14,
        gu15,
        gu20,
        gu21,
        gu22,
        gu23,
        gu24,
        gu25,
        gu30,
        gu31,
        gu32,
        gu33,
        gu34,
        gu35,
    ]

    bprop = BackendProperties(
        last_update_date=calib_time,
        backend_name="test_backend",
        qubits=qubit_list,
        backend_version="1.0.0",
        gates=gate_list,
        general=[],
    )
    return bprop


@unittest.skipIf(not optionals.HAS_Z3, "z3-solver not installed.")
class TestCrosstalk(QiskitTestCase):
    """
    Tests for crosstalk adaptivity
    """

    def test_schedule_length1(self):
        """Testing with high crosstalk between CNOT 0,1 and CNOT 2,3"""
        bprop = create_fake_machine()
        crosstalk_prop = {}
        crosstalk_prop[(0, 1)] = {(2, 3): 0.2}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=["u1", "u2", "u3", "cx"])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop)
        scheduled_dag = pass_.run(dag)
        self.assertEqual(scheduled_dag.depth(), 3)

    def test_schedule_length2(self):
        """Testing with no crosstalk between CNOT 0,1 and CNOT 2,3"""
        bprop = create_fake_machine()
        crosstalk_prop = {}
        crosstalk_prop[(0, 1)] = {(2, 3): 0.05}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=["u1", "u2", "u3", "cx"])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop)
        scheduled_dag = pass_.run(dag)
        self.assertEqual(scheduled_dag.depth(), 1)

    def test_schedule_length3(self):
        """Testing with repeated calls to run"""
        bprop = create_fake_machine()
        crosstalk_prop = {}
        crosstalk_prop[(0, 1)] = {(2, 3): 0.2}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=["u1", "u2", "u3", "cx"])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop)
        scheduled_dag1 = pass_.run(dag)
        scheduled_dag2 = pass_.run(dag)
        self.assertEqual(scheduled_dag1.depth(), 3)
        self.assertEqual(scheduled_dag2.depth(), 3)


if __name__ == "__main__":
    unittest.main()
