# -*- coding: utf-8 -*-

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

import unittest
from datetime import datetime
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Nduv, Gate
import pprint
from z3 import *


def make_noisy_qubit(t1=50.0, t2=50.0):
    """Create a qubit for BackendProperties"""
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    return [Nduv(name="T1", date=calib_time, unit="µs", value=t1),
            Nduv(name="T2", date=calib_time, unit="µs", value=t2),
            Nduv(name="frequency", date=calib_time, unit="GHz", value=5.0),
            Nduv(name="readout_error", date=calib_time, unit="", value=0.01)]


def create_fake_machine():
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    qubit_list = []
    for i in range(6):
        qubit_list.append(make_noisy_qubit())
    cx01 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=500.0)]
    cx12 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=501.0)]
    cx23 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=502.0)]
    cx34 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=503.0)]
    cx45 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=504.0)]
    gcx01 = Gate(name="CX0_1", gate="cx", parameters=cx01, qubits=[0, 1])
    gcx12 = Gate(name="CX1_2", gate="cx", parameters=cx12, qubits=[1, 2])
    gcx23 = Gate(name="CX2_3", gate="cx", parameters=cx23, qubits=[2, 3])
    gcx34 = Gate(name="CX3_4", gate="cx", parameters=cx34, qubits=[3, 4])
    gcx45 = Gate(name="CX4_5", gate="cx", parameters=cx45, qubits=[4, 5])
    u1 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.001),
          Nduv(date=calib_time, name='gate_length', unit='ns', value=100.0)]
    gu10 = Gate(name="u1_0", gate="u1", parameters=u1, qubits=[0])
    gu11 = Gate(name="u1_1", gate="u1", parameters=u1, qubits=[1])
    gu12 = Gate(name="u1_2", gate="u1", parameters=u1, qubits=[2])
    gu13 = Gate(name="u1_3", gate="u1", parameters=u1, qubits=[3])
    gu14 = Gate(name="u1_4", gate="u1", parameters=u1, qubits=[4])
    u2 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.001),
          Nduv(date=calib_time, name='gate_length', unit='ns', value=100.0)]
    gu20 = Gate(name="u2_0", gate="u2", parameters=u2, qubits=[0])
    gu21 = Gate(name="u2_1", gate="u2", parameters=u2, qubits=[1])
    gu22 = Gate(name="u2_2", gate="u2", parameters=u2, qubits=[2])
    gu23 = Gate(name="u2_3", gate="u2", parameters=u2, qubits=[3])
    gu24 = Gate(name="u2_4", gate="u2", parameters=u2, qubits=[4])
    u3 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.001),
          Nduv(date=calib_time, name='gate_length', unit='ns', value=100.0)]
    gu30 = Gate(name="u3_0", gate="u3", parameters=u3, qubits=[0])
    gu31 = Gate(name="u3_1", gate="u3", parameters=u3, qubits=[1])
    gu32 = Gate(name="u3_2", gate="u3", parameters=u3, qubits=[2])
    gu33 = Gate(name="u3_3", gate="u3", parameters=u3, qubits=[3])
    gu34 = Gate(name="u3_4", gate="u3", parameters=u3, qubits=[4])

    gate_list = [gcx01, gcx12, gcx23, gcx34, gcx45,
                 gu10, gu11, gu12, gu13, gu14,
                 gu20, gu21, gu22, gu23, gu24,
                 gu30, gu31, gu32, gu33, gu34]

    bprop = BackendProperties(last_update_date=calib_time, backend_name="test_backend",
                              qubits=qubit_list, backend_version="1.0.0", gates=gate_list,
                              general=[])
    return bprop


class TestCrosstalk(QiskitTestCase):
    def test_schedule_length1(self):
        bprop = create_fake_machine()
        crosstalk_prop = {}
        # Testing with high crosstalk between CNOT 0,1 and CNOT 2,3
        crosstalk_prop[(0, 1)] = {(2, 3): 0.2}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=['u1', 'u2', 'u3', 'cx'])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop, {})
        scheduled_dag = pass_.run(dag)
        sched_circ = dag_to_circuit(scheduled_dag)
        assert(scheduled_dag.depth() == 3)

    def test_schedule_length2(self):
        bprop = create_fake_machine()
        crosstalk_prop = {}
        # Testing with no crosstalk between CNOT 0,1 and CNOT 2,3
        crosstalk_prop[(0, 1)] = {(2, 3): 0.05}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=['u1', 'u2', 'u3', 'cx'])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop, {})
        scheduled_dag = pass_.run(dag)
        sched_circ = dag_to_circuit(scheduled_dag)
        assert(scheduled_dag.depth() == 1)

    def test_schedule_length3(self):
        bprop = create_fake_machine()
        crosstalk_prop = {}
        # Testing with high crosstalk between CNOT 0,1 and single qubit op on q2
        crosstalk_prop[(0, 1)] = {(2, 3): 0.05, (2): 0.2}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.h(qr[2])

        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=['u1', 'u2', 'u3', 'cx'])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop, {})
        scheduled_dag = pass_.run(dag)
        sched_circ = dag_to_circuit(scheduled_dag)
        assert(scheduled_dag.depth() == 4)

    def test_schedule_length4(self):
        bprop = create_fake_machine()
        crosstalk_prop = {}
        # Testing with high crosstalk between CNOT 0,1 and single qubit op on q2
        crosstalk_prop[(0, 1)] = {(2, 3): 0.05, (2): 0.05}
        crosstalk_prop[(2, 3)] = {(0, 1): 0.05, (4, 5): 0.05}
        crosstalk_prop[(4, 5)] = {(2, 3): 0.05}
        crosstalk_prop[(1, 2)] = {(3, 4): 0.05}
        crosstalk_prop[(3, 4)] = {(1, 2): 0.05}
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.h(qr[2])

        mapping = [0, 1, 2, 3, 4, 5]
        layout = Layout({qr[i]: mapping[i] for i in range(6)})
        new_circ = transpile(circuit, initial_layout=layout, basis_gates=['u1', 'u2', 'u3', 'cx'])
        dag = circuit_to_dag(new_circ)
        pass_ = CrosstalkAdaptiveSchedule(bprop, crosstalk_prop, {})
        scheduled_dag = pass_.run(dag)
        sched_circ = dag_to_circuit(scheduled_dag)
        assert(scheduled_dag.depth() == 2)


if __name__ == '__main__':
    unittest.main()
