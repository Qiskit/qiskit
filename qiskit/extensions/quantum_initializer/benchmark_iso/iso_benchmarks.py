# -*- coding: utf-8 -*

# Copyright 2019, IBM.å©
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import BasicAer
from scipy.stats import unitary_group
from qiskit import execute
import time

from qiskit.transpiler import transpile


def time_unitary():
    times = []
    max_num_qubits = 6
    for n in range(1, max_num_qubits+1):
        q = QuantumRegister(n)
        circuit = QuantumCircuit(q)
        circuit.iso(unitary_group.rvs(2 ** n), q, [])
        start = time.time()
        execute(circuit, BasicAer.get_backend('statevector_simulator'))
        end = time.time()
        times.append(end-start)
    print("Time required for the decomposition of a unitary on i qubits (starting with i = 1). Time_List=", times)


def time_state_prep_ucg():
    times = []
    max_num_qubits = 10
    for n in range(1, max_num_qubits+1):
        q = QuantumRegister(n)
        circuit = QuantumCircuit(q)
        circuit.iso(unitary_group.rvs(2 ** n)[:, 0], [], q)
        start = time.time()
        execute(circuit, BasicAer.get_backend('statevector_simulator'))
        end = time.time()
        times.append(end-start)
    print("Time required for preparing a state on i qubits (starting with i = 1) using UCGs. Time_List=", times)


def time_state_prep_uc_rot():
    times = []
    max_num_qubits = 10
    for n in range(1, max_num_qubits+1):
        q = QuantumRegister(n)
        circuit = QuantumCircuit(q)
        circuit.initialize((unitary_group.rvs(2 ** n)[:, 0]).tolist(), q)
        start = time.time()
        execute(circuit, BasicAer.get_backend('statevector_simulator'))
        end = time.time()
        times.append(end-start)
    print("Time required for preparing a state on i qubits (starting with i = 1) using UC-rotations. Time_List=", times)


def gate_counts_for_state_prep_ucg():
    max_num_qubits = 7
    for n in range(1, max_num_qubits+1):
        q = QuantumRegister(n)
        circuit = QuantumCircuit(q)
        circuit.iso((unitary_group.rvs(2 ** n)[:, 0]), [], q)
        backend = BasicAer.get_backend('statevector_simulator')
        circuit = transpile(circuit, backend)
        print("Gate count for state preparation on", n, "qubits", circuit.count_ops())


def gate_counts_for_state_prep_uc_rot():
    max_num_qubits = 7
    for n in range(1, max_num_qubits+1):
        q = QuantumRegister(n)
        circuit = QuantumCircuit(q)
        circuit.initialize((unitary_group.rvs(2 ** n)[:, 0]).tolist(), q)
        backend = BasicAer.get_backend('statevector_simulator')
        circuit = transpile(circuit, backend)
        print("Gate count for state preparation on", n, "qubits", circuit.count_ops())


