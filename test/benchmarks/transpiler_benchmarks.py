# -*- coding: utf-8 -*

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import os

import qiskit


class TranspilerBenchSuite:

    def _build_cx_circuit(self):
        if self.local_qasm_simulator is None:
            qp = qiskit.QuantumProgram()
            cx_register = qp.create_quantum_register('qr', 2)
            cx_circuit = qp.create_circuit("cx_circuit", [cx_register])
            cx_circuit.h(cx_register[0])
            cx_circuit.h(cx_register[0])
            cx_circuit.cx(cx_register[0], cx_register[1])
            cx_circuit.cx(cx_register[0], cx_register[1])
            cx_circuit.cx(cx_register[0], cx_register[1])
            cx_circuit.cx(cx_register[0], cx_register[1])
            return qp
        if self.local_qasm_simulator is not None:
            cx_register = qiskit.QuantumRegister(2)
            cx_circuit = qiskit.QuantumCircuit(cx_register)
            cx_circuit.h(cx_register[0])
            cx_circuit.h(cx_register[0])
            cx_circuit.cx(cx_register[0], cx_register[1])
            cx_circuit.cx(cx_register[0], cx_register[1])
            cx_circuit.cx(cx_register[0], cx_register[1])
            cx_circuit.cx(cx_register[0], cx_register[1])
            return cx_circuit
        return None

    def _build_single_gate_circuit(self):
        if self.local_qasm_simulator is None:
            qp = qiskit.QuantumProgram()
            single_register = qp.create_quantum_register('qr', 1)
            single_gate_circuit = qp.create_circuit('single_gate',
                                                    [single_register])
            single_gate_circuit.h(single_register[0])
            return qp
        if self.local_qasm_simulator is not None:
            single_register = qiskit.QuantumRegister(1)
            single_gate_circuit = qiskit.QuantumCircuit(single_register)
            single_gate_circuit.h(single_register[0])
            return single_gate_circuit
        return None

    def setup(self):
        version_parts = qiskit.__version__.split('.')

        if version_parts[0] == '0' and int(version_parts[1]) < 5:
            self.local_qasm_simulator = None
        elif hasattr(qiskit, 'BasicAer'):
            self.local_qasm_simulator = qiskit.BasicAer.get_backend(
                'qasm_simulator')
        elif hasattr(qiskit, 'get_backend'):
            self.local_qasm_simulator = qiskit.get_backend(
                'local_qasm_simulator')
        else:
            self.local_qasm_simulator = qiskit.BasicAer.get_backend(
                "qasm_simulator")
        self.has_compile = False
        if hasattr(qiskit, 'compile'):
            self.has_compile = True
        self.single_gate_circuit = self._build_single_gate_circuit()
        self.cx_circuit = self._build_cx_circuit()
        self.qasm_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'qasm'))
        large_qasm_path = os.path.join(self.qasm_path, 'test_eoh_qasm.qasm')

        if hasattr(qiskit, 'load_qasm_file'):
            self.large_qasm = qiskit.load_qasm_file(large_qasm_path)
        elif version_parts[0] == '0' and int(version_parts[1]) < 5:
            self.large_qasm = qiskit.QuantumProgram()
            self.large_qasm.load_qasm_file(large_qasm_path,
                                           name='large_qasm')
        else:
            self.large_qasm = qiskit.QuantumCircuit.from_qasm_file(
                large_qasm_path)

    def time_single_gate_transpile(self):
        if self.local_qasm_simulator is None:
            self.single_gate_circuit.compile('single_gate')
        else:
            if self.has_compile:
                qiskit.compile(self.single_gate_circuit,
                               self.local_qasm_simulator)
            else:
                circ = qiskit.compiler.transpile(self.single_gate_circuit,
                                                 self.local_qasm_simulator)
                qiskit.compiler.assemble(circ, self.local_qasm_simulator)

    def time_cx_transpile(self):
        if self.local_qasm_simulator is None:
            self.cx_circuit.compile('cx_circuit')
        else:
            if self.has_compile:
                qiskit.compile(self.cx_circuit, self.local_qasm_simulator)
            else:
                circ = qiskit.compiler.transpile(self.cx_circuit,
                                                 self.local_qasm_simulator)
                qiskit.compiler.assemble(circ, self.local_qasm_simulator)

    def time_transpile_from_large_qasm(self):
        if self.local_qasm_simulator is None:
            self.large_qasm.compile('large_qasm')
        else:
            if self.has_compile:
                qiskit.compile(self.large_qasm, self.local_qasm_simulator)
            else:
                circ = qiskit.compiler.transpile(self.large_qasm,
                                                 self.local_qasm_simulator)
                qiskit.compiler.assemble(circ, self.local_qasm_simulator)
