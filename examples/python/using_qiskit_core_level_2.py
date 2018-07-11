# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use Qiskit at level 2 (advanced).

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

# choose a remote device
import Qconfig
from qiskit.backends.ibmq import IBMQProvider
ibmqprovider = IBMQProvider(Qconfig.APItoken, Qconfig.config['url'])
backend_device = ibmqprovider.get_backend('ibmqx4')

# 0. build circuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
q = QuantumRegister(2)
c = ClassicalRegister(2)
circ = QuantumCircuit(q, c)
circ.cx(q[0], q[1])
circ.cx(q[0], q[1])
circ.cx(q[0], q[1])
circ.cx(q[0], q[1])
circ.measure(q, c)

# draw circuit
from qiskit.tools.visualization import plot_circuit
plot_circuit(circ)

# 1. standard compile -- standard qiskit passes, when no PassManager given
from qiskit import transpiler, load_qasm_string
qobj_standard = transpiler.compile(circ, backend_device)
compiled_standard = load_qasm_string(qobj_standard['circuits'][0]['compiled_circuit_qasm'])
plot_circuit(compiled_standard)

# 2. custom compile -- customize PassManager to run specific circuit transformations
from qiskit.transpiler.passes import CXCancellation
pm = transpiler.PassManager()
pm.add_pass(CXCancellation())
qobj_custom = transpiler.compile(circ, backend_device, pass_manager=pm)
compiled_custom = load_qasm_string(qobj_custom['circuits'][0]['compiled_circuit_qasm'])
plot_circuit(compiled_custom)
