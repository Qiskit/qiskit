# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use Qiskit at level 2 (advanced).

The order of the passes can determine the best way for a circuit to be complied. Here we
make a simple circuit of 4 repeated CNOTs and show that the default pass is not as good
as making a pass manager and telling it to start with CXCancellation.


"""

# choose a remote device
from qiskit import IBMQ, qobj_to_circuits

try:
    IBMQ.load_accounts()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")


backend_device = IBMQ.get_backend('ibmqx4')

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
print(circ.qasm())

# 1. standard compile -- standard qiskit passes, when no PassManager given
from qiskit import transpiler
dags = transpiler.transpile(circ, backend_device)
[compiled_standard] = dags
print(compiled_standard.qasm())

# 2. custom compile -- customize PassManager to run specific circuit transformations
from qiskit.transpiler.passes import CXCancellation
pm = transpiler.PassManager()
pm.add_passes(CXCancellation())
dags = transpiler.transpile(circ, backend_device, pass_manager=pm)
[compiled_custom] = dags
print(compiled_custom.qasm())
