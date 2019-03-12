# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use Qiskit at level 1 (intermediate).

This example shows how an intermediate user interacts with Terra.
It builds some circuits and transpiles them with transpile options.
It then makes a qobj object which is just a container to be run on a backend.
The same qobj can be submitted to many backends (as shown).
It is the user's responsibility to make sure it can be run (i.e. it conforms
to the restrictions of the backend, if any).
This is useful when you want to compare the same
circuit on different backends without recompiling the whole circuit,
or just want to change some runtime parameters.

To control the passes that transform the circuit, we have a pass manager
for the level 2 user.
"""

import pprint, time

# Import the Qiskit modules
from qiskit import IBMQ, BasicAer
from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile, assemble_circuits
from qiskit.compiler import TranspileConfig, RunConfig
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

try:
    IBMQ.load_accounts()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")

try:
    # Create a Quantum and Classical Register and give them names.
    qubit_reg = QuantumRegister(2, name='q')
    clbit_reg = ClassicalRegister(2, name='c')

    # Making first circuit: bell state
    qc1 = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
    qc1.h(qubit_reg[0])
    qc1.cx(qubit_reg[0], qubit_reg[1])
    qc1.measure(qubit_reg, clbit_reg)

    # Making another circuit: superpositions
    qc2 = QuantumCircuit(qubit_reg, clbit_reg, name="superposition")
    qc2.h(qubit_reg)
    qc2.measure(qubit_reg, clbit_reg)

    # Setting up the backend
    print("(Aer Backends)")
    for backend in BasicAer.backends():
        print(backend.status())
    qasm_simulator = BasicAer.get_backend('qasm_simulator')
    print("(QASM Simulator configuration) ")
    pprint.pprint(qasm_simulator.configuration())
    print("(QASM Simulator properties) ")
    pprint.pprint(qasm_simulator.properties())


    # Compile and run the circuit on a real device backend
    # See a list of available remote backends
    print("\n(IBMQ Backends)")
    for backend in IBMQ.backends():
        print(backend.status())

    try:
        # select least busy available device and execute.
        least_busy_device = least_busy(IBMQ.backends(simulator=False))
    except:
        print("All devices are currently unavailable.")

    print("Running on current least busy device: ", least_busy_device)
    print("(with configuration) ")
    pprint.pprint(least_busy_device.configuration())
    print("(with properties) ")
    pprint.pprint(least_busy_device.properties())

    # Transpile the circuits to make them compatible with the experimental backend
    [qc1_new, qc2_new] = transpile(circuits=[qc1, qc2],
                                   transpile_config=TranspileConfig(backend=least_busy_device))
    print("Bell circuit before transpile:")
    print(qc1)
    print("Bell circuit after transpile:")
    print(qc1_new)
    print("Superposition circuit before transpile:")
    print(qc2)
    print("Superposition circuit after transpile:")
    print(qc2_new)

    # Assemble the two circuits into a runnable qobj
    qobj = assemble_circuits([qc1_new, qc2_new], run_config=RunConfig(shots=1000))

    # Running qobj on the simulator
    sim_job = qasm_simulator.run(qobj)

    # Getting the result
    sim_result=sim_job.result()

    # Show the results
    print(sim_result.get_counts(qc1))
    print(sim_result.get_counts(qc2))

    # Running the job.
    exp_job = least_busy_device.run(qobj)

    job_monitor(exp_job)
    exp_result = exp_job.result()

    # Show the results
    print(exp_result.get_counts(qc1))
    print(exp_result.get_counts(qc2))

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
