# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use Qiskit at level 1 (intermediate).

This example shows how an intermediate user interacts with Terra. It builds some circuits
and compiles them from compile parameters. It makes a qobj object which is just and container to be 
run on a backend. The same qobj can run on many backends (as shown). It is the
user responsibility to make sure it can be run. This is useful when you want to compare the same
circuits on different backends or change the compile parameters.

To control the passes and we have a pass manager for level 2 user. 
"""

import pprint, time

# Import the Qiskit modules
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QiskitError
from qiskit import compile, IBMQ, Aer
from qiskit.backends.ibmq import least_busy

try:
    IBMQ.load_accounts()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")

try:
    # Create a Quantum and Classical Register and giving a name.
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
    for backend in Aer.backends():
        print(backend.status())
    my_backend = Aer.get_backend('local_qasm_simulator')
    print("(QASM Simulator configuration) ")
    pprint.pprint(my_backend.configuration())
    print("(QASM Simulator properties) ")
    pprint.pprint(my_backend.properties())


    print("\n(IMQ Backends)")
    for backend in IBMQ.backends():
        print(backend.status())

    # select least busy available device and execute.
    least_busy_device = least_busy(IBMQ.backends(simulator=False))
    print("Running on current least busy device: ", least_busy_device)
    print("(with configuration) ")
    pprint.pprint(least_busy_device.configuration())
    print("(with properties) ")
    pprint.pprint(least_busy_device.properties())


    # Compiling the job for the experimental backend 
    qobj = compile([qc1, qc2], backend=least_busy_device, shots=1024, max_credits=10)

    # Running the job
    sim_job = my_backend.run(qobj)

    # Getting the result
    sim_result=sim_job.result()

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc1))
    print(sim_result.get_counts(qc2))

    # Compile and run the Quantum Program on a real device backend
    # See a list of available remote backends
    try:
        # Running the job.
        exp_job = least_busy_device.run(qobj)

        lapse = 0
        interval = 10
        while exp_job.status().name != 'DONE':
            print('Status @ {} seconds'.format(interval * lapse))
            print(exp_job.status())
            time.sleep(interval)
            lapse += 1
        print(exp_job.status())

        exp_result = exp_job.result()

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts(qc1))
        print(exp_result.get_counts(qc2))
    except:
        print("All devices are currently unavailable.")

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
