# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use Qiskit at level 1 (intermediate).

In Qiskit 0.6 we will be working on a pass manager for level 2+ users

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

import pprint

# Import the Qiskit modules
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
from qiskit import available_backends, compile, register, get_backend, least_busy

try:
    import Qconfig
    register(Qconfig.APItoken, Qconfig.config['url'])
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a Qconfig.py file with your personal token?
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
    print("(Local Backends)")
    for backend_name in available_backends({'local': True}):
        backend = get_backend(backend_name)
        print(backend.status)
    my_backend_name = 'local_qasm_simulator'
    my_backend = get_backend(my_backend_name)
    print("(Local QASM Simulator configuration) ")
    pprint.pprint(my_backend.configuration)
    print("(Local QASM Simulator calibration) ")
    pprint.pprint(my_backend.calibration)
    print("(Local QASM Simulator parameters) ")
    pprint.pprint(my_backend.parameters)


    # Compiling the job
    qobj = compile([qc1, qc2], my_backend)
    # Note: in the near future qobj will become an object

    # Runing the job
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
        print("\n(Remote Backends)")
        for backend_name in available_backends({'local': False}):
            backend = get_backend(backend_name)
            s = backend.status
            print(s)

        # select least busy available device and execute.
        least_busy_device = least_busy(available_backends())
        print("Running on current least busy device: ", least_busy_device)

        my_backend = get_backend(least_busy_device)

        print("(with Configuration) ")
        pprint.pprint(my_backend.configuration)
        print("(with calibration) ")
        pprint.pprint(my_backend.calibration)
        print("(with parameters) ")
        pprint.pprint(my_backend.parameters)

        # Compiling the job
        # I want to make it so the compile is only done once and the needing
        # a backend is optional
        qobj = compile([qc1, qc2], backend=my_backend, shots=1024, max_credits=10)

        # Runing the job.
        exp_job = my_backend.run(qobj)

        exp_result = exp_job.result()

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts(qc1))
        print(exp_result.get_counts(qc2))
    except:
        print("All devices are currently unavailable.")

except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
