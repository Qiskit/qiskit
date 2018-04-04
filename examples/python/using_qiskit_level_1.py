"""
Example showing how to use qiskit (medium)

In QISKit 0.6 we will be working on a pass manager for level 2+ users

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

import pprint

# Import the QISKit
import qiskit
try:
    import Qconfig
    qiskit.register(Qconfig.APItoken, package=qiskit)
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a Qconfig.py file with your personal token?
             For now, there's only access to local simulator backends...""")


def lowest_pending_jobs(list_of_backends):
    """Returns the backend with lowest pending jobs."""
    device_status = [qiskit.get_backend(backend).status for backend in list_of_backends]

    best = min([x for x in device_status if x['available'] is True],
               key=lambda x: x['pending_jobs'])
    return best['name']

try:
    # Create a Quantum and Classical Register and giving a name.
    qubit_reg = qiskit.QuantumRegister(2, name='q')
    clbit_reg = qiskit.ClassicalRegister(2, name='c')

    # Making first circuit: bell state
    qc1 = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
    qc1.h(qubit_reg[0])
    qc1.cx(qubit_reg[0], qubit_reg[1])
    qc1.measure(qubit_reg, clbit_reg)

    # Making another circuit: superpositions
    qc2 = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="superposition")
    qc2.h(qubit_reg)
    qc2.measure(qubit_reg, clbit_reg)

    # Setting up the backend
    print("(Local Backends)")
    for backend_name in qiskit.available_backends({'local': True}):
        backend = qiskit.get_backend(backend_name)
        print(backend.status)
    my_backend = qiskit.get_backend('local_qasm_simulator')
    print("(Local QASM Simulator configuration) ")
    pprint.pprint(my_backend.configuration)
    print("(Local QASM Simulator calibration) ")
    pprint.pprint(my_backend.calibration)
    print("(Local QASM Simulator parameters) ")
    pprint.pprint(my_backend.parameters)


    # Compiling the job
    qobj = qiskit.compile([qc1, qc2], my_backend)
    # I think we need to make a qobj into a class

    # Runing the job
    sim_result = my_backend.run(qiskit.QuantumJob(qobj, preformatted=True))
    # ideally
    #   1. we need to make the run take as the input a qobj
    #   2. we need to make the run return a job object
    #
    # job = my_backend.run(qobj)
    # sim_result=job.retrieve
    # the job is a new object that runs when it does and i dont wait for it to
    # finish and can get results later
    # other job methods
    # job.abort -- use to abort the job
    # job.status   -- the status of the job

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc1))
    print(sim_result.get_counts(qc2))

    # Compile and run the Quantum Program on a real device backend
    try:
        # See a list of available remote backends
        print("\n(Remote Backends)")
        for backend_name in qiskit.available_backends({'local': False}):
            backend = qiskit.get_backend(backend_name)
            print(backend.status)

        # select least busy available device and execute.
        best_device = lowest_pending_jobs(qiskit.available_backends({'local': False,
                                                                     'simulator': False}))
        print("Running on current least busy device: ", best_device)

        my_backend = qiskit.get_backend(best_device)

        print("(with Configuration) ")
        pprint.pprint(my_backend.configuration)
        print("(with calibration) ")
        pprint.pprint(my_backend.calibration)
        print("(with parameters) ")
        pprint.pprint(my_backend.parameters)

        # Compiling the job
        compile_config = {
            'shots': 1024,
            'max_credits': 10
            }

        # I want to make it so the compile is only done once and the needing
        # a backend is optional
        qobj = qiskit.compile([qc1, qc2], my_backend, compile_config)
        # I think we need to make a qobj into a class

        # Runing the job
        q_job = qiskit.QuantumJob(qobj, preformatted=True, resources={
            'max_credits': qobj['config']['max_credits'], 'wait': 5, 'timeout': 300})

        exp_result = my_backend.run(q_job)
        # ideally
        #   1. we need to make the run take as the input a qobj
        #   2. we need to make the run return a job object
        #
        # job = my_backend.run(qobj, run_config)
        # sim_result=job.retrieve
        # the job is a new object that runs when it does and i dont wait for it
        # to finish and can get results later
        # other job methods
        # job.abort -- use to abort the job
        # job.status   -- the status of the job

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts(qc1))
        print(exp_result.get_counts(qc2))

    except:
        print("All devices are currently unavailable.")

except qiskit.QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
