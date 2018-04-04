"""
Example showing how to use qiskit at level 0 (novice).

See level 1 if you would like to understand how to compile

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

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
    # Create a Quantum and Classical Register.
    qubit_reg = qiskit.QuantumRegister(2)
    clbit_reg = qiskit.ClassicalRegister(2)

    # making first circuit: bell state
    qc1 = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
    qc1.h(qubit_reg[0])
    qc1.cx(qubit_reg[0], qubit_reg[1])
    qc1.measure(qubit_reg, clbit_reg)

    # making another circuit: superpositions
    qc2 = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
    qc2.h(qubit_reg)
    qc2.measure(qubit_reg, clbit_reg)

    # setting up the backend
    print("(Local Backends)")
    print(qiskit.available_backends({'local': True}))

    # runing the job
    sim_result = qiskit.execute([qc1, qc2], "local_qasm_simulator")

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc1))
    print(sim_result.get_counts(qc2))

    # see a list of available remote backends
    print("\n(Remote Backends)")
    print(qiskit.available_backends({'local': False, 'simulator': False}))

    # Compile and run the Quantum Program on a real device backend
    try:
        # select least busy available device and execute.
        best_device = lowest_pending_jobs(qiskit.available_backends({'local': False,
                                                                     'simulator': False}))
        print("Running on current least busy device: ", best_device)

        # running the job
        compile_config = {
            'shots': 1024,
            'max_credits': 10
            }
        exp_result = qiskit.execute([qc1, qc2], backend=best_device,
                                    compile_config=compile_config,
                                    wait=5, timeout=300)

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts(qc1))
        print(exp_result.get_counts(qc2))
    except:
        print("All devices are currently unavailable.")
except qiskit.QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
