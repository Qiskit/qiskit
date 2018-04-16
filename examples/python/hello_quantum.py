"""
Example used in the README. In this example a Bell state is made.

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

# Import the QISKit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
from qiskit.wrapper import available_backends, execute, register, get_backend

# Authenticate for access to remote backends
try:
    import Qconfig
    register(Qconfig.APItoken, Qconfig.config['url'])
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a Qconfig.py file with your personal token?
             For now, there's only access to local simulator backends...""")


def lowest_pending_jobs():
    """Returns the backend with lowest pending jobs."""
    list_of_backends = available_backends(
        {'local': False, 'simulator': False})
    device_status = [get_backend(backend).status for backend in list_of_backends]

    best = min([x for x in device_status if x['available'] is True],
               key=lambda x: x['pending_jobs'])
    return best['name']

try:
    # Create a Quantum Register with 2 qubits.
    q = QuantumRegister(2)
    # Create a Classical Register with 2 bits.
    c = ClassicalRegister(2)
    # Create a Quantum Circuit
    qc = QuantumCircuit(q, c)

    # Add a H gate on qubit 0, putting this qubit in superposition.
    qc.h(q[0])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    qc.cx(q[0], q[1])
    # Add a Measure gate to see the state.
    qc.measure(q, c)

    # See a list of available local simulators
    print("Local backends: ", available_backends({'local': True}))

    # Compile and run the Quantum circuit on a simulator backend
    sim_result = execute(qc, "local_qasm_simulator")

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc))

    # see a list of available remote backends
    remote_backends = available_backends({'local': False, 'simulator': False})

    print("Remote backends: ", remote_backends)
    # Compile and run the Quantum Program on a real device backend
    try:
        best_device = lowest_pending_jobs()
        print("Running on current least busy device: ", best_device)

        #runing the job
        compile_config = {
            'shots': 1024,
            'max_credits': 10
            }
        exp_result = execute(qc, best_device, compile_config, wait=5, timeout=300)

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts(qc))
    except:
        print("All devices are currently unavailable.")

except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
