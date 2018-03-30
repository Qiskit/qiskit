"""
Example used in the README. In this example a Bell state is made.

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

# Import the QISKit SDK
import qiskit
import qiskit.api
# Import the IBM Q Experience API 
from IBMQuantumExperience import IBMQuantumExperience
# XXX ideally rather than import IBMQuantumExperience you do sth like:
# api = qiskit.remote()

# Authenticate for access to remote backends
# XXX ideally instead of import QConfig we use some localised configuration (windows: registry
# unix: dotfile, etc)
qiskit.api
import Qconfig

try:
    import Qconfig
    
    qiskit.api.register(Qconfig.APItoken)
except:
    print("""WARNING: There's no connection with IBMQuantumExperience servers.
             Have you initialized a Qconfig.py file with your personal token?
             For now, there's only access to local simulator backends...""")
try:
    # Create a Quantum Register called "qr" with 2 qubits.
    qr = qiskit.QuantumRegister("qr", 2)
    # Create a Classical Register called "cr" with 2 bits.
    cr = qiskit.ClassicalRegister("cr", 2)

    # Create a Quantum Circuit called involving "qr" and "cr"
    qc1 = qiskit.QuantumCircuit(qr, cr)
    # Add a H gate on qubit 0, putting this qubit in superposition.
    qc1.h(qr[0])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    qc1.cx(qr[0], qr[1])
    # Add a Measure gate to see the state.
    qc1.measure(qr, cr)

    # making another circuit of all superpositions
    qc2 = qiskit.QuantumCircuit(qr, cr)
    qc2.h(qr)
    qc2.measure(qr, cr)

    #setting up the backend
    print("(Local Backends)")
    for backend in qiskit.backends.local_backends():
        print(backend)
    my_backend = qiskit.backends.get_backend_instance('local_qasm_simulator')
    # ideally this should be
    # my_backend = qiskit.backends.get_backend_instance(filter on local and qasm simulator)
    # backend methods that exist are .config, .status .calibration and .run and .parameters
    # new method is .validate which returns a ticket that goes though some simulaitons

    #compiling the job
    qp = qiskit.QuantumProgram()
    qp.add_circuit("bell", qc1)
    qp.add_circuit("superposition", qc2)
    circuit_runs = ["bell","superposition"]
    qobj = qp.compile(circuit_runs, backend='local_qasm_simulator', shots=1024, seed=1)
    q_job = qiskit.QuantumJob(qobj, preformatted=True)
    # I am not convince the q_job is the correct class i would make a qobj class
    # ideally this should be qobj = qiskit.compile([qc],config) or qobj = QuantumObject([qc]) then qobj.compile
    # set the congig 
    # with the coupling layout, qubit layout, gate set etc
    # qobj = qiskit.compile([qc],config) 


    #runing the job
    sim_result = my_backend.run(q_job)
    #ideally this would be
    #job = my_backend.run(qobj)
    #job.status
    #sim_result=job.results
    # the job is a new object that runs when it does and i dont wait for it to finish and can get results later
    # other job methods are job.abort


    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts("bell"))
    print(sim_result.get_counts("superposition"))

    # Compile and run the Quantum Program on a real device backend
    if qiskit.backends.remote_backends():


        # see a list of available remote backends
        print("\n(Remote Backends)")
        for backend in qiskit.backends.remote_backends():
            exp_backend = qiskit.backends.get_backend_instance(backend)
            backend_status = exp_backend.status
            print(backend, backend_status)

        # select least busy available device and execute
        device_status = [backend.status
                         for backend in qiskit.backends.remote_backends() if "simulator" not in backend]
        best_device = min([x for x in device_status if x['available']==True],
                          key=lambda x:x['pending_jobs'])
        print("Running on current least busy device: ", best_device['backend'])
        my_backend = qiskit.backends.get_backend_instance(best_device['backend'])
        # this gets replaced by 
        # my_backend = qiskit.backends.get_backend_instance(filter remote, device, smallest queue)

        #compiling the job
        qp = qiskit.QuantumProgram()
        qp.add_circuit("bell", qc1)
        qp.add_circuit("superposition", qc2)
        circuit_runs = ["bell","superposition"]
        qobj = qp.compile(circuit_runs, backend=best_device['backend'], shots=1024, seed=1)
        wait = 5
        timeout = 300
        q_job = qiskit.QuantumJob(qobj, preformatted=True, resources={
                    'max_credits': qobj['config']['max_credits'], 'wait': wait,
                    'timeout': timeout})
        # I am not convince the q_job is the correct class i would make a qobj class
        # ideally this should be qobj = qiskit.compile([qc],config) or qobj = QuantumObject([qc]) then qobj.compile

        #runing the job
        exp_result = my_backend.run(q_job)
        #ideally this would be
        #job = my_backend.run(qobj)
        #job.status
        #sim_result=job.results
        # the job is a new object that runs when it does and i dont wait for it to finish and can get results later
        # other job methods are job.abort
        #job = my_backend.run(qobj)
        #job.status()
        #exp_result=job.results()

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts("bell"))
        print(exp_result.get_counts("superposition"))

except qiskit.QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
