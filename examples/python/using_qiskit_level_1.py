"""
Example showing how to use qiskit.

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

# Import the QISKit
import qiskit

# Authenticate for access to remote backends
# XXX ideally instead of import QConfig we use some localized configuration (windows: registry
# unix: dotfile, etc)

# registering the backends from the IBM Q Experience
import Qconfig
qiskit.api.register(Qconfig.APItoken)
try:
    import Qconfig
    qiskit.api.register(Qconfig.APItoken)
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a Qconfig.py file with your personal token?
             For now, there's only access to local simulator backends...""")

local_backends = qiskit.backends.local_backends()
remote_backends = qiskit.backends.remote_backends()

try:
    # Create a Quantum and Classical Register and giving a name.
    qubit_reg = qiskit.QuantumRegister(2, name='q')
    clbit_reg = qiskit.ClassicalRegister(2, name='c')

    # making first circuit: bell state
    qc1 = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
    qc1.h(qubit_reg[0])
    qc1.cx(qubit_reg[0], qubit_reg[1])
    qc1.measure(qubit_reg, clbit_reg)

    # making another circuit: superpositions
    qc2 = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
    qc2.h(qubit_reg)
    qc2.measure(qubit_reg, clbit_reg)

    #setting up the backend
    print("(Local Backends)")
    for backend in local_backends:
        print(backend)
    my_backend = qiskit.backends.get_backend_instance('local_qasm_simulator')
    # ideally this should be
    # my_backend = qiskit.backends.get_backend_instance(filter on local and qasm simulator)
    # backend methods that exist are .config, .status .calibration and .run and .parameters
    # new method is .validate which returns a ticket that goes though some simulators

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
    print(sim_result.get_counts(qc1))
    print(sim_result.get_counts(qc2))

    # Compile and run the Quantum Program on a real device backend
    if remote_backends:


        # see a list of available remote backends
        print("\n(Remote Backends)")
        for backend in remote_backends:
            print(backend)

        # select least busy available device and execute
        # select least busy available device and execute
        device_status = [qiskit.backends.status(backend)
                         for backend in remote_backends if "simulator" not in backend]
        try:
            best_device = min([x for x in device_status if x['available']==True],
                              key=lambda x:x['pending_jobs'])
        except:
            print("All devices are currently unavailable.")
        # this gets replaced by 
        my_backend  = qiskit.backends.get_backend_instance(best_device['backend'])
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
