Example Real Chip Backend
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from qiskit import QuantumProgram
    
    # Creating Programs create your first QuantumProgram object instance.
    Q_program = QuantumProgram()

    # Set your API Token
    # You can get it from https://quantumexperience.ng.bluemix.net/qx/account, looking for "Personal Access Token" section.
    QX_TOKEN = "API_TOKEN"
    QX_URL = "https://quantumexperience.ng.bluemix.net/api"

    # Set up the API and execute the program.
    # You need the API Token and the QX URL. 
    Q_program.set_api(QX_TOKEN, QX_URL)

    # Creating Registers create your first Quantum Register called "qr" with 2 qubits
    qr = Q_program.create_quantum_register("qr", 2)
    # create your first Classical Register called "cr" with 2 bits
    cr = Q_program.create_classical_register("cr", 2)
    # Creating Circuits create your first Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = Q_program.create_circuit("superposition", [qr], [cr])

    # add the H gate in the Qubit 0, we put this Qubit in superposition
    qc.h(qr[0])

    # add measure to see the state
    qc.measure(qr, cr)

    # Compiled  and execute in the local_qasm_simulator

    result = Q_program.execute(["superposition"], backend='ibmqx2', shots=1024)

    # Show the results
    print(result)
    print(result.get_data("superposition"))