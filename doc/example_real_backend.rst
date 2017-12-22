Example Real Chip Backend
^^^^^^^^^^^^^^^^^^^^^^^^^

The following code is an example of how to execute a Quantum Program on a real
Quantum device:

.. code-block:: python
    :linenos:

    from qiskit import QuantumProgram
    
    # Creating Programs create your first QuantumProgram object instance.
    Q_program = QuantumProgram()

    # Set your API Token
    # You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
    # looking for "Personal Access Token" section.
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


Example Real Chip Backend using IBMQ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have access to the IBM Q features, the following code can be used for
executing the same example as described on the previous section, but using
the IBM Q features:

.. code-block:: python
    :linenos:

    from qiskit import QuantumProgram

    # Creating Programs create your first QuantumProgram object instance.
    Q_program = QuantumProgram()

    # Set your API Token and credentials.
    # You can get them from https://quantumexperience.ng.bluemix.net/qx/account,
    # looking for "Personal Access Token" section.
    QX_TOKEN = "API_TOKEN"
    QX_URL = "https://quantumexperience.ng.bluemix.net/api"
    QX_HUB = "MY_HUB"
    QX_GROUP = "MY_GROUP"
    QX_PROJECT = "MY_PROJECT"

    # Set up the API and execute the program.
    # You need the API Token, the QX URL, and your hub/group/project details.
    Q_program.set_api(QX_TOKEN, QX_URL,
                      hub=QX_HUB,
                      group=QX_GROUP,
                      project=QX_PROJECT)

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


Please check the Installation :ref:`qconfig-setup` section for more details on
how to setup your IBM Q credentials.
