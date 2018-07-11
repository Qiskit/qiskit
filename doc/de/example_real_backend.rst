Beispiel für ein reales Chip Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Der folgende Code ist ein Beispiel für das Ausführen eines Quantum Program
auf einem echten Quantum Device:

.. code-block:: python

    # Import the Qiskit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, register

    # Set your API Token.
    # You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
    # looking for "Personal Access Token" section.
    QX_TOKEN = "API_TOKEN"
    QX_URL = "https://quantumexperience.ng.bluemix.net/api"

    # Authenticate with the IBM Q API in order to use online devices.
    # You need the API Token and the QX URL.
    register(QX_TOKEN, QX_URL)

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

    # Compile and run the Quantum Program on a real device backend
    job_exp = execute(qc, 'ibmqx4', shots=1024, max_credits=10)
    result = job_exp.result()

    # Show the results
    print(result)
    print(result.get_data())


Beispiel für ein reales Chip Backend über IBMQ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wenn Sie Zugriff auf IBM Q Funktionalität haben, kann der folgende Code zum
Ausführen des obigen Beispiels verwendet werden:

.. code-block:: python

    # Import the Qiskit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, register

    # Set your API Token and credentials.
    # You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
    # looking for "Personal Access Token" section.
    QX_TOKEN = "API_TOKEN"
    QX_URL = "https://quantumexperience.ng.bluemix.net/api"
    QX_HUB = "MY_HUB"
    QX_GROUP = "MY_GROUP"
    QX_PROJECT = "MY_PROJECT"

    # Authenticate with the IBM Q API in order to use online devices.
    # You need the API Token and the QX URL.
    register(QX_TOKEN, QX_URL,
             hub=QX_HUB,
             group=QX_GROUP,
             project=QX_PROJECT)

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

    # Compile and run the Quantum Program on a real device backend
    job_exp = execute(qc, 'ibmqx4', shots=1024, max_credits=10)
    result = job_exp.result()

    # Show the results
    print(result)
    print(result.get_data())

Bitte überprüfen Sie den Abschnitt zur Installation :ref:`qconfig-setup` für
mehr Details zum Einrichten der IBM Q Anmeldedaten.


Verwenden des HPC Online Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Das Online Backend mit Bezeichner ``ibmq_qasm_simulator_hpc`` kann über
die folgenden Parameter konfiguriert werden:

- ``multi_shot_optimization``: Boolean (True oder False)
- ``omp_num_threads``: Integer zwischen 1 und 16.

Die Parameter können für :func:`qiskit.compile` und :func:`qiskit.execute`
über den ``hpc`` Parameter spezifiziert werden. Zum Beispiel:

.. code-block:: python

    qiskit.compile(circuits,
                   backend=backend,
                   shots=shots,
                   seed=88,
                   hpc={
                       'multi_shot_optimization': True,
                       'omp_num_threads': 16
                   })

Wird das ``ibmq_qasm_simulator_hpc`` Backend verwendet und der ``hpc``
Parameter nicht angegeben, werden folgende Einstellungen standardmäßig verendet:

.. code-block:: python

    hpc={
        'multi_shot_optimization': True,
        'omp_num_threads': 16
    }

Bitte beachten Sie, dass diese Parameter nur für das
``ibmq_qasm_simulator_hpc`` Backend verwendet werden sollen. Falls die
Parameter für ein anderes Backend spezifiziert werden, erfolgt eine
automatische Zurücksetzung auf ``None`` und es wird eine Warnung vom SDK
ausgegeben.