Example Real Chip Backend
^^^^^^^^^^^^^^^^^^^^^^^^^

The following code is an example of how to execute a Quantum Program on a real
Quantum device:

.. code-block:: python

    # Import the QISKit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, register

    # Set your API Token
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
    job = execute(qc, 'ibmqx4')
    result = job.result()

    # Show the results
    print(result)
    print(result.get_data())


Example Real Chip Backend using IBMQ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have access to the IBM Q features, the following code can be used for
executing the same example as described on the previous section, but using
the IBM Q features:

.. code-block:: python

    # Import the QISKit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, register

    # Set your API Token
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
    job = execute(qc, 'ibmqx4')
    result = job.result()

    # Show the results
    print(result)
    print(result.get_data())

Please check the Installation :ref:`qconfig-setup` section for more details on
how to setup your IBM Q credentials.


Using the HPC online backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ibmq_qasm_simulator_hpc`` online backend has the following configurable
parameters:

- ``multi_shot_optimization``: boolean (True or False)
- ``omp_num_threads``: integer between 1 and 16.

The parameters can be specified to :func:`qiskit.compile` and
:func:`qiskit.execute` via the ``hpc`` parameter. For example:

.. code-block:: python

    qiskit.compile(circuits,
                   backend=backend,
                   shots=shots,
                   seed=88,
                   hpc={
                       'multi_shot_optimization': True,
                       'omp_num_threads': 16
                   })

If the ``ibmq_qasm_simulator_hpc`` backend is used and the ``hpc`` parameter
is not specified, the following values will be used by default:

.. code-block:: python

    hpc={
        'multi_shot_optimization': True,
        'omp_num_threads': 16
    }


Please note that these parameters must only be used for the
``ibmq_qasm_simulator_hpc``, and will be reset to None along with emitting
a warning by the SDK if used with another backend.
