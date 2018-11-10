========================
Running Quantum Circuits
========================

The starting point for writing code is the :class:`~qiskit.QuantumCircuit`.
A circuit are
collections of :class:`~qiskit.ClassicalRegister` objects,
:class:`~qiskit.QuantumRegister` objects and
:mod:`gates <qiskit.extensions.standard>`. Through the
:ref:`top-level functions <qiskit_top_level_functions>`, the circuits can be
sent to remote quantum devices or local simulator backends and collect the
results for further analysis.

-------------------
Qiskit Aer backends
-------------------

To compose and run a circuit on a simulator one can do,

.. code-block:: python

    # Import the Qiskit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, Aer

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

    # Compile and run the Quantum circuit on a simulator backend
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = execute(qc, backend_sim)
    result_sim = job_sim.result()

    # Show the results
    print("simulation: ", result_sim )
    print(result_sim.get_counts(qc))

The :func:`~qiskit.Result.get_counts` method outputs a dictionary of
``state:counts`` pairs;

.. code-block:: python

    {'00': 531, '11': 493}

-------------------------
IBM Q cloud real backends
-------------------------

You can execute your circuits on a real chip by using the IBM Q cloud platform. For chip details, and 
realtime information `visit IBMQ devices page <https://www.research.ibm.com/ibm-q/technology/devices/>`_.


The following code is an example of how to execute a Quantum Program on a real
Quantum device:

.. code-block:: python

    # Import Qiskit Terra
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, IBMQ

    # Set your API Token.
    # You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
    IBMQ.enable_account("MY_API_TOKEN")

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

    # Compile and run the Quantum circuit on a device.
    backend_ibmq = IBMQ.get_backend('ibmqx4')
    job_ibmq = execute(qc, backend_ibmq)
    result_ibmq = job_ibmq.result()

    # Show the results.
    print("real execution results: ", result_ibmq)
    print(result_ibmq.get_counts(qc))

-----------------------
IBM Q cloud HPC backend
-----------------------

The ``ibmq_qasm_simulator`` online backend capable of simulating up to32 qubits. It can be used the 
same way as the real chips. 

.. code-block:: python

    # Import Qiskit Terra
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, IBMQ

    # Set your API Token.
    # You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
    IBMQ.enable_account("MY_API_TOKEN")

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

    # Compile and run the Quantum circuit on a device.
    backend_ibmq = IBMQ.get_backend('ibmq_qasm_simulator')
    job_ibmq_simulator = execute(qc, backend_ibmq)
    result_ibmq_simulator = job_ibmq_simulator.result()

    # Show the results.
    print("HPC simulation results: ", result_ibmq_simulator)
    print(result_ibmq_simulator.get_counts(qc))
