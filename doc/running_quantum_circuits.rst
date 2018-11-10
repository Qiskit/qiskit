========================
Running Quantum Circuits
========================

Qiskit Terra makes it simple to run quantum circuits on local or remote backends. As a simple example 
we consider a quantum circuit that make a Bell state, :math:`(|00>+|11>)/\sqrt(2)`. 

.. code-block:: python

    # Import Qiskit Terra
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute

    # Create a Quantum Register with 2 qubits.
    q = QuantumRegister(2)
    # Create a Quantum Circuit
    qc = QuantumCircuit(q)

    # Add a H gate on qubit 0, putting this qubit in superposition.
    qc.h(q[0])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    qc.cx(q[0], q[1])

This circuit consist of a quantum register of two qubits and a gate 
sequence H on qubit 0 and CX between qubit 0 and 1. This is simple to visualize using the `circuit_drawer`

.. code-block:: python

    from qiskit.tools.visualization import circuit_drawer
    circuit_drawer(qc)

.. code-block:: python

-------------------
Qiskit Aer backends
-------------------

To run this circuit on the Qiskit Aer we can use the `statevecoter_simulator` using

.. code-block:: python

    # Import Aer
    from qiskit import Aer
    import numpy as np

    # Run the quantum circuit on a statevector simulator backend
    backend_sim = Aer.get_backend('statevector_simulator')
    job_sim = execute(qc, backend_sim)
    result_sim = job_sim.result()

    # Show the results
    print("simulation: ", result_sim )
    print(np.around(result_sim.get_statevector(qc),4))

which returns the statevector 

.. code-block:: python
    
    [0.7071+0.j 0.+0.j 0.+0.j 0.7071+0.j]

Qiskit Aer also includes a `unitary_simulator` 

.. code-block:: python

    # Import Aer
    from qiskit import Aer
    import numpy as np

    # Run the quantum circuit on a unitary simulator backend
    backend_sim = Aer.get_backend('unitary_simulator')
    job_sim = execute(qc, backend_sim)
    result_sim = job_sim.result()

    # Show the results
    print("simulation: ", result_sim )
    print(np.around(result_sim.get_unitary(qc), 4))

which returns the unitary 

.. code-block:: python

    [[ 0.7071+0.j  0.7071-0.j  0.+0.j  0.+0.j]
    [ 0.+0.j  0.+0.j  0.7071+0.j -0.7071+0.j]
    [ 0.+0.j  0.+0.j  0.7071+0.j  0.7071-0.j]
    [ 0.7071+0.j -0.7071+0.j  0.+0.j  0.+0.j]]

.. note::
    The tensor order used in qiskit goes :math:`Q_n\otimes \cdots  \otimes  Q_1\otimes Q_0` which is not standard 
    and results in the CX where


https://nbviewer.jupyter.org/github/Qiskit/qiskit-tutorial/blob/master/qiskit/terra/using_different_gates.ipynb

followed by a measurement which maps 
the qubit outcomes to the classical register consisting of two bits

The :func:`~qiskit.Result.get_counts` method outputs a dictionary of
``bits:counts`` pairs;

.. code-block:: python

    {'00': 531, '11': 493}

Aer also offers a `statevector simulator` that allo

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
