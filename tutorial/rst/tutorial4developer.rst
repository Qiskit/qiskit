###############################
Getting Started with QISKit SDK
###############################

For more information about how to use the IBM Q experience (QX), consult
the
`tutorials <https://quantumexperience.ng.bluemix.net/qstage/#/tutorial?sectionId=c59b3710b928891a1420190148a72cce&pageIndex=0>`__,
or check out the
`community <https://quantumexperience.ng.bluemix.net/qstage/#/community>`__.


Contributors
============

Ismael Faro, Jay Gambetta, Andrew Cross

QISKit SDK Tutorial
===================

This tutorial aims to explain how to use the QISKit SDK from a
developer's point of view. We review the steps it takes to install and
start to use the SDK tools.

QISKit is a Python software development kit (SDK) that you can use to
create your quantum computing programs based on circuits defined through
the `OpenQASM 2.0
specification <https://github.com/IBM/qiskit-openqasm>`__, compile them,
and execute them on several backends (Real Quantum Processors online,
Simulators online, and Simulators on local). For the online backends,
QISKit uses our `python API
connector <https://github.com/IBM/qiskit-api-py>`__ to the `IBM Q
experience project <http://quantumexperience.ng.bluemix.net/>`__.

In addition to this tutorial, we have other tutorials that introduce you
to more complex concepts directly related to quantum computing.

More examples: - Familiarize yourself with the important concepts of
:doc:`superposition and entanglement <superposition_and_entanglement>`. - Go beyond and
explore a bit more in-depth in :doc:`entanglement revisited <entanglement_revisited>`.

Install QISKit
==============

The easiest way to install QISKit is with the Anaconda Python
distribution.

-  Install Anaconda: https://www.continuum.io/downloads

Next, install QISKit from the git repository

-  Clone the repo:

.. code:: sh

    git clone https://github.ibm.com/IBMQuantum/qiskit-sdk-py-dev
    cd qiskit-sdk-py-dev

-  Create the environment with the dependencies:

.. code:: sh

    make env

Use QISKit Python SDK
=====================

You can try out the examples easily with Jupyter or Python.

Add your personal API token to the file "Qconfig.py" (get it from your
`IBM Q experience <https://quantumexperience.ng.bluemix.net>`__ >
Account):

.. code:: sh

    cp tutorial/Qconfig.py.default Qconfig.py

Run Jupyter notebook.

.. code:: sh

    make run

Basic Concept
-------------

The basic concept of our quantum program is an array of quantum
circuits. The program workflow consists of three stages: :ref:`Build
<building_your_program>`, :ref:`Compile <compile_and_run>`, and
:ref:`Run <execute_on_real_device>`.  Build allows you to make
different quantum circuits that represent the problem you are solving;
Compile allows you to rewrite them to run on different backends
(simulators/real chips of different `quantum volumes
<http://ibm.biz/qiskit-quantum-volume>`__, sizes, fidelity, etc); and
Run launches the jobs.  After the jobs have been run, the data is
collected. There are methods for putting this data together, depending
on the program. This either gives you the answer you wanted or allows
you to make a better program for the next instance.

.. _building_your_program:

Building your program: Create it 
--------------------------------

First you need to import the QuantumProgram package from QISKit.

.. code:: python

    import sys
    sys.path.append("../../") # solve the relative dependencies if you clone QISKit from the Git repo and use like a global.
    
    from qiskit import QuantumProgram
    import Qconfig

The basic elements needed for your first program are the QuantumProgram,
a Circuit, a Quantum Register, and a Classical Register.

.. code:: python

    # Creating Programs
    # create your first QuantumProgram object instance.
    Q_program = QuantumProgram()
    
    # Creating Registers
    # create your first Quantum Register called "qr" with 2 qubits 
    qr = Q_program.create_quantum_registers("qr", 2)
    # create your first Classical Register  called "cr" with 2 bits
    cr = Q_program.create_classical_registers("cr", 2)
    
    # Creating Circuits
    # create your first Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = Q_program.create_circuit("qc", ["qr"], ["cr"])


.. parsed-literal::

    >> quantum_registers created: qr 2
    >> classical_registers created: cr 2


Another option for creating your QuantumProgram instance is to define a
dictionary with all the necessary components of your program.

.. code:: python

    Q_SPECS = {
        "circuits": [{
            "name": "Circuit",
            "quantum_registers": [{
                "name": "qr",
                "size": 4
            }],
            "classical_registers": [{
                "name": "cr",
                "size": 4
            }]}],
    }

The required element for a Program is a "circuits" array. Within
"circuits", the required field is "name"; it can have several Quantum
Registers and Classical Registers. Every register must have a name and
the number of each element (qubits or bits).

After that, you can use this dictionary definition as the specs of one
QuantumProgram object to initialize it.

.. code:: python

    Q_program = QuantumProgram(specs=Q_SPECS)


.. parsed-literal::

    >> quantum_registers created: qr 4
    >> classical_registers created: cr 4


You can also get every component from your new Q\_program to use.

.. code:: python

    # Get the components.
    
    # get the circuit by Name
    circuit = Q_program.get_circuit("Circuit")
    
    # get the Quantum Register by Name
    quantum_r = Q_program.get_quantum_registers("qr")
    
    # get the Classical Register by Name
    classical_r = Q_program.get_classical_registers('cr')

Building your program: Add Gates to your Circuit
------------------------------------------------

After you create the circuit with its registers, you can add gates to
manipulate the registers. Below is a list of the gates you can use in
the QX.

You can find extensive information about these gates and how use them in
our `Quantum Experience User
Guide <https://quantumexperience.ng.bluemix.net/qstage/#/tutorial?sectionId=71972f437b08e12d1f465a8857f4514c&pageIndex=2>`__.

.. code:: python

    # H (Hadamard) gate to the qubit 0 in the Quantum Register "qr" 
    circuit.h(quantum_r[0])
    
    # Pauli X gate to the qubit 1 in the Quantum Register "qr" 
    circuit.x(quantum_r[1])
    
    # Pauli Y gate to the qubit 2 in the Quantum Register "qr" 
    circuit.y(quantum_r[2])
    
    # Pauli Z gate to the qubit 3 in the Quantum Register "qr" 
    circuit.z(quantum_r[3])
    
    # CNOT (Controlled-NOT) gate from qubit 0 to the Qbit 2
    circuit.cx(quantum_r[0], quantum_r[2])
    
    # add a barrier to your circuit
    circuit.barrier()
    
    # first physical gate: u1(lambda) to qubit 0
    circuit.u1(0.3, quantum_r[0])
    
    # second physical gate: u2(phi,lambda) to qubit 1
    circuit.u2(0.3, 0.2, quantum_r[1])
    
    # second physical gate: u3(theta,phi,lambda) to qubit 2
    circuit.u3(0.3, 0.2, 0.1, quantum_r[2])
    
    # S Phase gate to qubit 0
    circuit.s(quantum_r[0])
    
    # T Phase gate to qubit 1
    circuit.t(quantum_r[1])
    
    # identity gate to qubit 1
    circuit.iden(quantum_r[1])
    
    # Note: "if" is not implemented in the local simulator right now,
    #       so we comment it out here. You can uncomment it and
    #       run in the online simulator if you'd like.
    
    # Classical if, from qubit2 gate Z to classical bit 1
    # circuit.z(quantum_r[2]).c_if(classical_r, 0)
    
    # measure gate from the qubit 0 to classical bit 0
    circuit.measure(quantum_r[0], classical_r[0])





.. parsed-literal::

    <qiskit._measure.Measure at 0x112c72518>



Extract QASM
------------

You can obtain a QASM representation of your code.

.. code:: python

    # QASM from a program
    
    QASM_source = Q_program.get_qasm("Circuit")
    
    print(QASM_source)


.. parsed-literal::

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg qr[4];
    creg cr[4];
    h qr[0];
    x qr[1];
    y qr[2];
    z qr[3];
    cx qr[0],qr[2];
    barrier qr[0],qr[1],qr[2],qr[3];
    u1(0.300000000000000) qr[0];
    u2(0.300000000000000,0.200000000000000) qr[1];
    u3(0.300000000000000,0.200000000000000,0.100000000000000) qr[2];
    s qr[0];
    t qr[1];
    id qr[1];
    measure qr[0] -> cr[0];
    

.. _compile_and_run:

Compile and Run or Execute 
--------------------------

.. code:: python

    device = 'ibmqx_qasm_simulator' # Backend to execute your program, in this case it is the online simulator
    circuits = ["Circuit"]  # Group of circuits to execute
    
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url




.. parsed-literal::

    True



.. code:: python

    Q_program.compile(circuits, device) # Compile your program
    
    # Run your program in the device and check the execution result every 2 seconds 
    result = Q_program.run(wait=2, timeout=240)
    
    print(result)


.. parsed-literal::

    running on backend: ibmqx_qasm_simulator
    {'status': 'COMPLETED', 'result': 'all done'}


When you run a program, the possible results will be:

::

    JOB_STATUS = {
        inProgress: 'RUNNING',
        errorOnCreate: 'ERROR_CREATING_JOB',
        errorExecuting: 'ERROR_RUNNING_JOB',
        completed: 'COMPLETED'
      };

The *run()* command waits until the job either times out, returns an
error message, or completes successfully.

.. code:: python

    Q_program.get_counts("Circuit")




.. parsed-literal::

    {'0000': 529, '0001': 495}



In addition to getting the number of times each output was seen, you can
get the compiled QASM. For this simulation, the compiled circuit is not
much different from the input circuit. Each single-qubit gate has been
expressed as a u1, u2, or u3 gate.

.. code:: python

    compiled_qasm = Q_program.get_compiled_qasm("Circuit")
    
    print(compiled_qasm)


.. parsed-literal::

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg qr[4];
    creg cr[4];
    u1(3.141592653589793) qr[3];
    u3(3.141592653589793,1.5707963267948966,1.5707963267948966) qr[2];
    u3(3.141592653589793,0.0,3.141592653589793) qr[1];
    u2(0.0,3.141592653589793) qr[0];
    cx qr[0],qr[2];
    barrier qr[0],qr[1],qr[2],qr[3];
    u1(0.3) qr[0];
    u1(1.5707963267948966) qr[0];
    measure qr[0] -> cr[0];
    u2(0.3,0.2) qr[1];
    u1(0.7853981633974483) qr[1];
    id qr[1];
    u3(0.3,0.2,0.1) qr[2];
    


You can use *execute()* to combine the compile and run in a single step.

.. code:: python

    Q_program.execute(circuits, device, wait=2, timeout=240)


.. parsed-literal::

    running on backend: ibmqx_qasm_simulator




.. parsed-literal::

    {'result': 'all done', 'status': 'COMPLETED'}



Compile Parameters
^^^^^^^^^^^^^^^^^^

Q\_program.compile(circuits, device="simulator", shots=1024,
max\_credits=3, basis\_gates=None, coupling\_map=None, seed=None)

* ``circuits`` array of circuits to compile
	
* ``device`` specifies the backend which is one of,
	
  - ``simulator`` online default simulator links to ibmqx\_qasm\_simulator
  - ``real`` online default real chip links to ibmqx2
  - ``ibmqx_qasm_simulator`` qasm simulator
  - ``ibmqx2`` online real chip with 5 qubits
  - ``ibmqx3`` online real chip with 16 qubits
  - ``local_unitary_simulator`` local unitary simulator
  - ``local_qasm_simulator`` local simulator
* ``shots`` number of shots, only for real chips and qasm simulators
	
* ``max_credits`` Maximum number of the credits to spend in the executions. If the executions cost
	more than your available credits, the job is aborted
	
* ``basis_gates``: the base gates by default are: u1, u2, u3, cx, id
	
* ``coupling_map``: object that represents the physical/topological layout of a chip.
	
* ``seed`` for the qasm simulator if you want to set the initial seed.

Run Parameters
^^^^^^^^^^^^^^
Q\_program.run(wait=5, timeout=60)

* ``wait`` time to wait before checking if the execution is COMPLETED.
* ``timeout`` timeout of the execution.

Execute Parameters
^^^^^^^^^^^^^^^^^^
*Execute has the combined parameters of compile and run.*

Q\_program.execute(circuits, device, shots=1024, max\_credits=3,
basis\_gates=None, wait=5, timeout=60, basis\_gates=None,
coupling\_map=None,)

.. _execute_on_real_device:

Execute on a Real Device
------------------------

.. code:: python

    device = 'ibmqx2'   # Backend where you execute your program; in this case, on the Real Quantum Chip online 
    circuits = ["Circuit"]   # Group of circuits to execute
    shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
    max_credits = 3          # Maximum number of credits to spend on executions. 
    
    result = Q_program.execute(circuits, device, shots, max_credits=3, wait=10, timeout=240)


.. parsed-literal::

    running on backend: ibmqx2
    status = RUNNING (10 seconds)
    status = RUNNING (20 seconds)


Result
^^^^^^

You can access the result via the function
*get\_counts("circuit\_name")*. By default, the last device is used, but
you can be more specific by using *get\_counts("circuit\_name",
device="device\_name")*.

.. code:: python

    Q_program.get_counts("Circuit")




.. parsed-literal::

    {'00000': 516, '00001': 508}



Execute on a local simulator
----------------------------

.. code:: python

    Q_program.compile(circuits, "local_qasm_simulator") # Compile your program
    
    # Run your program in the device and check the execution result every 2 seconds 
    result = Q_program.run(wait=2, timeout=240)
    
    Q_program.get_counts("Circuit")


.. parsed-literal::

    running on backend: local_qasm_simulator




.. parsed-literal::

    {'0000': 511, '0001': 513}



