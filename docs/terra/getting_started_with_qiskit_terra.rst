


Getting Started with Qiskit
===========================

Here, we provide an overview of working with Qiskit. Qiskit provides the
basic building blocks necessary to program quantum computers. The
foundation of Qiskit is the Terra element. The basic concept of Qiskit
Terra is an array of quantum circuits. A workflow using Terra consists
of two stages: **Build** and **Execute**. **Build** allows you to make
different quantum circuits that represent the problem you are solving,
and **Execute** allows you to run them on different backends. After the
jobs have been run, the data is collected. There are methods for putting
this data together, depending on the program. This either gives you the
answer you wanted, or allows you to make a better program for the next
instance.

**Contents**

`Circuit basics <#circuit_basics>`__

`Simulating circuits with Qiskit Aer <#aer_simulation>`__

`Running circuits using the IBMQ provider <#ibmq_provider>`__

**Code imports**

.. code:: ipython3

    import numpy as np
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute

Circuit Basics 
---------------

Building the circuit
~~~~~~~~~~~~~~~~~~~~

The basic elements needed for your first program are the QuantumCircuit,
and QuantumRegister.

.. code:: ipython3

    # Create a Quantum Register with 3 qubits.
    q = QuantumRegister(3, 'q')
    
    # Create a Quantum Circuit acting on the q register
    circ = QuantumCircuit(q)

.. raw:: html

   <div class="alert alert-block alert-info">

Note: Naming the QuantumRegister is optional and not required.

.. raw:: html

   </div>

After you create the circuit with its registers, you can add gates
(“operations”) to manipulate the registers. As you proceed through the
documentation you will find more gates and circuits; below is an
example of a quantum circuit that makes a three-qubit GHZ state

.. math:: |\psi\rangle = \left(|000\rangle+|111\rangle\right)/\sqrt{2}.

To create such a state, we start with a 3-qubit quantum register. By
default, each qubit in the register is initialized to :math:`|0\rangle`.
To make the GHZ state, we apply the following gates: \* A Hadamard gate
:math:`H` on qubit 0, which puts it into a superposition state. \* A
controlled-Not operation (:math:`C_{X}`) between qubit 0 and qubit 1. \*
A controlled-Not operation between qubit 0 and qubit 2.

On an ideal quantum computer, the state produced by running this circuit
would be the GHZ state above.

In Qiskit Terra, operations can be added to the circuit one-by-one, as
shown below.

.. code:: ipython3

    # Add a H gate on qubit 0, putting this qubit in superposition.
    circ.h(q[0])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    circ.cx(q[0], q[1])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
    # the qubits in a GHZ state.
    circ.cx(q[0], q[2])




.. parsed-literal::

    <qiskit.extensions.standard.cx.CnotGate at 0xa20709128>



Visualize Circuit
-----------------

You can visualize your circuit using Qiskit Terra
``QuantumCircuit.draw()``, which plots circuit in the form found in many
textbooks.

.. code:: ipython3

    circ.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">        ┌───┐          
    q_0: |0>┤ H ├──■────■──
            └───┘┌─┴─┐  │  
    q_1: |0>─────┤ X ├──┼──
                 └───┘┌─┴─┐
    q_2: |0>──────────┤ X ├
                      └───┘</pre>



In this circuit, the qubits are put in order with qubit zero at the top
and qubit two at the bottom. The circuit is read left-to-right (meaning
that gates which are applied earlier in the circuit show up further to
the left).

Simulating circuits using Qiskit Aer 
-------------------------------------

Qiskit Aer is our package for simulating quantum circuits. It provides
many different backends for doing a simulation. Here we use the basic
python version.

Statevector backend
~~~~~~~~~~~~~~~~~~~

The most common backend in Qiskit Aer is the ``statevector_simulator``.
This simulator returns the quantum state which is a complex vector of
dimensions :math:`2^n` where :math:`n` is the number of qubits (so be
careful using this as it will quickly get too large to run on your
machine).

.. raw:: html

   <div class="alert alert-block alert-info">

When representing the state of a multi-qubit system, the tensor order
used in qiskit is different than that use in most physics textbooks.
Suppose there are :math:`n` qubits, and qubit :math:`j` is labeled as
:math:`Q_{j}`. In most textbooks (such as Nielsen and Chuang’s “Quantum
Computation and Information”), the basis vectors for the :math:`n`-qubit
state space would be labeled as
:math:`Q_{0}\otimes Q_{1} \otimes \cdots \otimes Q_{n}`. **This is not
the ordering used by qiskit!** Instead, qiskit uses an ordering in which
the :math:`n^{\mathrm{th}}` qubit is on the *left* side of the tesnsor
product, so that the basis vectors are labeled as
:math:`Q_n\otimes \cdots \otimes Q_1\otimes Q_0`.

For example, if qubit zero is in state 0, qubit 1 is in state 0, and
qubit 2 is in state 1, qiskit would represent this state as
:math:`|100\rangle`, whereas most physics textbooks would represent it
as :math:`|001\rangle`.

This difference in labeling affects the way multi-qubit operations are
represented as matrices. For example, qiskit represents a controlled-X
(:math:`C_{X}`) operation with qubit 0 being the control and qubit 1
being the target as

.. math:: C_X = \begin{pmatrix} 1 & 0 & 0 & 0 \\  0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\\end{pmatrix}.

.. raw:: html

   </div>

To run the above circuit using the statevector simulator, first you need
to import Aer and then set the backend to ``statevector_simulator``.

.. code:: ipython3

    # Import Aer
    from qiskit import BasicAer
    
    # Run the quantum circuit on a statevector simulator backend
    backend = BasicAer.get_backend('statevector_simulator')

Now we have chosen the backend it’s time to compile and run the quantum
circuit. In Qiskit Terra we provide the ``execute`` function for this.
``execute`` returns a ``job`` object that encapsulates information about
the job submitted to the backend.

.. raw:: html

   <div class="alert alert-block alert-info">

Tip: You can obtain the above parameters in Jupyter. Simply place the
text cursor on a function and press Shift+Tab.

.. raw:: html

   </div>

.. code:: ipython3

    # Create a Quantum Program for execution 
    job = execute(circ, backend)

When you run a program, a job object is made that has the following two
useful methods: ``job.status()`` and ``job.result()`` which return the
status of the job and a result object respectively.

.. raw:: html

   <div class="alert alert-block alert-info">

Note: Jobs run asynchronously but when the result method is called it
switches to synchronous and waits for it to finish before moving on to
another task.

.. raw:: html

   </div>

.. code:: ipython3

    result = job.result()

The results object contains the data and Qiskit Terra provides the
method ``result.get_statevector(circ)`` to return the state vector for
the quantum circuit.

.. code:: ipython3

    outputstate = result.get_statevector(circ, decimals=3)
    print(outputstate)


.. parsed-literal::

    [0.707+0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j
     0.707+0.j]


Qiskit Terra also provides a visualization toolbox to allow you to view
these results.

Below, we use the visualization function to plot the real and imaginary
components of the state vector.

.. code:: ipython3

    from qiskit.tools.visualization import plot_state_city
    plot_state_city(outputstate)




.. image:: getting_started_with_qiskit_terra_files/getting_started_with_qiskit_terra_21_0.png



Unitary backend
~~~~~~~~~~~~~~~

Qiskit Aer also includes a ``unitary_simulator`` that works *provided
all the elements in the circuit are unitary operations*. This backend
calculates the :math:`2^n \times 2^n` matrix representing the gates in
the quantum circuit.

.. code:: ipython3

    # Run the quantum circuit on a unitary simulator backend
    backend = BasicAer.get_backend('unitary_simulator')
    job = execute(circ, backend)
    result = job.result()
    
    # Show the results
    print(result.get_unitary(circ, decimals=3))


.. parsed-literal::

    [[ 0.707+0.j  0.707+0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.707+0.j -0.707+0.j]
     [ 0.   +0.j  0.   +0.j  0.707+0.j  0.707+0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.707+0.j -0.707+0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.707+0.j  0.707+0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.707+0.j -0.707+0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.707+0.j  0.707+0.j]
     [ 0.707+0.j -0.707+0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]]


OpenQASM backend
~~~~~~~~~~~~~~~~

The simulators above are useful because they provide information about
the state output by the ideal circuit and the matrix representation of
the circuit. However, a real experiment terminates by *measuring* each
qubit (usually in the computational :math:`|0\rangle, |1\rangle` basis).
Without measurement, we cannot gain information about the state.
Measurements cause the quantum system to collapse into classical bits.

For example, suppose we make independent measurements on each qubit of
the three-qubit GHZ state

.. math:: |\psi\rangle = |000\rangle +|111\rangle)/\sqrt{2},

and let :math:`xyz` denote the bitstring that results. Recall that,
under the qubit labeling used by Qiskit, :math:`x` would correspond to
the outcome on qubit 2, :math:`y` to the outcome on qubit 1, and
:math:`z` to the outcome on qubit 0. This representation of the
bitstring puts the most significant bit (MSB) on the left, and the least
significant bit (LSB) on the right. This is the standard ordering of
binary bitstrings. We order the qubits in the same way, which is why
Qiskit uses a non-standard tensor product order.

The probability of obtaining outcome :math:`xyz` is given by

.. math:: \mathrm{Pr}(xyz) = |\langle xyz | \psi \rangle |^{2}.

By explicit computation, we see there are only two bitstrings that will
occur: :math:`000` and :math:`111`. If the bitstring :math:`000` is
obtained, the state of the qubits is :math:`|000\rangle`, and if the
bitstring is :math:`111`, the qubits are left in the state
:math:`|111\rangle`. The probability of obtaining 000 or 111 is the
same; namely, 1/2:

.. math::

   \begin{align}
   \mathrm{Pr}(000) &= |\langle 000 | \psi \rangle |^{2} = \frac{1}{2}\\
   \mathrm{Pr}(111) &= |\langle 111 | \psi \rangle |^{2} = \frac{1}{2}.
   \end{align}

To simulate a circuit that includes measurement, we need to add
measurements to the original circuit above, and use a different Aer
backend.

.. code:: ipython3

    # Create a Classical Register with 3 bits.
    c = ClassicalRegister(3, 'c')
    # Create a Quantum Circuit
    meas = QuantumCircuit(q, c)
    meas.barrier(q)
    # map the quantum measurement to the classical bits
    meas.measure(q,c)
    
    # The Qiskit circuit object supports composition using
    # the addition operator.
    qc = circ+meas
    
    #drawing the circuit
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">        ┌───┐           ░       ┌─┐
    q_0: |0>┤ H ├──■────■───░───────┤M├
            └───┘┌─┴─┐  │   ░    ┌─┐└╥┘
    q_1: |0>─────┤ X ├──┼───░────┤M├─╫─
                 └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║ 
    q_2: |0>──────────┤ X ├─░─┤M├─╫──╫─
                      └───┘ ░ └╥┘ ║  ║ 
     c_0: 0 ═══════════════════╬══╬══╩═
                               ║  ║    
     c_1: 0 ═══════════════════╬══╩════
                               ║       
     c_2: 0 ═══════════════════╩═══════
                                       </pre>



This circuit adds a classical register, and three measurements that are
used to map the outcome of qubits to the classical bits.

To simulate this circuit, we use the ``qasm_simulator`` in Qiskit Aer.
Each run of this circuit will yield either the bitstring 000 or 111. To
build up statistics about the distribution of the bitstrings (to, e.g.,
estimate :math:`\mathrm{Pr}(000)`), we need to repeat the circuit many
times. The number of times the circuit is repeated can be specified in
the ``execute`` function, via the ``shots`` keyword.

.. code:: ipython3

    # Use Aer's qasm_simulator
    backend_sim = BasicAer.get_backend('qasm_simulator')
    
    # Execute the circuit on the qasm simulator.
    # We've set the number of repeats of the circuit
    # to be 1024, which is the default.
    job_sim = execute(qc, backend_sim, shots=1024)
    
    # Grab the results from the job.
    result_sim = job_sim.result()

Once you have a result object, you can access the counts via the
function ``get_counts(circuit)``. This gives you the *aggregated* binary
outcomes of the circuit you submitted.

.. code:: ipython3

    counts = result_sim.get_counts(qc)
    print(counts)


.. parsed-literal::

    {'000': 497, '111': 527}


Approximately 50 percent of the time the output bitstring is 000. Qiskit
Terra also provides a function ``plot_histogram`` which allows you to
view the outcomes.

.. code:: ipython3

    from qiskit.tools.visualization import plot_histogram
    plot_histogram(counts)




.. image:: getting_started_with_qiskit_terra_files/getting_started_with_qiskit_terra_33_0.png



The estimated outcome probabilities :math:`\mathrm{Pr}(000)` and
:math:`\mathrm{Pr}(111)` are computed by taking the aggregate counts and
dividing by the number of shots (times the circuit was repeated). Try
changing the ``shots`` keyword in the ``execute`` function and see how
the estimated probabilities change.

Running circuits using the IBMQ provider 
-----------------------------------------

To faciliate access to real quantum computing hardware, we have provided
a simple API interface. To access IBMQ devices, you’ll need an API
token. For the public IBM Q devices, you can generate an API token
`here <https://quantumexperience.ng.bluemix.net/qx/account/advanced>`__
(create an account if you don’t already have one). For Q Network
devices, login to the q-console, click your hub, group, and project, and
expand “Get Access” to generate your API token and access url.

Our IBMQ provider lets you run your circuit on real devices or on our
HPC simulator. Currently, this provider exists within Qiskit, and can be
imported as shown below. For details on the provider, see `The IBMQ
Provider <the_ibmq_provider.ipynb>`__.

.. code:: ipython3

    from qiskit import IBMQ

After generating your API token, call,
``IBMQ.save_account('MY_TOKEN')``. For Q Network users, you’ll also need
to include your access url: ``IBMQ.save_account('MY_TOKEN', 'URL')``

This will store your IBMQ credentials in a local file. Unless your
registration information has changed, you only need to do this once. You
may now load your accounts by calling,

.. code:: ipython3

    IBMQ.load_accounts()

Once your account has been loaded, you can view the list of backends
available to you.

.. code:: ipython3

    print("Available backends:")
    IBMQ.backends()


.. parsed-literal::

    Available backends:




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>,
     <IBMQBackend('ibmq_20_tokyo') from IBMQ(ibm-q-internal, research, yorktown)>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ(ibm-q-internal, research, yorktown)>]



Running circuits on real devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Today’s quantum information processors are small and noisy, but are
advancing at a fast pace. They provide a great opportunity to explore
what `noisy, intermediate-scale quantum
(NISQ) <https://arxiv.org/abs/1801.00862>`__ computers can do.

The IBMQ provider uses a queue to allocate the devices to users. We now
choose a device with the least busy queue which can support our program
(has at least 3 qubits).

.. code:: ipython3

    from qiskit.providers.ibmq import least_busy
    
    large_enough_devices = IBMQ.backends(filters=lambda x: x.configuration().n_qubits > 4 and
                                                           not x.configuration().simulator)
    backend = least_busy(large_enough_devices)
    print("The best backend is " + backend.name())


.. parsed-literal::

    The best backend is ibmqx4


To run the circuit on the backend, we need to specify the number of
shots and the number of credits we are willing to spend to run the
circuit. Then, we execute the circuit on the backend using the
``execute`` function.

.. code:: ipython3

    from qiskit.tools.monitor import job_monitor
    shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
    max_credits = 3        # Maximum number of credits to spend on executions. 
    
    job_exp = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
    job_monitor(job_exp)



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")


``job_exp`` has a ``.result()`` method that lets us get the results from
running our circuit.

.. raw:: html

   <div class="alert alert-block alert-info">

Note: When the .result() method is called, the code block will wait
until the job has finished before releasing the cell.

.. raw:: html

   </div>

.. code:: ipython3

    result_exp = job_exp.result()

Like before, the counts from the execution can be obtained using
``get_counts(qc)``

.. code:: ipython3

    counts_exp = result_exp.get_counts(qc)
    plot_histogram([counts_exp,counts])




.. image:: getting_started_with_qiskit_terra_files/getting_started_with_qiskit_terra_49_0.png



Simulating circuits using a HPC simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IBMQ provider also comes with a remote optimized simulator called
``ibmq_qasm_simulator``. This remote simulator is capable of simulating
up to 32 qubits. It can be used the same way as the remote real
backends.

.. code:: ipython3

    backend = IBMQ.get_backend('ibmq_qasm_simulator', hub=None)

.. code:: ipython3

    shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
    max_credits = 3        # Maximum number of credits to spend on executions. 
    
    job_hpc = execute(qc, backend=backend, shots=shots, max_credits=max_credits)

.. code:: ipython3

    result_hpc = job_hpc.result()

.. code:: ipython3

    counts_hpc = result_hpc.get_counts(qc)
    plot_histogram(counts_hpc)




.. image:: getting_started_with_qiskit_terra_files/getting_started_with_qiskit_terra_54_0.png



Retrieving a previously ran job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your experiment takes longer to run then you have time to wait
around, or if you simply want to retrieve old jobs back, the IBMQ
backends allow you to do that. First you would need to note your job’s
ID:

.. code:: ipython3

    jobID = job_exp.job_id()
    
    print('JOB ID: {}'.format(jobID))        


.. parsed-literal::

    JOB ID: 5c1a2b4f39c21300575b61b0


Given a job ID, that job object can be later reconstructed from the
backend using retrieve_job:

.. code:: ipython3

    job_get=backend.retrieve_job(jobID)

and then the results can be obtained from the new job object.

.. code:: ipython3

    job_get.result().get_counts(qc)




.. parsed-literal::

    {'000': 393,
     '110': 32,
     '111': 340,
     '010': 43,
     '101': 124,
     '001': 14,
     '011': 48,
     '100': 30}


