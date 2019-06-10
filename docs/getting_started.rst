===========================
Getting Started with Qiskit
===========================

The workflow of using Qiskit consists of three high-level steps:

- **Build**: design a quantum circuit that represents the problem you are
  considering.
- **Execute**: run experiments on different backends (*which include both
  systems and simulators*).
- **Analyze**: calculate summary statistics and visualize the results of
  experiments.

Here is an example of the entire workflow, with each step explained in detail in
subsequent sections:

.. code-block:: python

    import numpy as np
    from qiskit import(
      QuantumCircuit,
      execute,
      Aer)
    from qiskit.visualization import plot_histogram

    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(2,2)

    # Add a H gate on qubit 0
    circuit.h(0)

    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
    circuit.cx(0,1)

    # Map the quantum measurement to the classical bits
    circuit.measure([0,1],[0,1])

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1000)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    print("\nTotal count for 00 and 11 are:",counts)

    # Draw the circuit
    circuit.draw(output='mpl')

.. code-block:: text

    Total count for 00 and 11 are: {'00': 479, '11': 521}

.. image:: /images/figures/getting_started_1_1.png
   :alt: Quantum Circuit with an H gate and controlled nots.

.. code-block:: python

    # Plot a histogram
    plot_histogram(counts)

.. image:: /images/figures/getting_started_2_0.png
   :alt: Probabilities of each state.



-----------------------
Workflow Step--by--Step
-----------------------

The program above can be broken down into six steps:

1. Import packages
2. Initialize variables
3. Add gates
4. Visualize the circuit
5. Simulate the experiment
6. Visualize the results


~~~~~~~~~~~~~~~~~~~~~~~~
Step 1 : Import Packages
~~~~~~~~~~~~~~~~~~~~~~~~

The basic elements needed for your program are imported as follows:

.. code-block:: python

  import numpy as np
  from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
  from qiskit.visualization import plot_histogram

In more detail, the imports are

- ``QuantumCircuit``: can be thought as the instructions of the quantum system.
  It holds all your quantum operations.
- ``execute``: runs your circuit / experiment.
- ``Aer``: handles simulator backends.
- ``plot_histogram``: creates histograms.



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 2 : Initialize Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the next line of code

.. code-block:: python

    circuit = QuantumCircuit(2,2)

Here, you are initializing with 2 qubits in the zero state; with 2
classical bits set to zero; and ``circuit`` is the quantum circuit.

Syntax:

- ``QuantumCircuit(int, int)``



~~~~~~~~~~~~~~~~~~
Step 3 : Add Gates
~~~~~~~~~~~~~~~~~~

You can add gates (operations) to manipulate the registers of your circuit.

Consider the following three lines of code:

.. code-block:: python

    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0,1],[0,1])

The gates are added to the circuit one-by-one to form the Bell state

.. math:: |\psi\rangle = \left(|00\rangle+|11\rangle\right)/\sqrt{2}.

The code above applies the following gates:

- ``QuantumCircuit.h(0)``: A Hadamard gate :math:`H` on qubit 0,
  which puts it into a **superposition state**.
- ``QuantumCircuit.cx(0,1)``: A controlled-Not operation
  (:math:`C_{X}`) on control qubit 0 and target qubit 1, putting the qubits in
  an **entangled state**.
- ``QuantumCircuit.measure([0,1], [0,1])``: if you pass
  the entire quantum and classical registers to ``measure``, the ith qubit’s
  measurement result will be stored in the ith classical bit.



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 4 : Visualize the Circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``QuantumCircuit.draw()`` to view the circuit that you have designed
in the :ref:`various forms <Visualizing a Quantum Circuit>` used in many
textbooks and research articles.

.. code-block:: python

    circuit.draw(output='mpl')

.. image:: images/figures/getting_started_1_1.png
   :alt: Quantum circuit to make a Bell state.

In this circuit, the qubits are ordered with qubit zero at the top and
qubit one at the bottom. The circuit is read left-to-right, meaning that gates
which are applied earlier in the circuit show up farther to the left.



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 5 : Simulate the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qiskit Aer is a high performance simulator framework for quantum circuits. It
provides :ref:`several backends <executing_quantum_programs>` to achieve
different simulation goals.

To simulate this circuit, you will use the ``qasm_simulator``. Each run of this
circuit will yield either the bit string 00 or 11.

.. code-block:: python

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print("\nTotal count for 00 and 11 are:",counts)


.. code-block:: text

    Total count for 00 and 11 are: {'00': 479, '11': 521}

As expected, the output bit string is 00 approximately 50 percent of the time.
The number of times the circuit is run can be specified via the ``shots``
argument of the ``execute`` method. The number of shots of the simulation was
set to be 1000 (the default is 1024).

Once you have a ``result`` object, you can access the counts via the method
``get_counts(circuit)``. This gives you the aggregate outcomes of the
experiment you ran.



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 6 : Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qiskit provides :ref:`many visualizations <plotting_data_in_qiskit>`, including
the function ``plot_histogram``, to view your results.

.. code-block:: python

    plot_histogram(counts)

.. image:: images/figures/getting_started_2_0.png
   :alt: Histogram of results.

The observed probabilities :math:`Pr(00)` and :math:`Pr(11)` are computed by
taking the respective counts and dividing by the total number of shots.

.. note::
  Try changing the ``shots`` keyword in the ``execute`` method to see how
  the estimated probabilities change.



----------
Next Steps
----------

Now that you have learnt the basics, consider these learning resources:

- `Notebook tutorials <https://nbviewer.jupyter.org/github/Qiskit/qiskit-tutorials/blob/master/qiskit/start_here.ipynb>`_
- `Video tutorials <https://www.youtube.com/channel/UClBNq7mCMf5xm8baE_VMl3A/featured>`_
- `Interactive tutorials in IBM Q Experience`_
- :ref:`Frequently Asked Questions <faq>`

.. _Interactive tutorials in IBM Q Experience:
   https://www.research.ibm.com/ibm-q/technology/experience/
