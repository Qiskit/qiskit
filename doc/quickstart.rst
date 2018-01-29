Getting Started
===============

The starting point for writing code is the
:py:class:`QuantumProgram <qiskit.QuantumProgram>` object. The
QuantumProgram is a collection of circuits, or scores if you are
coming from the Quantum Experience, quantum register objects, and
classical register objects. The QuantumProgram methods can send these
circuits to quantum hardware or simulator backends and collect the
results for further analysis.

To compose and run a circuit on a simulator, which is distributed with
this project, one can do,

.. code-block:: python

   from qiskit import QuantumProgram
   qp = QuantumProgram()

   qr = qp.create_quantum_register('qr', 2)
   cr = qp.create_classical_register('cr', 2)
   qc = qp.create_circuit('Bell', [qr], [cr])

   qc.h(qr[0])
   qc.cx(qr[0], qr[1])
   qc.measure(qr[0], cr[0])
   qc.measure(qr[1], cr[1])

   result = qp.execute('Bell')
   print(result.get_counts('Bell'))

The :code:`get_counts` method outputs a dictionary of state:counts pairs;

.. code-block:: python

    {'00': 531, '11': 493}

Quantum Chips
-------------

You can execute your QASM circuits on a real chip by using the IBM Q experience (QX) cloud platform. 
Currently through QX you can use the following chips:

-   ibmqx2: `5-qubit backend <https://ibm.biz/qiskit-ibmqx2>`_

-   ibmqx3: `16-qubit backend <https://ibm.biz/qiskit-ibmqx3>`_

For chip details visit the `IBM Q experience backend information <https://github.com/QISKit/ibmqx-backend-information>`_

.. include:: example_real_backend.rst

Project Organization
--------------------

Python example programs can be found in the *examples* directory, and test scripts are
located in *test*. The *qiskit* directory is the main module of the SDK.
