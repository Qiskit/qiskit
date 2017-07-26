.. QISKit documentation master file, created by
   sphinx-quickstart on Tue Jul 25 18:13:28 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
QISKit SDK
==========
Quantum Information Science Kit

Project Overview
================
The QISKit project comprises:


* `QISKit API <https://github.com/IBM/qiskit-api-py>`_: A thin Python
  wrapper around the Quantum Experience HTTP API that enables you to
  connect and and execute OPENQASM code.

* `QISKit SDK <https://github.com/IBM/qiskit-sdk-py>`_: Provides
  support for the Quantum Experience circuit generation phase and lets
  you use the QISKit API to access the Quantum Experience hardware and
  simulators. The SDK also includes example scripts written for
  Jupyter Notebooks.

* `QISKit OPENQASM <https://github.com/IBM/qiskit-openqasm>`_: Contains
  specifications, examples, documentation, and tools for the OPENQASM
  intermediate representation.

Getting Started
===============

The starting point for writing code is the QuantumProgram object. The
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
   qr = qp.create_quantum_registers('qr', 2)
   cr = qp.create_classical_registers('cr', 2)
   circuit = qp.create_circuit('super', ['qr'], ['cr'])
   circuit.h(qr[0])
   circuit.cx(qr[0], qr[1])
   circuit.measure(qr[0], cr[0])
   circuit.measure(qr[1], cr[1])
   qp.execute('super')
   qp.get_counts('super')

The :code:`get_counts` method outputs a dictionary of state:counts pairs;

.. code-block:: python

	 {'00': 531, '11': 493}


.. testoutput::
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   contributing
   tutorial4developer
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
