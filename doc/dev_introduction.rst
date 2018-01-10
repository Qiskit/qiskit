Structure
=========

Programming interface
---------------------

The *qiskit* directory is the main Python module and contains the
programming interface objects :py:class:`QuantumProgram <qiskit.QuantumProgram>`,
:py:class:`QuantumRegister <qiskit.QuantumRegister>`,
:py:class:`ClassicalRegister <qiskit.ClassicalRegister>`,
and :py:class:`QuantumCircuit <qiskit.QuantumCircuit>`.

At the highest level, users construct a *QuantumProgram* to create,
modify, compile, and execute a collection of quantum circuits. Each
*QuantumCircuit* has a set of data registers, each of type
*QuantumRegister* or *ClassicalRegister*. Methods of these objects are
used to apply instructions that define the circuit. The *QuantumCircuit*
can then generate **OpenQASM** code that can flow through other
components in the *qiskit* directory.

The :py:mod:`extensions <qiskit.extensions>` directory extends quantum circuits
as needed to support other gate sets and algorithms. Currently there is a
:py:mod:`standard <qiskit.extensions.standard>` extension defining some typical
quantum gates, and two additional extensions:
:py:mod:`qiskit_simulator <qiskit.extensions.qiskit_simulator>` and
:py:mod:`quantum_initializer <qiskit.extensions.quantum_initializer>`.

Internal modules
----------------

The directory also contains internal modules that are still under development:

- a :py:mod:`qasm <qiskit.qasm>` module for parsing **OpenQASM** circuits
- an :py:mod:`unroll <qiskit.unroll>` module to interpret and “unroll”
  **OpenQASM** to a target gate basis (expanding gate subroutines and loops as
  needed)
- a :py:mod:`dagcircuit <qiskit.dagcircuit>` module for working with circuits as
  graphs
- a :py:mod:`mapper <qiskit.mapper>` module for mapping all-to-all circuits to
  run on devices with fixed couplings
- a :py:mod:`backends <qiskit.backends>` module contains quantum circuit
  simulators
- a *tools* directory contains methods for applications, analysis, and visualization

Quantum circuits flow through the components as follows. The programming interface is used to
generate **OpenQASM** circuits, as text or *QuantumCircuit* objects. **OpenQASM** source, as a
file or string, is passed into a *Qasm* object, whose parse method produces an abstract syntax
tree (**AST**). The **AST** is passed to an *Unroller* that is attached to an *UnrollerBackend*.
There is a *PrinterBackend* for outputting text, a *JsonBackend* for producing input to
simulator and experiment backends, a *DAGBackend* for constructing *DAGCircuit* objects, and
a *CircuitBackend* for producing *QuantumCircuit* objects. The *DAGCircuit* object represents
an “unrolled” **OpenQASM** circuit as a directed acyclic graph (DAG). The *DAGCircuit* provides
methods for representing, transforming, and computing properties of a circuit and outputting the
results again as **OpenQASM**. The whole flow is used by the *mapper* module to rewrite a
circuit to execute on a device with fixed couplings given by a *CouplingGraph*. The structure of
these components is subject to change.

The circuit representations and how they are currently transformed into each other are summarized
in this figure:



.. image:: ../images/circuit_representations.png
    :width: 600px
    :align: center

Several unroller backends and their outputs are summarized here:



.. image:: ../images/unroller_backends.png
    :width: 600px
    :align: center


Logging
-------

The SDK uses the `standard Python "logging" library
<https://docs.python.org/3/library/logging.html>`_ for emitting several messages using the
family of "`qiskit.*`" loggers, and abides by the standard convention for the log levels:

.. tabularcolumns:: |l|L|

+--------------+----------------------------------------------+
| Level        | When it's used                               |
+==============+==============================================+
| ``DEBUG``    | Detailed information, typically of interest  |
|              | only when diagnosing problems.               |
+--------------+----------------------------------------------+
| ``INFO``     | Confirmation that things are working as      |
|              | expected.                                    |
+--------------+----------------------------------------------+
| ``WARNING``  | An indication that something unexpected      |
|              | happened, or indicative of some problem in   |
|              | the near future (e.g. 'disk space low').     |
|              | The software is still working as expected.   |
+--------------+----------------------------------------------+
| ``ERROR``    | Due to a more serious problem, the software  |
|              | has not been able to perform some function.  |
+--------------+----------------------------------------------+
| ``CRITICAL`` | A serious error, indicating that the program |
|              | itself may be unable to continue running.    |
+--------------+----------------------------------------------+


For convenience, :py:class:`QuantumProgram <qiskit.QuantumProgram>` provides two convenience
methods (:py:func:`enable_logs() <qiskit.QuantumProgram.enable_logs>` and
:py:func:`disable_logs() <qiskit.QuantumProgram.disable_logs>`) that modify the handlers
and the level of the `qiskit` logger. Using these methods might interfere with the global
logging setup of your environment - please take it into consideration if developing an
application on top of the SDK.

The convention for emitting log messages is declare a global variable in the module named
**logger**, which contains the logger with that module's **__name__**, and use it for emitting
the messages. For example, if the module is `qiskit/some/module.py`:

.. code-block:: python

   import logging

   logger = logging.getLogger(__name__)  # logger for "qiskit.some.module"
   ...
   logger.info("This is an info message)


Testing
-------

The SDK uses the `standard Pyton "unittest" framework
<https://docs.python.org/3/library/unittest.html>`_ for the testing of the
different components and functionality.

For executing the tests, a ``make test`` target is available. The execution
of the tests (both via the make target and during manual invocation) takes into
account the ``LOG_LEVEL`` environment variable. If present, a ``.log`` file
will be created on the test directory with the output of the log calls, which
will also be printed to stdout. You can adjust the verbosity via the content
of that variable, for example:

.. code-block:: bash

    $ LOG_LEVEL=DEBUG make test
    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py
