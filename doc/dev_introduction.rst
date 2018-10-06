Structure
=========

Programming interface
---------------------

The *qiskit* directory is the main Python module and contains the
programming interface objects:
:py:class:`QuantumRegister <qiskit.QuantumRegister>`,
:py:class:`ClassicalRegister <qiskit.ClassicalRegister>`,
and :py:class:`QuantumCircuit <qiskit.QuantumCircuit>`.

At the highest level, users construct a *QuantumCircuit* to create,
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
:py:mod:`qasm_simulator_cpp <qiskit.extensions.simulator>` and
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

Terra uses the `standard Python "logging" library
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


For convenience, two methods are provided in :py:mod<`qiskit_logging.py`>: (:py:func:<`set_qiskit_logger()>` and
:py:func:<`unset_qiskit_logger`>) that modify the handlers
and the level of the `qiskit` logger. Using these methods might interfere with the global
logging setup of your environment - please take it into consideration if developing an
application on top of Terra.

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

Terra uses the `standard Pyton "unittest" framework
<https://docs.python.org/3/library/unittest.html>`_ for the testing of the
different components and functionality.

As our build system is based on CMake, we need to perform what is called an
"out-of-source" build before running the tests.
This is as simple as executing these commands:

Linux and Mac:

.. code-block:: bash

    $ mkdir out
    $ cd out
    out$ cmake ..
    out$ make

Windows:

.. code-block:: bash

    C:\..\> mkdir out
    C:\..\> cd out
    C:\..\out> cmake -DUSER_LIB_PATH=C:\path\to\mingw64\lib\libpthreads.a -G "MinGW Makefiles" ..
    C:\..\out> make

This will generate all needed binaries for your specific platform.

For executing the tests, a ``make test`` target is available.
The execution of the tests (both via the make target and during manual invocation)
takes into account the ``LOG_LEVEL`` environment variable. If present, a ``.log``
file will be created on the test directory with the output of the log calls, which
will also be printed to stdout. You can adjust the verbosity via the content
of that variable, for example:

Linux and Mac:

.. code-block:: bash

    $ cd out
    out$ LOG_LEVEL="DEBUG" ARGS="-V" make test

Windows:

.. code-block:: bash

    $ cd out
    C:\..\out> set LOG_LEVEL="DEBUG"
    C:\..\out> set ARGS="-V"
    C:\..\out> make test

For executing a simple python test manually, we don't need to change the directory
to ``out``, just run this command:


Linux and Mac:

.. code-block:: bash

    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/test_apps.py

Testing options
^^^^^^^^^^^^^^^

By default, and if there is no user credentials available, the tests that require online access are run with recorded (mocked) information. This is, the remote requests are replayed from a ``test/cassettes`` and not real HTTP requests is generated.
If user credentials are found, in that cases it use them to make the network requests.

How and which tests are executed is controlled by a environment variable ``QISKIT_TESTS``. The options are (where ``uc_available = True`` if the user credentials are available, and ``False`` otherwise): 

+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
|  Option           | Description                                                                                                        | Default               |  If ``True``, forces                             |
+===================+====================================================================================================================+=======================+==================================================+
| ``skip_online``   | Skips tests that require remote requests (also, no mocked information is used). Does not require user credentials. | ``False``             | ``rec = False``                                  |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``mock_online``   | It runs the online tests using mocked information. Does not require user credentials.                              | ``not uc_available``  | ``skip_online = False``                          |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``run_slow``      | It runs tests tagged as *slow*.                                                                                    | ``False``             |                                                  |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``rec``           | It records the remote requests. It requires user credentials.                                                      | ``False``             | ``skip_online = False``                          |
|                   |                                                                                                                    |                       | ``run_slow = False``                             |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+

It is possible to provide more than one option separated with commas.
The order of precedence in the options is right to left. For example, ``QISKIT_TESTS=skip_online,rec`` will set the options as ``skip_online == False`` and ``rec == True``.	
