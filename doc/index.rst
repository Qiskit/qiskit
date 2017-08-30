.. QISKit documentation master file, created by
   sphinx-quickstart on Tue Jul 25 18:13:28 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================
QISKit Documentation
====================

Quantum Information Software Kit (QISKit), SDK Python version for working
with `OpenQASM <https://github.com/IBM/qiskit-openqasm>`_ and the IBM Q experience (QX).

Philosophy
==========

The basic concept of our quantum program is an array of quantum
circuits. The program workflow consists of three stages: Build, Compile,
and Run. Build allows you to make different quantum circuits that
represent the problem you are solving; Compile allows you to rewrite
them to run on different backends (simulators/real chips of different
quantum volumes, sizes, fidelity, etc); and Run launches the jobs. After
the jobs have been run, the data is collected. There are methods for
putting this data together, depending on the program. This either gives
you the answer you wanted or allows you to make a better program for the
next instance.

Project Overview
================
The QISKit project comprises:

* `QISKit API <https://github.com/IBM/qiskit-api-py>`_: A thin Python
  wrapper around the Quantum Experience HTTP API that enables you to
  connect and and execute OpenQASM code.

* `QISKit SDK <https://github.com/IBM/qiskit-sdk-py>`_: Provides
  support for the Quantum Experience circuit generation phase and lets
  you use the QISKit API to access the Quantum Experience hardware and
  simulators. The SDK also includes example scripts written for
  Jupyter Notebooks.

* `QISKit OpenQASM <https://github.com/IBM/qiskit-openqasm>`_: Contains
  specifications, examples, documentation, and tools for the OpenQASM
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

Project Organization
--------------------

Python example programs can be found in the *examples* directory, and test scripts are
located in *test*. The *qiskit* directory is the main module of the SDK.

Structure
=========

Programming interface
---------------------

The *qiskit* directory is the main Python module and contains the
programming interface objects `QuantumProgram <quantum_program.rst>`__, *QuantumRegister*,
*ClassicalRegister*, and *QuantumCircuit*.

At the highest level, users construct a *QuantumProgram* to create,
modify, compile, and execute a collection of quantum circuits. Each
*QuantumCircuit* has a set of data registers, each of type
*QuantumRegister* or *ClassicalRegister*. Methods of these objects are
used to apply instructions that define the circuit. The *QuantumCircuit*
can then generate **OpenQASM** code that can flow through other
components in the *qiskit* directory.

The *extensions* directory extends quantum circuits as needed to support
other gate sets and algorithms. Currently there is a *standard*
extension defining some typical quantum gates.

Internal modules
----------------

The directory also contains internal modules that are still under
development:

-  a *qasm* module for parsing **OpenQASM** circuits
-  an *unroll* module to interpret and "unroll" **OpenQASM** to a target
   gate basis (expanding gate subroutines and loops as needed)
-  a *circuit* module for working with circuits as graphs
-  a *mapper* module for mapping all-to-all circuits to run on devices
   with fixed couplings

Quantum circuits flow through the components as follows. The programming
interface is used to generate **OpenQASM** circuits. **OpenQASM**
source, as a file or string, is passed into a *Qasm* object, whose
*parse* method produces an abstract syntax tree (**AST**). The **AST**
is passed to an *Unroller* that is attached to an *UnrollerBackend*.
There is a *PrinterBackend* for outputting text, a *SimulatorBackend*
for outputting simulator input data for the local simulators, and a
*CircuitBackend* for constructing *Circuit* objects. The *Circuit*
object represents an "unrolled" **OpenQASM** circuit as a directed
acyclic graph (**DAG**). The *Circuit* provides methods for
representing, transforming, and computing properties of a circuit and
outputting the results again as **OpenQASM**. The whole flow is used by
the *mapper* module to rewrite a circuit to execute on a device with
fixed couplings given by a *CouplingGraph*.

The four circuit representations and how they are currently transformed
into each other are summarized in this figure:

.. image:: ../images/circuit_representations.png
    :width: 600px
    :align: center

Several unroller backends and their outputs are summarized here:

.. image:: ../images/unroller_backends.png
    :width: 600px
    :align: center


Python Modules
==============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   _autodoc/modules


Installation and setup
======================

1. Get the tools
----------------

You'll need:

-  Install `Python 3 <https://docs.python.org/3/using/index.html>`__.
-  `Jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`__
   client is needed to run the tutorials, not to use as a library.
-  Mac OS X users will find Xcode useful:
   https://developer.apple.com/xcode/
-  For Windows users we highly recommend to install `Anaconda 3 <https://www.continuum.io/downloads#windows>`_
-  Optionally download Git: https://git-scm.com/download/.

2. Get the code
---------------

Clone the QISKit SDK repository and navigate to its folder on your local
machine:

-  If you have Git installed, run the following commands:

.. code:: sh

    git clone https://github.com/QISKit/qiskit-sdk-py
    cd qiskit-sdk-py

-  If you don't have Git installed, click the "Clone or download" button
   at the URL shown in the git clone command, unzip the file if needed,
   then navigate to that folder in a terminal window.

3. Setup the environment
------------------------

To use as a library install the dependencies:

.. code:: sh

    # Depending on the system and setup to append "sudo -H" before could be needed.
    pip install -r requires.txt

To get the tutorials working set up an Anaconda environment for working
with QISKit, and install the required dependencies:

-  If running either Linux or Mac OS X with Xcode, simply run the
   following command:

.. code:: sh

    make env

-  If running on Mac OS X without Xcode, run the following set of commands:

.. code:: sh

    conda create -y -n QISKitenv python=3 pip scipy
    activate QISKitenv
    pip install -r requires.txt
    
-  If running on Windows, make sure to execute an Anaconda Prompt and run
   the following command:

.. code:: sh

    .\make env


4. Configure your API token
---------------------------

-  Create an `IBM Quantum
   Experience <https://quantumexperience.ng.bluemix.net>`__ account if
   you haven't already done so
-  Get an API token from the Quantum Experience website under “My
   Account” > “Personal Access Token”
-  You will insert your API token in a file called Qconfig.py. First
   copy the default version of this file from the tutorial folder to the
   main SDK folder (on Windows, replace ``cp`` with ``copy``):

.. code:: sh

    cp Qconfig.py.default Qconfig.py

-  Open your Qconfig.py, remove the ``#`` from the beginning of the API
   token line, and copy/paste your API token into the space between the
   quotation marks on that line. Save and close the file.
  
Starting the Jupyter-based tutorials
====================================

The SDK includes tutorials in the form of Jupyter notebooks, which are
essentially web pages that contain "cells" of embedded Python code. To
run a cell, click on it and hit ``Shift+Enter`` or use the toolbar at
the top of the page. Any output from a cell is displayed immediately
below it on the page. In most cases, the cells on each page must be run
in sequential order from top to bottom in order to avoid errors. To get
started with the tutorials, follow the instructions below.

-  If running either Linux or Mac OS X with Xcode, simply run the
   following command from the QISKit SDK folder:

.. code:: sh

    make run

-  If running on Mac OS X without Xcode, run the
   following set of commands from the QISKit SDK folder:

.. code:: sh

    activate QISKitenv
    cd tutorial
    jupyter notebook index.ipynb
    
-  If running on Windows, make sure you are running an Anaconda Prompt,
   and then run the following commands from the QISKit SDK folder:

.. code:: sh

    .\make run
    


FAQ
===

If you upgrade the dependencies and get the error below, try the fix
shown below the error:

.. code:: sh

    # Depending on the system and setup to append "sudo -H" before could be needed.
    pip install --upgrade IBMQuantumExperience
    

.. code:: sh

    # Fix: run the command below
    curl https://bootstrap.pypa.io/ez_setup.py -o - | python

For additional troubleshooting tips, see the QISKit troubleshooting page
on the project's GitHub wiki.

Authors (alphabetical)
======================

Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Juan Gomez, Paco Martin, Antonio Mezzacapo, Jesus Perez, and John Smolin, Erick Winston, Chris Wood.

In future releases, anyone who contributes code to this project can include their name here.

Other QISKit projects
=====================

- `ibmqx backend information <https://github.com/QISKit/ibmqx-backend-information>`__ Information about the different IBM Q experience backends.
- `ibmqx user guide <https://github.com/QISKit/ibmqx-user-guides>`__ The users guides for the IBM Q experience.
- `OpenQasm <https://github.com/QISKit/openqasm>`__ Examples and tools for the OpenQASM intermediate representation.
- `Python API <https://github.com/QISKit/qiskit-api-py>`__ API Client to use IBM Q experience in Python.
- `Tutorials <https://github.com/QISKit/qiskit-tutorial>`__ Jupyter notebooks for using QISKit.


License
=======

QISKit is released under the `Apache license, version
2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__.

Do you want to help?
====================

If you'd like to contribute please take a look to our
`contribution guidelines <CONTRIBUTING.rst>`__.


.. testoutput::
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
