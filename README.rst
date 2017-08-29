.. image:: tutorial/images/QISKit-c.gif
   :align: center

----------

|Build Status|

**QISKit**, Quantum Information Software Kit, is a software development kit (SDK)
and Jupyter notebooks for working with OpenQASM and the IBM Q experience (QX).
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

Organization
------------

Python example programs can be found in the *examples* directory, and test scripts are
located in *test*. The *qiskit* directory is the main module of the SDK.

Structure
---------

Programming interface
~~~~~~~~~~~~~~~~~~~~~

The *qiskit* directory is the main Python module and contains the
programming interface objects *QuantumProgram*, *QuantumRegister*,
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
~~~~~~~~~~~~~~~~

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

.. image:: images/circuit_representations.png
    :width: 200px
    :align: center

Several unroller backends and their outputs are summarized here:

.. image:: images/unroller_backends.png
    :width: 200px
    :align: center

Installation and setup
----------------------

1. Get the tools
~~~~~~~~~~~~~~~~

You'll need:

-  Install `Python 3 <https://docs.python.org/3/using/index.html>`__.
-  `Jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`__
   client is needed to run the tutorials, not to use as a library.
-  Mac OS X users will find Xcode useful:
   https://developer.apple.com/xcode/
-  For Windows users we highly recommend to install `Anaconda 3 <https://www.continuum.io/downloads#windows>`_
-  Optionally download Git: https://git-scm.com/download/.

2. Get the code
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

To use as a library install the dependencies:

.. code:: sh

    # Depending on the system and setup to append "sudo -H" before could be needed.
    pip3 install -r requires.txt

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

**We use GitHub issues for tracking requests and bugs. So please see**
`IBM Q experience Community <https://quantumexperience.ng.bluemix.net/qx/community>`__
**for questions and discussion.**


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



**Try your first QISKit program**

Now it's time to begin doing real work with QISKit.

First, you need to get your `API token and configure the Qconfig file <QISKitDETAILS.rst#APIToken>`_.

And then, You can run a QASM using QISKit!

.. code:: ipython3

    from qiskit import QuantumProgram
    import Qconfig

    # Creating Programs
    # create your first QuantumProgram object instance.
    Q_program = QuantumProgram()

    # Set up the API and execute the program. You need the APItoken and the QX URL.
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])

    # Creating Registers
    # create your first Quantum Register called "qr" with 2 qubits
    qr = Q_program.create_quantum_registers("qr", 2)
    # create your first Classical Register  called "cr" with 2 bits
    cr = Q_program.create_classical_registers("cr", 2)

    # Creating Circuits
    # create your first Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = Q_program.create_circuit("qc", ["qr"], ["cr"])

    # Create a GHZ state, for example
    qc.h(q[0])
    for i in range(4):
        qc.cx(q[i], q[i+1])
    # Insert a barrier before measurement
    qc.barrier()
    # Measure all of the qubits in the standard basis
    for i in range(5):
        qc.measure(q[i], c[i])

    # Compiled to qc5qv2 coupling graph
    result = qp.execute(["ghz"], backend='local_qasm_simulator',
                        coupling_map=coupling_map, shots=1024)

    # Show the results
    print(result)
    print(qp.get_counts("ghz"))


For more information
--------------------

 - `QISKit in depth <QISKitDETAILS.rst>`__
 - `QISKit Tutorials <tutorial/index.ipynb>`__
 - `QISKit for Developers <tutorial/rst/tutorial4developer.rst>`_

Learn more about the QISKit community at the community page of
`IBM Q experience <https://quantumexperience.ng.bluemix.net/qx/community>`__
for a few ways to participate.

.. |Build Status| image:: https://travis.ibm.com/IBMQuantum/qiskit-sdk-py-dev.svg?token=GMH4xFrA9iezVJKqw2zH&branch=master
   :target: https://travis.ibm.com/IBMQuantum/qiskit-sdk-py-dev
