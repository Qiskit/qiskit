Quantum Information Software Kit (QISKit)
=========================================

|Build Status|

The Quantum Information Software Kit (**QISKit** for short) is a
software development kit (SDK) for working with `OpenQASM`_ and the `IBM
Quantum Experience (QX)`_.

Use **QISKit** to create quantum computing programs, compile them, and
execute them on one of several backends (online Real Quantum Processors,
online Simulators, and local Simulators). For the online backends,
QISKit uses our `python API client`_ to connect to the IBM Quantum
Experience.

**We use GitHub issues for tracking requests and bugs. Please see the**
`IBM Q Experience community`_ **for questions and discussion.** **If
you’d like to contribute to QISKit, please take a look at our**
`contribution guidelines`_.

In addition, a basic understanding of quantum information is very
helpful when interacting with QISKit. If you’re new to quantum, Start
with our `User Guides`_! from Quantum Experience.

Links to Sections:

-  `Installation (Python)`_
-  `Installation (Anaconda)`_
-  `Getting Started`_
-  `More Information`_
-  `License`_

Python Installation
-------------------

For those more familiar with python, follow the QISKit install process
below:

Dependencies
~~~~~~~~~~~~

-  `Python`_ (3 or later required)
-  `Jupyter Notebooks`_ (recommended to interact with tutorials)

If you don't have installed any Python version or your operating system has a 2.x 
version, we recommend use the Anaconda_ Python Distribution, because it has all 
the components that you need like Jupyter.

QISKit Installation
~~~~~~~~~~~~~~~~~~~

**1.** Clone the QISKit SDK repository and navigate to its folder on
your local machine:

Select the “Clone or download” button at the top of this webpage (or
from URL shown in the git clone command), unzip the file if needed, and
navigate to **qiskit-sdk-py folder** in a terminal window.

Alternatively, if you have Git installed, run the following commands:

::

        git clone https://github.com/QISKit/qiskit-sdk-py
        cd qiskit-sdk-py

**2.** Next, install QISKit using ``pip``. For example, from the
command line:

::

        pip install qiskit

Once the install is complete, you are now ready to begin using QISKit.
See the `Getting Started`_ section for next steps.

Anaconda Installation
---------------------

For those who would prefer to use Anaconda, you can use the following
QISKit install process instead:

Dependencies > NEEDS REVIEW
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Anaconda`_ (**QUESTION: What version is needed?**)
-  `Jupyter Notebooks`_ (recommended to interact with tutorials)

A basic understanding of quantum information is also very helpful when
interacting with QISKit. If you’re new to quantum, Start with our `User
Guides`_!

User I
~~~~~~

.. _OpenQASM: https://github.com/QISKit/qiskit-openqasm
.. _IBM Quantum Experience (QX): https://quantumexperience.ng.bluemix.net/
.. _python API client: https://github.com/QISKit/qiskit-api-py
.. _IBM Q Experience community: https://quantumexperience.ng.bluemix.net/qx/community
.. _contribution guidelines: CONTRIBUTING.rst
.. _Installation (Python): #python-installation
.. _Installation (Anaconda): #anaconda-installation
.. _Getting Started: #getting-started
.. _More Information: #more-information
.. _License: #license
.. _Python: https://www.python.org/downloads/
.. _Jupyter Notebooks: https://jupyter.readthedocs.io/en/latest/install.html
.. _User Guides: https://github.com/QISKit/ibmqx-user-guides
.. _Anaconda: https://www.continuum.io/downloads
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

QISKit was originally developed by researchers and developers working on the IBM Q
Team within IBM Research organization to offer a high level development kit to work with
quantum computers.

**If you'd like to contribute to QISKit, please take a look 
to our** `contribution guidelines <CONTRIBUTING.rst>`__

**We use GitHub issues for tracking requests and bugs. So please see**
`IBM Q experience Community <https://quantumexperience.ng.bluemix.net/qx/community>`__
**for questions and discussion.**


Installation
------------

You can install me using `pip3`. For example, from the command line:

.. code:: sh

    pip3 install qiskit

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
