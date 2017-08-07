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
