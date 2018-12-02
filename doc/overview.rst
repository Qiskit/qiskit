
Overview
========

Terra, the ‘earth’ element, is the foundation on which the rest of Qiskit lies. 
Terra provides a bedrock for composing quantum programs at the level of circuits and pulses, 
to optimize them for the constraints of a particular device, and to manage the execution 
of batches of experiments on remote-access devices. Terra defines the interfaces 
for a desirable end-user experience, as well as the efficient handling of layers 
of optimization, pulse scheduling and backend communication.


Terra Organization
------------------

Python example programs can be found in the *examples* directory, and test scripts are
located in *test*. The *qiskit* directory is the main module of Terra. This moudule has six main parts.


Quantum Circuits
^^^^^^^^^^^^^^^^

A quantum circuit is a model for quantum computing in which a computation is done by performing a 
sequence of quantum operations (usually gates) on a register of qubits. A quantum circuit usually 
starts with the qubits in the :math:`|0,…,0>` state (Terra assumes this unless otherwise specified) and 
these gates evolve the qubits to states that cannot be efficiently represented on a classical computer. 
To extract information on the state a quantum circuit must have a measurement which maps the outcomes
(possible random due to the fundamental nature of quantum systems) to classical registers which 
can be efficiently represented.


Transpiler
^^^^^^^^^^

A major part of research on quantum computing is working out how to run a quantum 
circuits on real devices.  In these devices, experimental errors and decoherence introduce
errors during computation. Thus, to obtain a robust implementation it is essential 
to reduce the number of gates and the overall running time of the quantum circuit. 
The transpiler introduces the concept of a pass manager to allow users to explore
optimization and find better quantum circuits for their given algorithm. We call it a 
transpiler as the end result is still a circuit.


Tools
^^^^^

This directory contains tools that make working with Terra simpler. It contains functions that
allow the user to execute quantum circuits and not worry about the optimization for a given 
backend. It also contains a compiler which uses the transpiler to map an array of quantum circuits
to a `qobj` (quantum object) which can then be run on a backend. The `qobj` is a convenient 
representation (currently JSON) of the data that can be easily sent to the remote backends. 
It also has functions for monitoring jobs, backends, and parallelization of transpilation tasks. 


Backends and Results
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the user has made the `qobj` to run on the backend they need to have a convenient way of 
working with it. In Terra we do this using three parts: the *provider*, the *backend*, 
and the *job*. 

A *Provider* is an entity that provides access to a group of different backends (for example, 
backends available through the `IBM Q <https://www.research.ibm.com/ibm-q/technology/devices/>`_). 
It interacts with those backends to, for example, 
find out which ones are available, or retrieve an instance of a particular backend.

*Backends* represent either a simulator or a real quantum computer and are responsible 
for running quantum circuits and returning results. They have a run method which takes in a
`qobj` as input and returns a `BaseJob` object. This object allows asynchronous running of
jobs for retrieving results from a backend when the job is completed.

*Job* instances can be thought of as the “ticket” for a submitted job. 
They find out the execution’s state at a given point in time (for example, 
if the job is queued, running, or has failed) and also allow control over the job.

Once the job has finished Terra allows the results to be obtained from the remote backends 
using `result = job.result()`.  This result object holds the quantum data and the most 
common way of interacting with it is by using `result.get_counts(circuit)`. This method allows 
the user to get the raw counts from the quantum circuit and use them for more analysis with 
quantum inofrmation tools provided by Terra.



Quantum Information
^^^^^^^^^^^^^^^^^^^

To perform more advance algorithms and analyzation of the circuits run on the quantum computer it is
important to have tools to perform simple quantum information tasks. These include methods to estimate
metrics on and generate quantum states, operations, and channels. 


Visualization Tools
^^^^^^^^^^^^^^^^^^^

In Terra we have many tools to visualize a quantum circuit. This allows a quick inspection of the quantum 
circuit to make sure it is what the user wanted to implement. There is a text, python and latex version. 
Once the circuit has run it is important to be able to view the output. There is a simple function 
(`plot_histogram`) to plot the results from a quantum circuit including an interactive version. 
There is also a function `plot_state` and ` plot_bloch_vector` that allow the plotting of a 
quantum state. These functions are usually only used when using the `statevector_simulator` 
backend but can also be used on real data after running state tomography experiments (ignis). 


License
-------

This project uses the `Apache License Version 2.0 software
license <https://www.apache.org/licenses/LICENSE-2.0>`__.
