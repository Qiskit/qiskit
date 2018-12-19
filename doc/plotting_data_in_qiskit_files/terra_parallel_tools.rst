


Using the Qiskit Terra parallel_map
===================================

In this tutorial we will see how to leverage the ``parallel_map``
routine in Qiskit Terra to execute functions in parallel, and track the
progress of these parallel tasks using progress bars.

.. code:: ipython3

    from qiskit import *
    from qiskit.tools.parallel import parallel_map
    from qiskit.tools.events import TextProgressBar
    from qiskit.tools.jupyter import *  # Needed to load the Jupyter HTMLProgressBar

Define a function that builds a single Quantum Volume circuit
-------------------------------------------------------------

Here we will construct a set of 1000 Quantum Volume circuits of width
and depth 4. For a technical discussion of Quantum Volume see:
https://arxiv.org/abs/1811.12926.

.. code:: ipython3

    num_circuits = 1000
    width = 4
    depth = 4

.. code:: ipython3

    import copy
    import math
    import numpy as np
    from qiskit.tools.qi.qi import random_unitary_matrix
    from qiskit.mapper import two_qubit_kak

In preparation for executing in parallel, the code below takes an index
value, an array of random number seeds, and the width and depth of the
circuit as inputs.

.. code:: ipython3

    def build_qv_circuit(idx, seeds, width, depth):
        """Builds a single Quantum Volume circuit.  Two circuits,
        one with measurements, and one widthout, are returned.
    
        The model circuits consist of layers of Haar random
        elements of SU(4) applied between corresponding pairs
        of qubits in a random bipartition.
        
        See: https://arxiv.org/abs/1811.12926
        """
        np.random.seed(seeds[idx])
        q = QuantumRegister(width, "q")
        c = ClassicalRegister(width, "c")
        # Create measurement subcircuit
        qc = QuantumCircuit(q,c)
        # For each layer
        for j in range(depth):
            # Generate uniformly random permutation Pj of [0...n-1]
            perm = np.random.permutation(width)
            # For each pair p in Pj, generate Haar random SU(4)
            # Decompose each SU(4) into CNOT + SU(2) and add to Ci
            for k in range(math.floor(width/2)):
                qubits = [int(perm[2*k]), int(perm[2*k+1])]
                U = random_unitary_matrix(4)
                for gate in two_qubit_kak(U):
                    i0 = qubits[gate["args"][0]]
                    if gate["name"] == "cx":
                        i1 = qubits[gate["args"][1]]
                        qc.cx(q[i0], q[i1])
                    elif gate["name"] == "u1":
                        qc.u1(gate["params"][2], q[i0])
                    elif gate["name"] == "u2":
                        qc.u2(gate["params"][1], gate["params"][2],
                                     q[i0])
                    elif gate["name"] == "u3":
                        qc.u3(gate["params"][0], gate["params"][1],
                                     gate["params"][2], q[i0])
                    elif gate["name"] == "id":
                        pass  # do nothing
        qc_no_meas = copy.deepcopy(qc)
        # Create circuit with final measurement
        qc.measure(q,c)
        return qc, qc_no_meas

Generate 1000 circuits in parallel and track progress
-----------------------------------------------------

Becuase Quantum Volume circuits are generated randomly for the NumPy
random number generator, we must be careful when running in parallel. If
the random number generator is not explicitly seeded, the computer uses
the current time as a seed value. When running in parallel, this can
result in each process starting with the saem seed value, and thus not
giving random results. Here we generate all the random seed values
needed, and pass this into ``parallel_map`` as a extra argument in
``task_args``, along with ``width`` and ``depth``. The main function
argument passed in ``parallel_map`` is just an array that indexes the
processes and seed value.

.. code:: ipython3

    num_circuits = 1000
    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_circuits)
    TextProgressBar()
    parallel_map(build_qv_circuit, np.arange(num_circuits), task_args=(seeds, width, depth));


.. parsed-literal::

    |██████████████████████████████████████████████████| 1000/1000 [00:00:00:00]


Use a Jupyter progress bar
--------------------------

.. code:: ipython3

    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_circuits)
    HTMLProgressBar()
    parallel_map(build_qv_circuit, np.arange(num_circuits), task_args=(seeds, width, depth));



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, bar_style='info', max=1000)))

