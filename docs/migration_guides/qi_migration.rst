================================
Quantum Instance Migration Guide
================================

The :class:`~qiskit.utils.QuantumInstance` is a utility class that allowed to jointly
configure the circuit transpilation and execution steps, and provided useful tools for algorithm development,
such as basic error mitigation strategies. This functionality has now been delegated to the different
implementations of the :mod:`~qiskit.primitives`, and thus, the :class:`~qiskit.utils.QuantumInstance` is
being deprecated.

Contents
--------

* `How to select the right primitive to replace a QuantumInstance`_

How to select the right primitive to replace a QuantumInstance
--------------------------------------------------------------

The current pool of primitives include two different classes (:class:`~qiskit.primitives.Sampler` and
:class:`~qiskit.primitives.Estimator`) that can be imported from three different locations ( ``qiskit`` ,
``qiskit_aer``, ``qiskit_ibm_runtime``). In addition to these, ``qiskit`` also contains a
:class:`~qiskit.primitives.BackendSampler` and a :class:`~qiskit.primitives.Estimator` class.

Naively, the ``Sampler`` is the most direct replacement of the
:class:`~qiskit.utils.QuantumInstance`. Both serve as entry points to the backend, where circuits are
sent, and results from sampling are returned. The main difference is the format of the output.
:meth:`~qiskit.utils.QuantumInstance.execute()` returns a :class:`~qiskit.results.Result` with
the backend's output, while :class:`~qiskit.primitives.Sampler.run()`
returns a :class:`~qiskit.primitives.SamplerResult` that contains a quasi-probability distribution.

.. code-block:: python

    # Using quantum instance
    qi = QuantumInstance(backend)
    results = qi.execute(circuit).results

    # Using sampler
    sampler = Sampler()
    results = sampler.run(circuit).result()


However, for use cases that rely on expectation value calculations, the ``Estimator`` is the most
suited alternative for :class:`~qiskit.utils.QuantumInstance`, as it handles the calculation together with
the execution, and outputs an  :class:`~qiskit.primitives.EstimatorResult` containing the expectation values
directly.

.. code-block:: python

    # Using quantum instance
    ... # Prepare expectation circuit from circuit and observables
    qi = QuantumInstance(backend)
    results = qi.execute(circuit).results

    # Using estimator
    estimator = Estimator()
    results = estimator.run(circuit, observables).result()


.. hint::

    In order to know which primitive to use instead of :class:`~qiskit.utils.QuantumInstance`, you should ask
    yourself two questions:

    **1. What is what is the minimal unit of information used by my algorithm?**

    a. **Expectation value** - you will need an ``Estimator``
    b. **Probability distribution** (from sampling the device) - you will need a ``Sampler``

    **2. How do I want to execute my circuits?**

    a. Using **local** statevector simulators for quick prototyping: Reference Primitives.
      ``from qiskit.primitives import Sampler/Estimator``
    b. Using **local** noisy simulations for finer algorithm tuning: Aer Primitives.
      ``from qiskit_aer.primitives import Sampler/Estimator``
    c. Accessing **runtime-enabled backends** (or cloud simulators): Runtime Primitives.
      ``from qiskit_ibm_runtime import Sampler/Estimator``
    d. Accessing **non runtime-enabled backends** : Backend Primitives.
      ``from qiskit import BackendSampler/BackendEstimator``


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Circuit Sampling with Local Statevector Simulation (1a/1b)</font></a></summary>
    <br>

**Using Quantum Instance**

The only alternative for local simulations using the quantum instance was through the definition of an Aer Simulator
as backend:

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.utils import QuantumInstance

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    simulator = AerSimulator('aer_simulator_statevector')
    qi = QuantumInstance(backend=simulator, shots=200)
    result = qi.execute(circuit).results[0].data
    # result: ExperimentResultData(counts={'0x3': 1},
    #         statevector=Statevector([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j], dims=(2, 2)))
    counts = result.get_counts()
    # counts: {'11': 1}

**Using Primitives**

The primitives offer two alternatives for local statevector simulation:

**a. Using the Reference Primitives**

Basic statevector simulation:

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.primitives import Sampler

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    sampler = Sampler(options = {"shots":200})
    result = sampler.run(circuit).result()
    # result: SamplerResult(quasi_dists=[{3: 1.0}], metadata=[{'shots': 200}])

    quasi_dists = result.quasi_dists
    # quasi_dists: [{3: 1.0}]

**b. Using the Aer Primitives**

This option is closer to the QI example, as it accesses the same backend

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import Sampler

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    sampler = Sampler(run_options = {"backend":"aer_simulator_statevector", "shots":200})
    result = sampler.run(circuit).result()
    # result: SamplerResult(quasi_dists=[{3: 1.0}],
    # metadata=[{'shots': 200, 'simulator_metadata': {'parallel_state_update': 16,
    # 'parallel_shots': 1, 'sample_measure_time': 0.00022784, 'noise': 'ideal',
    # 'batched_shots_optimization': False, 'remapped_qubits': False, 'device': 'CPU',
    # 'active_input_qubits': [0, 1], 'measure_sampling': True, 'num_clbits': 2,
    # 'input_qubit_map': [[1, 1], [0, 0]], 'num_qubits': 2, 'method': 'stabilizer',
    # 'fusion': {'enabled': False}}}])

    quasi_dists = result.quasi_dists
    # quasi_dists: [{3: 1.0}]

.. raw:: html

    </details>

