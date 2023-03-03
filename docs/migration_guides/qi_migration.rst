################################
Quantum Instance Migration Guide
################################

The :class:`~qiskit.utils.QuantumInstance` is a utility class that allowed to jointly
configure the circuit transpilation and execution steps, and provided useful tools for algorithm development,
such as basic error mitigation strategies. The functionality of :class:`~qiskit.utils.QuantumInstance.execute` has
now been delegated to the different implementations of the :mod:`~qiskit.primitives` base classes,
while the explicit transpilation has been left to the :meth:`~qiskit.transpiler` module (see table below).
Thus, the :class:`~qiskit.utils.QuantumInstance` is being deprecated.

Summary of migration alternatives for the :class:`~qiskit.utils.QuantumInstance` class:

.. list-table::
   :header-rows: 1

   * - QuantumInstance method
     - Alternative
   * - ``QuantumInstance.execute``
     - ``Sampler.run`` or ``Estimator.run``
   * - ``QuantumInstance.transpile``
     - ``qiskit.transpiler.transpile``
   * - ``QuantumInstance.assemble``
     - Deprecated

Contents
========

* `Choosing the right primitive for your task`_
* `Choosing the right primitive for your settings`_
* `Code examples`_


.. |qiskit_aer.primitives| replace:: ``qiskit_aer.primitives``
.. _qiskit_aer.primitives: https://github.com/Qiskit/qiskit-aer/tree/main/qiskit_aer/primitives

.. |qiskit_ibm_runtime| replace:: ``qiskit_ibm_runtime``
.. _qiskit_ibm_runtime: https://qiskit.org/documentation/partners/qiskit_ibm_runtime/index.html

.. attention::

    The current pool of primitives includes **two** different **classes** (:class:`~qiskit.primitives.Sampler` and
    :class:`~qiskit.primitives.Estimator`) that can be imported from **three** different locations (
    :mod:`qiskit.primitives`, |qiskit_aer.primitives|_ and |qiskit_ibm_runtime|_ ). In addition to the
    reference Sampler and Estimator, :mod:`qiskit.primitives` also contains a
    :class:`~qiskit.primitives.BackendSampler` and a :class:`~qiskit.primitives.BackendEstimator` class. These are
    wrappers for ``backend.run()`` that follow the primitives interface.

    This guide uses the following naming standard to refer to the primitives:

    - *Primitives* - Any Sampler/Estimator implementation
    - *Reference Primitives* - The Sampler and Estimator in :mod:`qiskit.primitives` --> ``from qiskit.primitives import Sampler/Estimator``
    - *Aer Primitives* - The Sampler and Estimator in |qiskit_aer.primitives|_ --> ``from qiskit_aer.primitives import Sampler/Estimator``
    - *Runtime Primitives* - The Sampler and Estimator in |qiskit_ibm_runtime|_ --> ``from qiskit_ibm_runtime import Sampler/Estimator``
    - *Backend Primitives* - The BackendSampler and BackendEstimator in :mod:`qiskit.primitives` --> ``from qiskit import BackendSampler/BackendEstimator``

    For guidelines on which primitives to choose for your task. Please continue reading.

Choosing the right primitive for your task
===========================================

While the :class:`~qiskit.utils.QuantumInstance` was designed as as single, highly-configurable, task-agnostic class,
the primitives don't follow the same principle. There are multiple primitives, and each is optimized for a specific
purpose. Selecting the right primitive (``Sampler`` or ``Estimator``) requires some knowledge about
**what** is it expected to do and **where/how** is it expected to run.

.. note::

    The role of the primitives is two-fold. On one hand, they act as access points to backends and simulators.
    On the other hand, they are **algoritmic** abstractions with defined tasks:

    * The ``Estimator`` takes in circuits and observables and returns their **expectation values**.
    * The ``Sampler`` takes in circuits, measures them, and returns their  **quasi-probability distribution**.

    The :class:`~qiskit.utils.QuantumInstance` shares the role of access point to backends and simulators, but
    unlike the primitives, it returned the **raw** output of the execution, with a higher level of granularity.
    The minimal unit of information of this output was usually **measurement counts**. And in this sense, the closest
    primitive would be the ``Sampler``. However, you must keep in mind the difference in output formats.


In order to know which primitive to use instead of :class:`~qiskit.utils.QuantumInstance`, you should ask
yourself two questions:

1. What is the minimal unit of information used by your algorithm?
    a. **Expectation value** - you will need an ``Estimator``
    b. **Probability distribution** (from sampling the device) - you will need a ``Sampler``

2. How do you want to execute your circuits?
    a. Using **local** statevector simulators for quick prototyping: **Reference Primitives**
    b. Using **local** noisy simulations for finer algorithm tuning: **Aer Primitives**
    c. Accessing **runtime-enabled backends** (or cloud simulators): **Runtime Primitives**
    d. Accessing **non runtime-enabled backends** : **Backend Primitives**



Choosing the right primitive for your settings
==============================================

Certain :class:`~qiskit.utils.QuantumInstance` features are only available in certain primitive implementations.
The following table summarizes the most common :class:`~qiskit.utils.QuantumInstance` settings and which
primitives **expose a similar setting through their interface**:

.. attention::

    In some cases, a setting might not be exposed through the interface, but there might be workarounds to make
    it work. This is the case for custom transpiler passes, which cannot be set through the primitives interface,
    but pre-transpiled circuits can be sent if setting the option ``skip_transpilation=True``. For more information,
    please refer to the API reference or source code of the desired primitive implementation.

.. list-table::
   :header-rows: 1

   * - QuantumInstance
     - Reference Primitives
     - Aer Primitives
     - Runtime Primitives
     - Backend Primitives
   * - Select ``backend``
     - No
     - No
     - Yes
     - Yes
   * - Set ``shots``
     - Yes
     - Yes
     - Yes
     - Yes
   * - Simulator settings: ``basis_gates``, ``coupling_map``, ``initial_layout``, ``noise_model``, ``backend_options``
     - No
     - Yes
     - Yes
     - No
   * - Transpiler settings: ``seed_transpiler``, ``optimization_level``
     - No
     - No
     - Yes (via ``options``)
     - Yes (via ``set_transpile_options``)
   * - Set unbound ``pass_manager``
     - No
     - No
     - No (but can ``skip_transpilation``)
     - No (but can ``skip_transpilation``)
   * - Set ``bound_pass_manager``
     - No
     - No
     - No
     - Yes
   * - Set ``backend_options``: common ones were ``memory`` and ``meas_level``
     - No
     - No
     - No (only ``qubit_layout``)
     - No
   * - Measurement error mitigation: ``measurement_error_mitigation_cls``, ``cals_matrix_refresh_period``,
       ``measurement_error_mitigation_shots``, ``mit_pattern``
     - No
     - No
     - Sampler default -> M3 (*)
     - No
   * - Job management: ``job_callback``, ``max_job_retries``, ``timeout``, ``wait``
     - No
     - No
     - Sessions, callback (**)
     - No


(*) For more information on error mitigation options on Runtime Primitives, visit
`this link <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/stubs/qiskit_ibm_runtime.options.Options.html#qiskit_ibm_runtime.options.Options>`_.

(**) For more information on Runtime sessions, visit `this how-to <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/how_to/run_session.html>`_.

Code examples
=============

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Circuit Sampling with Local Statevector Simulation</font></a></summary>
    <br>

**Using Quantum Instance**

The only alternative for local simulations using the quantum instance was through the definition of an Aer Simulator
as backend:

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from qiskit_aer import AerSimulator
    >>> from qiskit.utils import QuantumInstance

    >>> circuit = QuantumCircuit(2)
    >>> circuit.x(0)
    >>> circuit.x(1)
    >>> circuit.measure_all()

    >>> simulator = AerSimulator()
    >>> qi = QuantumInstance(backend=simulator, shots=200, backend_options={"method": "statevector"})
    >>> result = qi.execute(circuit).results[0]
    >>> result
    ExperimentResult(shots=200, success=True, meas_level=2, data=ExperimentResultData(counts={'0x3': 200}), header=QobjExperimentHeader(clbit_labels=[['meas', 0], ['meas', 1]], creg_sizes=[['meas', 2]], global_phase=0.0, memory_slots=2, metadata={}, n_qubits=2, name='circuit-112', qreg_sizes=[['q', 2]], qubit_labels=[['q', 0], ['q', 1]]), status=DONE, seed_simulator=3116700546, metadata={'parallel_state_update': 16, 'parallel_shots': 1, 'sample_measure_time': 6.0573e-05, 'noise': 'ideal', 'batched_shots_optimization': False, 'remapped_qubits': False, 'device': 'CPU', 'active_input_qubits': [0, 1], 'measure_sampling': True, 'num_clbits': 2, 'input_qubit_map': [[1, 1], [0, 0]], 'num_qubits': 2, 'method': 'statevector', 'fusion': {'applied': False, 'max_fused_qubits': 5, 'threshold': 14, 'enabled': True}}, time_taken=0.000426016)

    >>> data = result.data
    >>> data
    ExperimentResultData(counts={'0x3': 200})

    >>> counts = data.counts
    >>> counts
    {'0x3': 200}

**Using Primitives**

The primitives offer two alternatives for local statevector simulation:

**a. Using the Reference Primitives**

Basic statevector simulation based on :class:`qiskit.quantum_info.Statevector` class.

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from qiskit.primitives import Sampler

    >>> circuit = QuantumCircuit(2)
    >>> circuit.x(0)
    >>> circuit.x(1)
    >>> circuit.measure_all()

    >>> sampler = Sampler(options = {"shots":200})
    >>> result = sampler.run(circuit).result()
    >>> result
    SamplerResult(quasi_dists=[{3: 1.0}], metadata=[{'shots': 200}])

    >>> quasi_dists = result.quasi_dists
    >>> quasi_dists
    [{3: 1.0}]

**b. Using the Aer Primitives**

Aer simulation following the statevector method. This would be the direct 1-1 replacement of the Quantum Instance
exeample, as they are both accessing the same simulator. For this reason, the output metadata is richer, and
closer to the Quantum Instance's output.

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from qiskit_aer.primitives import Sampler

    >>> circuit = QuantumCircuit(2)
    >>> circuit.x(0)
    >>> circuit.x(1)
    >>> circuit.measure_all()

    >>> sampler = Sampler(run_options = {"method":"statevector", "shots":200})
    >>> result = sampler.run(circuit).result()
    >>> result
    SamplerResult(quasi_dists=[{3: 1.0}], metadata=[{'shots': 200, 'simulator_metadata': {'parallel_state_update': 16, 'parallel_shots': 1, 'sample_measure_time': 9.016e-05, 'noise': 'ideal', 'batched_shots_optimization': False, 'remapped_qubits': False, 'device': 'CPU', 'active_input_qubits': [0, 1], 'measure_sampling': True, 'num_clbits': 2, 'input_qubit_map': [[1, 1], [0, 0]], 'num_qubits': 2, 'method': 'statevector', 'fusion': {'applied': False, 'max_fused_qubits': 5, 'threshold': 14, 'enabled': True}}}])

    >>> quasi_dists = result.quasi_dists
    >>> quasi_dists
    [{3: 1.0}]

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: Expectation Value Calculation with Local Noisy Simulation</font></a></summary>
    <br>

**Using Quantum Instance**

The most common use case for computing expectation values with the Quantum Instance was as in combination with the
:mod:`~qiskit.opflow` library. You can see more information in the `opflow migration guide <http://qisk.it/opflow_migration>`_.

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from qiskit.opflow import StateFn, PauliSumOp, PauliExpectation, CircuitSampler
    >>> from qiskit.utils import QuantumInstance
    >>> from qiskit_aer import AerSimulator
    >>> from qiskit_aer.noise import NoiseModel
    >>> from qiskit_ibm_provider import IBMProvider

    # Define problem
    >>> op = PauliSumOp.from_list([("XY",1)])
    >>> qc = QuantumCircuit(2)
    >>> qc.x(0)
    >>> qc.x(1)
    >>> state = StateFn(qc)
    >>> measurable_expression = StateFn(op, is_measurement=True).compose(state)
    >>> expectation = PauliExpectation().convert(measurable_expression)

    # Define Quantum Instance with noisy simulator
    >>> provider = IBMProvider()
    >>> device = provider.get_backend("ibmq_manila")
    >>> noise_model = NoiseModel.from_backend(device)
    >>> coupling_map = device.configuration().coupling_map

    >>> backend = AerSimulator()
    >>> qi = QuantumInstance(backend=backend, shots=1024,
    ...                     seed_simulator=42, seed_transpiler=42,
    ...                     coupling_map=coupling_map, noise_model=noise_model)

    # Run
    >>> sampler = CircuitSampler(qi).convert(expectation)
    >>> expectation_value = sampler.eval().real
    >>> expectation_value
    -0.04687500000000008

**Using Primitives**

Now, the primitives have allowed to combine the opflow and quantum instance functionality in a single ``Estimator``.
In this case, for local noisy simulation, this will be the Aer Estimator.

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from qiskit.quantum_info import SparsePauliOp
    >>> from qiskit_aer.noise import NoiseModel
    >>> from qiskit_aer.primitives import Estimator
    >>> from qiskit_ibm_provider import IBMProvider

    # Define problem
    >>> op = SparsePauliOp("XY")
    >>> qc = QuantumCircuit(2)
    >>> qc.x(0)
    >>> qc.x(1)

    # Define Aer Estimator with noisy simulator
    >>> device = provider.get_backend("ibmq_manila")
    >>> noise_model = NoiseModel.from_backend(device)
    >>> coupling_map = device.configuration().coupling_map

    >>> estimator = Estimator(
    ...            backend_options={
    ...                "method": "density_matrix",
    ...                "coupling_map": coupling_map,
    ...                "noise_model": noise_model,
    ...            },
    ...            run_options={"seed": 42, "shots": 1024},
    ...           transpile_options={"seed_transpiler": 42},
    ...        )

    # Run
    >>> expectation_value = estimator.run(qc, op).result().values
    >>> expectation_value
    [-0.04101562]

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 3: Circuit Sampling on IBM Backend with Error Mitigation</font></a></summary>
    <br>

**Using Quantum Instance**

The QuantumInstance interface allowed to configure measurement error mitigation settings such as the method, the
matrix refresh period or the mitigation pattern.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.utils import QuantumInstance
    from qiskit.utils.mitigation import CompleteMeasFitter
    from qiskit_ibm_provider import IBMProvider

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    provider = IBMProvider()
    backend = provider.get_backend("ibmq_manila")

    qi = QuantumInstance(
        backend=backend,
        shots=1000,
        measurement_error_mitigation_cls=CompleteMeasFitter,
        cals_matrix_refresh_period=0,
    )

    result = qi.execute(circuit).results[0]

**Using Primitives**

The Runtime Primitives offer a suite of error mitigation methods that can be easily "turned on" with the
``resilience_level`` option. These are, however, not configurable. The sampler's ``resilience_level=1``
is the closest alternative to the Quantum Instance's error mitigation implementation, but this
is not a 1-1 replacement.

For more information on the error mitigation options in the Runtime Primitives, you can check out the following
`link <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/stubs/qiskit_ibm_runtime.options.Options.html#qiskit_ibm_runtime.options.Options>`_.


.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibmq_manila")

    options = Options(resilience_level = 1) # 1 = measurement error mitigation
    sampler = Sampler(session=backend, options=options)

    # Run
    result = sampler.run(circuit).result()

    quasi_dists = result.quasi_dists

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 4: Circuit Sampling with Custom Bound and Unbound Pass Managers</font></a></summary>
    <br>

The management of transpilation is quite different between the QuantumInstance and the Primitives.

The Quantum Instance allowed you to:

* Define bound and unbound pass managers that will be called during ``.execute()``.
* Explicitly call its ``.transpile()`` method with a specific pass manager.

However:

* The Quantum Instance **did not** manage parameter bindings on parametrized quantum circuits. This would
  mean that if a ``bound_pass_manager`` was set, the circuit sent to ``QuantumInstance.execute()`` could
  not have any free parameters.

On the other hand, when using the primitives:

* You cannot explicitly access their transpilation routine.
* The mechanism to apply custom transpilation passes to the Aer, Runtime and Backend primitives is to pre-transpile
  locally and set ``skip_transpilation=True`` in the corresponding primitive.
* The only primitives that currently accept a custom **bound** transpiler pass manager are the **Backend Primitives**.
  If a ``bound_pass_manager`` is defined, the ``skip_transpilation=True`` option will **not** skip this bound pass.

Note that the primitives **do** handle parameter bindings, meaning that even if a ``bound_pass_manager`` is defined in a
Backend Primitive, you do not have to manually assign parameters as expected in the Quantum Instance workflow.

Let's see an example with a parametrized quantum circuit and different custom transpiler passes, ran on an ``AerSimulator``.

**Using Quantum Instance**

.. code-block:: python

    >>> from qiskit.circuit import QuantumRegister, Parameter, QuantumCircuit
    >>> from qiskit.transpiler import PassManager, CouplingMap
    >>> from qiskit.transpiler.passes import BasicSwap, Unroller
    >>> from qiskit_ibm_provider import IBMProvider

    >>> from qiskit.utils import QuantumInstance
    >>> from qiskit_aer.noise import NoiseModel
    >>> from qiskit_aer import AerSimulator

    >>> q = QuantumRegister(7, 'q')
    >>> p = Parameter('p')
    >>> circuit = QuantumCircuit(q)
    >>> circuit.h(q[0])
    >>> circuit.cx(q[0], q[4])
    >>> circuit.cx(q[2], q[3])
    >>> circuit.cx(q[6], q[1])
    >>> circuit.cx(q[5], q[0])
    >>> circuit.rz(p, q[2])
    >>> circuit.cx(q[5], q[0])
    >>> circuit.measure_all()

    # Set up simulation based on real device
    >>> provider = IBMProvider()
    >>> backend = AerSimulator()
    >>> device = provider.get_backend("ibm_oslo")
    >>> noise_model = NoiseModel.from_backend(device)
    >>> coupling_map = device.configuration().coupling_map

    # Define unbound pass manager
    >>> unbound_pm = PassManager(BasicSwap(CouplingMap(couplinglist=coupling_map)))

    # Define bound pass manager
    >>> bound_pm = PassManager(Unroller(['u1', 'u2', 'u3', 'cx']))

    # Define quantum instance
    >>> qi = QuantumInstance(
    ...    backend=backend,
    ...    shots=1000,
    ...    seed_simulator=42,
    ...    noise_model=noise_model,
    ...    coupling_map=coupling_map,
    ...    pass_manager=unbound_pm,
    ...    bound_pass_manager=bound_pm
    ... )

    # You can transpile the unbound circuit
    >>> transpiled_circuit = qi.transpile(circuit, pass_manager=unbound_pm)
    >>> print(transpiled_circuit)

    # You can bind the parameter and transpile
    >>> bound_circuit = circuit.bind_parameters({p: 0.1})
    >>> transpiled_bound_circuit = qi.transpile(bound_circuit, pass_manager=bound_pm)
    >>> print(transpiled_bound_circuit)

    # Or you can execute bound circuit with passes defined during init.
    >>> result = qi.execute(bound_circuit).results[0]
    >>> result
    ExperimentResult(shots=1000, success=True, meas_level=2, data=ExperimentResultData(counts={'0x39': 1, '0x3': 3, '0x1f': 4, '0x43': 2, '0x14': 1, '0x22': 1, '0x5': 1, '0x15': 3, '0xc': 5, '0x1d': 4, '0x50': 1, '0x44': 1, '0x32': 1, '0x1': 73, '0x1a': 1, '0x1b': 2, '0x30': 1, '0x9': 1, '0x12': 4, '0x13': 14, '0x53': 2, '0xe': 4, '0x21': 1, '0x10': 89, '0x19': 7, '0x31': 5, '0x17': 1, '0x11': 326, '0x41': 1, '0x8': 12, '0x1e': 1, '0x20': 13, '0x42': 6, '0x4': 9, '0x51': 6, '0x40': 19, '0x52': 2, '0x2': 8, '0x0': 364}), header=QobjExperimentHeader(clbit_labels=[['meas', 0], ['meas', 1], ['meas', 2], ['meas', 3], ['meas', 4], ['meas', 5], ['meas', 6]], creg_sizes=[['meas', 7]], global_phase=6.233185307179586, memory_slots=7, metadata={}, n_qubits=7, name='circuit-1845', qreg_sizes=[['q', 7]], qubit_labels=[['q', 0], ['q', 1], ['q', 2], ['q', 3], ['q', 4], ['q', 5], ['q', 6]]), status=DONE, seed_simulator=42, metadata={'parallel_state_update': 16, 'parallel_shots': 1, 'sample_measure_time': 0.000634964, 'noise': 'superop', 'batched_shots_optimization': False, 'remapped_qubits': False, 'device': 'CPU', 'active_input_qubits': [0, 1, 2, 3, 4, 5, 6], 'measure_sampling': True, 'num_clbits': 7, 'input_qubit_map': [[6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]], 'num_qubits': 7, 'method': 'density_matrix', 'fusion': {'applied': False, 'max_fused_qubits': 2, 'threshold': 7, 'enabled': True}}, time_taken=0.045343491)

    >>> result.data.counts
    {'0x39': 1, '0x3': 3, '0x1f': 4, '0x43': 2, '0x14': 1, '0x22': 1, '0x5': 1, '0x15': 3, '0xc': 5, '0x1d': 4, '0x50': 1, '0x44': 1, '0x32': 1, '0x1': 73, '0x1a': 1, '0x1b': 2, '0x30': 1, '0x9': 1, '0x12': 4, '0x13': 14, '0x53': 2, '0xe': 4, '0x21': 1, '0x10': 89, '0x19': 7, '0x31': 5, '0x17': 1, '0x11': 326, '0x41': 1, '0x8': 12, '0x1e': 1, '0x20': 13, '0x42': 6, '0x4': 9, '0x51': 6, '0x40': 19, '0x52': 2, '0x2': 8, '0x0': 364}

**Using Primitives**

Let's see how the workflow changes with the Backend Sampler:

.. code-block:: python

    >>> from qiskit.circuit import QuantumRegister, Parameter
    >>> from qiskit.transpiler import PassManager, CouplingMap
    >>> from qiskit.transpiler.passes import BasicSwap, Unroller
    >>> from qiskit_ibm_provider import IBMProvider
    >>> from qiskit import QuantumCircuit
    >>> from qiskit.primitives import BackendSampler
    >>> from qiskit_aer.noise import NoiseModel
    >>> from qiskit_aer import AerSimulator

    >>> q = QuantumRegister(7, 'q')
    >>> p = Parameter('p')
    >>> circuit = QuantumCircuit(q)
    >>> circuit.h(q[0])
    >>> circuit.cx(q[0], q[4])
    >>> circuit.cx(q[2], q[3])
    >>> circuit.cx(q[6], q[1])
    >>> circuit.cx(q[5], q[0])
    >>> circuit.rz(p, q[2])
    >>> circuit.cx(q[5], q[0])
    >>> circuit.measure_all()

    # Set up simulation based on real device
    >>> provider = IBMProvider()
    >>> backend = AerSimulator()
    >>> device = provider.get_backend("ibm_oslo")
    >>> noise_model = NoiseModel.from_backend(device)
    >>> coupling_map = device.configuration().coupling_map
    >>> backend.set_options(seed_simulator=42, noise_model=noise_model, coupling_map=coupling_map)

    # Pre-run transpilation using pass manager
    >>> unbound_pm = PassManager(BasicSwap(CouplingMap(couplinglist=coupling_map)))
    >>> transpiled_circuit = unbound_pm.run(circuit)
    >>> print(transpiled_circuit)
            ┌───┐                                                     ░       ┌─┐
       q_0: ┤ H ├───────────────X─────────────────────────────────────░───────┤M├────────────
            └───┘     ┌───────┐ │                                     ░       └╥┘         ┌─┐
       q_1: ──X────■──┤ Rz(p) ├─X──X──────────────────────────X───■───░────────╫──────────┤M├
              │    │  └───────┘    │                          │ ┌─┴─┐ ░    ┌─┐ ║          └╥┘
       q_2: ──X────┼───────────────┼──────────────────────────┼─┤ X ├─░────┤M├─╫───────────╫─
                 ┌─┴─┐             │                          │ └───┘ ░    └╥┘ ║ ┌─┐       ║
       q_3: ─────┤ X ├─────────────X──X────────■────■──────X──X───────░─────╫──╫─┤M├───────╫─
                 └───┘                │ ┌───┐  │    │      │          ░     ║  ║ └╥┘┌─┐    ║
       q_4: ──────────────────────────┼─┤ X ├──┼────┼──────┼──────────░─────╫──╫──╫─┤M├────╫─
                                      │ └─┬─┘┌─┴─┐┌─┴─┐    │          ░     ║  ║  ║ └╥┘┌─┐ ║
       q_5: ──────────────────────────X───■──┤ X ├┤ X ├─X──X──────────░─────╫──╫──╫──╫─┤M├─╫─
                                             └───┘└───┘ │             ░ ┌─┐ ║  ║  ║  ║ └╥┘ ║
       q_6: ────────────────────────────────────────────X─────────────░─┤M├─╫──╫──╫──╫──╫──╫─
                                                                      ░ └╥┘ ║  ║  ║  ║  ║  ║
    meas: 7/═════════════════════════════════════════════════════════════╩══╩══╩══╩══╩══╩══╩═
                                                                         0  1  2  3  4  5  6

    # Define bound pass manager
    >>> bound_pm = PassManager(Unroller(['u1', 'u2', 'u3', 'cx']))

    # Set up sampler with skip_transpilation and bound_pass_manager
    >>> sampler = BackendSampler(backend=backend, skip_transpilation=True, bound_pass_manager=bound_pm)

    # Run
    >>> result = sampler.run(transpiled_circuit, [[0.1]], shots=1024).result().quasi_dists
    >>> result
    [{20: 0.0009765625,
      18: 0.001953125,
      80: 0.00390625,
      6: 0.001953125,
      29: 0.0048828125,
      66: 0.0048828125,
      24: 0.00390625,
      8: 0.0166015625,
      65: 0.0009765625,
      14: 0.0029296875,
      19: 0.01171875,
      83: 0.001953125,
      64: 0.0068359375,
      81: 0.0029296875,
      49: 0.005859375,
      25: 0.0087890625,
      16: 0.072265625,
      33: 0.001953125,
      53: 0.0009765625,
      82: 0.001953125,
      2: 0.0107421875,
      31: 0.0048828125,
      5: 0.0009765625,
      21: 0.005859375,
      48: 0.0048828125,
      9: 0.00390625,
      44: 0.0009765625,
      3: 0.0068359375,
      1: 0.0693359375,
      12: 0.0048828125,
      4: 0.005859375,
      89: 0.001953125,
      32: 0.0068359375,
      67: 0.0048828125,
      73: 0.0009765625,
      38: 0.0009765625,
      0: 0.376953125,
      17: 0.330078125}]

.. raw:: html

    </details>
