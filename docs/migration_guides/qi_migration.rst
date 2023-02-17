================================
Quantum Instance Migration Guide
================================

The :class:`~qiskit.utils.QuantumInstance` is a utility class that allowed to jointly
configure the circuit transpilation and execution steps, and provided useful tools for algorithm development,
such as basic error mitigation strategies. This functionality has now been delegated to the different
implementations of the :mod:`~qiskit.primitives` base classes, and thus,
the :class:`~qiskit.utils.QuantumInstance` is being deprecated.

.. attention::

    The current pool of primitives includes **two** different **classes** (:class:`~qiskit.primitives.Sampler` and
    :class:`~qiskit.primitives.Estimator`) that can be imported from **three** different locations (
    :mod:`qiskit.primitives`,
    `qiskit_aer.primitives <https://github.com/Qiskit/qiskit-aer/tree/main/qiskit_aer/primitives>`_ and
    `qiskit_ibm_runtime <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/index.html>`_ ). In addition to the
    reference Sampler and Estimator, :mod:`qiskit.primitives` also contains a
    :class:`~qiskit.primitives.BackendSampler` and a :class:`~qiskit.primitives.BackendEstimator` class. These are
    wrappers for ``backend.run()`` that follow the primitives interface.

    This guide uses the following naming standard to refer to the primitives:

    - *Primitives* - Any Sampler/Estimator implementation
    - *Reference Primitives* - The Sampler and Estimator in :mod:`qiskit.primitives` --> ``from qiskit.primitives import Sampler/Estimator``
    - *Aer Primitives* - The Sampler and Estimator in :mod:`qiskit_aer.primitives` --> ``from qiskit_aer.primitives import Sampler/Estimator``
    - *Runtime Primitives* - The Sampler and Estimator in :mod:`qiskit_ibm_runtime` --> ``from qiskit_ibm_runtime import Sampler/Estimator``
    - *Backend Primitives* - The BackendSampler and BackendEstimator in :mod:`qiskit.primitives` --> ``from qiskit import BackendSampler/BackendEstimator``


Contents
--------
* `Choosing the right primitive for your task`_
* `Choosing the right primitive for your settings`_
* `Examples`_

Choosing the right primitive for your task
------------------------------------------

While the :class:`~qiskit.utils.QuantumInstance` was designed as as single, highly-configurable, task-agnostic class,
the primitives don't follow the same principle. There are multiple primitives, and each is optimized for a specific
purpose. Selecting the right primitive (``Sampler`` or ``Estimator``) requires some knowledge about
**what** is it expected to do and **where/how** is it expected to run.

.. important::

    In order to know which primitive to use instead of :class:`~qiskit.utils.QuantumInstance`, you should ask
    yourself two questions:

    **I. What is what is the minimal unit of information used by my algorithm?**

    a. **Expectation value** - you will need an ``Estimator``
    b. **Probability distribution** (from sampling the device) - you will need a ``Sampler``

    **II. How do I want to execute my circuits?**

    1. Using **local** statevector simulators for quick prototyping: Reference Primitives
    2. Using **local** noisy simulations for finer algorithm tuning: Aer Primitives
    3. Accessing **runtime-enabled backends** (or cloud simulators): Runtime Primitives
    4. Accessing **non runtime-enabled backends** : Backend Primitives

Choosing the right primitive for your settings
----------------------------------------------

Certain :class:`~qiskit.utils.QuantumInstance` features are only available in certain primitive implementations.
The following chart summarizes the most common :class:`~qiskit.utils.QuantumInstance` settings and which
primitives provide a similar feature:

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
     - Yes
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
   * - M3 error mitigation: ``measurement_error_mitigation_cls``, ``cals_matrix_refresh_period``,
       ``measurement_error_mitigation_shots``, ``mit_pattern``
     - No
     - No
     - Sampler default
     - No
   * - Job management: ``job_callback``, ``max_job_retries``, ``timeout``, ``wait``
     - No
     - No
     - Sessions, callback
     - No

Examples
--------

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Circuit Sampling with Local Statevector Simulation</font></a></summary>
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

    simulator = AerSimulator()
    qi = QuantumInstance(backend=simulator, shots=200, backend_options={"method": "statevector"})
    result = qi.execute(circuit).results[0]
    # result: ExperimentResult(shots=200, success=True, meas_level=2,
    #         data=ExperimentResultData(counts={'0x3': 200}, statevector=Statevector([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
    #         dims=(2, 2))), header=QobjExperimentHeader(clbit_labels=[['meas', 0], ['meas', 1]],
    #         creg_sizes=[['meas', 2]], global_phase=0.0, memory_slots=2,
    #         metadata={}, n_qubits=2, name='circuit-136', qreg_sizes=[['q', 2]], qubit_labels=[['q', 0], ['q', 1]]),
    #         status=DONE, seed_simulator=1625693156, metadata={'noise': 'ideal', 'batched_shots_optimization': False,
    #         'remapped_qubits': False, 'parallel_state_update': 1, 'parallel_shots': 16, 'device': 'CPU',
    #         'active_input_qubits': [0, 1], 'measure_sampling': False, 'num_clbits': 2, 'input_qubit_map': [[1, 1], [0, 0]],
    #         'num_qubits': 2, 'method': 'statevector', 'result_types': {'statevector': 'save_statevector'},
    #         'result_subtypes': {'statevector': 'single'}, 'fusion': {'applied': False, 'max_fused_qubits': 5,
    #         'threshold': 14, 'enabled': True}}, time_taken=0.011046995)
    data = result.data
    # result: ExperimentResultData(counts={'0x3': 1},
    #         statevector=Statevector([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j], dims=(2, 2)))
    counts = data.get_counts()
    # counts: {'11': 1}

**Using Primitives**

The primitives offer two alternatives for local statevector simulation:

**a. Using the Reference Primitives**

Basic statevector simulation based on :class:`qiskit.quantum_info.Statevector` class.

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

Aer simulation following the statevector method. This would be the direct 1-1 replacement of the Quantum Instance
exeample, as they are both accessing the same simulator. For this reason, the output metadata is richer, and
closer to the Quantum Instance's output.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import Sampler

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    sampler = Sampler(run_options = {"method":"statevector", "shots":200})
    result = sampler.run(circuit).result()
    # result: SamplerResult(quasi_dists=[{3: 1.0}],
    #         metadata=[{'shots': 200, 'simulator_metadata': {'parallel_state_update': 16, 'parallel_shots': 1,
    #         'sample_measure_time': 7.3952e-05, 'noise': 'ideal', 'batched_shots_optimization': False,
    #         'remapped_qubits': False, 'device': 'CPU', 'active_input_qubits': [0, 1], 'measure_sampling': True,
    #         'num_clbits': 2, 'input_qubit_map': [[1, 1], [0, 0]], 'num_qubits': 2, 'method': 'statevector',
    #         'fusion': {'applied': False, 'max_fused_qubits': 5, 'threshold': 14, 'enabled': True}}}])

    quasi_dists = result.quasi_dists
    # quasi_dists: [{3: 1.0}]

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: Expectation Value Calculation with Local Noisy Simulation</font></a></summary>
    <br>

**Using Quantum Instance**

The most common use case for computing expectation values with the Quantum Instance was as in combination with the
Opflow library. You can see more information in the opflow migration guide [link].

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.opflow import StateFn, PauliSumOp, PauliExpectation, CircuitSampler
    from qiskit.utils import QuantumInstance
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit.providers.fake_provider import FakeVigo

    # Define problem
    op = PauliSumOp.from_list([("XY",1)])
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    state = StateFn(qc)
    measurable_expression = StateFn(op, is_measurement=True).compose(state)
    expectation = PauliExpectation().convert(measurable_expression)

    # Define Quantum Instance with noisy simulator
    device = FakeVigo()
    noise_model = NoiseModel.from_backend(device)
    coupling_map = device.configuration().coupling_map

    backend = AerSimulator()
    qi = QuantumInstance(backend=backend, shots=1024,
                         seed_simulator=42, seed_transpiler=42,
                         coupling_map=coupling_map, noise_model=noise_model)

    # Run
    sampler = CircuitSampler(qi).convert(expectation)
    expectation_value = sampler.eval().real
    # expectation_value: -0.04101562500000017

**Using Primitives**

Now, the primitives have allowed to combine the opflow and quantum instance functionality in a single ``Estimator``.
In this case, for local noisy simulation, this will be the Aer Estimator.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.primitives import Estimator
    from qiskit.providers.fake_provider import FakeVigo

    # Define problem
    op = SparsePauliOp("XY")
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)

    # Define Aer Estimator with noisy simulator
    device = FakeVigo()
    noise_model = NoiseModel.from_backend(device)
    coupling_map = device.configuration().coupling_map

    estimator = Estimator(
                backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                },
                run_options={"seed": 42, "shots": 1024},
                transpile_options={"seed_transpiler": 42},
            )

    # Run
    expectation_value = estimator.run(qc,op).result().values
    # expectation_value = array([-0.04101562])

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
    from qiskit import IBMQ # USE NON-IBMQ syntax!!!

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()

    IBMQ.load_account()
    provider = IBMQ.get_provider()
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
resources:

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
    <summary><a><font size="+1">Example 4: Circuit Sampling on IBM Backend with Bound and Unbound Pass Managers</font></a></summary>
    <br>

(This is a dummy example, the passes chosen might not make much sense)

**Using Quantum Instance**

.. code-block:: python

    from qiskit.circuit import QuantumRegister, Parameter
    from qiskit.utils import QuantumInstance
    from qiskit import IBMQ # USE NON-IBMQ syntax!!!
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import BasicSwap, Unroller

    q = QuantumRegister(7, 'q')
    p = Parameter('p')
    circuit = QuantumCircuit(q)
    circuit.h(q[0])
    circuit.cx(q[0], q[4])
    circuit.cx(q[2], q[3])
    circuit.cx(q[6], q[1])
    circuit.cx(q[5], q[0])
    circuit.rz(p, q[2])
    circuit.cx(q[5], q[0])
    circuit.measure_all()

    coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    coupling_map = CouplingMap(couplinglist=coupling)
    unbound_pm = PassManager(BasicSwap(coupling_map))

    pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
    bound_pm = PassManager(pass_)

    # Define backend!

    qi = QuantumInstance(
        backend=backend,
        shots=1000,
        pass_manager=unbound_pm,
        bound_pass_manager=bound_pm
    )

    result = qi.execute(circuit).results[0]

**Using Primitives**

The only primitives that currently accept custom pass managers are the Backend Primitives. For the Runtime and
Aer primitives, it is possible to still perform custom unbound transpilation passes by pre-transpiling locally
and activating the ``skip_transpilation=True`` option. However, this option will not work for bound pass managers.

.. code-block:: python

    from qiskit.primitives import BackendSampler
    from qiskit.circuit import QuantumRegister, Parameter
    from qiskit.utils import QuantumInstance
    from qiskit import IBMQ # USE NON-IBMQ syntax!!!
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import BasicSwap, Unroller

    q = QuantumRegister(7, 'q')
    p = Parameter('p')
    circuit = QuantumCircuit(q)
    circuit.h(q[0])
    circuit.cx(q[0], q[4])
    circuit.cx(q[2], q[3])
    circuit.cx(q[6], q[1])
    circuit.cx(q[5], q[0])
    circuit.rz(p, q[2])
    circuit.cx(q[5], q[0])
    circuit.measure_all()

    coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    coupling_map = CouplingMap(couplinglist=coupling)
    unbound_pm = PassManager(BasicSwap(coupling_map))

    pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
    bound_pm = PassManager(pass_)

    # Define backend!

    # can you set the unbound pm?
    sampler = BackendSampler(backend=backend, bound_pass_manager=bound_pm)
    sampler.set_transpile_options(pass_manager=unbound_pm) #?????

    result = sampler.run(circuit).quasi_dists

.. raw:: html

    </details>


https://qiskit.org/documentation/partners/qiskit_ibm_runtime/apidocs/options.html
