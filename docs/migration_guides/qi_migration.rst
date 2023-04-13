################################
Quantum Instance Migration Guide
################################

The :class:`~qiskit.utils.QuantumInstance` is a utility class that allows the joint
configuration of the circuit transpilation and execution steps, and provides functions
at a higher level of abstraction for a more convenient integration with algorithms.
These include measurement error mitigation, splitting/combining execution to
conform to job limits,
and ensuring reliable execution of circuits with additional job management tools.

The :class:`~qiskit.utils.QuantumInstance` is being deprecated for several reasons:
On one hand, the functionality of :meth:`~qiskit.utils.QuantumInstance.execute` has
now been delegated to the different implementations of the :mod:`~qiskit.primitives` base classes.
On the other hand, with the direct implementation of transpilation at the primitives level,
the algorithms no longer
need to manage that aspect of execution, and thus :meth:`~qiskit.utils.QuantumInstance.transpile` is no longer
required by the workflow. If desired, custom transpilation routines can still be performed at the
user level through the :mod:`~qiskit.transpiler` module (see table below).


The following table summarizes the migration alternatives for the :class:`~qiskit.utils.QuantumInstance` class:

.. list-table::
   :header-rows: 1

   * - QuantumInstance method
     - Alternative
   * - :meth:`.QuantumInstance.execute`
     - :meth:`qiskit.primitives.Sampler.run` or :meth:`qiskit.primitives.Estimator.run`
   * - :meth:`.QuantumInstance.transpile`
     - :meth:`qiskit.compiler.transpile`
   * - :meth:`.QuantumInstance.assemble`
     - :meth:`qiskit.compiler.assemble`

The remainder of this guide will focus on the :meth:`.QuantumInstance.execute` to
:mod:`~qiskit.primitives` migration path.

Contents
========

* `Choosing the right primitive for your task`_
* `Choosing the right primitive for your settings`_
* `Code examples`_

.. attention::

    **Background on the Qiskit Primitives**

    The Qiskit Primitives are algorithmic abstractions that encapsulate the access to backends or simulators
    for an easy integration into algorithm workflows.

    The current pool of primitives includes **two** different **classes** (:class:`~qiskit.primitives.Sampler` and
    :class:`~qiskit.primitives.Estimator`) that can be imported from **three** different locations (
    :mod:`qiskit.primitives`, :mod:`qiskit_aer.primitives` and :mod:`qiskit_ibm_runtime` ). In addition to the
    reference Sampler and Estimator, :mod:`qiskit.primitives` also contains a
    :class:`~qiskit.primitives.BackendSampler` and a :class:`~qiskit.primitives.BackendEstimator` class. These are
    wrappers for ``backend.run()`` that follow the primitives interface.

    This guide uses the following naming standard to refer to the primitives:

    - *Primitives* - Any Sampler/Estimator implementation
    - *Reference Primitives* - The Sampler and Estimator in :mod:`qiskit.primitives` --> ``from qiskit.primitives import Sampler/Estimator``
    - *Aer Primitives* - The Sampler and Estimator in :mod:`qiskit_aer.primitives` --> ``from qiskit_aer.primitives import Sampler/Estimator``
    - *Runtime Primitives* - The Sampler and Estimator in :mod:`qiskit_ibm_runtime` --> ``from qiskit_ibm_runtime import Sampler/Estimator``
    - *Backend Primitives* - The BackendSampler and BackendEstimator in :mod:`qiskit.primitives` --> ``from qiskit import BackendSampler/BackendEstimator``

    For guidelines on which primitives to choose for your task, please continue reading.

Choosing the right primitive for your task
===========================================

The :class:`~qiskit.utils.QuantumInstance` was designed to be an abstraction over transpile/run.
It took inspiration from :func:`~qiskit.execute_function.execute`, but retained config information that could be set
at the algorithm level, to save the user from defining the same parameters for every transpile/execute call.

The :mod:`qiskit.primitives` share some of these features, but unlike the :class:`~qiskit.utils.QuantumInstance`,
there are multiple primitive classes, and each is optimized for a specific
purpose. Selecting the right primitive (``Sampler`` or ``Estimator``) requires some knowledge about
**what** it is expected to do and **where/how** it is expected to run.

.. note::

    The role of the primitives is two-fold. On one hand, they act as access points to backends and simulators.
    On the other hand, they are **algorithmic** abstractions with defined tasks:

    * The ``Estimator`` takes in circuits and observables and returns **expectation values**.
    * The ``Sampler`` takes in circuits, measures them, and returns their  **quasi-probability distributions**.

In order to know which primitive to use instead of :class:`~qiskit.utils.QuantumInstance`, you should ask
yourself two questions:

1. What is the minimal unit of information used by your algorithm?
    a. **Expectation value** - you will need an ``Estimator``
    b. **Probability distribution** (from sampling the device) - you will need a ``Sampler``

2. How do you want to execute your circuits?

    This question is not new. In the legacy algorithm workflow, you would have to decide to set up a
    :class:`~qiskit.utils.QuantumInstance` with either a real backend from a provider, or a simulator.
    Now, this "backend selection" process is translated to **where** do you import your primitives
    from:

    a. Using **local** statevector simulators for quick prototyping: **Reference Primitives**
    b. Using **local** noisy simulations for finer algorithm tuning: **Aer Primitives**
    c. Accessing **runtime-enabled backends** (or cloud simulators): **Runtime Primitives**
    d. Accessing **non runtime-enabled backends** : **Backend Primitives**

Arguably, the ``Sampler`` is the closest primitive to :class:`~qiskit.utils.QuantumInstance`, as they
both execute circuits and provide a result back. However, with the :class:`~qiskit.utils.QuantumInstance`,
the result data was backend dependent (it could be a counts ``dict``, a :class:`numpy.array` for
statevector simulations, etc), while the ``Sampler`` normalizes its ``SamplerResult`` to
return a :class:`~qiskit.result.QuasiDistribution` object with the resulting quasi-probability distribution.

The ``Estimator`` provides a specific abstraction for the expectation value calculation that can replace
the use of :class:`.QuantumInstance` as well as the associated pre- and post-processing steps, usually performed
with an additional library such as :mod:`qiskit.opflow`.

Choosing the right primitive for your settings
==============================================

Certain :class:`~qiskit.utils.QuantumInstance` features are only available in certain primitive implementations.
The following table summarizes the most common :class:`~qiskit.utils.QuantumInstance` settings and which
primitives **expose a similar setting through their interface**:

.. attention::

    In some cases, a setting might not be exposed through the interface, but there might an alternative path to make
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
     - No (inferred from internal ``backend``)
   * - Transpiler settings: ``seed_transpiler``, ``optimization_level``
     - No
     - No
     - Yes (via ``options``) (*)
     - Yes (via ``.set_transpile_options()``)
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
     - Does not apply
     - Does not apply
     - Sessions, callback (**)
     - No


(*) For more information on error mitigation and setting options on Runtime Primitives, visit
`this link <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/stubs/qiskit_ibm_runtime.options.Options.html#qiskit_ibm_runtime.options.Options>`_.

(**) For more information on Runtime sessions, visit `this how-to <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/how_to/run_session.html>`_.

Code examples
=============

.. dropdown:: Example 1: Circuit Sampling with Local Simulation
    :animate: fade-in-slide-down

    **Using Quantum Instance**

    The only alternative for local simulations using the quantum instance was using an Aer simulator backend.
    If no simulation method is specified, the Aer simulator will default to an exact simulation
    (statevector/stabilizer), if shots are specified, it will add shot noise.
    Please note that ``QuantumInstance.execute()`` returned the counts in hexadecimal format.

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit.utils import QuantumInstance

        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.measure_all()

        simulator = AerSimulator()
        qi = QuantumInstance(backend=simulator, shots=200)
        result = qi.execute(circuit).results[0]
        data = result.data
        counts = data.counts

        print("Counts: ", counts)
        print("Data: ", data)
        print("Result: ", result)

    .. code-block:: text

        Counts: {'0x3': 200}
        Data: ExperimentResultData(counts={'0x3': 200})
        Result:  ExperimentResult(shots=200, success=True, meas_level=2, data=ExperimentResultData(counts={'0x3': 200}), header=QobjExperimentHeader(clbit_labels=[['meas', 0], ['meas', 1]], creg_sizes=[['meas', 2]], global_phase=0.0, memory_slots=2, metadata={}, n_qubits=2, name='circuit-99', qreg_sizes=[['q', 2]], qubit_labels=[['q', 0], ['q', 1]]), status=DONE, seed_simulator=2846213898, metadata={'parallel_state_update': 16, 'parallel_shots': 1, 'sample_measure_time': 0.00025145, 'noise': 'ideal', 'batched_shots_optimization': False, 'remapped_qubits': False, 'device': 'CPU', 'active_input_qubits': [0, 1], 'measure_sampling': True, 'num_clbits': 2, 'input_qubit_map': [[1, 1], [0, 0]], 'num_qubits': 2, 'method': 'stabilizer', 'fusion': {'enabled': False}}, time_taken=0.000672166)

    **Using Primitives**

    The primitives offer two alternatives for local simulation, one with the Reference primitives
    and one with the Aer primitives. As mentioned above the closest alternative to ``QuantumInstance.execute()``
    for sampling is the ``Sampler`` primitive.

    **a. Using the Reference Primitives**

    Basic simulation implemented using the :mod:`qiskit.quantum_info` module. If shots are
    specified, the results will include shot noise. Please note that
    the resulting quasi-probability distribution does not use bitstrings but **integers** to identify the states.

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit.primitives import Sampler

        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.measure_all()

        sampler = Sampler()
        result = sampler.run(circuit, shots=200).result()
        quasi_dists = result.quasi_dists

        print("Quasi-dists: ", quasi_dists)
        print("Result: ", result)

    .. code-block:: text

        Quasi-dists: [{3: 1.0}]
        Result: SamplerResult(quasi_dists=[{3: 1.0}], metadata=[{'shots': 200}])

    **b. Using the Aer Primitives**

    Aer simulation following the statevector method. This would be the closer replacement of the
    :class:`~qiskit.utils.QuantumInstance`
    example, as they are both accessing the same simulator. For this reason, the output metadata is
    closer to the Quantum Instance's output. Please note that
    the resulting quasi-probability distribution does not use bitstrings but **integers** to identify the states.

    .. note::

        The :class:`qiskit.result.QuasiDistribution` class returned as part of the :class:`qiskit.primitives.SamplerResult`
        exposes two methods to convert the result keys from integer to binary strings/hexadecimal:

            - :meth:`qiskit.result.QuasiDistribution.binary_probabilities`
            - :meth:`qiskit.result.QuasiDistribution.hex_probabilities`


    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit_aer.primitives import Sampler

        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.measure_all()

        # if no Noise Model provided, the aer primitives
        # perform an exact (statevector) simulation
        sampler = Sampler()
        result = sampler.run(circuit, shots=200).result()
        quasi_dists = result.quasi_dists
        # convert keys to binary bitstrings
        binary_dist = quasi_dists[0].binary_probabilities()

        print("Quasi-dists: ", quasi_dists)
        print("Result: ", result)
        print("Binary quasi-dist: ", binary_dist)

    .. code-block:: text

        Quasi-dists: [{3: 1.0}]
        Result: SamplerResult(quasi_dists=[{3: 1.0}], metadata=[{'shots': 200, 'simulator_metadata': {'parallel_state_update': 16, 'parallel_shots': 1, 'sample_measure_time': 9.016e-05, 'noise': 'ideal', 'batched_shots_optimization': False, 'remapped_qubits': False, 'device': 'CPU', 'active_input_qubits': [0, 1], 'measure_sampling': True, 'num_clbits': 2, 'input_qubit_map': [[1, 1], [0, 0]], 'num_qubits': 2, 'method': 'statevector', 'fusion': {'applied': False, 'max_fused_qubits': 5, 'threshold': 14, 'enabled': True}}}])
        Binary quasi-dist:  {'11': 1.0}

.. dropdown:: Example 2: Expectation Value Calculation with Local Noisy Simulation
    :animate: fade-in-slide-down

    While this example does not include a direct call to ``QuantumInstance.execute()``, it shows
    how to migrate from a :class:`~qiskit.utils.QuantumInstance`-based to a :mod:`~qiskit.primitives`-based
    workflow.

    **Using Quantum Instance**

    The most common use case for computing expectation values with the Quantum Instance was as in combination with the
    :mod:`~qiskit.opflow` library. You can see more information in the `opflow migration guide <http://qisk.it/opflow_migration>`_.

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit.opflow import StateFn, PauliSumOp, PauliExpectation, CircuitSampler
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_provider import IBMProvider

        # Define problem using opflow
        op = PauliSumOp.from_list([("XY",1)])
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)

        state = StateFn(qc)
        measurable_expression = StateFn(op, is_measurement=True).compose(state)
        expectation = PauliExpectation().convert(measurable_expression)

        # Define Quantum Instance with noisy simulator
        provider = IBMProvider()
        device = provider.get_backend("ibmq_manila")
        noise_model = NoiseModel.from_backend(device)
        coupling_map = device.configuration().coupling_map

        backend = AerSimulator()
        qi = QuantumInstance(backend=backend, shots=1024,
                            seed_simulator=42, seed_transpiler=42,
                            coupling_map=coupling_map, noise_model=noise_model)

        # Run
        sampler = CircuitSampler(qi).convert(expectation)
        expectation_value = sampler.eval().real

        print(expectation_value)

    .. code-block:: text

        -0.04687500000000008

    **Using Primitives**

    The primitives now allow the combination of the opflow and quantum instance functionality in a single ``Estimator``.
    In this case, for local noisy simulation, this will be the Aer Estimator.

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import Estimator
        from qiskit_ibm_provider import IBMProvider

        # Define problem
        op = SparsePauliOp("XY")
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)

        # Define Aer Estimator with noisy simulator
        device = provider.get_backend("ibmq_manila")
        noise_model = NoiseModel.from_backend(device)
        coupling_map = device.configuration().coupling_map

        # if Noise Model provided, the aer primitives
        # perform a "qasm" simulation
        estimator = Estimator(
                   backend_options={ # method chosen automatically to match options
                       "coupling_map": coupling_map,
                       "noise_model": noise_model,
                   },
                   run_options={"seed": 42, "shots": 1024},
                  transpile_options={"seed_transpiler": 42},
               )

        # Run
        expectation_value = estimator.run(qc, op).result().values

        print(expectation_value)

    .. code-block:: text

        [-0.04101562]

.. dropdown:: Example 3: Circuit Sampling on IBM Backend with Error Mitigation
    :animate: fade-in-slide-down

    **Using Quantum Instance**

    The ``QuantumInstance`` interface allowed the configuration of measurement error mitigation settings such as the method, the
    matrix refresh period or the mitigation pattern. This configuration is no longer available in the primitives
    interface.

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
        backend = provider.get_backend("ibmq_montreal")

        qi = QuantumInstance(
            backend=backend,
            shots=4000,
            measurement_error_mitigation_cls=CompleteMeasFitter,
            cals_matrix_refresh_period=0,
        )

        result = qi.execute(circuit).results[0].data
        print(result)

    .. code-block:: text

        ExperimentResultData(counts={'11': 4000})


    **Using Primitives**

    The Runtime Primitives offer a suite of error mitigation methods that can be easily turned on with the
    ``resilience_level`` option. These are, however, not configurable. The sampler's ``resilience_level=1``
    is the closest alternative to the Quantum Instance's measurement error mitigation implementation, but this
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
        backend = service.backend("ibmq_montreal")

        options = Options(resilience_level = 1) # 1 = measurement error mitigation
        sampler = Sampler(session=backend, options=options)

        # Run
        result = sampler.run(circuit, shots=4000).result()
        quasi_dists = result.quasi_dists

        print("Quasi dists: ", quasi_dists)

    .. code-block:: text

        Quasi dists: [{2: 0.0008492371522941081, 3: 0.9968874384378738, 0: -0.0003921227905920063,
		 1: 0.002655447200424097}]

.. dropdown:: Example 4: Circuit Sampling with Custom Bound and Unbound Pass Managers
    :animate: fade-in-slide-down

    The management of transpilation is different between the ``QuantumInstance`` and the Primitives.

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

    .. attention::

        Care is needed when setting ``skip_transpilation=True`` with the ``Estimator`` primitive.
        Since operator and circuit size need to match for the Estimator, should the custom transpilation change
        the circuit size, then the operator must be adapted before sending it
        to the Estimator, as there is no currently no mechanism to identify the active qubits it should consider.

    ..
        In opflow, the ansatz would always have the basis change and measurement gates added before transpilation,
        so if the circuit ended up on more qubits it did not matter.

    Note that the primitives **do** handle parameter bindings, meaning that even if a ``bound_pass_manager`` is defined in a
    Backend Primitive, you do not have to manually assign parameters as expected in the Quantum Instance workflow.

    The use-case that motivated the addition of the two-stage transpilation to the ``QuantumInstance`` was to allow
    running pulse-efficient transpilation passes with the :class:`~qiskit.opflow.CircuitSampler` class. The following
    example shows to migrate this particular use-case, where the ``QuantumInstance.execute()`` method is called
    under the hood by the :class:`~qiskit.opflow.CircuitSampler`.

    **Using Quantum Instance**

    .. code-block:: python

        from qiskit.circuit.library.standard_gates.equivalence_library import StandardEquivalenceLibrary as std_eqlib
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.opflow import CircuitSampler, StateFn
        from qiskit.providers.fake_provider import FakeBelem
        from qiskit.transpiler import PassManager, PassManagerConfig, CouplingMap
        from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
        from qiskit.transpiler.passes import (
            Collect2qBlocks, ConsolidateBlocks, Optimize1qGatesDecomposition,
            RZXCalibrationBuilderNoEcho, UnrollCustomDefinitions, BasisTranslator
        )
        from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import EchoRZXWeylDecomposition
        from qiskit.utils import QuantumInstance

        # Define backend
        backend = FakeBelem()

        # Build the pass manager for the parameterized circuit
        rzx_basis = ['rzx', 'rz', 'x', 'sx']
        coupling_map = CouplingMap(backend.configuration().coupling_map)
        config = PassManagerConfig(basis_gates=rzx_basis, coupling_map=coupling_map)
        pre = level_1_pass_manager(config)
        inst_map = backend.defaults().instruction_schedule_map

        # Build a pass manager for the CX decomposition (works only on bound circuits)
        post = PassManager([
            # Consolidate consecutive two-qubit operations.
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=['rz', 'sx', 'x', 'rxx']),

            # Rewrite circuit in terms of Weyl-decomposed echoed RZX gates.
            EchoRZXWeylDecomposition(inst_map),

            # Attach scaled CR pulse schedules to the RZX gates.
            RZXCalibrationBuilderNoEcho(inst_map),

            # Simplify single-qubit gates.
            UnrollCustomDefinitions(std_eqlib, rzx_basis),
            BasisTranslator(std_eqlib, rzx_basis),
            Optimize1qGatesDecomposition(rzx_basis),
        ])

        # Instantiate qi
        quantum_instance = QuantumInstance(backend, pass_manager=pre, bound_pass_manager=post)

        # Define parametrized circuit and parameter values
        qc = RealAmplitudes(2)
        print(qc.decompose())
        param_dict = {p: 0.5 for p in qc.parameters}

        # Instantiate CircuitSampler
        sampler = CircuitSampler(quantum_instance)

        # Run
        quasi_dists = sampler.convert(StateFn(qc), params=param_dict).sample()
        print("Quasi-dists: ", quasi_dists)

    .. code-block:: text

             ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
        q_0: ┤ Ry(θ[0]) ├──■──┤ Ry(θ[2]) ├──■──┤ Ry(θ[4]) ├──■──┤ Ry(θ[6]) ├
             ├──────────┤┌─┴─┐├──────────┤┌─┴─┐├──────────┤┌─┴─┐├──────────┤
        q_1: ┤ Ry(θ[1]) ├┤ X ├┤ Ry(θ[3]) ├┤ X ├┤ Ry(θ[5]) ├┤ X ├┤ Ry(θ[7]) ├
             └──────────┘└───┘└──────────┘└───┘└──────────┘└───┘└──────────┘
        Quasi-dists: {'11': 0.443359375, '10': 0.21875, '01': 0.189453125, '00': 0.1484375}

    **Using Primitives**

    Let's see how the workflow changes with the Backend Sampler:

    .. code-block:: python

        from qiskit.circuit.library.standard_gates.equivalence_library import StandardEquivalenceLibrary as std_eqlib
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.primitives import BackendSampler
        from qiskit.providers.fake_provider import FakeBelem
        from qiskit.transpiler import PassManager, PassManagerConfig, CouplingMap
        from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
        from qiskit.transpiler.passes import (
            Collect2qBlocks, ConsolidateBlocks, Optimize1qGatesDecomposition,
            RZXCalibrationBuilderNoEcho, UnrollCustomDefinitions, BasisTranslator
        )
        from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import EchoRZXWeylDecomposition

        # Define backend
        backend = FakeBelem()

        # Build the pass manager for the parameterized circuit
        rzx_basis = ['rzx', 'rz', 'x', 'sx']
        coupling_map = CouplingMap(backend.configuration().coupling_map)
        config = PassManagerConfig(basis_gates=rzx_basis, coupling_map=coupling_map)
        pre = level_1_pass_manager(config)

        # Build a pass manager for the CX decomposition (works only on bound circuits)
        inst_map = backend.defaults().instruction_schedule_map
        post = PassManager([
            # Consolidate consecutive two-qubit operations.
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=['rz', 'sx', 'x', 'rxx']),

            # Rewrite circuit in terms of Weyl-decomposed echoed RZX gates.
            EchoRZXWeylDecomposition(inst_map),

            # Attach scaled CR pulse schedules to the RZX gates.
            RZXCalibrationBuilderNoEcho(inst_map),

            # Simplify single-qubit gates.
            UnrollCustomDefinitions(std_eqlib, rzx_basis),
            BasisTranslator(std_eqlib, rzx_basis),
            Optimize1qGatesDecomposition(rzx_basis),
        ])

        # Define parametrized circuit and parameter values
        qc = RealAmplitudes(2)
        qc.measure_all() # add measurements!
        print(qc.decompose())

        # Instantiate backend sampler with skip_transpilation
        sampler = BackendSampler(backend=backend, skip_transpilation=True, bound_pass_manager=post)

        # Run unbound transpiler pass
        transpiled_circuit = pre.run(qc)

        # Run sampler
        quasi_dists = sampler.run(transpiled_circuit, [[0.5] * len(qc.parameters)]).result().quasi_dists
        print("Quasi-dists: ", quasi_dists)

    .. code-block:: text

                ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐ ░ ┌─┐
           q_0: ┤ Ry(θ[0]) ├──■──┤ Ry(θ[2]) ├──■──┤ Ry(θ[4]) ├──■──┤ Ry(θ[6]) ├─░─┤M├───
                ├──────────┤┌─┴─┐├──────────┤┌─┴─┐├──────────┤┌─┴─┐├──────────┤ ░ └╥┘┌─┐
           q_1: ┤ Ry(θ[1]) ├┤ X ├┤ Ry(θ[3]) ├┤ X ├┤ Ry(θ[5]) ├┤ X ├┤ Ry(θ[7]) ├─░──╫─┤M├
                └──────────┘└───┘└──────────┘└───┘└──────────┘└───┘└──────────┘ ░  ║ └╥┘
        meas: 2/═══════════════════════════════════════════════════════════════════╩══╩═
                                                                                   0  1
        Quasi-dists:  [{1: 0.18359375, 2: 0.2333984375, 0: 0.1748046875, 3: 0.408203125}]
