
*********
Changelog
*********

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_.

  **Types of changes:**

  - **Added**: for new features.
  - **Changed**: for changes in existing functionality.
  - **Deprecated**: for soon-to-be removed features.
  - **Removed**: for now removed features.
  - **Fixed**: for any bug fixes.
  - **Security**: in case of vulnerabilities.

`UNRELEASED`_
=============

Removed
-------
- The previously deprecated functions ``qiksit.visualization.plot_state`` and
  ``qiskit.visualization.iplot_state`` have been removed. Instead use the
  specific functions for each plot type (#2325).
- International documentation of outdated readme etc (#2302)
- Removed deprecated options in execute, transpile, and assemble. Removed deprecated compiler. 
-  Removed deprecated qcvv in tools. Removed deprecated converters qobj_to_circuits and circuits_to_qobj (#2301)
- The previously deprecated ``qiskit._util`` module has been removed. Use
  ``qiskit.util`` instead. (#2329)

`0.8.0`_ - 2019-05-02
=====================


Added
-----

- Added exact and approximate decomposition of SU(4) to arbitrary supercontrolled basis
- Introduced schedule lo configuration. (#2115)
- Introduced pulse schedule assembler. (#2115)
- Builtin library of continuous pulses and builtin library of discrete pulses which are obtained
  by sampling continuous pulses with default sampling strategy.
- Sampler decorator and standard sampler library for conversion of continuous pulses
  to discrete ``SamplePulse`` (#2042).
- Core StochasticSwap routine implimented in Cython (#1789).
- Added QuantumChannel classes SuperOp, Choi, Kraus, Stinespring, PTM, Chi to
  quantum_info for manipulating quantum channels and CPTP maps.
- Added Operator object to quantum_info for representing matrix operators.
- Introduced the backend defaults model and endpoint for pulse backends (#2101).
- `meas_level` to result schema (#2085).
- Core StochasticSwap routine implemented in Cython (#1789).
- New EnlargeWithAncilla pass for adding ancilla qubits after a Layout
  selection pass (#1603).
- New Unroll2Q pass for unrolling gates down to just 1q or 2q gates (#1614).
- Added support for register slicing when applying operations to a register (#1643).
- Added in new parameter ``justify`` to the text, mpl and latex circuit drawers to say how the
  circuit should be aligned. (#1725, #1797, #1977)
- Added function for purity of a mixed state in ``qiskit.quantum_information``
  (#1733)
- Added parameter to the TextProgressBar to allow the output to be sent to a
  different output stream
- Added a ``__qiskit_version__`` parameter to the qiskit namespace. This will
  contain a dictionary of versions for all installed qiskit elements. (#1885).
- Added a ``RunConfig`` object for configurations related to running an
  experiment (e.g. shots, memory) (#1856)
- Added a ``TranspileConfig`` object for configurations related to transforming
  circuits (e.g. basis_gates, coupling_map, initial_layout) (#1856)
- Added a ``qiskit.compiler`` namespace for all functions that transpile, schedule
  and assemble circuits and pulses (#1856)
- Added support for passing a list of ``basis_gates``, ``coupling_map`` etc. to the
  ``qiskit.compiler.transpile()`` function, each corresponding to one of the circuits (#2163)
- Added a ``qiskit.compiler.assemble_circuits()`` function to generate qobj from some
  circuits and a RunConfig (#1856)
- ``execute()`` and ``assemble()`` allow setting a qobj_header, of type
  QobjHeader or dict, to add extra information to the qobj (and thus result).
- Register indexing supports negative indices (#1875)
- Added new resource estimation passes: ``Depth``, ``Width``, ``Size``, ``CountOps``, and
  ``NumTensorFactors``, all grouped in the ``ResourceEstimation`` analysis pass.
- Added ``nodes_on_wire()`` to DAGCircuit which returns an iterator over all the
  operations on the given wire
- Added new properties to an Instruction:
  ``num_qubits``, ``num_clbits`` (#1816).
- Added a ``QuantumCircuit.append`` public method for appending arbitrary instructions
  to some qubits and clbits in the circuit (#1816).
- Added an ``Instruction.definition`` property that defines a composite instruction
  in terms of other, simpler instructions (#1816).
- Added an ``Instruction.mirror()`` method that mirrors a composite instruction
  (reverses its sub-instructions) (#1816).
- Added an ``PassManager.passes()`` method that returns a list of the passes that
  have been added to the pass manager, including options and flow controllers.
- Added a ``PassManager.run()`` that transforms a ``QuantumCircuit`` according to its
  pass schedule and returns a ``QuantumCircuit``.
- Added a ``qiskit.quantum_info.random`` for generating random states, unitaries, etc (#2119).
- Added a ``qiskit.quantum_info.synthesis`` for algorithms that synthesize circuits (#2119).
- Added a ``NoiseAdaptiveLayout`` pass to compute a backend calibration-data aware initial
  qubit layout. (#2089)
- Gates and instructions in a circuit accept integers as parameters to refer to
  wires instead of named bits.
- Added a ``OptimizeSwapBeforeMeasure`` pass that removes the swap gates when they
  are followed by a measurement instruction, moving the latter to the proper wire. (#1890)
- Added a ``RemoveDiagonalGatesBeforeMeasure`` pass that removes the diagonal gates when they
  are followed by a measurement instruction. (#2208)
- Added a ``CommutativeCancellation`` pass that cancels self-inverse gates and combines
  rotations about the Z axis, leveraging previously-found gate commutation relations. (#2012)
- Add an option for using a user config file to enable changing default
  settings for various functions in qiskit. Right now it only supports setting
  the default circuit drawing backend. (#2122)
- Added a ``Collect2qBlocks`` pass that analyzes the circuit for uninterrupted sequences
  of gates (blocks) acting on 2 qubits. (#2134)
- Added a ``ConsolidateBlocks`` that turns previously-collected blocks of any size
  into equivalent Unitary operators in the circuit. (#2134)
- Added support for parameterized circuits. (#2103)
- Added preset PassManagers that offer predetermined pipelines of transpiler passes. (#2163)

Changed
-------

- require scipy>=1.0, use `scipy.stats.unitary_group.rvs` for `random_unitary()`.
- two_qubit_kak decomposition works with Operator or raw matrix input objects.
- process_fidelity works with QuantumChannel and Operator object inputs.
- Backend defaults values are no longer required (#2101).
- QuantumCircuit properties more self-consistent and no longer need DAG (#1993).
- The most connected subset in DenseLayout is now reduced bandwidth (#2021).
- plot_histogram now allows sorting by Hamming distance from target_string (#2064).
- FunctionalPulse is no longer a class and instead is a decorator, `functional_pulse`
  that returns a `SamplePulse` when called. (#2043)
- Changed ``average_data`` to accept observable input in matrix form (#1858)
- Change random_state to take in dim over number of qubits (#1857)
- The ``Exception`` subclasses have been moved to an ``.exceptions`` module
  within each package (for example, ``qiskit.exceptions.QiskitError``) (#1600).
- The ``QiskitTestCase`` and testing utilities are now included as part of
  ``qiskit.test`` and thus available for third-party implementations, with
  convenience test cases for providers and backends. (#1616, #1844)
- The snapshot instruction now takes ``label`` and ``snap_type`` instead of
  ``slot`` (#1615).
- The test folders have been reorganized to match the python modules (#1625)
- The circuits_to_qobj no longers uses the unrollers (#1629)
- The previously deprecated default output of ``circuit_drawer()`` (using latex
  and falling back to mpl) is no longer present. Instead the default output
  is the ascii art ``text`` output backend.
- Changed param to params in Instruction (#1665).
- ``dag_drawer`` and ``plot_gate_map`` are available via importing
  ``qiskit.tools.visualization``. They will raise at the point of use, if
  dependencies are not installed (#1669).
- The ``qiskit.validation`` schemas are now strict and raise a more specific
  ``ModelValidationError`` (#1695).
- The default transpile pipeline will now add a barrier before the set of
  final measurements when compiling for both simulators and devices (#1591).
- Purity function in ``qiskit.tools.qi.qi`` calls new version in
  ``qiskit.quantum_information`` and issues deprecation warning (#1733)
- Updated `dag.node_counter` to return the current number of nodes (#1763)
- The argument ``basis_gates`` used in ``compile``, ``execute``, and ``transpile``
  is not longer a comma-separated string but a list of strings. For example,
  this basis ``['u1','u2','u3','cx']`` should be used instead of ``'u1,u2,u3,cx'``
  (#1333)
- Methods on the ``DAGCircuit`` which previously returned node_ids and/or dicts now
  return ``DAGNodes``
- The ``Qobj`` classes have been reimplemented using models and schemas, as the
  rest of spec-defined entities. (#1909).
- The rzz gate is now represented as a line when printed in text (#1957).
- Text drawer has support for multi-q gates (#1939).
- Separate ``Qobj`` into ``PulseQobj`` and ``QasmQobj`` (#1969).
- It is possible to define a layout as a list of integers. This maps the ordered list
  of virtual circuit qubits to physical qubits as defined by the list of integers (#1946).
- Instructions no longer have context about where they are in a circuit. Instead,
  the circuit keeps this context. So Instructions are now light-weight and only
  have a name, num_qubits, num_clbits and params (#1816).
- The old syntax for attaching a gate to the circuit then modifying it is no longer
  supported (e.g. ``circuit.s(qr).inverse()`` or ``circuit.s(qr).c_if(cr, 4)``).
  Instead, you must first modify the gate then attach it (#1816).
- ``QuantumCircuit.data`` now contains a list of tuples, where each tuple is a
  (instruction, qarg, carg) (#1816).
- The visualization subpackage has moved from ``qiskit.tools.visualization`` to
  ``qiskit.visualization``. The public API (which was declared stable in
  the 0.7 release) is still accessible off of ``qiskit.tools.visualization``.
  (#1878)
- Layout object can now only be constructed from a dictionary, and must be bijective (#2157).
- ``transpile()`` accepts ``initial_layout`` in the form of dict, list or Layout (#2157).
- Not specifying a basis in ``execute()`` or ``transpile()`` no longer defaults to unrolling
  to the ['u1', 'u2', 'u3', 'cx'] basis. Instead the default behavior is to not unroll,
  unless specifically requested (#2166).
- Instruction.copy() is now a shallow copy instead of deep (#2214)
- Layout and CouplingMap classes are now accessible from qiskit.transpiler (#2222).

Deprecated
----------

- The methods prefixed by `_get` in the DAGCircuit object are being renamed
  without that prefix (see #1346)
- Changed elements in ``couplinglist`` of ``CouplingMap`` from tuples to lists
  (#1666).
- Unroller bases must now be explicit, and violation raises an informative
  ``QiskitError`` (#1802).
- The ``qiskit.tools.qcvv`` package is deprecated in favor of Qiskit Ignis (#1884).
- The ``qiskit.compile()`` function is now deprecated in favor of explicitly
  using the ``qiskit.compiler.transpile()`` function to transform a circuit followed
  by ``qiskit.compiler.assemble()`` to make a qobj out of it.
- ``qiskit.converters.qobj_to_circuits()`` has been deprecated and will be
  removed in a future release. Instead
  ``qiskit.compiler.disassemble_circuits()`` should be used to extract
  ``QuantumCircuit`` objects from a compiled qobj. (#2137)
- The ``qiskit.transpiler.transpile()`` function is deprecated in favor of
  ``qiskit.compiler.transpile()`` (#2166).
- The ``seed_mapper`` argument in ``transpile()`` and ``execute()`` is deprecated in favor of
  ``seed_transpile()``, which sets the seed for all stochastic stages of the transpiler (#2166).
- The ``seed`` argument is ``execute()`` is deprecated in favor of ``seed_simulator`` (#2166).
- The ``pass_manager`` argument in ``transpile()`` is deprecated. Instead, the
  ``pass_manager.run()`` methdod can be used directly to transform the circuit (#2166).
- The ``qiskit._util`` module is deprecated and replaced by ``qiskit.util``.
  ``qiskit._util`` will be removed in the 0.9 release. (#2154)

Fixed
-----

- Fixed #1892, whereby inheriting from QuantumRegister or ClassicalRegister would
  cause a QiskitError in instruction.py (#1908).
- Fixed #829 by removing dependence on scipy unitary_group (#1857).
- Fixed a bug with measurement sampling optimization in BasicAer
  qasm_simulator (#1624).
- Fixed a bug where barriers didn't plot over all qubits when using matplotlib (#1718).
- Fixed a minor conda env bug in Makefile (#1691).
- Fixed a bug in BasicMapper pass operating over multiple registers (#1611).
- Fixed a bug in BarrierBeforeFinalMeasurements which incorrectly moved measurements
  used in conditional operations (#1705).
- Fixed a bug that with transpile ignoring initial layout when
  coupling map is provided (#1711).
- Fixed a bug in the definition of the rzz gate (#1940).
- Fixed a bug in DAGCircuit.collect_runs() that did not exclude conditional gates (#1943).
- Fixed a mapping issue with layouts on non-adjacent qubits, by adding ancillas (#2023).
- Fixed a bug in which an `initial_layout` could be changed even if it made the circuit
  compatible with the device `coupling_map` (#2036).
- Fixed ``qobj_to_circuits`` for circuits that contain initialize instructions
  (#2138)


Removed
-------

- The previously deprecated functions ``plot_circuit()``,
  ``latex_circuit_drawer()``, ``generate_latex_source()``, and
  ``matplotlib_circuit_drawer()`` from ``qiskit.tools.visualization`` have
  been removed. The ``circuit_drawer()`` function from the same module should
  be used instead.
- The previously deprecated keys ``plot_barriers`` and ``reverse_bits`` keys in
  the ``style`` kwarg dict are deprecated, instead the
  ``qiskit.tools.visualization.circuit_drawer()`` kwargs ``plot_barriers`` and
  ``reverse_bits`` should be used instead.
- Removed the wrapper folder as part of the post 0.7 cleanup (#1613).
- Removed the python wrappers of the legacy simualtors now that
  Qiskit Aer is out (#1615).
- Removed simulator instructions ``save``, ``load``, ``wait``, ``noise``
  as unsupported in Aer (#1615).
- Removed circuit.add as deprecated (#1627)
- Removed the unroller (#1629)
- Removed deprecated ``result`` methods (#1659)
- Removed deprecated ``couplingdict`` kwarg from ``CouplingMap`` (#1666)
- Removed deprecated ``transpile_dag()`` ``format`` kwarg (#1664)
- Removed deprecated ``Pauli`` ``v``, ``w``, and ``pauli_group`` case arg as int (#1680)
- Removed deprecated ``state_fidelity()`` function from ``tools.qi`` (#1681)
- Removed ``QISKitError`` in favor of ``QiskitError``. (#1684)
- The IBMQ provider (``qiskit.providers.ibmq``) has been moved to its own
  package (``pip install qiskit-ibmq-provider``). (#1700)
- ``compiled_circuit_qasm`` has been removed from the Qobj header, since it
  was part of the pre-qobj specification (#1715).
- Removed the wigner plotting functions ``plot_wigner_function``,
  ``plot_wigner_curve``, ``plot_wigner_plaquette``, and ``plot_wigner_data``
  (#1860).
- Removed ``Instruction.reapply()`` method (#1816).


`0.7.2`_ - 2019-05-01
=====================


Fixed
-----

- A potential issue where the backend configuration schema validation would
  improperly reject valid responses from the API (#2258)


`0.7.1`_ - 2019-03-04
=====================


Fixed
-----

- Fixed a bug with measurement sampling optimization in BasicAer
  qasm_simulator (#1624).


`0.7.0`_ - 2018-12-19
=====================


Added
-----

- Added DAG visualizer which requires `Graphivz <https://www.graphviz.org/>`_
  (#1059)
- Added an ASCII art circuit visualizer (#909)
- The QuantumCircuit class now returns an ASCII art visualization when treated
  as a string (#911)
- The QuantumCircuit class now has a `draw()` method which behaves the same
  as the `qiskit.tools.visualization.circuit_drawer()` function for visualizing
  the quantum circuit (#911)
- A new method `hinton` can be used on
  `qiskit.tools.visualization.plot_state()` to draw a hinton diagram (#1246)
- Two new constructor methods, `from_qasm_str()` and `from_qasm_file()`, to
  create a QuantumCircuit object from OpenQASM were added to the
  QuantumCircuit class. (#1172)
- New methods in QuantumCircuit for common circuit metrics:
  `size()`, `depth()`, `width()`, `count_ops()`, `num_tensor_factors()` (#1285)
- Added `backend_monitor` and `backend_overview` Jupyter magics,
  as well as `plot_coupling_map` (#1231)
- Added a `Layout` object (#1313)
- New `plot_bloch_multivector()` to plot Bloch vectors from a tensored state
  vector or density matrix. (#1359)
- Per-shot measurement results are available in simulators and select devices.
  Request them by setting ``memory=True`` in ``compile()``/``execute()``,
  and retrieve them from ``result.get_memory()`` (#1385).
- Added a ``qiskit.converters`` module for translation between commonly used
  representations of a circuit: ``dag_to_circuits``, ``circuits_to_dag``,
  ``qobj_to_circuits``, ``circuits_to_qobj``, ``ast_to_dag``.
- PassManager can schedule passes at __init__ time (#1510).
- Added a ``.qobj()`` method for IBMQ and local simulator Jobs (#1532).
- New Decompose pass for decomposing a gate according to a rule (#1487).
- New Unroller pass in the transpiler for unrolling up to some basis (#1455).
- New BarrierBeforeFinalMeasurements pass for preventing final
  measure reorder (#1538).
- New CommutationAnalysis and CommutationTransformation transpiler
  passes for modifying a DAG based on gate commutativity relations (#1500).
- New transpiler mapper pass: BasicSwap (#1270).
- New transpiler mapper pass: LookaheadSwap (#1140).
- New transpiler mapper pass: StochasticSwap (#1520).
- New CXDirection pass for fixing the direction of cx gates (#1410).
- New CheckMap pass for checking if circuit meets mapping requirements (#1433).
- New Optimize1QGate pass for combining chains of 1q rotations (#1442).


Changed
-------

- Schedules and underlying classes are now immutable. (#2186)
- Evolved pass-based transpiler to support advanced functionality (#1060)
- `.retrieve_job()` and `.jobs()` no longer returns results by default,
  instead the result must be accessed by the `result()` method on the job
  objects (#1082).
- Make `backend.status()` dictionary conform with schema.
- The different output backends for the circuit_drawer() visualizations
  have been moved into separate private modules in
  `qiskit.tools.visualizations`. (#1105, #1111)
- DAG nodes contain pointers to Register and Instruction objects, rather
  than their string names (#1189).
- Upgraded some external dependencies to:
   -  networkx>=2.2 (#1267).
- The `qiskit.tools.visualization.circuit_drawer()` method now returns
  a matplotlib.Figure object when the `mpl` output is used and a
  `TextDrawer` object when `text` output is used. (#1224, #1181)
- Speed up the Pauli class and extended its operators (#1271 #1166).
- `IBMQ.save_account()` now takes an `overwrite` option to replace an existing
  account on disk. Default is False (#1295).
- Backend and Provider methods defined in the specification use model objects
  rather than dicts, along with validation against schemas (#1249, #1277,
  #1350). The updated methods include:
  - ``backend.status()`` (#1301).
  - ``backend.configuration()`` (and ``__init__``) (#1323).
  - ``backend.properties()``, returning ``None`` for sims (#1331, #1401).
  - ``qiskit.Result`` (#1360).
- ``backend.provider()`` is now a method instead of a property (#1312).
- Remove local backend (Aer) fallback (#1303)
- The signatures for the plotting functions in
  `qiskit.tools.visualization._counts_visualization.py`,
  `qiskit.tools.visualization._state_visualization.py`, and
  `qiskit.tools.visualization.interactive` have been modified to make them
  in-line with standard Matplotlib calling conventions (#1359).
- Remove local backend (Aer) fallback (#1303).
- DAGCircuits store Instruction and Register objects, instead of name
  references. The DAGCircuit class methods are updated accordingly (#1210).
- ``transpile()`` now takes QuantumCircuit(s) to QuantumCircuit(s), and DAG
  processing is only done internally (#1397).
- The different unrollers are deprecated. The only unrolling happens
  from DAG to DAG (#1210).
- Moved all the circuit modules into a circuit module but for most users it
  is still imported in the top level for QuantumCircuit, QuantumRegister,
  ClassicalRegister
- ``qiskit.backends`` has been renamed to ``qiskit.providers`` (#1531).
- ``qiskit.backends.aer`` has been removed in favor of
  ``qiskit.providers.builtinsimulators`` (Python simulators) and
  ``qiskit.providers.legacysimulators`` (C++ simulators) (#1484)
- ``Aer`` in ``qiskit`` root module depends on having the
  qiskit-aer package installed, by default it is not present. Instead there are
  2 new provider instances in the root module ``BasicAer`` which provides the
  Python simulators and ``LegacySimulators`` which provides the old C++
  simulators in qiskit-terra. (#1484)


Deprecated
----------

- ``plot_circuit()``, ``latex_circuit_drawer()``, ``generate_latex_source()``,
   and ``matplotlib_circuit_drawer()`` from qiskit.tools.visualization are
   deprecated. Instead the ``circuit_drawer()`` function from the same module
   should be used. (#1055)
- The current default output of ``circuit_drawer()`` (using latex and falling
   back on python) is deprecated and will be changed in the future. (#1055)
- The `qiskit.wrapper.load_qasm_string()` and `qiskit.wrapper.load_qasm_file()`
  functions are deprecated and the `QuantumCircuit.from_qasm_str()` and
  `QuantumCircuit.from_qasm_file()` contstructor methods should be used instead
  (#1172)
- The ``plot_barriers`` and ``reverse_bits`` keys in the ``style`` kwarg dict
  are deprecated, instead the `qiskit.tools.visualization.circuit_drawer()`
  kwargs ``plot_barriers`` and ``reverse_bits`` should be used instead. (#1180)
- The ``transpile_dag()`` function ``format`` kwarg for emitting different
  output formats is deprecated (#1319).
- Several methods of ``qiskit.Result`` have been deprecated (#1360).
- The functions `plot_state()` and `iplot_state()` have been depreciated.
  Instead the functions `plot_state_*()` and `iplot_state_*()` should be
  called. (#1359)
- The ``skip_transpiler`` arg has been deprecated from ``compile()`` and
  ``execute()`` in favor of using the PassManager directly.

Fixed
-----

- Fixed a variety of typos throughout sources (#1139)
- Fixed horizontal spacing when drawing barriers before CCNOT gates in latex
  circuit plots (#1051)
- Use case insensitive matching when comparing premium account URLs. (#1102)
- Fixed AerJob status when the submitted Job is in a PENDING state. (#1215)
- Add fallback for when CPU count can't be determined (#1214)
- Fix `random_state` from returning nan (#1258)
- The Clifford simulator `run()` method now works correctly with the updated
  AerJob usage (#1125)
- Fixed an edge case when connection checks would raise an unhandled exception
  (#1226)
- Fixed a bug where the transpiler moved middle-of-circuit measurements to the
  end (#1334)
- The `number_to_keep` kwarg in ``plot_histgram()`` now functions correctly
  (#1359).
- parallel_map no longer creates a progress bar for a single circuit (#1394).
- The `timeout` parameter is now passed into the inner ``_wait_for_submission``
  function in ``IBMQJob`` from ``_wait_for_result`` (#1542).

Removed
-------

- Remove register, available_backends (#1131).
- Remove tools/apps (#1184).
- Removed the dependency on ``IBMQuantumExperience``, as it is now included
  in ``qiskit.backends.IBMQ`` (#1198).
- ``matplotlib`` is no longer in the package requirements and is now an
  optional dependency. In order to use any matplotlib based visualizations
  (which includes the ``qiskit.tools.visualization.circuit_drawer()``
  ``mpl`` output,
  ``qiskit.tools.visualization.plot_state``,
  ``qiskit.tools.visualization.plot_histogram``, and
  ``qiskit.tools.visualization.plot_bloch_vector`` you will now need to ensure
  you manually install and configure matplotlib independently.
- The ``basis`` kwarg for the ``circuit_drawer()`` function to provide an
  alternative list of basis gates has been removed. Instead users should adjust
  the basis gates prior to visualizing the circuit. (#1151)
- ``backend.parameters()`` and ``backend.calibration()`` have been fully
  deprecated, in favour of ``backend.properties()`` (#1305).
- The ``qiskit.tools.file_io`` module has been removed. Conversion between
  ``qiskit.Result`` and json can be achieved using ``.to_dict()`` and
  ``.from_dict()`` directly (#1360).
- The ``qiskit.Result`` class method for ``len()`` and indexing have been
  removed, along with the functions that perform post-processing (#1351).
- The ``get_snapshot()`` and ``get_snapshots()`` method from the ``Result``
  class has been removed. Instead you can access the snapshots in a Result
  using ``Result.data()['snapshots']``.
- Completed the deprecation of ``job.backend_name()``, ``job.id()``, and the
  ``backend_name`` parameter in its constructor.
- The ``qiskit.Result`` class now does post-processing of results returned
  from backends if they are called via the ``Result.get_xxx()`` methods
  (i.e. ``get_counts()``, ``get_memory()``, ``get_statevector()``,
  ``get_unitary()``). The raw data is accessible through ``Result.data()``
  (#1404).
- The ``transpile()`` function kwarg ``format`` has been removed and will
  always return a circuit object. Instead you'll need to manually convert the
  output with the functions provided in ``qiskit.converters``.

`0.6.0`_ - 2018-10-04
=====================


Added
-----

- Added `SchemaValidationError` to be thrown when schema validation fails
  (#881)
- Generalized Qobj schema validation functions for all qiskit schemas (#882).
- Added decorator to check for C++ simulator availability (#662)
- It is possible to cancel jobs in non comercial backends (#687)
- Introduced new `qiskit.IBMQ` provider, with centralized handling of IBMQ
  credentials (qiskitrc file, environment variables). (#547, #948, #1000)
- Add OpenMP parallelization for Apple builds of the cpp simulator (#698).
- Add parallelization utilities (#701)
- Parallelize transpilation (#701)
- New interactive visualizations (#765).
- Added option to reverse the qubit order when plotting a circuit. (#762, #786)
- Jupyter notebook magic function qiskit_job_status, qiskit_progress_bar
  (#701, #734)
- Add a new function ``qobj_to_circuits`` to convert a Qobj object to
  a list of QuantumCircuit objects (#877)
- Allow selective loading of accounts from disk via hub/group/project
  filters to `IBMQ.load_accounts()`.
- Add new `job_monitor` function to automaically check the status of a job
  (#975).


Changed
-------

- Schema tests in `tests/schemas/test_schemas.py` replaced with proper
  unit test (#834).
- Renamed ``QISKit`` to ``Qiskit`` in the documentation. (#634)
- Use ``Qobj`` as the formally defined schema for sending information to the
  devices:
  - introduce the ``qiskit.qobj`` module. (#589, #655)
  - update the ``Qobj`` JSON schema. (#668, #677, #703, #709)
  - update the local simulators for accepting ``Qobj`` as input. (#667)
  - update the ``Result`` class. (#773)
- Use ``get_status_job()`` for checking IBMQJob status. (#641)
- Q network hub/group/project credentials replaced by new url format. (#740)
- Breaking change: ``Jobs`` API simplification. (#686)
- Breaking change: altered tomography APIs to not use QuantumProgram. (#818)
- Breaking change: ``BaseBackend`` API changed, properties are now methods
  (#858)
- When ``plot_histogram()`` or ``plot_state()`` are called from a jupyter
  notebook if there is network connectivity the interactive plots will be used
  by default (#862, #866)
- Breaking change: ``BaseJob`` API changed, any job constructor must be passed
  the backend used to run them and a unique job id (#936).
- Add support for drawing circuit barriers to the latex circuit drawer. This
  requires having the LaTeX qcircuit package version >=2.6.0 installed (#764)


Deprecated
----------

- The ``number_to_keep`` kwarg on the ``plot_histogram()`` function is now
  deprecated. A field of the same name should be used in the ``option``
  dictionary kwarg instead. (#866)
- Breaking change: ``backend.properties()`` instead of
  ``backend.calibration()`` and ``backend.parameters()`` (#870)


Removed
-------

- Removed the QuantumProgram class. (#724)


Fixed
-----

- Fixed ``get_ran_qasm`` methods on ``Result`` instances (#688).
- Fixed ``probabilities_ket`` computation in C++ simulator (#580).
- Fixed bug in the definition of ``cswap`` gate and its test (#685).
- Fixed the examples to be compatible with version 0.5+ (#672).
- Fixed swap mapper using qubits after measurement (#691).
- Fixed error in cpp simulator for 3+ qubit operations (#698).
- Fixed issue with combining or extending circuits that contain CompositeGate
  (#710).
- Fixed the random unitary generation from the Haar measure (#760).
- Fixed the issue with control lines spanning through several classical
  registers (#762).
- Fixed visualizations crashing when using simulator extensions (#885).
- Fixed check for network connection when loading interactive visualizations
  (#892).
- Fixed bug in checking that a circuit already matches a coupling map (#1024).


`0.5.7`_ - 2018-07-19
=====================


Changed
-------

- Add new backend names support, with aliasing for the old ones.


`0.5.6`_ - 2018-07-06
=====================


Changed
-------

- Rename repository to ``qiskit-terra`` (#606).
- Update Bloch sphere to QuTiP version (#618).
- Adjust margin of matplotlib_circuit_drawer (#632)


Removed
-------

- Remove OpenQuantumCompiler (#610).


Fixed
-----

- Fixed broken process error and simulator slowdown on Windows (#613).
- Fixed yzy_to_zyz bugs (#520, #607) by moving to quaternions (#626).


`0.5.5`_ - 2018-07-02
=====================


Added
-----

- Retrieve IBM Q jobs from server (#563, #585).
- Add German introductory documentation (``doc/de``) (#592).
- Add ``unregister()`` for removing previously registered providers (#584).
- Add matplotlib-based circuit drawer (#579).
- Adding backend filtering by least busy (#575).
- Allow running with new display names for IBMQ devices,
  and return those from ``available_backends()`` (#566)
- Introduce Qiskit Transpiler and refactor compilation flow (#578)
- Add CXCancellation pass (#578)


Changed
-------

- Remove backend filtering in individual providers, keep only in wrapper
  (#575).
- Single source of version information (#581)
- Bumped IBMQuantumExperience dependency to 1.9.6 (#600).
- For backend status, `status['available']` is now `status['operational']`
  (#609).
- Added support for registering third-party providers in `register()` (#602).
- Order strings in the output of ``available_backends()`` (#566)


Removed
-------

- Remove Clifford simulator from default available_backends, until its stable
  release (#555).
- Remove ProjectQ simulators for moving to new repository (#553).
- Remove QuantumJob class (#616)


Fixed
-----

- Fix issue with unintended inversion of initializer gates (#573).
- Fix issue with skip_transpiler causing some gates to be ignored silently
  (#562).


`0.5.4`_ - 2018-06-11
=====================


Added
-----

- Performance improvements:
    - remove deepcopies from dagcircuit, and extra check on qasm() (#523).


Changed
-------

- Rename repository to ``qiskit-core`` (#530).
- Repository improvements: new changelog format (#535), updated issue templates
  (#531).
- Renamed the specification schemas (#464).
- Convert ``LocalJob`` tests into unit-tests. (#526)
- Move wrapper ``load_qasm_*`` methods to a submodule (#533).


Removed
-------

- Remove Sympy simulators for moving to new repository (#514)


Fixed
-----

- Fix erroneous density matrix and probabilities in C++ simulator (#518)
- Fix hardcoded backend mapping tests (#521)
- Removed ``_modifiers call`` from ``reapply`` (#534)
- Fix circuit drawer issue with filename location on windows (#543)
- Change initial qubit layout only if the backend coupling map is not satisfied
  (#527)
- Fix incorrect unrolling of t to tdg in CircuitBackend (#557)
- Fix issue with simulator extension commands not reapplying correctly (#556)


`0.5.3`_ - 2018-05-29
=====================


Added
-----

- load_qasm_file / load_qasm_string methods


Changed
-------

- Dependencies version bumped


Fixed
-----

- Crash in the cpp simulator for some linux platforms
- Fixed some minor bugs


`0.5.2`_ - 2018-05-21
=====================


Changed
-------

- Adding Result.get_unitary()


Deprecated
----------

- Deprecating ``ibmqx_hpc_qasm_simulator`` and ``ibmqx_qasm_simulator`` in
  favor of ``ibmq_qasm_simulator``.


Fixed
-----

- Fixing a Mapper issue.
- Fixing Windows 7 builds.


`0.5.1`_ - 2018-05-15
=====================

- There are no code changes.

  MacOS simulator has been rebuilt with external user libraries compiled
  statically, so there’s no need for users to have a preinstalled gcc
  environment.

  Pypi forces us to bump up the version number if we want to upload a new
  package, so this is basically what have changed.


`0.5.0`_ - 2018-05-11
=====================


Improvements
------------

- Introduce providers and rework backends (#376).
    - Split backends into ``local`` and ``ibmq``.
    - Each provider derives from the following classes for its specific
      requirements (``BaseProvider``, ``BaseBackend``, ``BaseJob``).
    - Allow querying result by both circuit name and QuantumCircuit instance.
- Introduce the Qiskit ``wrapper`` (#376).
    - Introduce convenience wrapper functions around commonly used Qiskit
      components (e.g. ``compile`` and ``execute`` functions).
    - Introduce the DefaultQISKitProvider, which acts as a context manager for
      the current session (e.g. providing easy access to all
      ``available_backends``).
    - Avoid relying on QuantumProgram (eventual deprecation).
    - The functions are also available as top-level functions (for example,
      ``qiskit.get_backend()``).
- Introduce ``BaseJob`` class and asynchronous jobs (#403).
    - Return ``BaseJob`` after ``run()``.
    - Mechanisms for querying ``status`` and ``results``, or to ``cancel`` a
      job.
- Introduce a ``skip_transpiler`` flag for ``compile()`` (#411).
- Introduce schemas for validating interfaces between qiskit and backends
  (#434)
  - qobj_schema
  - result_schema
  - job_status_schema
  - default_pulse_config_schema
  - backend_config_schema
  - backend_props_schema
  - backend_status_schema
- Improve C++ simulator (#386)
    - Add ``tensor_index.hpp`` for multi-partite qubit vector indexing.
    - Add ``qubit_vector.hpp`` for multi-partite qubit vector algebra.
    - Rework C++ simulator backends to use QubitVector class instead of
      ``std::vector``.
- Improve interface to simulator backends (#435)
    - Introduce ``local_statevector_simulator_py`` and
      ``local_statevector_simulator_cpp``.
    - Introduce aliased and deprecated backend names and mechanisms for
      resolving them.
    - Introduce optional ``compact`` flag to query backend names only by unique
      function.
    - Introduce result convenience functions ``get_statevector``,
      ``get_unitary``
    - Add ``snapshot`` command for caching a copy of the current simulator
      state.
- Introduce circuit drawing via ``circuit_drawer()`` and
  ``plot_circuit()`` (#295, #414)
- Introduce benchmark suite for performance testing
  (``test/performance``) (#277)
- Introduce more robust probability testing via assertDictAlmostEqual (#390)
- Allow combining circuits across both depth and width (#389)
- Enforce string token names (#395)


Fixed
-----

- Fix coherent error bug in ``local_qasm_simulator_cpp`` (#318)
- Fix the order and format of result bits obtained from device backends (#430)
- Fix support for noises in the idle gate of
  ``local_clifford_simulator_cpp`` (#440)
- Fix JobProcessor modifying input qobj (#392) (and removed JobProcessor
  during #403)
- Fix ability to apply all gates on register (#369)


Deprecated
----------

- Some methods of ``QuantumProgram`` are soon to be deprecated. Please use the
  top-level functions instead.
- The ``Register`` instantiation now expects ``size, name``. Using
  ``name, size`` is still supported but will be deprecated in the future.
- Simulators no longer return wavefunction by setting shots=1. Instead,
  use the ``local_statevector_simulator``, or explicitly ask for ``snapshot``.
- Return ``job`` instance after ``run()``, rather than ``result``.
- Rename simulators according to
  ``PROVIDERNAME_SIMPLEALIAS_simulator_LANGUAGEORPROJECT``
- Move simulator extensions to ``qiskit/extensions/simulator``
- Move Rzz and CSwap to standard extension library


`0.4.15`_ - 2018-05-07
======================


Fixed
-----

- Fixed an issue with legacy code that was affecting Developers Challenge.


`0.4.14`_ - 2018-04-18
======================


Fixed
-----

- Fixed an issue about handling Basis Gates parameters on backend
  configurations.


`0.4.13`_ - 2018-04-16
======================


Changed
-------

- OpenQuantumCompiler.dag2json() restored for backward compatibility.


Fixed
-----

- Fixes an issue regarding barrier gate misuse in some circumstances.


`0.4.12`_ - 2018-03-11
======================


Changed
-------

- Improved circuit visualization.
- Improvements in infrastructure code, mostly tests and build system.
- Better documentation regarding contributors.


Fixed
-----

- A bunch of minor bugs have been fixed.


`0.4.11`_ - 2018-03-13
======================


Added
-----

- More testing :)


Changed
-------

- Stabilizing code related to external dependencies.


Fixed
-----

- Fixed bug in circuit drawing where some gates in the standard library
  were not plotting correctly.


`0.4.10`_ - 2018-03-06
======================


Added
-----

- Chinese translation of README.


Changed
-------

- Changes related with infrastructure (linter, tests, automation)
  enhancement.


Fixed
-----

- Fix installation issue when simulator cannot be built.
- Fix bug with auto-generated CNOT coherent error matrix in C++ simulator.
- Fix a bug in the async code.


`0.4.9`_ - 2018-02-12
=====================


Changed
-------

- CMake integration.
- QASM improvements.
- Mapper optimizer improvements.


Fixed
-----

- Some minor C++ Simulator bug-fixes.


`0.4.8`_ - 2018-01-29
=====================


Fixed
-----

- Fix parsing U_error matrix in C++ Simulator python helper class.
- Fix display of code-blocks on ``.rst`` pages.


`0.4.7`_ - 2018-01-26
=====================


Changed
-------

- Changes some naming conventions for ``amp_error`` noise parameters to
  ``calibration_error``.


Fixed
-----

- Fixes several bugs with noise implementations in the simulator.
- Fixes many spelling mistakes in simulator README.


`0.4.6`_ - 2018-01-22
=====================


Changed
-------

- We have upgraded some of out external dependencies to:

   -  matplotlib >=2.1,<2.2
   -  networkx>=1.11,<2.1
   -  numpy>=1.13,<1.15
   -  ply==3.10
   -  scipy>=0.19,<1.1
   -  Sphinx>=1.6,<1.7
   -  sympy>=1.0


`0.4.4`_ - 2018-01-09
=====================


Changed
-------

- Update dependencies to more recent versions.


Fixed
-----

- Fix bug with process tomography reversing qubit preparation order.


`0.4.3`_ - 2018-01-08
=====================


Removed
-------

- Static compilation has been removed because it seems to be failing while
  installing Qiskit via pip on Mac.


`0.4.2`_ - 2018-01-08
=====================


Fixed
-----

- Minor bug fixing related to pip installation process.


`0.4.0`_ - 2018-01-08
=====================


Added
-----

- Job handling improvements.
    - Allow asynchronous job submission.
    - New JobProcessor class: utilizes concurrent.futures.
    - New QuantumJob class: job description.
- Modularize circuit "compilation".
    Takes quantum circuit and information about backend to transform circuit
    into one which can run on the backend.
- Standardize job description.
    All backends take QuantumJob objects which wraps ``qobj`` program
    description.
- Simplify addition of backends, where circuits are run/simulated.
    - ``qiskit.backends`` package added.
    - Real devices and simulators are considered "backends" which inherent from
      ``BaseBackend``.
- Reorganize and improve Sphinx documentation.
- Improve unittest framework.
- Add tools for generating random circuits.
- New utilities for fermionic Hamiltonians (``qiskit/tools/apps/fermion``).
- New utilities for classical optimization and chemistry
  (``qiskit/tools/apps/optimization``).
- Randomized benchmarking data handling.
- Quantum tomography (``qiskit/tools/qcvv``).
    Added functions for generating, running and fitting process tomography
    experiments.
- Quantum information functions (``qiskit/tools/qi``).
    - Partial trace over subsystems of multi-partite vector.
    - Partial trace over subsystems of multi-partite matrix.
    - Flatten an operator to a vector in a specified basis.
    - Generate random unitary matrix.
    - Generate random density matrix.
    - Generate normally distributed complex matrix.
    - Generate random density matrix from Hilbert-Schmidt metric.
    - Generate random density matrix from the Bures metric.
    - Compute Shannon entropy of probability vector.
    - Compute von Neumann entropy of quantum state.
    - Compute mutual information of a bipartite state.
    - Compute the entanglement of formation of quantum state.
- Visualization improvements (``qiskit/tools``).
    - Wigner function representation.
    - Latex figure of circuit.
- Use python logging facility for info, warnings, etc.
- Auto-deployment of sphinx docs to github pages.
- Check IBMQuantumExperience version at runtime.
- Add QuantumProgram method to reconfigure already generated qobj.
- Add Japanese introductory documentation (``doc/ja``).
- Add Korean translation of readme (``doc/ko``).
- Add appveyor for continuous integration on Windows.
- Enable new IBM Q parameters for hub/group/project.
- Add QuantumProgram methods for destroying registers and circuits.
- Use Sympy for evaluating expressions.
- Add support for ibmqx_hpc_qasm_simulator backend.
- Add backend interface to Project Q C++ simulator.
    Requires installation of Project Q.
- Introduce ``Initialize`` class.
    Generates circuit which initializes qubits in arbitrary state.
- Introduce ``local_qiskit_simulator`` a C++ simulator with realistic noise.
    Requires C++ build environment for ``make``-based build.
- Introduce ``local_clifford_simulator`` a C++ Clifford simulator.
    Requires C++ build environment for ``make``-based build.


Changed
-------

- The standard extension for creating U base gates has been modified to be
  consistent with the rest of the gate APIs (see #203).


Removed
-------

- The ``silent`` parameter has been removed from a number of ``QuantumProgram``
  methods. The same behaviour can be achieved now by using the
  ``enable_logs()`` and ``disable_logs()`` methods, which use the standard
  Python logging.


Fixed
-----

- Fix basis gates (#76).
- Enable QASM parser to work in multiuser environments.
- Correct operator precedence when parsing expressions (#190).
- Fix "math domain error" in mapping (#111, #151).

.. _UNRELEASED: https://github.com/Qiskit/qiskit-terra/compare/0.8.0...HEAD
.. _0.8.0: https://github.com/Qiskit/qiskit-terra/compare/0.7.2...0.8.0
.. _0.7.2: https://github.com/Qiskit/qiskit-terra/compare/0.7.1...0.7.2
.. _0.7.1: https://github.com/Qiskit/qiskit-terra/compare/0.7.0...0.7.1
.. _0.7.0: https://github.com/Qiskit/qiskit-terra/compare/0.6.0...0.7.0
.. _0.6.0: https://github.com/Qiskit/qiskit-terra/compare/0.5.7...0.6.0
.. _0.5.7: https://github.com/Qiskit/qiskit-terra/compare/0.5.6...0.5.7
.. _0.5.6: https://github.com/Qiskit/qiskit-terra/compare/0.5.5...0.5.6
.. _0.5.5: https://github.com/Qiskit/qiskit-terra/compare/0.5.4...0.5.5
.. _0.5.4: https://github.com/Qiskit/qiskit-terra/compare/0.5.3...0.5.4
.. _0.5.3: https://github.com/Qiskit/qiskit-terra/compare/0.5.2...0.5.3
.. _0.5.2: https://github.com/Qiskit/qiskit-terra/compare/0.5.1...0.5.2
.. _0.5.1: https://github.com/Qiskit/qiskit-terra/compare/0.5.0...0.5.1
.. _0.5.0: https://github.com/Qiskit/qiskit-terra/compare/0.4.15...0.5.0
.. _0.4.15: https://github.com/Qiskit/qiskit-terra/compare/0.4.14...0.4.15
.. _0.4.14: https://github.com/Qiskit/qiskit-terra/compare/0.4.13...0.4.14
.. _0.4.13: https://github.com/Qiskit/qiskit-terra/compare/0.4.12...0.4.13
.. _0.4.12: https://github.com/Qiskit/qiskit-terra/compare/0.4.11...0.4.12
.. _0.4.11: https://github.com/Qiskit/qiskit-terra/compare/0.4.10...0.4.11
.. _0.4.10: https://github.com/Qiskit/qiskit-terra/compare/0.4.9...0.4.10
.. _0.4.9: https://github.com/Qiskit/qiskit-terra/compare/0.4.8...0.4.9
.. _0.4.8: https://github.com/Qiskit/qiskit-terra/compare/0.4.7...0.4.8
.. _0.4.7: https://github.com/Qiskit/qiskit-terra/compare/0.4.6...0.4.7
.. _0.4.6: https://github.com/Qiskit/qiskit-terra/compare/0.4.5...0.4.6
.. _0.4.4: https://github.com/Qiskit/qiskit-terra/compare/0.4.3...0.4.4
.. _0.4.3: https://github.com/Qiskit/qiskit-terra/compare/0.4.2...0.4.3
.. _0.4.2: https://github.com/Qiskit/qiskit-terra/compare/0.4.1...0.4.2
.. _0.4.0: https://github.com/Qiskit/qiskit-terra/compare/0.3.16...0.4.0

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
