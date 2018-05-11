QISKit SDK 0.5.0
================

2018/05/11

Improvements
------------

- Introduce providers and rework backends (#376).
    - Split backends into ``local`` and ``ibmq``.
    - Each provider derives from the following classes for its specific requirements.
        ``BaseProvider`` class.
        ``BaseBackend`` class.
        ``BaseJob`` class.
    - Allow querying result by both circuit name and QuantumCircuit instance.
- Introduce the QISKit ``wrapper`` (#376).
    - Introduce convenience wrapper functions around commonly used QISKit components
      (e.g. ``compile`` and ``execute`` functions).
    - Introduce the DefaultQISKitProvider, which acts as a context manager for the current session
      (e.g. providing easy access to all ``available_backends``).
    - Avoid relying on QuantumProgram (eventual deprecation).
    - The functions are also available as top-level functions (for example,
      ``qiskit.get_backend()``).
- Introduce ``BaseJob`` class and asynchronous jobs (#403).
    - Return ``BaseJob`` after ``run()``.
    - Mechanisms for querying ``status`` and ``results``, or to ``cancel`` a job
- Introduce a ``skip_transpiler`` flag for ``compile()`` (#411).
- Introduce schemas for validating interfaces between qiskit and backends (#434)
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
    - Rework C++ simulator backends to use QubitVector class instead of std::vector.
- Improve interface to simulator backends (#435)
    - Introduce ``local_statevector_simulator_py`` and ``local_statevector_simulator_cpp``.
    - Introduce aliased and deprecated backend names and mechanisms for resolving them.
    - Introduce optional ``compact`` flag to query backend names only by unique function.
    - Introduce result convenience functions ``get_statevector``, ``get_unitary``
    - Add ``snapshot`` command for caching a copy of the current simulator state.
- Introduce circuit drawing via ``circuit_drawer()`` and ``plot_circuit()`` (#295, #414)
- Introduce benchmark suite for performance testing (``test/performance``) (#277)
- Introduce more robust probability testing via assertDictAlmostEqual (#390)
- Allow combining circuits across both depth and width (#389)
- Enforce string token names (#395)

Bug Fixes
---------

- Fix coherent error bug in ``local_qasm_simulator_cpp`` (#318)
- Fix the order and format of result bits obtained from device backends (#430)
- Fix support for noises in the idle gate of ``local_clifford_simulator_cpp`` (#440)
- Fix JobProcessor modifying input qobj (#392) (and removed JobProcessor during #403)
- Fix ability to apply all gates on register (#369)

Backward-incompatible changes
-----------------------------

- Some methods of ``QuantumProgram`` are soon to be deprecated. Please use the
  top-level functions instead.
- The ``Register`` instantiation now expects ``size, name``. Using
  ``name, size`` is still supported but will be deprecated in the future.
- Simulators no longer return wavefunction by setting shots=1. Instead,
  use the ``local_statevector_simulator``, or explicitly ask for ``snapshot``.
- Return ``job`` instance after ``run()``, rather than ``result``.
- Rename simulators according to ``PROVIDERNAME_SIMPLEALIAS_simulator_LANGUAGEORPROJECT``
- Move simulator extensions to ``qiskit/extensions/simulator``
- Move Rzz and CSwap to standard extension library



QISKit SDK 0.4.0
================

2018/01/08

Improvements
------------

- Job handling improvements.
    - Allow asynchronous job submission.
    - New JobProcessor class: utilizes concurrent.futures.
    - New QuantumJob class: job description.
- Modularize circuit "compilation".
    Takes quantum circuit and information about backend to transform
    circuit into one which can run on the backend.
- Standardize job description.
    All backends take QuantumJob objects which wraps ``qobj`` program description.
- Simplify addition of backends, where circuits are run/simulated.
    - ``qiskit.backends`` package added.
    - Real devices and simulators are considered "backends" which inherent from ``BaseBackend``.
- Reorganize and improve Sphinx documentation.
- Improve unittest framework.
- Add tools for generating random circuits.
- New utilities for fermionic Hamiltonians (``qiskit/tools/apps/fermion``).
- New utilities for classical optimization and chemistry (``qiskit/tools/apps/optimization``).
- Randomized benchmarking data handling.
- Quantum tomography (``qiskit/tools/qcvv``).
    Added functions for generating, running and fitting process tomography experiments.
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
- Introduce ``InitializeGate`` class.
    Generates circuit which initializes qubits in arbitrary state.
- Introduce ``local_qiskit_simulator`` a C++ simulator with realistic noise.
    Requires C++ build environment for ``make``-based build.
- Introduce ``local_clifford_simulator`` a C++ Clifford simulator.
    Requires C++ build environment for ``make``-based build.

Bug Fixes
---------

- Fix basis gates (#76).
- Enable QASM parser to work in multiuser environments.
- Correct operator precedence when parsing expressions (#190).
- Fix "math domain error" in mapping (#111, #151).

Backward-incompatible changes
-----------------------------

- The standard extension for creating U base gates has been modified to be
  consistent with the rest of the gate APIs (see #203).
- The ``silent`` parameter has been removed from a number of ``QuantumProgram``
  methods. The same behaviour can be achieved now by using the
  ``enable_logs()`` and ``disable_logs()`` methods, which use the standard
  Python logging.
