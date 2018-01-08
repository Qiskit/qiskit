Release history
###############

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
