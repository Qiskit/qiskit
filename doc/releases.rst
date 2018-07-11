Release history
###############

Release notes
*************

Qiskit SDK 0.5.0
================

This release brings a number of improvements to Qiskit, both for the user
experience and under the hood. Please refer to the full changelog for a
detailed description of the changes - the highlights are:

* new ``statevector`` :mod:`simulators <qiskit.backends.local>` and feature and
  performance improvements to the existing ones (in particular to the C++
  simulator), along with a reorganization of how to work with backends focused
  on extensibility and flexibility (using aliases and backend providers).
* reorganization of the asynchronous features, providing a friendlier interface
  for running jobs asynchronously via :class:`Job` instances.
* numerous improvements and fixes throughout the SDK as a whole, both for
  convenience of the users (such as allowing anonymous registers) and for
  enhanced functionality (such as improved plotting of circuits).

Upgrading to 0.5.0
------------------

Please note that several backwards-incompatible changes have been introduced
during this release as a result of the ongoing development. While some of these
features will continue to be supported during a period of time before being
fully deprecated, it is recommended to update your programs in order to prepare
for the new versions and take advantage of the new functionality.

``QuantumProgram`` changes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Several methods of the :class:`~qiskit.QuantumProgram` class are on their way
to being deprecated:

* methods for interacting **with the backends and the API**:

  The recommended way for opening a connection to the IBMQ API and for using
  the backends is through the
  :ref:`top-level functions <qiskit_top_level_functions>` directly instead of
  the ``QuantumProgram`` methods. In particular, the
  :func:`qiskit.register` method provides the equivalent of the previous
  :func:`qiskit.QuantumProgram.set_api` call. In a similar vein, there is a new
  :func:`qiskit.available_backends`, :func:`qiskit.get_backend` and related
  functions for querying the available backends directly. For example, the
  following snippet for version 0.4::

    from qiskit import QuantumProgram

    quantum_program = QuantumProgram()
    quantum_program.set_api(token, url)
    backends = quantum_program.available_backends()
    print(quantum_program.get_backend_status('ibmqx4')

  would be equivalent to the following snippet for version 0.5::

    from qiskit import register, available_backends, get_backend

    register(token, url)
    backends = available_backends()
    backend = get_backend('ibmqx4')
    print(backend.status)

* methods for **compiling and executing programs**:

  The :ref:`top-level functions <qiskit_top_level_functions>` now also provide
  equivalents for the :func:`qiskit.QuantumProgram.compile` and
  :func:`qiskit.QuantumProgram.execute` methods. For example, the following
  snippet from version 0.4::

    quantum_program.execute(circuit, args, ...)

  would be equivalent to the following snippet for version 0.5::

    from qiskit import execute

    execute(circuit, args, ...)

In general, from version 0.5 onwards we encourage to try to make use of the
individual objects and classes directly instead of relying on
``QuantumProgram``. For example, a :class:`~qiskit.QuantumCircuit` can be
instantiated and constructed by appending :class:`~qiskit.QuantumRegister`,
:class:`~qiskit.ClassicalRegister`, and gates directly. Please check the
update example in the Quickstart section, or the
``using_qiskit_core_level_0.py`` and ``using_qiskit_core_level_1.py``
examples on the main repository.

Backend name changes
^^^^^^^^^^^^^^^^^^^^

In order to provide a more extensible framework for backends, there have been
some design changes accordingly:

* **local simulator names**

  The names of the local simulators have been homogenized in order to follow
  the same pattern: ``PROVIDERNAME_TYPE_simulator_LANGUAGEORPROJECT`` -
  for example, the C++ simulator previously named ``local_qiskit_simulator``
  is now ``local_qasm_simulator_cpp``. An overview of the current
  simulators:

  * ``QASM`` simulator is supposed to be like an experiment. You apply a
    circuit on some qubits, and observe measurement results - and you repeat
    for many shots to get a histogram of counts via ``result.get_counts()``.
  * ``Statevector`` simulator is to get the full statevector (:math:`2^n`
    amplitudes) after evolving the zero state through the circuit, and can be
    obtained via ``result.get_statevector()``.
  * ``Unitary`` simulator is to get the unitary matrix equivalent of the
    circuit, returned via ``result.get_unitary()``.
  * In addition, you can get intermediate states from a simulator by applying
    a ``snapshot(slot)`` instruction at various spots in the circuit. This will
    save the current state of the simulator in a given slot, which can later
    be retrieved via ``result.get_snapshot(slot)``.

* **backend aliases**:

  The SDK now provides an "alias" system that allows for automatically using
  the most performant simulator of a specific type, if it is available in your
  system. For example, with the following snippet::

    from qiskit import get_backend

    backend = get_backend('local_statevector_simulator')

  the backend will be the C++ statevector simulator if available, falling back to
  the Python statevector simulator if not present.

More flexible names and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several functions of the SDK have been made more flexible and user-friendly:

* **automatic circuit and register names**

  :class:`qiskit.ClassicalRegister`, :class:`qiskit.QuantumRegister` and
  :class:`qiskit.QuantumCircuit` can now be instantiated without explicitly
  giving them a name - a new autonaming feature will automatically assign them
  an identifier::

    q = QuantumRegister(2)

  Please note as well that the order of the parameters have been swapped
  ``QuantumRegister(size, name)``.

* **methods accepting names or instances**

  In combination with the autonaming changes, several methods such as
  :func:`qiskit.Result.get_data` now accept both names and instances for
  convenience. For example, when retrieving the results for a job that has a
  single circuit such as::

    qc = QuantumCircuit(..., name='my_circuit')
    job = execute(qc, ...)
    result = job.result()

  The following calls are equivalent::

    data = result.get_data('my_circuit')
    data = result.get_data(qc)
    data = result.get_data()


Changelog
*********

.. include:: CHANGELOG.rst
