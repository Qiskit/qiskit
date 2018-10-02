Release history
###############

Release notes
*************

Qiskit SDK 0.6.0
================

This release includes a redesign of internal components centered around a new,
formal communication format (``Qobj``), along with long awaited features to
improve the user experience as a whole. The highlights, compared to the 0.5
release, are:

* Improvements for inter-operability (based on the ``Qobj`` specification) and
  extensibility (facilities for extending Qiskit with new backends in a seamless
  way).
* New options for handling credentials and authentication for the IBM Q
  backends, aimed at simplifying the process and supporting automatic loading
  of user credentials.
* A revamp of the visualization utilities: stylish interactive visualizations
  are now available for Jupyter users, along with refinements for the circuit
  drawer (including a matplotlib-based version).
* Performance improvements centered around circuit transpilation: the basis for
  a more flexible and modular architecture have been set, including
  paralellization of the circuit compilation and numerous optimizations.

Upgrading to 0.6.0
------------------

Please note that some backwards-incompatible changes have been introduced
during this release - the following notes contain information on how to adapt to
the new changes.

Removal of ``QuantumProgram``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As hinted during the 0.5 release, the deprecation of the  ``QuantumProgram``
class has now been completed and is no longer available, in favor of working
with the individual components (:class:`~qiskit.backends.basebackend.BaseJob`,
:class:`~qiskit._quantumcircuit.QuantumCircuit`,
:class:`~qiskit._classicalregister.ClassicalRegister`,
:class:`~qiskit._quantumregister.QuantumRegister`,
:mod:`~qiskit`) directly.

Please check the :ref:`0.5 release notes <quantum-program-0-5>` and the
:doc:`quickstart` examples for details about the transition ::


  from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
  from qiskit import available_backends, execute

  q = QuantumRegister(2)
  c = ClassicalRegister(2)
  qc = QuantumCircuit(q, c)

  qc.h(q[0])
  qc.cx(q[0], q[1])
  qc.measure(q, c)

  print("Local backends: ", available_backends({'local': True}))

  job_sim = execute(qc, "local_qasm_simulator")
  sim_result = job_sim.result()

  print("simulation: ", sim_result)
  print(sim_result.get_counts(qc))

IBM Q Authentication and ``Qconfig.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The managing of credentials for authenticating when using the QX backends has
been expanded, and there are new options that can be used for convenience:

1. save your credentials in disk once, and automatically load them in future
   sessions. This provides a one-off mechanism::

     from qiskit import IBMQ
     IBQM.save_account('MY_API_TOKEN', 'MY_API_URL')

   afterwards, your credentials can be automatically loaded from disk by invoking
   :meth:`~qiskit.backends.ibmq.IBMQ.load_accounts`::

     from qiskit import IBMQ
     IBMQ.load_accounts()

   or you can load only specific accounts if you only want to use those in a session::
   
     IBMQ.load_accounts(project='MY_PROJECT')

2. use environment variables. If ``QE_TOKEN`` and ``QE_URL`` is set, the
   ``IBMQ.load_accounts()`` call will automatically load the credentials from
   them.

Additionally, the previous method of having a ``Qconfig.py`` file in the program
folder and passing the credentials explicitly is still supported. Please check
the :ref:`qconfig-setup` section for more details about combining and using
the different authentication options.

Working with backends
^^^^^^^^^^^^^^^^^^^^^

A new mechanism has been introduced in 0.6 as the recommended way for obtaining
a backend, allowing for more powerful and unified filtering and integrated with
the new credentials system. The previous top-level methods
:meth:`~qiskit.wrapper._wrapper.register`,
:meth:`~qiskit.wrapper._wrapper.available_backends` and
:meth:`~qiskit.wrapper._wrapper.get_backend` are still supported, but will
deprecated in upcoming versions in favor of using the `qiskit.IBMQ` and
`qiskit.Aer` objects directly, which allow for more complex filtering.

For example, to list and use a local backend::

  from qiskit import Aer

  all_local_backends = Aer.backends(local=True)  # returns a list of instances
  qasm_simulator = Aer.backends('local_qasm_simulator')

And for listing and using remote backends::

  from qiskit import IBMQ

  IBMQ.enable_account('MY_API_TOKEN')
  5_qubit_devices = IBMQ.backends(simulator=True, n_qubits=5)
  ibmqx4 = IBMQ.get_backend('ibmqx4')

Backend and Job API changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Jobs submitted to IBM Q backends have improved capabilities. It is possible to
  cancel them and replenish credits (``job.cancel()``), and to retrieve previous jobs
  executed on a specific backend either by job id (``backend.retrieve_job(job_id)``) or
  in batch of latest jobs (``backend.jobs(limit)``)

* Properties for checking each individual job status (``queued``, ``running``,
  ``validating``, ``done`` and ``cancelled``) no longer exist. If you
  want to check the job status, use the identity comparison against
  ``job.status``::

    from qiskit.backends import JobStatus

    job = execute(circuit, backend)
    if job.status() is JobStatus.RUNNING:
        handle_job(job)

Please consult the new documentation of the
:class:`~qiskit.backends.ibmq.ibmqjob.IBMQJob` class to get further insight in
how to use the simplified API.

* A number of members of :class:`~qiskit.backends.basebackend.BaseBackend` and 
:class:`~qiskit.backends.basejob.BaseJob` are no longer properties, 
but methods, and as a result they need to be invoked as functions.

=====================  ========================
Qiskit 0.5             Qiskit 0.6
=====================  ========================
backend.name           backend.name()
backend.status         backend.status()
backend.configuration  backend.configuration()
backend.calibration    backend.properties()
backend.parameters     backend.jobs()
                       backend.retrieve_job(job_id)
=====================  ==========
job.status             job.status()
job.cancelled          job.queue_position()
job.running            job.cancel()
job.queued
job.done
=====================  ========================

Better Jupyter tools
^^^^^^^^^^^^^^^^^^^^
The new release contains improvements to the user experience while using Jupyter notebooks.

First, new interactive visualizations of counts histograms and quantum states are provided:
:meth:`~qiskit.tools.visualization.plot_histogram` and :meth:`~qiskit.tools.visualization.plot_state`.
These methods will default to the new interactive kind when the environment is Jupyter and internet
connection exists.

Secondly, the new release provides Jupyter cell magics for keeping track of the progress of your code.
Use ``%%qiskit_job_status`` to keep track of the status of submitted jobs to IBMQ backends.
Use ``%%qiskit_progress_bar`` to keep track of the progress of compilation/execution.


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

.. _quantum-program-0-5:

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
