###############
Release history
###############

*************
Release notes
*************

Qiskit Terra 0.7.0
==================

This release includes several new features and many bug fixes. With this
release the interfaces for circuit diagram, histogram, bloch vectors,
and state visualizations are declared stable. Additionally, this release includes a
defined and standardized bit order/endianness throughout all aspects of Qiskit.
These are all declared as stable interfaces in this release which won't have
breaking changes made moving forward, unless there is appropriate and lengthy
deprecation periods warning of any coming changes.

There is also the introduction of the following new features:

* A new ASCII art circuit drawing output mode
* A new ciruit drawing interface off of QuantumCircuit objects. Now you can
  call ``circuit.draw()`` or ``print(circuit)`` and render a drawing of
  the circuit.
* A visualizer for drawing the DAG representation of a circuit
* A new quantum state plot type for hinton diagrams in the local matplotlib
  based state plots.
* 2 new constructor methods off the ``QuantumCircuit`` class
  ``from_qasm_str()`` and ``from_qasm_file()`` which let you easily create a
  circuit object from OpenQASM.
* A new function ``plot_bloch_multivector()`` to plot Bloch vectors from a
  tensored state vector or density matrix
* Per-shot measurement results are available in simulators and select devices.
  These can be accessed by setting the ``memory`` kwarg to ``True`` when
  calling ``compile()`` or ``execute()`` and then accessed using the
  ``get_memory()`` method on the ``Result`` object.
* A ``qiskit.quantum_info`` module with revamped Pauli objects and methods for
  working with quantum states.
* New transpile passes for circuit analysis and transformation:
  CommutationAnalysis, CommutationTransformation, CXCancellation, Decompose,
  Unroll, Optimize1QGates, CheckMap, CXDirection, BarrierBeforeFinalMeasurements
* New alternative swap mapper passes in the transpiler:
  BasicSwap, LookaheadSwap, StochasticSwap
* More advanced transpiler infrastructure with support for analysis passes,
  transformation passes, a global property_set for the pass manager, and
  repeat-until control of passes.


Upgrading to 0.7.0
------------------

Please note that some backwards-incompatible changes have been made during this
release. The following notes contain information on how to adapt to these
changes.

Changes to Result objects
^^^^^^^^^^^^^^^^^^^^^^^^^

As part of the rewrite of the Results object to be more consistent and a
stable interface moving forward a few changes have been made to how you access
the data stored in the result object. First the ``get_data()`` method has been
renamed to just ``data()``. Accompanying that change is a change in the data
format returned by the function. It is now returning the raw data from the
backends instead of doing any post-processing. For example, in previous
versions you could call::

   result = execute(circuit, backend).result()
   unitary = result.get_data()['unitary']
   print(unitary)

and that would return the unitary matrix like::

   [[1+0j, 0+0.5j], [0-0.5j][-1+0j]]

But now if you call (with the renamed method)::

   result.data()['unitary']

it will return something like::

   [[[1, 0], [0, -0.5]], [[0, -0.5], [-1, 0]]]

To get the post processed results in the same format as before the 0.7 release
you must use the ``get_counts()``, ``get_statevector()``, and ``get_unitary()``
methods on the result object instead of ``get_data()['counts']``,
``get_data()['statevector']``, and ``get_data()['unitary']`` respectively.

Additionally, support for ``len()`` and indexing on a ``Result`` object has
been removed. Instead you should deal with the output from the post processed
methods on the Result objects.

Also, the ``get_snapshot()`` and ``get_snapshots()`` methods from the
``Result`` class have been removed. Instead you can access the snapshots
using ``Result.data()['snapshots']``.


Changes to visualization
^^^^^^^^^^^^^^^^^^^^^^^^

The biggest change made to visualization in the 0.7 release is the removal of
Matplotlib and other visualization dependencies from the project requirements.
This was done to simplify the requirements and configuration required for
installing Qiskit. If you plan to use any visualizations (including all the
jupyter magics) except for the ``text``, ``latex``, and ``latex_source``
output for the circuit drawer you'll you must manually ensure that
the visualization dependencies are installed. You can leverage the optional
requirements to the Qiskit-Terra package to do this::

   pip install qiskit-terra[visualization]

Aside from this there have been changes made to several of the interfaces
as part of the stabilization which may have an impact on existing code.
The first is the the ``basis`` kwarg in the ``circuit_drawer()`` function
is no longer accepted. If you were relying on the ``circuit_drawer()`` to
adjust the basis gates used in drawing a circuit diagram you will have to
do this priort to calling ``circuit_drawer()``. For example::

   from qiskit.tools import visualization
   visualization.circuit_drawer(circuit, basis_gates='x,U,CX')

will have to be adjust to be::

   from qiskit import BasicAer
   from qiskit import transpiler
   from qiskit.tools import visualization
   backend = BasicAer.backend('qasm_simulator')
   draw_circ = transpiler.transpile(circuit, backend, basis_gates='x,U,CX')
   visualization.circuit_drawer(draw_circ)

Moving forward the ``circuit_drawer()`` function will be the sole interface
for circuit drawing in the visualization module. Prior to the 0.7 release there
were several other functions which either used different output backends or
changed the output for drawing circuits. However, all those other functions
have been deprecated and that functionality has been integrated as options
on ``circuit_drawer()``.

For the other visualization functions, ``plot_histogram()`` and
``plot_state()`` there are also a few changes to check when upgrading. First
is the output from these functions has changed, in prior releases these would
interactively show the output visualization. However that has changed to
instead return a ``matplotlib.Figure`` object. This provides much more
flexibility and options to interact with the visualization prior to saving or
showing it. This will require adjustment to how these functions are consumed.
For example, prior to this release when calling::

   plot_histogram(counts)
   plot_state(rho)

would open up new windows (depending on matplotlib backend) to display the
visualization. However starting in the 0.7 you'll have to call ``show()`` on
the output to mirror this behavior. For example::

   plot_histogram(counts).show()
   plot_state(rho).show()

or::

   hist_fig = plot_histogram(counts)
   state_fig = plot_state(rho)
   hist_fig.show()
   state_fig.show()

Note that this is only for when running outside of Jupyter. No adjustment is
required inside a Jupyter environment because Jupyter notebooks natively
understand how to render ``matplotlib.Figure`` objects.

However, returning the Figure object provides additional flexibility for
dealing with the output. For example instead of just showing the figure you
can now directly save it to a file by leveraging the ``savefig()`` method.
For example::

   hist_fig = plot_histogram(counts)
   state_fig = plot_state(rho)
   hist_fig.savefig('histogram.png')
   state_fig.savefig('state_plot.png')

The other key aspect which has changed with these functions is when running
under jupyter. In the 0.6 release ``plot_state()`` and ``plot_histogram()``
when running under jupyter the default behavior was to use the interactive
Javascript plots if the externally hosted Javascript library for rendering
the visualization was reachable over the network. If not it would just use
the matplotlib version. However in the 0.7 release this no longer the case,
and separate functions for the interactive plots, ``iplot_state()`` and
``iplot_histogram()`` are to be used instead. ``plot_state()`` and
``plot_histogram()`` always use the matplotlib versions.

Additionally, starting in this release the ``plot_state()`` function is
deprecated in favor of calling individual methods for each method of plotting
a quantum state. While the ``plot_state()`` function will continue to work
until the 0.9 release, it will emit a warning each time it is used. The

==================================  ========================
Qiskit Terra 0.6                    Qiskit Terra 0.7+
==================================  ========================
plot_state(rho)                     plot_state_city(rho)
plot_state(rho, method='city')      plot_state_city(rho)
plot_state(rho, method='paulivec')  plot_state_paulivec(rho)
plot_state(rho, method='qsphere')   plot_state_qsphere(rho)
plot_state(rho, method='bloch')     plot_bloch_multivector(rho)
plot_state(rho, method='hinton')    plot_state_hinton(rho)
==================================  ========================

The same is true for the interactive JS equivalent, ``iplot_state()``. The
function names are all the same, just with a prepended `i` for each function.
For example, ``iplot_state(rho, method='paulivec')`` is
``iplot_state_paulivec(rho)``.

Changes to Backends
^^^^^^^^^^^^^^^^^^^

With the improvements made in the 0.7 release there are a few things related
to backends to keep in mind when upgrading. The biggest change is the
restructuring of the provider instances in the root  ``qiskit``` namespace.
The ``Aer`` provider is not installed by default and requires the installation
of the ``qiskit-aer`` package. This package contains the new high performance
fully featured simulator. If you installed via ``pip install qiskit`` you'll
already have this installed. The python simulators are now available under
``qiskit.BasicAer`` and the old C++ simulators are available with
``qiskit.LegacySimulators``. This also means that the implicit fallback to
python based simulators when the C++ simulators are not found doesn't exist
anymore. If you ask for a local C++ based simulator backend, and it can't be
found an exception will be raised instead of just using the python simulator
instead.

Additionally the previously deprecation top level functions ``register()`` and
``available_backends()`` have been removed. Also, the deprecated
``backend.parameters()`` and ``backend.calibration()`` methods have been
removed in favor of ``backend.properties()``. You can refer to the 0.6 release
notes section :ref:`backends` for more details on these changes.

The ``backend.jobs()`` and ``backend.retrieve_jobs()`` calls no longer return
results from those jobs. Instead you must call the ``result()`` method on the
returned jobs objects.

Changes to the compiler, transpiler, and unrollers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As part of an effort to stabilize the compiler interfaces there have been
several changes to be aware of when leveraging the compiler functions.
First it is important to note that the ``qiskit.transpiler.transpile()``
function now takes a QuantumCircuit object (or a list of them) and returns
a QuantumCircuit object (or a list of them). The DAG processing is done
internally now.

You can also easily switch between circuits, DAGs, and Qobj now using the
functions in ``qiskit.converters``.


Deprecations
------------

As part of the part of the 0.7 release the following things have been
deprecated and will either be removed or changed in a backwards incompatible
manner in a future release. While not strictly necessary these are things to
adjust for before the next release to avoid a breaking change.

* ``plot_circuit()``, ``latex_circuit_drawer()``, ``generate_latex_source()``,
   and ``matplotlib_circuit_drawer()`` from qiskit.tools.visualization are
   deprecated. Instead the ``circuit_drawer()`` function from the same module
   should be used, there are kwarg options to mirror the functionality of all
   the deprecated functions.
* The current default output of ``circuit_drawer()`` (using latex and falling
   back on python) is deprecated and will be changed to just use the ``text``
   output by default in future releases.
* The `qiskit.wrapper.load_qasm_string()` and `qiskit.wrapper.load_qasm_file()`
  functions are deprecated and the `QuantumCircuit.from_qasm_str()` and
  `QuantumCircuit.from_qasm_file()` constructor methods should be used instead.
* The ``plot_barriers`` and ``reverse_bits`` keys in the ``style`` kwarg dict
  are deprecated, instead the `qiskit.tools.visualization.circuit_drawer()`
  kwargs ``plot_barriers`` and ``reverse_bits`` should be used instead.
* The functions `plot_state()` and `iplot_state()` have been depreciated.
  Instead the functions `plot_state_*()` and `iplot_state_*()` should be
  called for the visualization method required.
* The ``skip_transpiler`` arg has been deprecated from ``compile()`` and
  ``execute()``. Instead you can use the PassManager directly, just set
  the ``pass_manager`` to a blank PassManager object with ``PassManager()``
* The ``transpile_dag()`` function ``format`` kwarg for emitting different
  output formats is deprecated, instead you should convert the default output
  ``DAGCircuit`` object to the desired format.
* The unrollers have been deprecated, moving forward only DAG to DAG unrolling
  will be supported.


Qiskit Terra 0.6.0
==================

This release includes a redesign of internal components centered around a new,
formal communication format (`qobj`), along with long awaited features to
improve the user experience as a whole. The highlights, compared to the 0.5
release, are:

* Improvements for inter-operability (based on the `qobj` specification) and
  extensibility (facilities for extending Qiskit with new backends in a
  seamless way).
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
during this release - the following notes contain information on how to adapt
to the new changes.

Removal of ``QuantumProgram``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As hinted during the 0.5 release, the deprecation of the  ``QuantumProgram``
class has now been completed and is no longer available, in favor of working
with the individual components (:class:`~qiskit.backends.basejob.BaseJob`,
:class:`~qiskit._quantumcircuit.QuantumCircuit`,
:class:`~qiskit._classicalregister.ClassicalRegister`,
:class:`~qiskit._quantumregister.QuantumRegister`,
:mod:`~qiskit`) directly.

Please check the :ref:`0.5 release notes <quantum-program-0-5>` and the
examples for details about the transition::


  from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
  from qiskit import Aer, execute

  q = QuantumRegister(2)
  c = ClassicalRegister(2)
  qc = QuantumCircuit(q, c)

  qc.h(q[0])
  qc.cx(q[0], q[1])
  qc.measure(q, c)

  backend = get_backend('qasm_simulator')

  job_sim = execute(qc, backend)
  sim_result = job_sim.result()

  print("simulation: ", sim_result)
  print(sim_result.get_counts(qc))


IBM Q Authentication and ``Qconfig.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The managing of credentials for authenticating when using the IBMQ backends has
been expanded, and there are new options that can be used for convenience:

1. save your credentials in disk once, and automatically load them in future
   sessions. This provides a one-off mechanism::

     from qiskit import IBMQ
     IBQM.save_account('MY_API_TOKEN', 'MY_API_URL')

   afterwards, your credentials can be automatically loaded from disk by invoking
   :meth:`~qiskit.backends.ibmq.ibmqprovider.IBMQ.load_accounts`::

     from qiskit import IBMQ
     IBMQ.load_accounts()

   or you can load only specific accounts if you only want to use those in a session::

     IBMQ.load_accounts(project='MY_PROJECT')

2. use environment variables. If ``QE_TOKEN`` and ``QE_URL`` is set, the
   ``IBMQ.load_accounts()`` call will automatically load the credentials from
   them.

Additionally, the previous method of having a ``Qconfig.py`` file in the
program folder and passing the credentials explicitly is still supported.


.. _backends:

Working with backends
^^^^^^^^^^^^^^^^^^^^^

A new mechanism has been introduced in Terra 0.6 as the recommended way for
obtaining a backend, allowing for more powerful and unified filtering and
integrated with the new credentials system. The previous top-level methods
:meth:`~qiskit.wrapper._wrapper.register`,
:meth:`~qiskit.wrapper._wrapper.available_backends` and
:meth:`~qiskit.wrapper._wrapper.get_backend` are still supported, but will
deprecated in upcoming versions in favor of using the `qiskit.IBMQ` and
`qiskit.Aer` objects directly, which allow for more complex filtering.

For example, to list and use a local backend::

  from qiskit import Aer

  all_local_backends = Aer.backends(local=True)  # returns a list of instances
  qasm_simulator = Aer.backends('qasm_simulator')

And for listing and using remote backends::

  from qiskit import IBMQ

  IBMQ.enable_account('MY_API_TOKEN')
  5_qubit_devices = IBMQ.backends(simulator=True, n_qubits=5)
  ibmqx4 = IBMQ.get_backend('ibmqx4')

Please note as well that the names of the local simulators have been
simplified. The previous names can still be used, but it is encouraged to use
the new, shorter names:

=============================  ========================
Qiskit Terra 0.5               Qiskit Terra 0.6
=============================  ========================
'local_qasm_simulator'         'qasm_simulator'
'local_statevector_simulator'  'statevector_simulator'
'local_unitary_simulator_py'   'unitary_simulator'
=============================  ========================


Backend and Job API changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Jobs submitted to IBM Q backends have improved capabilities. It is possible
  to cancel them and replenish credits (``job.cancel()``), and to retrieve
  previous jobs executed on a specific backend either by job id
  (``backend.retrieve_job(job_id)``) or in batch of latest jobs
  (``backend.jobs(limit)``)

* Properties for checking each individual job status (``queued``, ``running``,
  ``validating``, ``done`` and ``cancelled``) no longer exist. If you
  want to check the job status, use the identity comparison against
  ``job.status``::

    from qiskit.backends import JobStatus

    job = execute(circuit, backend)
    if job.status() is JobStatus.RUNNING:
        handle_job(job)

Please consult the new documentation of the
:class:`~qiskit.backends.ibmq.ibmqjob.IBMQJob` class to get further insight
in how to use the simplified API.

* A number of members of :class:`~qiskit.backends.basebackend.BaseBackend`
  and :class:`~qiskit.backends.basejob.BaseJob` are no longer properties,
  but methods, and as a result they need to be invoked as functions.

=====================  ========================
Qiskit Terra 0.5       Qiskit Terra 0.6
=====================  ========================
backend.name           backend.name()
backend.status         backend.status()
backend.configuration  backend.configuration()
backend.calibration    backend.properties()
backend.parameters     backend.jobs()
                       backend.retrieve_job(job_id)
job.status             job.status()
job.cancelled          job.queue_position()
job.running            job.cancel()
job.queued
job.done
=====================  ========================


Better Jupyter tools
^^^^^^^^^^^^^^^^^^^^

The new release contains improvements to the user experience while using
Jupyter notebooks.

First, new interactive visualizations of counts histograms and quantum states
are provided:
:meth:`~qiskit.tools.visualization.plot_histogram` and
:meth:`~qiskit.tools.visualization.plot_state`.
These methods will default to the new interactive kind when the environment
is Jupyter and internet connection exists.

Secondly, the new release provides Jupyter cell magics for keeping track of
the progress of your code. Use ``%%qiskit_job_status`` to keep track of the
status of submitted jobs to IBMQ backends. Use ``%%qiskit_progress_bar`` to
keep track of the progress of compilation/execution.


Qiskit Terra 0.5.0
==================

This release brings a number of improvements to Qiskit, both for the user
experience and under the hood. Please refer to the full changelog for a
detailed description of the changes - the highlights are:

* new ``statevector`` :mod:`simulators <qiskit.backends.local>` and feature and
  performance improvements to the existing ones (in particular to the C++
  simulator), along with a reorganization of how to work with backends focused
  on extensibility and flexibility (using aliases and backend providers).
* reorganization of the asynchronous features, providing a friendlier interface
  for running jobs asynchronously via :class:`Job` instances.
* numerous improvements and fixes throughout the Terra as a whole, both for
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
  top-level functions directly instead of
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

  the backend will be the C++ statevector simulator if available, falling
  back to the Python statevector simulator if not present.

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

