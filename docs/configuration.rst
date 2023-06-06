Local Configuration
===================

Once you have Qiskit installed and running there are some optional configuration
steps you can take to change the default behavior of Qiskit for your specific
use case.

User Config File
----------------

The main location for local configuration of Qiskit is the user config file.
This is an `ini <https://en.wikipedia.org/wiki/INI_file>`__  format file that
can be used to change defaults in Qiskit.

For example:

.. code-block:: ini

    [default]
    circuit_drawer = mpl
    circuit_mpl_style = default
    circuit_mpl_style_path = ~:~/.qiskit
    state_drawer = hinton
    transpile_optimization_level = 3
    parallel = False
    num_processes = 15

By default this file lives in ``~/.qiskit/settings.conf`` but the path used
can be overriden with the ``QISKIT_SETTINGS`` environment variable. If
``QISKIT_SETTINGS`` is set its value will used as the path to the user config
file.

Available options:

 * ``circuit_drawer``: This is used to change the default backend for
   the circuit drawer :meth:`qiskit.circuit.QuantumCircuit.draw` and
   :func:`qiskit.visualization.circuit_drawer`. It can be set to ``latex``,
   ``mpl``, ``text``, or ``latex_source`` and when the ``ouptut`` kwarg is
   not explicitly set that drawer backend will be used.
 * ``circuit_mpl_style``: This is the default style sheet used for the
   ``mpl`` output backend for the circuit drawer
   :meth:`qiskit.circuit.QuantumCircuit.draw` and
   :func:`qiskit.visualization.circuit_drawer`. It can be set to ``default``
   or ``bw``.
 * ``circuit_mpl_style_path``: This can be used to set the path(s) to have the
   circuit drawer, :meth:`qiskit.circuit.QuantumCircuit.draw` or
   :func:`qiskit.visualization.circuit_drawer`, use to look for json style
   sheets when using the ``mpl`` output mode.
 * ``state_drawer``: This is used to change the default backend for the
   state visualization draw methods :meth:`qiskit.quantum_info.Statevector.draw`
   and :meth:`qiskit.quantum_info.DensityMatrix.draw`. It can be set to
   ``repr``, ``text``', ``latex``, ``latex_source``, ``qsphere``, ``hinton``,
   or bloch ``bloch`` and when the ``output`` kwarg is not explicitly set on
   the :meth:`~qiskit.quantum_info.DensityMatrix.draw` method that output
   method will be used.
 * ``transpile_optimization_level``: This takes an integer between 0-3 and is
   used to change the default optimization level for
   :func:`~qiskit.compiler.transpile` and :func:`~qiskit.execute.execute`.
 * ``parallel``: This option takes a boolean value (either ``True`` or
   ``False``) and is used to configure whether
   `Python multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`__
   is enabled for operations that support running in parallel (for example
   transpilation of multiple :class:`~qiskit.circuit.QuantumCircuit` objects).
   The default setting in the user config file can be overriden by
   the ``QISKIT_PARALLEL`` environment variable.
 * ``num_processes``: This option takes an integer value (> 0) that is used
   to specify the maximum number of parallel processes to launch for parallel
   operations if parallel execution is enabled. The default setting in the
   user config file can be overriden by the ``QISKIT_NUM_PROCS`` environment
   variable.

Environment Variables
---------------------

There are also a few environment variables that can be set to alter the default
behavior of Qiskit.

 * ``QISKIT_PARALLEL``: if this variable is set to ``TRUE`` it will enable
   the use of
   `Python multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`__
   to parallelize certain operations (for example transpilation over multiple
   circuits) in Qiskit Terra.
 * ``QISKIT_NUM_PROCS``: Specifies the maximum number of parallel processes to
   launch for parallel operations if parallel execution is enabled. It takes an
   integer > 0 as the expected value.
 * ``RAYON_NUM_THREADS``: Specifies the number of threads to run multithreaded
   operations in Qiskit Terra. By default this multithreaded code will launch
   a thread for each logical CPU, if you'd like to adjust the number of threads
   Qiskit will use you can set this to an integer value. For example, setting
   ``RAYON_NUM_THREADS=4`` will only launch 4 threads for multithreaded
   functions.
 * ``QISKIT_FORCE_THREADS``: Specify that multithreaded code should always
   execute in multiple threads. By default if you're running multithreaded code
   in a section of Qiskit that is already running in parallel processes Qiskit
   will not launch multiple threads and instead execute that function serially.
   This is done to avoid potentially overloading limited CPU resources. However,
   if you would like to force the use of multiple threads even when in a
   multiprocess context you can set ``QISKIT_FORCE_THREADS=TRUE`` to do this.
 * ``QISKIT_IBMQ_PROVIDER_LOG_LEVEL``: Specifies the log level to use, for the
   ``qiskit-ibmq-provider`` modules. If an invalid level is set, the log level
   defaults to WARNING. The valid log levels are ``DEBUG``, ``INFO``,
   ``WARNING``, ``ERROR``, and ``CRITICAL`` (case-insensitive). If the
   environment variable is not set, then the parent logger’s level is used,
   which also defaults to ``WARNING``.
 * ``QISKIT_IBMQ_PROVIDER_LOG_FILE``: Specifies the name of the log file to
   use from log messages originating from ``qiskit-ibmq-provider``. If
   specified, messages will be logged to the file only. Otherwise messages will
   be logged to the standard error (usually the screen).
 * ``QISKIT_AQUA_MAX_GATES_PER_JOB``: An optional parameter to set a threshold
   for splitting Aqua generated circuits up into multiple jobs submitted to a
   backend based on the number of gates.
