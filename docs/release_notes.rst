%%%%%%%%%%%%%
Release Notes
%%%%%%%%%%%%%


###############
Version History
###############

This table tracks the meta-package versions and the version of each Qiskit element installed:

.. list-table:: **Version History**
   :header-rows: 1

   * - Qiskit Metapackage Version
     - qiskit-terra
     - qiskit-aer
     - qiskit-ignis
     - qiskit-ibmq-provider
     - qiskit-aqua
   * - 0.12.0
     - 0.9.0
     - 0.3.0
     - 0.2.0
     - 0.3.2
     - 0.6.0
   * - 0.11.2
     - 0.8.2
     - 0.2.3
     - 0.1.1
     - 0.3.2
     - 0.5.5
   * - 0.11.1
     - 0.8.2
     - 0.2.3
     - 0.1.1
     - 0.3.1
     - 0.5.3
   * - 0.11.0
     - 0.8.2
     - 0.2.3
     - 0.1.1
     - 0.3.0
     - 0.5.2
   * - 0.10.5
     - 0.8.2
     - 0.2.1
     - 0.1.1
     - 0.2.2
     - 0.5.2
   * - 0.10.4
     - 0.8.2
     - 0.2.1
     - 0.1.1
     - 0.2.2
     - 0.5.1
   * - 0.10.3
     - 0.8.1
     - 0.2.1
     - 0.1.1
     - 0.2.2
     - 0.5.1
   * - 0.10.2
     - 0.8.0
     - 0.2.1
     - 0.1.1
     - 0.2.2
     - 0.5.1
   * - 0.10.1
     - 0.8.0
     - 0.2.0
     - 0.1.1
     - 0.2.2
     - 0.5.0
   * - 0.10.0
     - 0.8.0
     - 0.2.0
     - 0.1.1
     - 0.2.1
     - 0.5.0
   * - 0.9.0
     - 0.8.0
     - 0.2.0
     - 0.1.1
     - 0.1.1
     - 0.5.0
   * - 0.8.1
     - 0.7.2
     - 0.1.1
     - 0.1.0
     -
     -
   * - 0.8.0
     - 0.7.1
     - 0.1.1
     - 0.1.0
     -
     -
   * - 0.7.3
     - 0.7.x
     - 0.1.x
     -
     -
     -
   * - 0.7.2
     - 0.7.x
     - 0.1.x
     -
     -
     -
   * - 0.7.1
     - 0.7.x
     - 0.1.x
     -
     -
     -
   * - 0.7.0
     - 0.7.x
     - 0.1.x
     -
     -
     -

.. note::

  ``0.7.x`` and ``0.1.x`` mean any patch version on that minor version. For,
  example ``0.7.x`` will install the latest ``0.7`` version found on pypi which
  would be ``0.7.2``. For the ``0.7.0``, ``0.7.1``, and ``0.7.2`` meta-package
  releases the :ref:`versioning_strategy` policy was formalized yet.

###############
Notable Changes
###############

*************
Qiskit 0.12.0
*************

.. _Release Notes_0.9.0:

Terra 0.9
=========

.. _Release Notes_0.9.0_Prelude:

Prelude
-------

The 0.9 release includes many new features and many bug fixes. The biggest
changes for this release are new debugging capabilities for PassManagers. This
includes a function to visualize a PassManager, the ability to add a callback
function to a PassManager, and logging of passes run in the PassManager.
Additionally, this release standardizes the way that you can set an initial
layout for your circuit. So now you can leverage ``initial_layout`` the kwarg
parameter on ``qiskit.compiler.transpile()`` and ``qiskit.execute()`` and the
qubits in the circuit will get laid out on the desire qubits on the device.
Visualization of circuits will now also show this clearly when visualizing a
circuit that has been transpiled with a layout.

.. _Release Notes_0.9.0_New Features:

New Features
------------

- A ``DAGCircuit`` object (i.e. the graph representation of a QuantumCircuit where
  operation dependencies are explicit) can now be visualized with the ``.draw()``
  method. This is in line with Qiskit's philosophy of easy visualization.
  Other objects which support a ``.draw()`` method are ``QuantumCircuit``,
  ``PassManager``, and ``Schedule``.

- Added a new visualization function
  ``qiskit.visualization.plot_error_map()`` to plot the error map for a given
  backend. It takes in a backend object from the qiskit-ibmq-provider and
  will plot the current error map for that device.

- Both ``qiskit.QuantumCircuit.draw()`` and
  ``qiskit.visualization.circuit_drawer()`` now support annotating the
  qubits in the visualization with layout information. If the
  ``QuantumCircuit`` object being drawn includes layout metadata (which is
  normally only set on the circuit output from ``transpile()`` calls) then
  by default that layout will be shown on the diagram. This is done for all
  circuit drawer backends. For example::

      from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
      from qiskit.compiler import transpile

      qr = QuantumRegister(2, 'userqr')
      cr = ClassicalRegister(2, 'c0')
      qc = QuantumCircuit(qr, cr)
      qc.h(qr[0])
      qc.cx(qr[0], qr[1])
      qc.y(qr[0])
      qc.x(qr[1])
      qc.measure(qr, cr)

      # Melbourne coupling map
      coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                      [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                      [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                      [13, 12]]
      qc_result = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                            coupling_map=coupling_map, optimization_level=0)
      qc.draw(output='text')

  will yield a diagram like::

                        ┌──────────┐┌──────────┐┌───┐┌──────────┐┌──────────────────┐┌─┐
         (userqr0) q0|0>┤ U2(0,pi) ├┤ U2(0,pi) ├┤ X ├┤ U2(0,pi) ├┤ U3(pi,pi/2,pi/2) ├┤M├───
                        ├──────────┤└──────────┘└─┬─┘├──────────┤└─┬─────────────┬──┘└╥┘┌─┐
         (userqr1) q1|0>┤ U2(0,pi) ├──────────────■──┤ U2(0,pi) ├──┤ U3(pi,0,pi) ├────╫─┤M├
                        └──────────┘                 └──────────┘  └─────────────┘    ║ └╥┘
        (ancilla0) q2|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla1) q3|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla2) q4|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla3) q5|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla4) q6|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla5) q7|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla6) q8|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
        (ancilla7) q9|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
       (ancilla8) q10|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
       (ancilla9) q11|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
      (ancilla10) q12|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
      (ancilla11) q13|0>──────────────────────────────────────────────────────────────╫──╫─
                                                                                      ║  ║
                c0_0: 0 ══════════════════════════════════════════════════════════════╩══╬═
                                                                                         ║
                c0_1: 0 ═════════════════════════════════════════════════════════════════╩═

  If you do not want the layout to be shown on transpiled circuits (or any
  other circuits with a layout set) there is a new boolean kwarg for both
  functions, ``with_layout`` (which defaults ``True``), which when set
  ``False`` will disable the layout annotation in the output circuits.

- A new analysis pass ``CountOpsLongest`` was added to retrieve the number
  of operations on the longest path of the DAGCircuit. When used it will
  add a ``count_ops_longest_path`` key to the property set dictionary.
  You can add it to your a passmanager with something like::

      from qiskit.transpiler.passes import CountOpsLongestPath
      from qiskit.transpiler.passes import CxCancellation
      from qiskit.transpiler import PassManager

      pm = PassManager()
      pm.append(CountOpsLongestPath())

  and then access the longest path via the property set value with something
  like::

      pm.append(
          CxCancellation(),
          condition=lambda property_set: property_set[
              'count_ops_longest_path'] < 5)

  which will set a condition on that pass based on the longest path.

- Two new functions, ``sech()`` and ``sech_deriv()`` were added to the pulse
  library module ``qiskit.pulse.pulse_lib`` for creating an unnormalized
  hyperbolic secant ``SamplePulse`` object and an unnormalized hyperbolic
  secant derviative ``SamplePulse`` object resepctively.

- A new kwarg option ``vertical_compression`` was added to the
  ``QuantumCircuit.draw()`` method and the
  ``qiskit.visualization.circuit_drawer()`` function. This option only works
  with the ``text`` backend. This option can be set to either ``high``,
  ``medium`` (the default), or ``low`` to adjust how much vertical space is
  used by the output visualization.

- A new kwarg boolean option ``idle_wires`` was added to the
  ``QuantumCircuit.draw()`` method and the
  ``qiskit.visualization.circuit_drawer()`` function. It works for all drawer
  backends. When ``idle_wires`` is set False in a drawer call the drawer will
  not draw any bits that do not have any circuit elements in the output
  quantum circuit visualization.

- A new PassManager visualizer function
  ``qiskit.visualization.pass_mamanger_drawer()`` was added. This function
  takes in a PassManager object and will generate a flow control diagram
  of all the passes run in the PassManager.

- When creating a PassManager you can now specify a callback function that
  if specified will be run after each pass is executed. This function gets
  passed a set of kwargs on each call with the state of the pass maanger after
  each pass execution. Currently these kwargs are:

  * ``pass_`` (``Pass``): the pass being run
  * ``dag`` (``DAGCircuit``): the dag output of the pass
  * ``time`` (``float``): the time to execute the pass
  * ``property_set`` (``PropertySet``): the property set
  * ``count`` (``int``): the index for the pass execution

  However, it's worth noting that while these arguments are set for the 0.9
  release they expose the internals of the pass manager and are subject to
  change in future release.

  For example you can use this to create a callback function that will
  visualize the circuit output after each pass is executed::

      from qiskit.transpiler import PassManager

      def my_callback(**kwargs):
          print(kwargs['dag'])

      pm = PassManager(callback=my_callback)

  Additionally you can specify the callback function when using
  ``qiskit.compiler.transpile()``::

      from qiskit.compiler import transpile

      def my_callback(**kwargs):
          print(kwargs['pass'])

      transpile(circ, callback=my_callback)

- A new method ``filter()`` was added to the ``qiskit.pulse.Schedule`` class.
  This enables filtering the instructions in a schedule. For example,
  filtering by instruction type::

      from qiskit.pulse import Schedule
      from qiskit.pulse.commands import Acquire
      from qiskit.pulse.commands import AcquireInstruction
      from qiskit.pulse.commands import FrameChange

      sched = Schedule(name='MyExperiment')
      sched.insert(0, FrameChange(phase=-1.57)(device))
      sched.insert(60, Acquire(5))
      acquire_sched = sched.filter(instruction_types=[AcquireInstruction])

- Additonal decomposition methods for several types of gates. These methods
  will use different decomposition techniques to break down a gate into
  a squence of CNOTs and single qubit gates. The following methods are
  added:

  +--------------------------------+---------------------------------------+
  | Method                         | Description                           |
  +================================+=======================================+
  | ``QuantumCircuit.iso()``       | Add an arbitrary isometry from m to n |
  |                                | qubits to a circuit. This allows for  |
  |                                | attaching arbitrary unitaries on n    |
  |                                | qubits (m=n) or to prepare any state  |
  |                                | of n qubits (m=0)                     |
  +--------------------------------+---------------------------------------+
  | ``QuantumCircuit.diag_gate()`` | Add a diagonal gate to the circuit    |
  +--------------------------------+---------------------------------------+
  | ``QuantumCircuit.squ()``       | Decompose an arbitrary 2x2 unitary    |
  |                                | into three rotation gates and add to  |
  |                                | a circuit                             |
  +--------------------------------+---------------------------------------+
  | ``QuantumCircuit.ucg()``       | Attach an uniformly controlled gate   |
  |                                | (also called a multiplexed gate) to a |
  |                                | circuit                               |
  +--------------------------------+---------------------------------------+
  | ``QuantumCircuit.ucx()``       | Attach a uniformly controlled (also   |
  |                                | called multiplexed) Rx rotation gate  |
  |                                | to a circuit                          |
  +--------------------------------+---------------------------------------+
  | ``QuantumCircuit.ucy()``       | Attach a uniformly controlled (also   |
  |                                | called multiplexed) Ry rotation gate  |
  |                                | to a circuit                          |
  +--------------------------------+---------------------------------------+
  | ``QuantumCircuit.ucz()``       | Attach a uniformly controlled (also   |
  |                                | called multiplexed) Rz rotation gate  |
  |                                | to a circuit                          |
  +--------------------------------+---------------------------------------+

- Addition of Gray-Synth and Patel–Markov–Hayes algorithms for
  synthesis of CNOT-Phase and CNOT-only linear circuits. These functions
  allow the synthesis of circuits that consist of only CNOT gates given
  a linear function or a circuit that consists of only CNOT and phase gates
  given a matrix description.

- A new function ``random_circuit`` was added to the
  ``qiskit.circuit.random`` module. This function will generate a random
  circuit of a specified size by randomly selecting different gates and
  adding them to the circuit. For example, you can use this to generate a
  5 qubit circuit with a depth of 10 using::

      from qiskit.circuit.random import random_circuit

      circ = random_circuit(5, 10)

- A new kwarg ``output_names`` was added to the
  ``qiskit.compiler.transpile()`` function. This kwarg takes in a string
  or a list of strings and uses those as the value of the circuit name for
  the output circuits that get returned by the ``transpile()`` call. For
  example::

      from qiskit.compiler import transpile
      my_circs = [circ_a, circ_b]
      tcirc_a, tcirc_b = transpile(my_circs,
                                   output_names=['Circuit A', 'Circuit B'])

  the ``name`` attribute on tcirc_a and tcirc_b will be ``'Circuit A'`` and
  ``'Circuit B'`` respectively.

- A new method ``equiv()`` was added to the ``qiskit.quantum_info.Operator``
  and ``qiskit.quantum_info.Statevector`` classes. These methods are used
  to check whether a second ``Operator`` object or ``Statevector`` is
  equivalent up to global phase.

- The user config file has several new options:

  * The ``circuit_drawer`` field now accepts an `auto` value. When set as
    the value for the ``circuit_drawer`` field the default drawer backend
    will be `mpl` if it is available, otherwise the `text` backend will be
    used.
  * A new field ``circuit_mpl_style`` can be used to set the default style
    used by the matplotlib circuit drawer. Valid values for this field are
    ``bw`` and ``default`` to set the default to a black and white or the
    default color style respectively.
  * A new field ``transpile_optimization_level`` can be used to set the
    default transpiler optimization level to use for calls to
    ``qiskit.compiler.transpile()``. The value can be set to either 0, 1, 2,
    or 3.

- Introduced a new pulse command ``Delay`` which may be inserted into a pulse
  ``Schedule``. This command accepts a ``duration`` and may be added to any
  ``Channel``. Other commands may not be scheduled on a channel during a delay.

  The delay can be added just like any other pulse command. For example::

    from qiskit import pulse

    drive_channel = pulse.DriveChannel(0)
    delay = pulse.Delay(20)

    sched = pulse.Schedule()
    sched += delay(drive_channel)


.. _Release Notes_0.9.0_Upgrade Notes:

Upgrade Notes
-------------

- The previously deprecated ``qiskit._util`` module has been removed.
  ``qiskit.util`` should be used instead.

- The ``QuantumCircuit.count_ops()`` method now returns an ``OrderedDict``
  object instead of a ``dict``. This should be compatible for most use cases
  since ``OrderedDict`` is a ``dict`` subclass. However type checks and
  other class checks might need to be updated.

- The ``DAGCircuit.width()`` method now returns the total number quantum bits
  and classical bits. Before it would only return the number of quantum bits.
  If you require just the number of quantum bits you can use
  ``DAGCircuit.num_qubits()`` instead.

- The function ``DAGCircuit.num_cbits()`` has been removed. Instead you can
  use ``DAGCircuit.num_clbits()``.

- Individual quantum bits and classical bits are no longer represented as
  ``(register, index)`` tuples. They are now instances of `Qubit` and
  `Clbit` classes. If you're dealing with individual bits make sure that
  you update any usage or type checks to look for these new classes instead
  of tuples.

- The preset passmanager classes
  ``qiskit.transpiler.preset_passmanagers.default_pass_manager`` and
  ``qiskit.transpiler.preset_passmanagers.default_pass_manager_simulator``
  (which were the previous default pass managers for
  ``qiskit.compiler.transpile()`` calls) have been removed. If you were
  manually using this pass managers switch to the new default,
  ``qiskit.transpile.preset_passmanagers.level1_pass_manager``.

- The ``LegacySwap`` pass has been removed. If you were using it in a custom
  pass manager, it's usage can be replaced by the ``StochasticSwap`` pass,
  which is a faster more stable version. All the preset passmanagers have
  been updated to use ``StochasticSwap`` pass instead of the ``LegacySwap``.

- The following deprecated ``qiskit.dagcircuit.DAGCircuit`` methods have been
  removed:

  * ``DAGCircuit.get_qubits()`` - Use ``DAGCircuit.qubits()`` instead
  * ``DAGCircuit.get_bits()`` - Use ``DAGCircuit.clbits()`` instead
  * ``DAGCircuit.qasm()`` - Use a combination of
    ``qiskit.converters.dag_to_circuit()`` and ``QuantumCircuit.qasm()``. For
    example::

      from qiskit.dagcircuit import DAGCircuit
      from qiskit.converters import dag_to_circuit
      my_dag = DAGCircuit()
      qasm = dag_to_circuit(my_dag).qasm()

  * ``DAGCircuit.get_op_nodes()`` - Use ``DAGCircuit.op_nodes()`` instead.
    Note that the return type is a list of ``DAGNode`` objects for
    ``op_nodes()`` instead of the list of tuples previously returned by
    ``get_op_nodes()``.
  * ``DAGCircuit.get_gate_nodes()`` - Use ``DAGCircuit.gate_nodes()``
    instead. Note that the return type is a list of ``DAGNode`` objects for
    ``gate_nodes()`` instead of the list of tuples previously returned by
    ``get_gate_nodes()``.
  * ``DAGCircuit.get_named_nodes()`` - Use ``DAGCircuit.named_nodes()``
    instead. Note that the return type is a list of ``DAGNode`` objects for
    ``named_nodes()`` instead of the list of node_ids previously returned by
    ``get_named_nodes()``.
  * ``DAGCircuit.get_2q_nodes()`` - Use ``DAGCircuit.twoQ_gates()``
    instead. Note that the return type is a list of ``DAGNode`` objects for
    ``twoQ_gates()`` instead of the list of data_dicts previously returned by
    ``get_2q_nodes()``.
  * ``DAGCircuit.get_3q_or_more_nodes()`` - Use
    ``DAGCircuit.threeQ_or_more_gates()`` instead. Note that the return type
    is a list of ``DAGNode`` objects for ``threeQ_or_more_gates()`` instead
    of the list of tuples previously returned by ``get_3q_or_more_nodes()``.

- The following ``qiskit.dagcircuit.DAGCircuit`` methods had deprecated
  support for accepting a ``node_id`` as a parameter. This has been removed
  and now only ``DAGNode`` objects are accepted as input:

  * ``successors()``
  * ``predecessors()``
  * ``ancestors()``
  * ``descendants()``
  * ``bfs_successors()``
  * ``quantum_successors()``
  * ``remove_op_node()``
  * ``remove_ancestors_of()``
  * ``remove_descendants_of()``
  * ``remove_nonancestors_of()``
  * ``remove_nondescendants_of()``
  * ``substitute_node_with_dag()``

- The ``qiskit.dagcircuit.DAGCircuit`` method ``rename_register()`` has been
  removed. This was unused by all the qiskit code. If you were relying on it
  externally you'll have to re-implement is an external function.

- The ``qiskit.dagcircuit.DAGCircuit`` property ``multi_graph`` has been
  removed. Direct access to the underlying ``networkx`` ``multi_graph`` object
  isn't supported anymore. The API provided by the ``DAGCircuit`` class should
  be used instead.

- The deprecated exception class ``qiskit.qiskiterror.QiskitError`` has been
  removed. Instead you should use ``qiskit.exceptions.QiskitError``.

- The boolean kwargs, ``ignore_requires`` and ``ignore_preserves`` from
  the ``qiskit.transpiler.PassManager`` constructor have been removed. These
  are no longer valid options.

- The module ``qiskit.tools.logging`` has been removed. This module was not
  used by anything and added nothing over the interfaces that Python's
  standard library ``logging`` module provides. If you want to set a custom
  formatter for logging use the standard library ``logging`` module instead.

- The ``CompositeGate`` class has been removed. Instead you should
  directly create a instruction object from a circuit and append that to your
  circuit. For example, you can run something like::

      custom_gate_circ = qiskit.QuantumCircuit(2)
      custom_gate_circ.x(1)
      custom_gate_circ.h(0)
      custom_gate_circ.cx(0, 1)
      custom_gate = custom_gate_circ.to_instruction()

- The previously deprecated kwargs, ``seed`` and ``config`` for
  ``qiskit.compiler.assemble()`` have been removed use ``seed_simulator`` and
  ``run_config`` respectively instead.

- The previously deprecated converters
  ``qiskit.converters.qobj_to_circuits()`` and
  ``qiskit.converters.circuits_to_qobj()`` have been removed. Use
  ``qiskit.assembler.disassemble()`` and ``qiskit.compiler.assemble()``
  respectively instead.

- The previously deprecated kwarg ``seed_mapper`` for
  ``qiskit.compiler.transpile()`` has been removed. Instead you should use
  ``seed_transpiler``

- The previously deprecated kwargs ``seed``, ``seed_mapper``, ``config``,
  and ``circuits`` for the ``qiskit.execute()`` function have been removed.
  Use ``seed_simulator``, ``seed_transpiler``, ``run_config``, and
  ``experiments`` arguments respectively instead.

- The previously deprecated ``qiskit.tools.qcvv`` module has been removed
  use qiskit-ignis instead.

- The previously deprecated functions ``qiskit.transpiler.transpile()`` and
  ``qiskit.transpiler.transpile_dag()`` have been removed. Instead you should
  use ``qiskit.compiler.transpile``. If you were using ``transpile_dag()``
  this can be replaced by running::

      circ = qiskit.converters.dag_to_circuit(dag)
      out_circ = qiskit.compiler.transpile(circ)
      qiskit.converters.circuit_to_dag(out_circ)

- The previously deprecated function ``qiskit.compile()`` has been removed
  instead you should use ``qiskit.compiler.transpile()`` and
  ``qiskit.compiler.assemble()``.

- The jupyter cell magic ``%%qiskit_progress_bar`` from
  ``qiskit.tools.jupyter`` has been changed to a line magic. This was done
  to better reflect how the magic is used and how it works. If you were using
  the ``%%qiskit_progress_bar`` cell magic in an existing notebook, you will
  have to update this to be a line magic by changing it to be
  ``%qiskit_progress_bar`` instead. Everything else should behave
  identically.

- The deprecated function ``qiskit.tools.qi.qi.random_unitary_matrix()``
  has been removed. You should use the
  ``qiskit.quantum_info.random.random_unitary()`` function instead.

- The deprecated function ``qiskit.tools.qi.qi.random_density_matrix()``
  has been removed. You should use the
  ``qiskit.quantum_info.random.random_density_matrix()`` function
  instead.

- The deprecated function ``qiskit.tools.qi.qi.purity()`` has been removed.
  You should the ``qiskit.quantum_info.purity()`` function instead.

- The deprecated ``QuantumCircuit._attach()`` method has been removed. You
  should use ``QuantumCircuit.append()`` instead.

- The ``qiskit.qasm.Qasm`` method ``get_filename()`` has been removed.
  You can use the ``return_filename()`` method instead.

- The deprecated ``qiskit.mapper`` module has been removed. The list of
  functions and classes with their alternatives are:

  * ``qiskit.mapper.CouplingMap``: ``qiskit.transpiler.CouplingMap`` should
    be used instead.
  * ``qiskit.mapper.Layout``: ``qiskit.transpiler.Layout`` should be used
    instead
  * ``qiskit.mapper.compiling.euler_angles_1q()``:
    ``qiskit.quantum_info.synthesis.euler_angles_1q()`` should be used
    instead
  * ``qiskit.mapper.compiling.two_qubit_kak()``:
    ``qiskit.quantum_info.synthesis.two_qubit_cnot_decompose()`` should be
    used instead.

  The deprecated exception classes ``qiskit.mapper.exceptions.CouplingError``
  and ``qiskit.mapper.exceptions.LayoutError`` don't have an alternative
  since they serve no purpose without a ``qiskit.mapper`` module.

- The ``qiskit.pulse.samplers`` module has been moved to
  ``qiskit.pulse.pulse_lib.samplers``. You will need to update imports of
  ``qiskit.pulse.samplers`` to ``qiskit.pulse.pulse_lib.samplers``.

- `seaborn`_ is now a dependency for the function
  ``qiskit.visualization.plot_state_qsphere()``. It is needed to generate
  proper angular color maps for the visualization. The
  ``qiskit-terra[visualization]`` extras install target has been updated to
  install ``seaborn>=0.9.0`` If you are using visualizations and specifically
  the ``plot_state_qsphere()`` function you can use that to install
  ``seaborn`` or just manually run ``pip install seaborn>=0.9.0``

  .. _seaborn: https://seaborn.pydata.org/

- The previously deprecated functions ``qiksit.visualization.plot_state`` and
  ``qiskit.visualization.iplot_state`` have been removed. Instead you should
  use the specific function for each plot type. You can refer to the
  following tables to map the deprecated functions to their equivalent new
  ones:

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

- The ``pylatexenc`` and ``pillow`` dependencies for the ``latex`` and
  ``latex_source`` circuit drawer backends are no longer listed as
  requirements. If you are going to use the latex circuit drawers ensure
  you have both packages installed or use the setuptools extras to install
  it along with qiskit-terra::

      pip install qiskit-terra[visualization]

- The root of the ``qiskit`` namespace will now emit a warning on import if
  either ``qiskit.IBMQ`` or ``qiskit.Aer`` could not be setup. This will
  occur whenever anything in the ``qiskit`` namespace is imported. These
  warnings were added to make it clear for users up front if they're running
  qiskit and the qiskit-aer and qiskit-ibmq-provider packages could not be
  found. It's not always clear if the packages are missing or python
  packaging/pip installed an element incorrectly until you go to use them and
  get an empty ``ImportError``. These warnings should make it clear up front
  if there these commonly used aliases are missing.

  However, for users that choose not to use either qiskit-aer or
  qiskit-ibmq-provider this might cause additional noise. For these users
  these warnings are easily suppressable using Python's standard library
  ``warnings``. Users can suppress the warnings by putting thse two lines
  before any imports from qiskit::

      import warnings
      warnings.filterwarnings('ignore', category=RuntimeWarning,
                              module='qiskit')

  This will suppress the warnings emitted by not having qiskit-aer or
  qiskit-ibmq-provider installed, but still preserve any other warnings
  emitted by qiskit or any other package.


.. _Release Notes_0.9.0_Deprecation Notes:

Deprecation Notes
-----------------

- The ``U`` and ``CX`` gates have been deprecated. If you're using these gates
  in your code you should update them to use ``u3`` and ``cx`` instead. For
  example, if you're using the circuit gate functions ``circuit.u_base()``
  and ``circuit.cx_base()`` you should update these to be ``circuit.u3()`` and
  ``circuit.cx()`` respectively.

- The ``u0`` gate has been deprecated in favor of using multiple ``iden``
  gates and it will be removed in the future. If you're using the ``u0`` gate
  in your circuit you should update your calls to use ``iden``. For example,
  f you were using ``circuit.u0(2)`` in your circuit before that should be
  updated to be::

      circuit.iden()
      circuit.iden()

  instead.

- The ``qiskit.pulse.DeviceSpecification`` class is deprecated now. Instead
  you should use ``qiskit.pulse.PulseChannelSpec``.

- Accessing a ``qiskit.circuit.Qubit``, ``qiskit.circuit.Clbit``, or
  ``qiskit.circuit.Bit`` class by index is deprecated (for compatibility
  with the ``(register, index)`` tuples that these classes replaced).
  Instead you should use the ``register`` and ``index`` attributes.

- Passing in a bit to the ``qiskit.QuantumCircuit`` method ``append`` as
  a tuple ``(register, index)`` is deprecated. Instead bit objects should
  be used directly.

- Accessing the elements of a ``qiskit.transpiler.Layout`` object with a
  tuple ``(register, index)`` is deprecated. Instead a bit object should
  be used directly.

- The ``qiskit.transpiler.Layout`` constructor method
  ``qiskit.transpiler.Layout.from_tuplelist()`` is deprecated. Instead the
  constructor ``qiskit.transpiler.Layout.from_qubit_list()`` should be used.

- The module ``qiskit.pulse.ops`` has been deprecated. All the functions it
  provided:

  * ``union``
  * ``flatten``
  * ``shift``
  * ``insert``
  * ``append``

  have equivalent methods available directly on the ``qiskit.pulse.Schedule``
  and ``qiskit.pulse.Instruction`` classes. Those methods should be used
  instead.

- The ``qiskit.qasm.Qasm`` method ``get_tokens()`` is deprecated. Instead
  you should use the ``generate_tokens()`` method.

- The ``qiskit.qasm.qasmparser.QasmParser`` method ``get_tokens()`` is
  deprecated. Instead you should use the ``read_tokens()`` method.

- The ``as_dict()`` method for the Qobj class has been deprecated and will
  be removed in the future. You should replace calls to it with ``to_dict()``
  instead.


.. _Release Notes_0.9.0_Bug Fixes:

Bug Fixes
---------

- The definition of the ``CU3Gate`` has been changed to be equivalent to the
  canonical definition of a controlled ``U3Gate``.

- The handling of layout in the pass manager has been standardized. This
  fixes several reported issues with handling layout. The ``initial_layout``
  kwarg parameter on ``qiskit.compiler.transpile()`` and
  ``qiskit.execute()`` will now lay out your qubits from the circuit onto
  the desired qubits on the device when transpiling circuits.

- Support for n-qubit unitaries was added to the BasicAer simulator and
  ``unitary`` (arbitrary unitary gates) was added to the set of basis gates
  for the simulators

- The ``qiskit.visualization.plost_state_qsphere()`` has been updated to fix
  several issues with it. Now output Q Sphere visualization will be correctly
  generated and the following aspects have been updated:

  * All complementary basis states are antipodal
  * Phase is indicated by color of line and marker on sphere's surface
  * Probability is indicated by translucency of line and volume of marker on
     sphere's surface


.. _Release Notes_0.9.0_Other Notes:

Other Notes
-----------

- The default PassManager for ``qiskit.compiler.transpile()`` and
  ``qiskit.execute()`` has been changed to optimization level 1 pass manager
  defined at ``qiskit.transpile.preset_passmanagers.level1_pass_manager``.

- All the circuit drawer backends now willl express gate parameters in a
  circuit as common fractions of pi in the output visualization. If the value
  of a parameter can be expressed as a fraction of pi that will be used
  instead of the numeric equivalent.

- When using ``qiskit.assembler.assemble_schedules()`` if you do not provide
  the number of memory_slots to use the number will be infered based on the
  number of acquisitions in the input schedules.

- The deprecation warning on the ``qiskit.dagcircuit.DAGCircuit`` property
  ``node_counter`` has been removed. The behavior change being warned about
  was put into effect when the warning was added, so warning that it had
  changed served no purpose.

- Calls to ``PassManager.run()`` now will emit python logging messages at the
  INFO level for each pass execution. These messages will include the Pass
  name and the total execution time of the pass. Python's standard logging
  was used because it allows Qiskit-Terra's logging to integrate in a standard
  way with other applications and libraries. All logging for the transpiler
  occurs under the ``qiskit.transpiler`` namespace, as used by
  ``logging.getLogger('qiskit.transpiler``). For example, to turn on DEBUG
  level logging for the transpiler you can run::

      import logging

      logging.basicConfig()
      logging.getLogger('qiskit.transpiler').setLevel(logging.DEBUG)

  which will set the log level for the transpiler to DEBUG and configure
  those messages to be printed to stderr.

Aer 0.3
=======
- There's a new high-performance Density Matrix Simulator that can be used in
  conjunction with our noise models, to better simulate real world scenarios.
- We have added a Matrix Product State (MPS) simulator. MPS allows for
  efficient simulation of several classes of quantum circuits, even under
  presence of strong correlations and highly entangled states. For cases
  amenable to MPS, circuits with several hundred qubits and more can be exactly
  simulated, e.g., for the purpose of obtaining expectation values of observables.
- Snapshots can be performed in all of our simulators.
- Now we can measure sampling circuits with read-out errors too, not only ideal
  circuits.
- We have increased some circuit optimizations with noise presence.
- A better 2-qubit error aproximations have been included.
- Included some tools for making certain noisy simulations easier to craft and
  faster to simulate.
- Increased performance with simulations that require less floating point
  numerical precision.

Ignis 0.2
=========

New Features
------------

- `Logging Module <https://github.com/Qiskit/qiskit-iqx-tutorials/blob/stable/0.12.x/qiskit/advanced/ignis/9_ignis_logging.ipynb>`_
- `Purity RB <https://github.com/Qiskit/qiskit-iqx-tutorials/blob/stable/0.12.x/qiskit/advanced/ignis/5c_purity_rb.ipynb>`_
- `Interleaved RB <https://github.com/Qiskit/qiskit-iqx-tutorials/blob/stable/0.12.x/qiskit/advanced/ignis/5b_interleaved_rb.ipynb>`_
- `Repetition Code for Verification <https://github.com/Qiskit/qiskit-iqx-tutorials/blob/stable/0.12.x/qiskit/advanced/ignis/8_repetition_code.ipynb>`_
- Seed values can now be aribtrarily added to RB (not just in order)
- Support for adding multiple results to measurement mitigation
- RB Fitters now support providing guess values

Bug Fixes
---------

- Fixed a bug in RB fit error
- Fixed a bug in the characterization fitter when selecting a qubit index to
  fit

Other Notes
-----------

- Measurement mitigation now operates in parallel when applied to multiple
  results
- Guess values for RB fitters are improved

Aqua 0.6
========

Added
-----

- Relative-Phase Toffoli gates ``rccx`` (with 2 controls) and ``rcccx``
  (with 3 controls).
- Variational form ``RYCRX``
- A new ``'basic-no-ancilla'`` mode to ``mct``.
- Multi-controlled rotation gates ``mcrx``, ``mcry``, and ``mcrz`` as a general
  ``u3`` gate is not supported by graycode implementation
- Chemistry: ROHF open-shell support

  * Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
  * HartreeFock initial state, UCCSD variational form and two qubit reduction for
    parity mapping now support different alpha and beta particle numbers for open
    shell support

- Chemistry: UHF open-shell support

  * Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
  * QMolecule extended to include integrals, coeffiecients etc for separate beta

- Chemistry: QMolecule extended with integrals in atomic orbital basis to
  facilitate common access to these for experimentation

  * Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4

- Chemistry: Additional PyQuante and PySCF driver configuration

  * Convergence tolerance and max convergence iteration controls.
  * For PySCF initial guess choice

- Chemistry: Processing output added to debug log from PyQuante and PySCF
  computations (Gaussian16 and PSI4 outputs were already added to debug log)
- Chemistry: Merged qiskit-chemistry into qiskit-aqua
- Add ``MatrixOperator``, ``WeightedPauliOperator`` and
  ``TPBGroupedPauliOperator`` class.
- Add ``evolution_instruction`` function to get registerless instruction of
  time evolution.
- Add ``op_converter`` module to unified the place in charge of converting
  different types of operators.
- Add ``Z2Symmetries`` class to encapsulate the Z2 symmetries info and has
  helper methods for tapering an Operator.
- Amplitude Estimation: added maximum likelihood postprocessing and confidence
  interval computation.
- Maximum Likelihood Amplitude Estimation (MLAE): Implemented new algorithm for
  amplitude estimation based on maximum likelihood estimation, which reduces
  number of required qubits and circuit depth.
- Added (piecewise) linearly and polynomially controlled Pauli-rotation
  circuits.
- Add ``q_equation_of_motion`` to study excited state of a molecule, and add
  two algorithms to prepare the reference state.

Changed
-------

- Improve ``mct``'s ``'basic'`` mode by using relative-phase Toffoli gates to
  build intermediate results.
- Adapt to Qiskit Terra's newly introduced ``Qubit`` class.
- Prevent ``QPE/IQPE`` from modifying input ``Operator`` objects.
- The PyEDA dependency was removed;
  corresponding oracles' underlying logic operations are now handled by SymPy.
- Refactor the ``Operator`` class, each representation has its own class
  ``MatrixOperator``, ``WeightedPauliOperator`` and ``TPBGroupedPauliOperator``.
- The ``power`` in ``evolution_instruction`` was applied on the theta on the
  CRZ gate directly, the new version repeats the circuits to implement power.
- CircuitCache is OFF by default, and it can be set via environment variable now
  ``QISKIT_AQUA_CIRCUIT_CACHE``.

Bug Fixes
---------

- A bug where ``TruthTableOracle`` would build incorrect circuits for truth
  tables with only a single ``1`` value.
- A bug caused by ``PyEDA``'s indeterminism.
- A bug with ``QPE/IQPE``'s translation and stretch computation.
- Chemistry: Bravyi-Kitaev mapping fixed when num qubits was not a power of 2
- Setup ``initial_layout`` in ``QuantumInstance`` via a list.

Removed
-------

- General multi-controlled rotation gate ``mcu3`` is removed and replaced by
  multi-controlled rotation gates ``mcrx``, ``mcry``, and ``mcrz``

Deprecated
----------
- The ``Operator`` class is deprecated, in favor of using ``MatrixOperator``,
  ``WeightedPauliOperator`` and ``TPBGroupedPauliOperator``.


IBM Q Provider 0.3
==================

No change


*************
Qiskit 0.11.1
*************

We have bumped up Qiskit micro version to 0.11.1 because IBM Q Provider has
bumped its micro version as well.

Terra 0.8
=========

No Change

Aer 0.2
=======

No change

Ignis 0.1
=========

No Change

Aqua 0.5
========

``qiskit-aqua`` has been updated to ``0.5.3`` to fix code related to
changes in how gates inverses are done.

IBM Q Provider 0.3
==================

The ``IBMQProvider`` has been updated to version ``0.3.1`` to fix
backward compatibility issues and work with the default 10 job
limit in single calls to the IBM Q API v2.


***********
Qiskit 0.11
***********

We have bumped up Qiskit minor version to 0.11 because IBM Q Provider has bumped up
its minor version too.
On Aer, we have jumped from 0.2.1 to 0.2.3 because there was an issue detected
right after releasing 0.2.2 and before Qiskit 0.11 went online.

Terra 0.8
=========

No Change

Aer 0.2
=======

New features
------------

- Added support for multi-controlled phase gates
- Added optimized anti-diagonal single-qubit gates

Improvements
------------

- Introduced a technique called Fusion that increments performance of circuit execution
  Tuned threading strategy to gain performance in most common scenarios.
- Some of the already implemented error models have been polished.



Ignis 0.1
=========

No Change

Aqua 0.5
========

No Change

IBM Q Provider 0.3
==================

The ``IBMQProvider`` has been updated in order to default to using the new
`IBM Q Experience v2 <https://quantum-computing.ibm.com>`__. Accessing the legacy IBM Q Experience v1 and QConsole
will still be supported during the 0.3.x line until its final deprecation one
month from the release. It is encouraged to update to the new IBM Q
Experience to take advantage of the new functionality and features.

Updating to the new IBM Q Experience v2
---------------------------------------

If you have credentials for the legacy IBM Q Experience stored on disk, you
can make use of the interactive helper::

    from qiskit import IBMQ

    IBMQ.update_account()


For more complex cases or fine tuning your configuration, the following methods
are available:

* the ``IBMQ.delete_accounts()`` can be used for resetting your configuration
  file.
* the ``IBMQ.save_account('MY_TOKEN')`` method can be used for saving your
  credentials, following the instructions in the `IBM Q Experience v2 <https://quantum-computing.ibm.com>`__
  account page.

Updating your programs
----------------------

When using the new IBM Q Experience v2 through the provider, access to backends
is done via individual ``provider`` instances (as opposed to accessing them
directly through the ``qiskit.IBMQ`` object as in previous versions), which
allows for more granular control over the project you are using.

You can get a reference to the ``providers`` that you have access to using the
``IBMQ.providers()`` and ``IBMQ.get_provider()`` methods::

    from qiskit import IBMQ

    provider = IBMQ.load_account()
    my_providers = IBMQ.providers()
    provider_2 = IBMQ.get_provider(hub='A', group='B', project='C')


For convenience, ``IBMQ.load_account()`` and ``IBMQ.enable_account()`` will
return a provider for the open access project, which is the default in the new
IBM Q Experience v2.

For example, the following program in previous versions::

    from qiskit import IBMQ

    IBMQ.load_accounts()
    backend = IBMQ.get_backend('ibmqx4')
    backend_2 = IBMQ.get_backend('ibmq_qasm_simulator', hub='HUB2')

Would be equivalent to the following program in the current version::

    from qiskit import IBMQ

    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmqx4')
    provider_2 = IBMQ.get_provider(hub='HUB2')
    backend_2 = provider_2.get_backend('ibmq_qasm_simulator')

You can find more information and details in the `IBM Q Provider documentation <https://github.com/Qiskit/qiskit-ibmq-provider>`__.


***********
Qiskit 0.10
***********

Terra 0.8
=========

No Change

Aer 0.2
=======

No Change

Ignis 0.1
=========

No Change

Aqua 0.5
========

No Change

IBM Q Provider 0.2
==================

New Features
------------

- The ``IBMQProvider`` supports connecting to the new version of the IBM Q API.
  Please note support for this version is still experimental :pull_ibmq-provider:`78`.
- Added support for Circuits through the new API :pull_ibmq-provider:`79`.


Bug Fixes
---------

- Fixed incorrect parsing of some API hub URLs :pull_ibmq-provider:`77`.
- Fixed noise model handling for remote simulators :pull_ibmq-provider:`84`.


**********
Qiskit 0.9
**********

Terra 0.8
=========



Highlights
----------

- Introduction of the Pulse module under ``qiskit.pulse``, which includes
  tools for building pulse commands, scheduling them on pulse channels,
  visualization, and running them on IBM Q devices.
- Improved QuantumCircuit and Instruction classes, allowing for the
  composition of arbitrary sub-circuits into larger circuits, and also
  for creating parametrized circuits.
- A powerful Quantum Info module under ``qiskit.quantum_info``, providing
  tools to work with operators and channels and to use them inside circuits.
- New transpiler optimization passes and access to predefined transpiling
  routines.



New Features
------------

- The core ``StochasticSwap`` routine is implemented in `Cython <https://cython.org/>`__.
- Added ``QuantumChannel`` classes for manipulating quantum channels and CPTP
  maps.
- Support for parameterized circuits.
- The ``PassManager`` interface has been improved and new functions added for
  easier interaction and usage with custom pass managers.
- Preset ``PassManager``\s are now included which offer a predetermined pipeline
  of transpiler passes.
- User configuration files to let local environments override default values
  for some functions.
- New transpiler passes: ``EnlargeWithAncilla``, ``Unroll2Q``,
  ``NoiseAdaptiveLayout``, ``OptimizeSwapBeforeMeasure``,
  ``RemoveDiagonalGatesBeforeMeasure``, ``CommutativeCancellation``,
  ``Collect2qBlocks``, and ``ConsolidateBlocks``.


Compatibility Considerations
----------------------------

As part of the 0.8 release the following things have been deprecated and will
either be removed or changed in a backwards incompatible manner in a future
release. While not strictly necessary these are things to adjust for before the
0.9 (unless otherwise noted) release to avoid a breaking change in the future.

* The methods prefixed by ``_get`` in the ``DAGCircuit`` object are being
  renamed without that prefix.
* Changed elements in ``couplinglist`` of ``CouplingMap`` from tuples to lists.
* Unroller bases must now be explicit, and violation raises an informative
  ``QiskitError``.
* The ``qiskit.tools.qcvv`` package is deprecated and will be removed in the in
  the future. You should migrate to using the Qiskit Ignis which replaces this
  module.
* The ``qiskit.compile()`` function is now deprecated in favor of explicitly
  using the ``qiskit.compiler.transpile()`` function to transform a circuit,
  followed by ``qiskit.compiler.assemble()`` to make a Qobj out of
  it. Instead of ``compile(...)``, use ``assemble(transpile(...), ...)``.
* ``qiskit.converters.qobj_to_circuits()`` has been deprecated and will be
  removed in a future release. Instead
  ``qiskit.assembler.disassemble()`` should be used to extract
  ``QuantumCircuit`` objects from a compiled Qobj.
* The ``qiskit.mapper`` namespace has been deprecated. The ``Layout`` and
  ``CouplingMap`` classes can be accessed via ``qiskit.transpiler``.
* A few functions in ``qiskit.tools.qi.qi`` have been deprecated and
  moved to ``qiskit.quantum_info``.

Please note that some backwards incompatible changes have been made during this
release. The following notes contain information on how to adapt to these
changes.

IBM Q Provider
^^^^^^^^^^^^^^

The IBM Q provider was previously included in Terra, but it has been split out
into a separate package ``qiskit-ibmq-provider``. This will need to be
installed, either via pypi with ``pip install qiskit-ibmq-provider`` or from
source in order to access ``qiskit.IBMQ`` or ``qiskit.providers.ibmq``. If you
install qiskit with ``pip install qiskit``, that will automatically install
all subpackages of the Qiskit project.



Cython Components
^^^^^^^^^^^^^^^^^

Starting in the 0.8 release the core stochastic swap routine is now implemented
in `Cython <https://cython.org/>`__. This was done to significantly improve the performance of the
swapper, however if you build Terra from source or run on a non-x86 or other
platform without prebuilt wheels and install from source distribution you'll
need to make sure that you have Cython installed prior to installing/building
Qiskit Terra. This can easily be done with pip/pypi: ``pip install Cython``.




Compiler Workflow
^^^^^^^^^^^^^^^^^

The ``qiskit.compile()`` function has been deprecated and replaced by first
calling ``qiskit.compiler.transpile()`` to run optimization and mapping on a
circuit, and then ``qiskit.compiler.assemble()`` to build a Qobj from that
optimized circuit to send to a backend. While this is only a deprecation it
will emit a warning if you use the old ``qiskit.compile()`` call.

**transpile(), assemble(), execute() parameters**

These functions are heavily overloaded and accept a wide range of inputs.
They can handle circuit and pulse inputs. All kwargs except for ``backend``
for these functions now also accept lists of the previously accepted types.
The ``initial_layout`` kwarg can now be supplied as a both a list and dictionary,
e.g. to map a Bell experiment on qubits 13 and 14, you can supply:
``initial_layout=[13, 14]`` or ``initial_layout={qr[0]: 13, qr[1]: 14}``



Qobj
^^^^

The Qobj class has been split into two separate subclasses depending on the
use case, either ``PulseQobj`` or ``QasmQobj`` for pulse and circuit jobs
respectively. If you're interacting with Qobj directly you may need to
adjust your usage accordingly.

The ``qiskit.qobj.qobj_to_dict()`` is removed. Instead use the ``to_dict()``
method of a Qobj object.



Visualization
^^^^^^^^^^^^^

The largest change to the visualization module is it has moved from
``qiskit.tools.visualization`` to ``qiskit.visualization``. This was done to
indicate that the visualization module is more than just a tool. However, since
this interface was declared stable in the 0.7 release the public interface off
of ``qiskit.tools.visualization`` will continue to work. That may change in a
future release, but it will be deprecated prior to removal if that happens.

The previously deprecated functions, ``plot_circuit()``,
``latex_circuit_drawer()``, ``generate_latex_source()``, and
``matplotlib_circuit_drawer()`` from ``qiskit.tools.visualization`` have been
removed. Instead of these functions, calling
``qiskit.visualization.circuit_drawer()`` with the appropriate arguments should
be used.

The previously deprecated ``plot_barriers`` and ``reverse_bits`` keys in
the ``style`` kwarg dictionary are deprecated, instead the
``qiskit.visualization.circuit_drawer()`` kwargs ``plot_barriers`` and
``reverse_bits`` should be used.

The Wigner plotting functions ``plot_wigner_function``, ``plot_wigner_curve``,
``plot_wigner_plaquette``, and ``plot_wigner_data`` previously in the
``qiskit.tools.visualization._state_visualization`` module have been removed.
They were never exposed through the public stable interface and were not well
documented. The code to use this feature can still be accessed through the
qiskit-tutorials repository.



Mapper
^^^^^^

The public api from ``qiskit.mapper`` has been moved into ``qiskit.transpiler``.
While it has only been deprecated in this release, it will be removed in the
0.9 release so updating your usage of ``Layout`` and ``CouplingMap`` to import
from ``qiskit.transpiler`` instead of ``qiskit.mapper`` before that takes place
will avoid any surprises in the future.






Aer 0.2
=======

New Features
------------

- Added multiplexer gate :pull_aer:`192`
- Added ``remap_noise_model`` function to ``noise.utils`` :pull_aer:`181`
- Added ``__eq__`` method to ``NoiseModel``, ``QuantumError``, ``ReadoutError``
  :pull_aer:`181`
- Added support for labelled gates in noise models :pull_aer:`175`
- Added optimized ``mcx``, ``mcy``, ``mcz``, ``mcu1``, ``mcu2``, ``mcu3``,
  gates to ``QubitVector`` :pull_aer:`124`
- Added optimized controlled-swap gate to ``QubitVector`` :pull_aer:`142`
- Added gate-fusion optimization for ``QasmContoroller``, which is enabled by
  setting ``fusion_enable=true`` :pull_aer:`136`
- Added better management of failed simulations :pull_aer:`167`
- Added qubits truncate optimization for unused qubits :pull_aer:`164`
- Added ability to disable depolarizing error on device noise model
  :pull_aer:`131`
- Added initialize simulator instruction to ``statevector_state``
  :pull_aer:`117`, :pull_aer:`137`
- Added coupling maps to simulators :pull_aer:`93`
- Added circuit optimization framework :pull_aer:`83`
- Added benchmarking :pull_aer:`71`, :pull_aer:`177`
- Added wheels support for Debian-like distributions :pull_aer:`69`
- Added autoconfiguration of threads for qasm simulator :pull_aer:`61`
- Added Simulation method based on Stabilizer Rank Decompositions :pull_aer:`51`
- Added ``basis_gates`` kwarg to ``NoiseModel`` init :pull_aer:`175`.
- Added an optional parameter to ``NoiseModel.as_dict()`` for returning
  dictionaries that can be serialized using the standard json library directly
  :pull_aer:`165`
- Refactor thread management :pull_aer:`50`
- Improve noise transformations :pull_aer:`162`
- Improve error reporting :pull_aer:`160`
- Improve efficiency of parallelization with ``max_memory_mb`` a new parameter
  of ``backend_opts`` :pull_aer:`61`
- Improve u1 performance in ``statevector`` :pull_aer:`123`


Bug Fixes
---------

- Fixed OpenMP clashing problems on macOS for the Terra add-on :pull_aer:`46`




Compatibility Considerations
----------------------------

- Deprecated ``"initial_statevector"`` backend option for ``QasmSimulator`` and
  ``StatevectorSimulator`` :pull_aer:`185`
- Renamed ``"chop_threshold"`` backend option to ``"zero_threshold"`` and
  changed default value to 1e-10 :pull_aer:`185`



Ignis 0.1
=========

New Features
------------

* Quantum volume
* Measurement mitigation using tensored calibrations
* Simultaneous RB has the option to align Clifford gates across subsets
* Measurement correction can produce a new calibration for a subset of qubits



Compatibility Considerations
----------------------------

* RB writes to the minimal set of classical registers (it used to be
  Q[i]->C[i]). This change enables measurement correction with RB.
  Unless users had external analysis code, this will not change outcomes.
  RB circuits from 0.1 are not compatible with 0.1.1 fitters.




Aqua 0.5
========

New Features
------------

* Implementation of the HHL algorithm supporting ``LinearSystemInput``
* Pluggable component ``Eigenvalues`` with variant ``EigQPE``
* Pluggable component ``Reciprocal`` with variants ``LookupRotation`` and
  ``LongDivision``
* Multiple-Controlled U1 and U3 operations ``mcu1`` and ``mcu3``
* Pluggable component ``QFT`` derived from component ``IQFT``
* Summarized the transpiled circuits at the DEBUG logging level
* ``QuantumInstance`` accepts ``basis_gates`` and ``coupling_map`` again.
* Support to use ``cx`` gate for the entanglement in ``RY`` and ``RYRZ``
  variational form (``cz`` is the default choice)
* Support to use arbitrary mixer Hamiltonian in QAOA, allowing use of QAOA
  in constrained optimization problems [arXiv:1709.03489]
* Added variational algorithm base class ``VQAlgorithm``, implemented by
  ``VQE`` and ``QSVMVariational``
* Added ``ising/docplex.py`` for automatically generating Ising Hamiltonian
  from optimization models of DOcplex
* Added ``'basic-dirty-ancilla``' mode for ``mct``
* Added ``mcmt`` for Multi-Controlled, Multi-Target gate
* Exposed capabilities to generate circuits from logical AND, OR, DNF
  (disjunctive normal forms), and CNF (conjunctive normal forms) formulae
* Added the capability to generate circuits from ESOP (exclusive sum of
  products) formulae with optional optimization based on Quine-McCluskey and ExactCover
* Added ``LogicalExpressionOracle`` for generating oracle circuits from
  arbitrary Boolean logic expressions (including DIMACS support) with optional
  optimization capability
* Added ``TruthTableOracle`` for generating oracle circuits from truth-tables
  with optional optimization capability
* Added ``CustomCircuitOracle`` for generating oracle from user specified
  circuits
* Added implementation of the Deutsch-Jozsa algorithm
* Added implementation of the Bernstein-Vazirani algorithm
* Added implementation of the Simon's algorithm
* Added implementation of the Shor's algorithm
* Added optional capability for Grover's algorithm to take a custom
  initial state (as opposed to the default uniform superposition)
* Added capability to create a ``Custom`` initial state using existing
  circuit
* Added the ADAM (and AMSGRAD) optimization algorithm
* Multivariate distributions added, so uncertainty models now have univariate
  and multivariate distribution components
* Added option to include or skip the swaps operations for qft and iqft
  circuit constructions
* Added classical linear system solver ``ExactLSsolver``
* Added parameters ``auto_hermitian`` and ``auto_resize`` to ``HHL`` algorithm
  to support non-Hermitian and non :math:`2^n` sized matrices by default
* Added another feature map, ``RawFeatureVector``, that directly maps feature
  vectors to qubits' states for classification
* ``SVM_Classical`` can now load models trained by ``QSVM``



Bug Fixes
---------

* Fixed ``ising/docplex.py`` to correctly multiply constant values in constraints
* Fixed package setup to correctly identify namespace packages using
  ``setuptools.find_namespace_packages``



Compatibility Considerations
----------------------------

* ``QuantumInstance`` does not take ``memory`` anymore.
* Moved command line and GUI to separate repo
  (``qiskit_aqua_uis``)
* Removed the ``SAT``-specific oracle (now supported by
  ``LogicalExpressionOracle``)
* Changed ``advanced`` mode implementation of ``mct``: using simple ``h`` gates
  instead of ``ch``, and fixing the old recursion step in ``_multicx``
* Components ``random_distributions`` renamed to ``uncertainty_models``
* Reorganized the constructions of various common gates (``ch``, ``cry``,
  ``mcry``, ``mct``, ``mcu1``, ``mcu3``, ``mcmt``, ``logic_and``, and
  ``logic_or``) and circuits (``PhaseEstimationCircuit``,
  ``BooleanLogicCircuits``, ``FourierTransformCircuits``,
  and ``StateVectorCircuits``) under the ``circuits`` directory
* Renamed the algorithm ``QSVMVariational`` to ``VQC``, which stands for
  Variational Quantum Classifier
* Renamed the algorithm ``QSVMKernel`` to ``QSVM``
* Renamed the class ``SVMInput`` to ``ClassificationInput``
* Renamed problem type ``'svm_classification'`` to ``'classification'``
* Changed the type of ``entangler_map`` used in ``FeatureMap`` and
  ``VariationalForm`` to list of lists



IBM Q Provider 0.1
==================

New Features
------------

- This is the first release as a standalone package. If you
  are installing Terra standalone you'll also need to install the
  ``qiskit-ibmq-provider`` package with ``pip install qiskit-ibmq-provider`` if
  you want to use the IBM Q backends.

- Support for non-Qobj format jobs has been removed from
  the provider. You'll have to convert submissions in an older format to
  Qobj before you can submit.



**********
Qiskit 0.8
**********

In Qiskit 0.8 we introduced the Qiskit Ignis element. It also includes the
Qiskit Terra element 0.7.1 release which contains a bug fix for the BasicAer
Python simulator.

Terra 0.7
=========

No Change

Aer 0.1
=======

No Change

Ignis 0.1
=========

This is the first release of Qiskit Ignis.



**********
Qiskit 0.7
**********

In Qiskit 0.7 we introduced Qiskit Aer and combined it with Qiskit Terra.



Terra 0.7
=========

New Features
------------

This release includes several new features and many bug fixes. With this release
the interfaces for circuit diagram, histogram, bloch vectors, and state
visualizations are declared stable. Additionally, this release includes a
defined and standardized bit order/endianness throughout all aspects of Qiskit.
These are all declared as stable interfaces in this release which won't have
breaking changes made moving forward, unless there is appropriate and lengthy
deprecation periods warning of any coming changes.

There is also the introduction of the following new features:

- A new ASCII art circuit drawing output mode
- A new circuit drawing interface off of ``QuantumCircuit`` objects that
  enables calls of ``circuit.draw()`` or ``print(circuit)`` to render a drawing
  of circuits
- A visualizer for drawing the DAG representation of a circuit
- A new quantum state plot type for hinton diagrams in the local matplotlib
  based state plots
- 2 new constructor methods off the ``QuantumCircuit`` class
  ``from_qasm_str()`` and ``from_qasm_file()`` which let you easily create a
  circuit object from OpenQASM
- A new function ``plot_bloch_multivector()`` to plot Bloch vectors from a
  tensored state vector or density matrix
- Per-shot measurement results are available in simulators and select devices.
  These can be accessed by setting the ``memory`` kwarg to ``True`` when
  calling ``compile()`` or ``execute()`` and then accessed using the
  ``get_memory()`` method on the ``Result`` object.
- A ``qiskit.quantum_info`` module with revamped Pauli objects and methods for
  working with quantum states
- New transpile passes for circuit analysis and transformation:
  ``CommutationAnalysis``, ``CommutationTransformation``, ``CXCancellation``,
  ``Decompose``, ``Unroll``, ``Optimize1QGates``, ``CheckMap``,
  ``CXDirection``, ``BarrierBeforeFinalMeasurements``
- New alternative swap mapper passes in the transpiler:
  ``BasicSwap``, ``LookaheadSwap``, ``StochasticSwap``
- More advanced transpiler infrastructure with support for analysis passes,
  transformation passes, a global ``property_set`` for the pass manager, and
  repeat-until control of passes



Compatibility Considerations
----------------------------

As part of the 0.7 release the following things have been deprecated and will
either be removed or changed in a backwards incompatible manner in a future
release. While not strictly necessary these are things to adjust for before the
next release to avoid a breaking change.

- ``plot_circuit()``, ``latex_circuit_drawer()``, ``generate_latex_source()``,
  and ``matplotlib_circuit_drawer()`` from qiskit.tools.visualization are
  deprecated. Instead the ``circuit_drawer()`` function from the same module
  should be used, there are kwarg options to mirror the functionality of all
  the deprecated functions.
- The current default output of ``circuit_drawer()`` (using latex and falling
  back on python) is deprecated and will be changed to just use the ``text``
  output by default in future releases.
- The ``qiskit.wrapper.load_qasm_string()`` and
  ``qiskit.wrapper.load_qasm_file()`` functions are deprecated and the
  ``QuantumCircuit.from_qasm_str()`` and
  ``QuantumCircuit.from_qasm_file()`` constructor methods should be used
  instead.
- The ``plot_barriers`` and ``reverse_bits`` keys in the ``style`` kwarg
  dictionary are deprecated, instead the
  ``qiskit.tools.visualization.circuit_drawer()`` kwargs ``plot_barriers`` and
  ``reverse_bits`` should be used instead.
- The functions ``plot_state()`` and ``iplot_state()`` have been depreciated.
  Instead the functions ``plot_state_*()`` and ``iplot_state_*()`` should be
  called for the visualization method required.
- The ``skip_transpiler`` argumentt has been deprecated from ``compile()`` and
  ``execute()``. Instead you can use the ``PassManager`` directly, just set
  the ``pass_manager`` to a blank ``PassManager`` object with ``PassManager()``
- The ``transpile_dag()`` function ``format`` kwarg for emitting different
  output formats is deprecated, instead you should convert the default output
  ``DAGCircuit`` object to the desired format.
- The unrollers have been deprecated, moving forward only DAG to DAG unrolling
  will be supported.

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


Changes to Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

The largest change made to visualization in the 0.7 release is the removal of
Matplotlib and other visualization dependencies from the project requirements.
This was done to simplify the requirements and configuration required for
installing Qiskit. If you plan to use any visualizations (including all the
jupyter magics) except for the ``text``, ``latex``, and ``latex_source``
output for the circuit drawer you'll you must manually ensure that
the visualization dependencies are installed. You can leverage the optional
requirements to the Qiskit Terra package to do this::

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




Aer 0.1
=======

New Features
------------

Aer provides three simulator backends:

- ``QasmSimulator``: simulate experiments and return measurement outcomes
- ``StatevectorSimulator``: return the final statevector for a quantum circuit
  acting on the all zero state
- ``UnitarySimulator``: return the unitary matrix for a quantum circuit

``noise`` module: contains advanced noise modeling features for the
``QasmSimulator``

- ``NoiseModel``, ``QuantumError``, ``ReadoutError`` classes for simulating a
  Qiskit quantum circuit in the presence of errors
- ``errors`` submodule including functions for generating ``QuantumError``
  objects for the following types of quantum errors: Kraus, mixed unitary,
  coherent unitary, Pauli, depolarizing, thermal relaxation, amplitude damping,
  phase damping, combined phase and amplitude damping
- ``device`` submodule for automatically generating a noise model based on the
  ``BackendProperties`` of a device

``utils`` module:

- ``qobj_utils`` provides functions for directly modifying a Qobj to insert
  special simulator instructions not yet supported through the Qiskit Terra API.


Aqua 0.4
========

New Features
------------

- Programmatic APIs for algorithms and components -- each component can now be
  instantiated and initialized via a single (non-emptY) constructot call
- ``QuantumInstance`` API for algorithm/backend decoupling --
  ``QuantumInstance`` encapsulates a backend and its settings
- Updated documentation and Jupyter Notebooks illustrating the new programmatic
  APIs
- Transparent parallelization for gradient-based optimizers
- Multiple-Controlled-NOT (cnx) operation
- Pluggable algorithmic component ``RandomDistribution``
- Concrete implementations of ``RandomDistribution``:
  ``BernoulliDistribution``, ``LogNormalDistribution``,
  ``MultivariateDistribution``, ``MultivariateNormalDistribution``,
  ``MultivariateUniformDistribution``, ``NormalDistribution``,
  ``UniformDistribution``, and ``UnivariateDistribution``
- Concrete implementations of ``UncertaintyProblem``:
  ``FixedIncomeExpectedValue``, ``EuropeanCallExpectedValue``, and
  ``EuropeanCallDelta``
- Amplitude Estimation algorithm
- Qiskit Optimization: New Ising models for optimization problems exact cover,
  set packing, vertex cover, clique, and graph partition
- Qiskit AI:

  - New feature maps extending the ``FeatureMap`` pluggable interface:
    ``PauliExpansion`` and ``PauliZExpansion``
  - Training model serialization/deserialization mechanism

- Qiskit Finance:

  - Amplitude estimation for Bernoulli random variable: illustration of
    amplitude estimation on a single qubit problem
  - Loading of multiple univariate and multivariate random distributions
  - European call option: expected value and delta (using univariate
    distributions)
  - Fixed income asset pricing: expected value (using multivariate
    distributions)

- The Pauli string in ``Operator`` class is aligned with Terra 0.7. Now the
  order of a n-qubit pauli string is ``q_{n-1}...q{0}`` Thus, the (de)serialier
  (``save_to_dict`` and ``load_from_dict``) in the ``Operator`` class are also
  changed to adopt the changes of ``Pauli`` class.

Compatibility Considerations
----------------------------

- ``HartreeFock`` component of pluggable type ``InitialState`` moved to Qiskit
  Chemistry
- ``UCCSD`` component of pluggable type ``VariationalForm`` moved to Qiskit
  Chemistry


**********
Qiskit 0.6
**********

Terra 0.6
=========

Highlights
----------

This release includes a redesign of internal components centered around a new,
formal communication format (Qobj), along with long awaited features to
improve the user experience as a whole. The highlights, compared to the 0.5
release, are:

- Improvements for inter-operability (based on the Qobj specification) and
  extensibility (facilities for extending Qiskit with new backends in a
  seamless way)
- New options for handling credentials and authentication for the IBM Q
  backends, aimed at simplifying the process and supporting automatic loading
  of user credentials
- A revamp of the visualization utilities: stylish interactive visualizations
  are now available for Jupyter users, along with refinements for the circuit
  drawer (including a matplotlib-based version)
- Performance improvements centered around circuit transpilation: the basis for
  a more flexible and modular architecture have been set, including
  parallelization of the circuit compilation and numerous optimizations


Compatibility Considerations
----------------------------

Please note that some backwards-incompatible changes have been introduced
during this release -- the following notes contain information on how to adapt
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

The managing of credentials for authenticating when using the IBM Q backends has
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
status of submitted jobs to IBM Q backends. Use ``%%qiskit_progress_bar`` to
keep track of the progress of compilation/execution.



**********
Qiskit 0.5
**********

Terra 0.5
=========

Highlights
----------

This release brings a number of improvements to Qiskit, both for the user
experience and under the hood. Please refer to the full changelog for a
detailed description of the changes - the highlights are:

* new ``statevector`` :mod:`simulators <qiskit.backends.local>` and feature and
  performance improvements to the existing ones (in particular to the C++
  simulator), along with a reorganization of how to work with backends focused
  on extensibility and flexibility (using aliases and backend providers)
* reorganization of the asynchronous features, providing a friendlier interface
  for running jobs asynchronously via :class:`Job` instances
* numerous improvements and fixes throughout the Terra as a whole, both for
  convenience of the users (such as allowing anonymous registers) and for
  enhanced functionality (such as improved plotting of circuits)


Compatibility Considerations
----------------------------

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

  The recommended way for opening a connection to the IBM Q API and for using
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

  The top-level functions now also provide
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
