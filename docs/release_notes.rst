%%%%%%%%%%%%%
Release Notes
%%%%%%%%%%%%%


###############
Version History
###############

This table tracks the meta-package versions and the version of each Qiskit element installed:

.. version-history:: **Version History**

.. note::

   For the ``0.7.0``, ``0.7.1``, and ``0.7.2`` meta-package releases the
   :ref:`versioning_strategy` policy was not formalized yet.


###############
Notable Changes
###############

*************
Qiskit 0.28.0
*************

.. _Release Notes_0.18.0:

Terra 0.18.0
============

.. _Release Notes_0.18.0_Prelude:

Prelude
-------

This release includes many new features and bug fixes. The highlights of
this release are the introduction of two new transpiler
passes, :class:`~qiskit.transpiler.passes.BIPMapping` and
:class:`~qiskit.transpiler.passes.DynamicalDecoupling`, which when combined
with the new ``pulse_optimize`` kwarg on the
:class:`~qiskit.transpiler.passes.UnitarySynthesis` pass enables recreating
the Quantum Volume 64 results using the techniques
described in: https://arxiv.org/abs/2008.08571. These new transpiler passes
and options and are also generally applicable to optimizing any circuit.


.. _Release Notes_0.18.0_New Features:

New Features
------------

- The ``measurement_error_mitgation`` kwarg for the
  :class:`~qiskit.utils.QuantumInstance` constructor can now be set to the
  :class:`~qiskit.ignis.mitigation.TensoredMeasFitter` class from
  qiskit-ignis in addition to
  :class:`~qiskit.ignis.mitigation.CompleteMeasFitter` that was already
  supported. If you use :class:`~qiskit.ignis.mitigation.TensoredMeasFitter`
  you will also be able to set the new ``mit_pattern`` kwarg to specify the
  qubits on which to use :class:`~qiskit.ignis.mitigation.TensoredMeasFitter`
  You can refer to the documentation for ``mit_pattern`` in the
  :class:`~qiskit.ignis.mitigation.TensoredMeasFitter` documentation for
  the expected format.

- The decomposition methods for single-qubit gates, specified via the
  ``basis`` kwarg, in
  :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` has been expanded to
  now also include the ``'ZSXX'`` basis, for making use of direct
  :math:`X` gate as well as :math:`\sqrt{X}` gate.

- Added two new passes :class:`~qiskit.transpiler.passes.AlignMeasures` and
  :class:`~qiskit.transpiler.passes.ValidatePulseGates` to the
  :mod:`qiskit.transpiler.passes` module. These passes are a hardware-aware
  optimization, and a validation routine that are used to manage alignment
  restrictions on time allocation of instructions for a backend.

  If a backend has a restriction on the alignment of
  :class:`~qiskit.circuit.Measure` instructions (in terms of quantization in time), the
  :class:`~qiskit.transpiler.passes.AlignMeasures` pass is used to adjust
  delays in a scheduled circuit to ensure that any
  :class:`~qiskit.circuit.Measure` instructions in the circuit
  are aligned given the constraints of the backend. The
  :class:`~qiskit.transpiler.passes.ValidatePulseGates` pass is used to
  check if any custom pulse gates (gates that have a custom pulse definition
  in the :attr:`~qiskit.circuit.QuantumCircuit.calibrations` attribute of
  a :class:`~qiskit.circuit.QuantumCircuit` object) are valid given
  an alignment constraint for the target backend.

  In the built-in :mod:`~qiskit.transpiler.preset_passmangers` used by the
  :func:`~qiskit.compiler.transpile` function, these passes get automatically
  triggered if the alignment constraint, either via the dedicated
  ``timing_constraints`` kwarg on :func:`~qiskit.compiler.transpile` or has an
  ``timing_constraints`` attribute in the
  :class:`~qiskit.providers.models.BackendConfiguration` object of the
  backend being targetted.

  The backends from IBM Quantum Services (accessible via the
  `qiskit-ibmq-provider <https://pypi.org/project/qiskit-ibmq-provider/>`__
  package) will provide the alignment information in the near future.

   For example:

  .. jupyter-execute::

    from qiskit import circuit, transpile
    from qiskit.test.mock import FakeArmonk

    backend = FakeArmonk()

    qc = circuit.QuantumCircuit(1, 1)
    qc.x(0)
    qc.delay(110, 0, unit="dt")
    qc.measure(0, 0)
    qc.draw('mpl')

  .. jupyter-execute::

    qct = transpile(qc, backend, scheduling_method='alap',
                    timing_constraints={'acquire_alignment': 16})
    qct.draw('mpl')

- A new transpiler pass class :class:`qiskit.transpiler.passes.BIPMapping`
  that tries to find the best layout and routing at once by solving a BIP
  (binary integer programming) problem as described in
  `arXiv:2106.06446 <https://arxiv.org/abs/2106.06446>`__ has been added.

  The ``BIPMapping`` pass (named "mapping" to refer to "layout and routing")
  represents the mapping problem as a BIP (binary integer programming)
  problem and relies on CPLEX (``cplex``) to solve the BIP problem.
  The dependent libraries including CPLEX can be installed along with qiskit-terra:

  .. code-block::

    pip install qiskit-terra[bip-mapper]

  Since the free version of CPLEX can solve only small BIP problems, i.e. mapping
  of circuits with less than about 5 qubits, the paid version of CPLEX may be
  needed to map larger circuits.

  The BIP mapper scales badly with respect to the number of qubits or gates.
  For example, it would not work with ``coupling_map`` beyond 10 qubits because
  the BIP solver (CPLEX) could not find any solution within the default time limit.

  Note that, if you want to fix physical qubits to be used in the mapping
  (e.g. running Quantum Volume (QV) circuits), you need to specify ``coupling_map``
  which contains only the qubits to be used.

  Here is a minimal example code to build pass manager to transpile a QV circuit:

  .. code-block:: python

    num_qubits = 4  # QV16
    circ = QuantumVolume(num_qubits=num_qubits)

    backend = ...
    basis_gates = backend.configuration().basis_gates
    coupling_map = CouplingMap.from_line(num_qubits)  # supply your own coupling map

    def _not_mapped(property_set):
        return not property_set["is_swap_mapped"]

    def _opt_control(property_set):
        return not property_set["depth_fixed_point"]

    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
    pm = PassManager()
    # preparation
    pm.append([
        Unroll3qOrMore(),
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        BarrierBeforeFinalMeasurements()
    ])
    # mapping
    pm.append(BIPMapping(coupling_map))
    pm.append(CheckMap(coupling_map))
    pm.append(Error(msg="BIP mapper failed to map", action="raise"),
              condition=_not_mapped)
    # post optimization
    pm.append([
        Depth(),
        FixedPoint("depth"),
        Collect2qBlocks(),
        ConsolidateBlocks(basis_gates=basis_gates),
        UnitarySynthesis(basis_gates),
        Optimize1qGatesDecomposition(basis_gates),
        CommutativeCancellation(),
        UnrollCustomDefinitions(sel, basis_gates),
        BasisTranslator(sel, basis_gates)
    ], do_while=_opt_control)

    transpile_circ = pm.run(circ)

- A new constructor method
  :meth:`~qiskit.pulse.Schedule.initialize_from` was added to the
  :class:`~qiskit.pulse.Schedule` and :class:`~qiskit.pulse.ScheduleBlock`
  classes. This method initializes a new empty schedule which
  takes the attributes from other schedule. For example:

  .. code-block:: python

      sched = Schedule(name='my_sched')
      new_sched = Schedule.initialize_from(sched)

      assert sched.name == new_sched.name

- A new kwarg, ``line_discipline``, has been added to the :func:`~qiskit.tools.job_monitor`
  function. This kwarg enables changing the carriage return characters used in the
  ``job_monitor`` output. The ``line_discipline`` kwarg defaults to ``'\r'``, which is what was
  in use before.

- The abstract ``Pulse`` class (which is the parent class for classes such
  as :class:`~qiskit.pulse.library.Waveform`,
  :class:`~qiskit.pulse.library.Constant`, and
  :class:`~qiskit.pulse.library.Gaussian` now has a new kwarg on the
  constructor, ``limit_amplitude``, which can be set to ``False`` to disable
  the previously hard coded amplitude limit of ``1``. This can also be set as
  a class attribute directly to change the global default for a Pulse class.
  For example::

    from qiskit.pulse.library import Waveform

    # Change the default value of limit_amplitude to False
    Waveform.limit_amplitude = False
    wave = Waveform(2.0 * np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000)))


- A new class, :class:`~qiskit.quantum_info.PauliList`, has been added to
  the :mod:`qiskit.quantum_info` module. This class is used to
  efficiently represent a list of :class:`~qiskit.quantum_info.Pauli`
  operators. This new class inherets from the same parent class as the
  existing :class:`~qiskit.quantum_info.PauliTable` (and therefore can be
  mostly used interchangeably), however it differs from the
  :class:`~qiskit.quantum_info.PauliTable`
  because the :class:`qiskit.quantum_info.PauliList` class
  can handle Z4 phases.

- Added a new transpiler pass, :class:`~qiskit.transpiler.passes.RemoveBarriers`,
  to :mod:`qiskit.transpiler.passes`. This pass is used to remove all barriers in a
  circuit.

- Add a new optimizer class,
  :class:`~qiskit.algorithms.optimizers.SciPyOptimizer`, to the
  :mod:`qiskit.algorithms.optimizers` module. This class is a simple wrapper class
  of the ``scipy.optimize.minimize`` function
  (`documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__)
  which enables the use of all optimization solvers and all
  parameters (e.g. callback) which are supported by ``scipy.optimize.minimize``.
  For example:

  .. code-block:: python

      from qiskit.algorithms.optimizers import SciPyOptimizer

      values = []

      def callback(x):
          values.append(x)

      optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000}, callback=callback)

- The :class:`~qiskit.transpiler.passes.HoareOptimizer` pass has been
  improved so that it can now replace a
  :class:`~qiskit.circuit.ControlledGate` in a circuit with
  with the base gate if all the control qubits are in the
  :math:`|1\rangle` state.

- Added two new methods, :meth:`~qiskit.dagcircuit.DAGCircuit.is_successor` and
  :meth:`~qiskit.dagcircuit.DAGCircuit.is_predecessor`, to the
  :class:`~qiskit.dagcircuit.DAGCircuit` class. These functions are used to check if a node
  is either a successor or predecessor of another node on the
  :class:`~qiskit.dagcircuit.DAGCircuit`.

- A new transpiler pass,
  :class:`~qiskit.transpiler.passes.RZXCalibrationBuilderNoEcho`, was added
  to the :mod:`qiskit.transpiler.passes` module. This pass is similar
  to the existing :class:`~qiskit.transpiler.passes.RZXCalibrationBuilder`
  in that it creates calibrations for an ``RZXGate(theta)``,
  however :class:`~qiskit.transpiler.passes.RZXCalibrationBuilderNoEcho`
  does this without inserting the echo pulses in the pulse schedule. This
  enables exposing the echo in the cross-resonance sequence as gates so that
  the transpiler can simplify them. The
  :class:`~qiskit.transpiler.passes.RZXCalibrationBuilderNoEcho` pass only
  supports the hardware-native direction of the
  :class:`~qiskit.circuit.library.CXGate`.

- A new kwarg, ``wrap``, has been added to the
  :meth:`~qiskit.circuit.QuantumCircuit.compose` method of
  :class:`~qiskit.circuit.QuantumCircuit`. This enables choosing whether
  composed circuits should be wrapped into an instruction or not. By
  default this is ``False``, i.e. no wrapping. For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      circuit = QuantumCircuit(2)
      circuit.h([0, 1])
      other = QuantumCircuit(2)
      other.x([0, 1])
      print(circuit.compose(other, wrap=True))  # wrapped
      print(circuit.compose(other, wrap=False))  # not wrapped

- A new attribute,
  :attr:`~qiskit.providers.models.PulseBackendConfiguration.control_channels`,
  has been added to the
  :class:`~qiskit.providers.models.PulseBackendConfiguration` class. This
  attribute represents the control channels on a backend as a mapping of
  qubits to a list of :class:`~qiskit.pulse.channels.ControlChannel` objects.

- A new kwarg, ``epsilon``, has been added to the constructor for the
  :class:`~qiskit.extensions.Isometry` class and the corresponding
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.isometry`. This kwarg enables
  optionally setting the epsilon tolerance used by an
  :class:`~qiskit.extensions.Isometry` gate. For example::

      import numpy as np
      from qiskit import QuantumRegister, QuantumCircuit

      tolerance = 1e-8
      iso = np.eye(2,2)
      num_q_output = int(np.log2(iso.shape[0]))
      num_q_input = int(np.log2(iso.shape[1]))
      q = QuantumRegister(num_q_output)
      qc = QuantumCircuit(q)

      qc.isometry(iso, q[:num_q_input], q[num_q_input:], epsilon=tolerance)

- Added a transpiler pass,
  :class:`~qiskit.transpiler.passes.DynamicalDecoupling`, to
  :mod:`qiskit.transpiler.passes` for inserting dynamical decoupling sequences
  in idle periods of a circuit (after mapping to physical qubits and
  scheduling). The pass allows control over the sequence of DD gates, the
  spacing between them, and the qubits to apply on. For example:

  .. jupyter-execute::

      from qiskit.circuit import QuantumCircuit
      from qiskit.circuit.library import XGate
      from qiskit.transpiler import PassManager, InstructionDurations
      from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
      from qiskit.visualization import timeline_drawer

      circ = QuantumCircuit(4)
      circ.h(0)
      circ.cx(0, 1)
      circ.cx(1, 2)
      circ.cx(2, 3)
      circ.measure_all()

      durations = InstructionDurations(
          [("h", 0, 50), ("cx", [0, 1], 700), ("reset", None, 10),
           ("cx", [1, 2], 200), ("cx", [2, 3], 300),
           ("x", None, 50), ("measure", None, 1000)]
      )

      dd_sequence = [XGate(), XGate()]

      pm = PassManager([ALAPSchedule(durations),
                        DynamicalDecoupling(durations, dd_sequence)])
      circ_dd = pm.run(circ)
      timeline_drawer(circ_dd)

- The :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.qasm` has a new kwarg, ``encoding``,
  which can be used to optionally set the character encoding of an output QASM
  file generated by the function. This can be set to any valid codec or alias
  string from the Python standard library's
  `codec module <https://docs.python.org/3/library/codecs.html#standard-encodings>`__.

- Added a new class, :class:`~qiskit.circuit.library.EvolvedOperatorAnsatz`,
  to the :mod:`qiskit.circuit.library` module. This library circuit, which
  had previously been located in
  `Qiskit Nature <https://qiskit.org/documentation/nature/>`__ , can be used
  to construct ansatz circuits that consist of time-evolved operators, where
  the evolution time is a variational parameter. Examples of such ansatz
  circuits include ``UCCSD`` class in the ``chemistry`` module of
  Qiskit Nature or the :class:`~qiskit.circuit.library.QAOAAnsatz` class.

- A new fake backend class is available under ``qiskit.test.mock`` for the
  ``ibmq_guadalupe`` backend.  As with the other fake backends, this includes
  a snapshot of calibration data (i.e. ``backend.defaults()``) and error data
  (i.e. ``backend.properties()``) taken from the real system, and can be used
  for local testing, compilation and simulation.

- A new method :meth:`~qiskit.pulse.Schedule.children` for the
  :class:`~qiskit.pulse.Schedule` class has been added. This method is used
  to return the child schedule components of the
  :class:`~qiskit.pulse.Schedule` object as a tuple. It returns nested
  schedules without flattening. This method is equivalent to the private
  ``_children()`` method but has a public and stable interface.

- A new optimizer class,
  :class:`~qiskit.algorithms.optimizers.GradientDescent`, has been added
  to the :mod:`qiskit.algorithms.optimizers` module. This optimizer class
  implements a standard gradient descent optimization algorithm for use
  with quantum variational algorithms, such as
  :class:`~qiskit.algorithms.VQE`.
  For a detailed description and examples on how to use this class, please
  refer to the :class:`~qiskit.algorithms.optimizers.GradientDescent` class
  documentation.

- A new optimizer class, :class:`~qiskit.algorithms.optimizers.QNSPSA`,
  has been added to the :mod:`qiskit.algorithms.optimizers` module. This
  class implements the
  `Quantum Natural SPSA (QN-SPSA) <https://arxiv.org/abs/2103.09232>`__
  algorithm, a generalization of the 2-SPSA algorithm, and estimates the
  Quantum Fisher Information Matrix instead of the Hessian to obtain a
  stochastic estimate of the Quantum Natural Gradient. For examples on
  how to use this new optimizer refer to the
  :class:`~qiskit.algorithms.optimizers.QNSPSA` class documentation.

- A new kwarg, ``second_order``, has been added to the constructor
  of the :class:`~qiskit.algorithms.optimizers.SPSA` class in the
  :mod:`qiskit.algorithms.optimizers` module. When set to ``True`` this
  enables using
  `second-order SPSA <https://ieeexplore.ieee.org/document/657661>`__.
  Second order SPSA, or 2-SPSA, is an extension of the ordinary SPSA algorithm that
  enables estimating the Hessian alongside the gradient, which is used
  to precondition the gradient before the parameter update step. As a
  second-order method, this tries to improve convergence of SPSA.
  For examples on how to use this option refer to the
  :class:`~qiskit.algorithms.optimizers.SPSA` class documentation.

- When using the ``latex`` or ``latex_source`` output mode of
  :meth:`~qiskit.visualization.circuit_drawer` or the
  :meth:`~qiskit.circuit.QuantumCircuit.draw` of
  :class:`~qiskit.circuit.QuantumCircuit` the ``style`` kwarg
  can now be used just as with the ``mpl`` output formatting.
  However, unlike the ``mpl`` output mode only the ``displaytext``
  field will be used when using the ``latex`` or ``latex_source`` output
  modes (because neither supports color).

- When using the ``mpl`` or ``latex`` output methods for the
  :meth:`~qiskit.visualization.circuit_drawer` function or the
  :meth:`~qiskit.circuit.QuantumCircuit.draw` of
  :class:`~qiskit.circuit.QuantumCircuit`, you can now use math mode
  formatting for text and set color formatting (``mpl`` only)
  by setting the ``style`` kwarg as a dict
  with a user-generated name or label. For example, to add subscripts and to
  change a gate color:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      from qiskit.circuit.library import HGate
      qc = QuantumCircuit(3)
      qc.append(HGate(label='h1'), [0])
      qc.append(HGate(label='h2'), [1])
      qc.append(HGate(label='h3'), [2])
      qc.draw('mpl', style={'displaytext': {'h1': 'H_1', 'h2': 'H_2', 'h3': 'H_3'},
          'displaycolor': {'h2': ('#EEDD00', '#FF0000')}})

- Added three new classes,
  :class:`~qiskit.circuit.library.CDKMRippleCarryAdder`,
  :class:`~qiskit.circuit.library.ClassicalAdder` and
  :class:`~qiskit.circuit.library.DraperQFTAdder`, to the
  :mod:`qiskit.circuit.library` module. These new circuit classes are used to
  perform classical addition of two equally-sized qubit registers. For two
  registers :math:`|a\rangle_n` and :math:`|b\rangle_n` on :math:`n`
  qubits, the three new classes perform the operation:

  .. math::

    |a\rangle_n |b\rangle_n \mapsto |a\rangle_n |a + b\rangle_{n + 1}.

  For example::

    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import CDKMRippleCarryAdder
    from qiskit.quantum_info import Statevector

    # a encodes |01> = 1
    a = QuantumCircuit(2)
    a.x(0)

    # b encodes |10> = 2
    b = QuantumCircuit(2)
    b.x(1)

    # adder on 2-bit numbers
    adder = CDKMRippleCarryAdder(2)

    # add the state preparations to the front of the circuit
    adder.compose(a, [0, 1], inplace=True, front=True)
    adder.compose(b, [2, 3], inplace=True, front=True)

    # simulate and get the state of all qubits
    sv = Statevector(adder)
    counts = sv.probabilities_dict()
    state = list(counts.keys())[0]  # we only have a single state

    # skip the input carry (first bit) and the register |a> (last two bits)
    result = state[1:-2]
    print(result)  # '011' = 3 = 1 + 2

- Added two new classes,
  :class:`~qiskit.circuit.library.RGQFTMultiplier` and
  :class:`~qiskit.circuit.library.HRSCumulativeMultiplier`, to the
  :mod:`qiskit.circuit.library` module. These classes are used to perform
  classical multiplication of two equally-sized qubit registers. For two
  registers :math:`|a\rangle_n` and :math:`|b\rangle_n` on :math:`n`
  qubits, the two new classes perform the operation

  .. math::

    |a\rangle_n |b\rangle_n |0\rangle_{2n} \mapsto |a\rangle_n |b\rangle_n |a \cdot b\rangle_{2n}.

  For example::

    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RGQFTMultiplier
    from qiskit.quantum_info import Statevector

    num_state_qubits = 2

    # a encodes |11> = 3
    a = QuantumCircuit(num_state_qubits)
    a.x(range(num_state_qubits))

    # b encodes |11> = 3
    b = QuantumCircuit(num_state_qubits)
    b.x(range(num_state_qubits))

    # multiplier on 2-bit numbers
    multiplier = RGQFTMultiplier(num_state_qubits)

    # add the state preparations to the front of the circuit
    multiplier.compose(a, [0, 1], inplace=True, front=True)
    multiplier.compose(b, [2, 3], inplace=True, front=True)

    # simulate and get the state of all qubits
    sv = Statevector(multiplier)
    counts = sv.probabilities_dict(decimals=10)
    state = list(counts.keys())[0]  # we only have a single state

    # skip both input registers
    result = state[:-2*num_state_qubits]
    print(result)  # '1001' = 9 = 3 * 3

- The :class:`~qiskit.circuit.Delay` class now can accept a
  :class:`~qiskit.circuit.ParameterExpression` or
  :class:`~qiskit.circuit.Parameter` value for the ``duration`` kwarg on its
  constructor and for its :attr:`~qiskit.circuit.Delay.duration` attribute.

  For example::

      idle_dur = Parameter('t')
      qc = QuantumCircuit(1, 1)
      qc.x(0)
      qc.delay(idle_dur, 0, 'us')
      qc.measure(0, 0)
      print(qc)  # parameterized delay in us (micro seconds)

      # assign before transpilation
      assigned = qc.assign_parameters({idle_dur: 0.1})
      print(assigned)  # delay in us
      transpiled = transpile(assigned, some_backend_with_dt)
      print(transpiled)  # delay in dt

      # assign after transpilation
      transpiled = transpile(qc, some_backend_with_dt)
      print(transpiled)  # parameterized delay in dt
      assigned = transpiled.assign_parameters({idle_dur: 0.1})
      print(assigned)  # delay in dt

- A new binary serialization format, `QPY`, has been introduced. It is
  designed to be a fast binary serialization format that is backwards
  compatible (QPY files generated with older versions of Qiskit can be
  loaded by newer versions of Qiskit) that is native to Qiskit. The QPY
  serialization tooling is available  via the
  :mod:`qiskit.circuit.qpy_serialization` module. For example, to generate a
  QPY file::

    from datetime import datetime

    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit import qpy_serialization

    qc = QuantumCircuit(
      2, metadata={'created_at': datetime.utcnow().isoformat()}
    )
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    circuits = [qc] * 5

    with open('five_bells.qpy', 'wb') as qpy_file:
        qpy_serialization.dump(circuits, qpy_file)

  Then the five circuits saved in the QPY file can be loaded with::

    from qiskit.circuit.qpy_serialization

    with open('five_bells.qpy', 'rb') as qpy_file:
        circuits = qpy_serialization.load(qpy_file)

  The QPY file format specification is available in the module documentation.

- The :class:`~qiskit.quantum_info.TwoQubitBasisDecomposer` class has been
  updated to perform pulse optimal decompositions for a basis with CX, √X, and
  virtual Rz gates as described in
  https://arxiv.org/pdf/2008.08571. Pulse optimal here means that
  the duration of gates between the CX gates of the decomposition is
  reduced in exchange for possibly more local gates before or after
  all the CX gates such that, when composed into a circuit, there is the
  possibility of single qubit compression with neighboring gates
  reducing the overall sequence duration.

  A new keyword argument, ```pulse_optimize``, has been added to the constructor
  for :class:`~qiskit.quantum_info.TwoQubitBasisDecomposer` to control this:

  * ``None``: Attempt pulse optimal decomposition. If a pulse optimal
    decomposition is unknown for the basis of the decomposer, drop
    back to the standard decomposition without warning. This is the default
    setting.
  * ``True``: Attempt pulse optimal decomposition. If a pulse optimal
    decomposition is unknown for the basis of the decomposer, raise
    `QiskitError`.
  * ``False``: Do not attempt pulse optimal decomposition.

  For example:

  .. code-block:: python

       from qiskit.quantum_info import TwoQubitBasisDecomposer
       from qiskit.circuit.library import CXGate
       from qiskit.quantum_info import random_unitary

       unitary_matrix = random_unitary(4)

       decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True)
       circuit = decomposer(unitary_matrix)

- The transpiler pass :class:`~qiskit.transpiler.passes.synthesis.UnitarySynthesis`
  located in :mod:`qiskit.transpiler.passes` has been updated to support performing
  pulse optimal decomposition. This is done primarily with the the
  ``pulse_optimize`` keyword argument which was added to the constructor and
  used to control whether pulse optimal synthesis is performed. The behavior of
  this kwarg mirrors the ``pulse_optimize`` kwarg in the
  :class:`~qiskit.quantum_info.TwoQubitBasisDecomposer` class's constructor.
  Additionally, the constructor has another new keyword argument, ``synth_gates``,
  which is used to specify the list of gate names over which synthesis should be attempted. If
  ``None`` and ``pulse_optimize`` is ``False`` or ``None``, use ``"unitary"``.
  If `None` and `pulse_optimize` is ``True``, use ``"unitary"`` and ``"swap"``.
  Since the direction of the CX gate in the synthesis is arbitrary, another
  keyword argument, ``natural_direction``, is added to consider first
  a coupling map and then :class:`~qiskit.circuit.library.CXGate` durations in
  choosing for which direction of CX to generate the synthesis.

  .. code-block:: python

      from qiskit.circuit import QuantumCircuit
      from qiskit.transpiler import PassManager, CouplingMap
      from qiskit.transpiler.passes import TrivialLayout, UnitarySynthesis
      from qiskit.test.mock import FakeVigo
      from qiskit.quantum_info.random import random_unitary

      backend = FakeVigo()
      conf = backend.configuration()
      coupling_map = CouplingMap(conf.coupling_map)
      triv_layout_pass = TrivialLayout(coupling_map)
      circ = QuantumCircuit(2)
      circ.unitary(random_unitary(4), [0, 1])
      unisynth_pass = UnitarySynthesis(
          basis_gates=conf.basis_gates,
          coupling_map=None,
          backend_props=backend.properties(),
          pulse_optimize=True,
          natural_direction=True,
          synth_gates=['unitary'])
      pm = PassManager([triv_layout_pass, unisynth_pass])
      optimal_circ = pm.run(circ)

- A new basis option, ``'XZX'``, was added for the ``basis`` argument
  :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` class.

- Added a new method, :meth:`~qiskit.circuit.QuantumCircuit.get_instructions`,
  was added to the :class:`~qiskit.circuit.QuantumCircuit` class. This method
  is used to return all :class:`~qiskit.circuit.Instruction` objects in the
  circuit which have a :attr:`~qiskit.circuit.Instruction.name` that matches
  the provided ``name`` argument along with its associated ``qargs`` and
  ``cargs`` lists of :class:`~qiskit.circuit.Qubit` and
  :class:`~qiskit.circuit.Clbit` objects.

- A new optional extra ``all`` has been added to the qiskit-terra package.
  This enables installing all the optional requirements with a single
  extra, for example: ``pip install 'qiskit-terra[all]'``, Previously, it
  was necessary to list all the extras individually to install all the
  optional dependencies simultaneously.

- Added two new classes :class:`~qiskit.result.ProbDistribution` and
  :class:`~qiskit.result.QuasiDistribution` for dealing with probability
  distributions and quasiprobability distributions respectively. These objects
  both are dictionary subclasses that add additional methods for working
  with probability and quasiprobability distributions.

- Added a new :attr:`~qiskit.algorithms.optimizers.Optimizer.settings`
  property to the :class:`~qiskit.algorithms.optimizers.Optimizer` abstract
  base class that all the optimizer classes in the
  :mod:`qiskit.algorithms.optimizers` module are based on. This property
  will return a Python dictionary of the settings for the optimizer
  that can be used to instantiate another instance of the same optimizer
  class. For example::

    from qiskit.algorithms.optimizers import GradientDescent

    optimizer = GradientDescent(maxiter=10, learning_rate=0.01)
    settings = optimizer.settings
    new_optimizer = GradientDescent(**settings)

  The ``settings`` dictionary is also potentially useful for serializing
  optimizer objects using JSON or another serialization format.

- A new function, :func:`~qiskit.user_config.set_config`, has been added
  to the :mod:`qiskit.user_config` module. This function enables setting
  values in a user config from the Qiskit API. For example::

    from qiskit.user_config import set_config
    set_config("circuit_drawer", "mpl", section="default", file="settings.conf")

  which will result in adding a value of ``circuit_drawer = mpl`` to the
  ``default`` section in the ``settings.conf`` file.

  If no ``file_path`` argument is specified, the currently used path to the
  user config file (either the value of the ``QISKIT_SETTINGS`` environment
  variable if set or the default location ``~/.qiskit/settings.conf``) will be
  updated. However, changes to the existing config file will not be reflected in
  the current session since the config file is parsed at import time.

- Added a new state class, :class:`~qiskit.quantum_info.StabilizerState`,
  to the :mod:`qiskit.quantum_info` module. This class represents a
  stabilizer simulator state using the convention from
  `Aaronson and Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`__.

- Two new options, ``'value'`` and ``'value_desc'`` were added to the
  ``sort`` kwarg of the :func:`qiskit.visualization.plot_histogram` function.
  When ``sort`` is set to either of these options the output visualization
  will sort the x axis based on the maximum probability for each bitstring.
  For example:

  .. jupyter-execute::

    from qiskit.visualization import plot_histogram

    counts = {
      '000': 5,
      '001': 25,
      '010': 125,
      '011': 625,
      '100': 3125,
      '101': 15625,
      '110': 78125,
      '111': 390625,
    }
    plot_histogram(counts, sort='value')


.. _Release Notes_0.18.0_Known Issues:

Known Issues
------------

- When running :func:`~qiskit.tools.parallel_map` (and functions that
  internally call :func:`~qiskit.tools.parallel_map` such as
  :func:`~qiskit.compiler.transpile` and :func:`~qiskit.compiler.assemble`)
  on Python 3.9 with ``QISKIT_PARALLEL`` set to True in some scenarios it is
  possible for the program to deadlock and never finish running. To avoid
  this from happening the default for Python 3.9 was changed to not run in
  parallel, but if ``QISKIT_PARALLEL`` is explicitly enabled then this
  can still occur.


.. _Release Notes_0.18.0_Upgrade Notes:

Upgrade Notes
-------------

- The minimum version of the `retworkx <https://pypi.org/project/retworkx/>`_ dependency
  was increased to version `0.9.0`. This was done to use new APIs introduced in that release
  which improved the performance of some transpiler passes.

- The default value for ``QISKIT_PARALLEL`` on Python 3.9 environments has
  changed to ``False``, this means that when running on Python 3.9 by default
  multiprocessing will not be used. This was done to avoid a potential
  deadlock/hanging issue that can occur when running multiprocessing on
  Python 3.9 (see the known issues section for more detail). It is still
  possible to manual enable it by explicitly setting the ``QISKIT_PARALLEL``
  environment variable to ``TRUE``.

- The existing fake backend classes in ``qiskit.test.mock`` now strictly
  implement the :class:`~qiskit.providers.BackendV1` interface. This means
  that if you were manually constructing :class:`~qiskit.qobj.QasmQobj` or
  :class:`~qiskit.qobj.PulseQobj` object for use with the ``run()`` method
  this will no longer work. The ``run()`` method only accepts
  :class:`~qiskit.circuit.QuantumCircuit` or :class:`~qiskit.pulse.Schedule`
  objects now. This was necessary to enable testing of new backends
  implemented without qobj which previously did not have any testing inside
  qiskit terra. If you need to leverage the fake backends with
  :class:`~qiskit.qobj.QasmQobj` or :class:`~qiskit.qobj.PulseQobj` new
  fake legacy backend objects were added to explicitly test the legacy
  providers interface. This will be removed after the legacy interface is
  deprecated and removed. Moving forward new fake backends will only
  implement the :class:`~qiskit.providers.BackendV1` interface and will not
  add new legacy backend classes for new fake backends.

- When creating a :class:`~qiskit.quantum_info.Pauli` object with an invalid
  string label, a :class:`~qiskit.exceptions.QiskitError` is now raised.
  This is a change from previous releases which would raise an
  ``AttributeError`` on an invalid string label. This change was made to
  ensure the error message is more informative and distinct from a generic
  ``AttributeError``.

- The output program representation from the pulse builder
  (:func:`qiskit.pulse.builder.build`) has changed from a
  :class:`~qiskit.pulse.Schedule` to a
  :class:`~qiskit.pulse.ScheduleBlock`. This new representation disables
  some timing related operations such as shift and insert. However, this
  enables parameterized instruction durations within the builder context.
  For example:

  .. code-block:: python

      from qiskit import pulse
      from qiskit.circuit import Parameter

      dur = Parameter('duration')

      with pulse.build() as sched:
          with pulse.align_sequential():
              pulse.delay(dur, pulse.DriveChannel(1))
              pulse.play(pulse.Gaussian(dur, 0.1, dur/4), pulse.DriveChannel(0))

      assigned0 = sched.assign_parameters({dur: 100})
      assigned1 = sched.assign_parameters({dur: 200})

  You can directly pass the duration-assigned schedules to the assembler (or backend),
  or you can attach them to your quantum circuit as pulse gates.

- The `tweedledum <https://pypi.org/project/tweedledum/>`__ library which
  was previously an optional dependency has been made a requirement. This
  was done because of the wide use of the
  :class:`~qiskit.circuit.library.PhaseOracle` (which depends on
  having tweedledum installed) with several algorithms
  from :mod:`qiskit.algorithms`.

- The optional extra ``full-featured-simulators`` which could previously used
  to install ``qiskit-aer`` with something like
  ``pip install qiskit-terra[full-featured-simulators]`` has been removed
  from the qiskit-terra package. If this was being used to install
  ``qiskit-aer`` with ``qiskit-terra`` instead you should rely on the
  `qiskit <https://pypi.org/project/qiskit/>`__ metapackage or just install
  qiskit-terra and qiskit-aer together with
  ``pip install qiskit-terra qiskit-aer``.

- A new requirement `symengine <https://pypi.org/project/symengine>`__ has
  been added for Linux (on x86_64, aarch64, and ppc64le) and macOS users
  (x86_64 and arm64). It is an optional dependency on Windows (and available
  on PyPi as a precompiled package for 64bit Windows) and other
  architectures. If it is installed it provides significantly improved
  performance for the evaluation of :class:`~qiskit.circuit.Parameter` and
  :class:`~qiskit.circuit.ParameterExpression` objects.

- All library circuit classes, i.e. all :class:`~qiskit.circuit.QuantumCircuit` derived
  classes in :mod:`qiskit.circuit.library`, are now wrapped in a
  :class:`~qiskit.circuit.Instruction` (or :class:`~qiskit.circuit.Gate`, if they are
  unitary). For example, importing and drawing the :class:`~qiskit.circuit.library.QFT`
  circuit:

  .. code-block::python

      from qiskit.circuit.library import QFT

      qft = QFT(3)
      print(qft.draw())

  before looked like

  .. code-block::

                                                ┌───┐
      q_0: ────────────────────■────────■───────┤ H ├─X─
                         ┌───┐ │        │P(π/2) └───┘ │
      q_1: ──────■───────┤ H ├─┼────────■─────────────┼─
           ┌───┐ │P(π/2) └───┘ │P(π/4)                │
      q_2: ┤ H ├─■─────────────■──────────────────────X─
           └───┘

  and now looks like

  .. code-block::

           ┌──────┐
      q_0: ┤0     ├
           │      │
      q_1: ┤1 QFT ├
           │      │
      q_2: ┤2     ├
           └──────┘

  To obtain the old circuit, you can call the
  :meth:`~qiskit.circuit.QuantumCircuit.decompose` method on the circuit

  .. code-block::python

      from qiskit.circuit.library import QFT

      qft = QFT(3)
      print(qft.decompose().draw())

  This change was primarily made for consistency as before this release some
  circuit classes in :mod:`qiskit.circuit.library` were previously wrapped
  in an :class:`~qiskit.circuit.Instruction` or :class:`~qiskit.circuit.Gate`
  but not all.


.. _Release Notes_0.18.0_Deprecation Notes:

Deprecation Notes
-----------------

- The class :class:`qiskit.exceptions.QiskitIndexError`
  is deprecated and will be removed in a future release. This exception
  was not actively being used by anything in Qiskit, if you were using
  it you can create a custom exception class to replace it.

- The kwargs ``epsilon`` and ``factr`` for the
  :class:`qiskit.algorithms.optimizers.L_BFGS_B` constructor
  and ``factr`` kwarg of the :class:`~qiskit.algorithms.optimizers.P_BFGS`
  optimizer class are deprecated and will be removed in a future release. Instead, please
  use the ``eps`` karg instead of ``epsilon``. The ``factr`` kwarg is
  replaced with ``ftol``. The relationship between the two is
  :code:`ftol = factr * numpy.finfo(float).eps`. This change was made
  to be consistent with the usage of the ``scipy.optimize.minimize``
  functions ``'L-BFGS-B'`` method. See the:
  ``scipy.optimize.minimize(method='L-BFGS-B')``
  `documentation <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`__
  for more information on how these new parameters are used.

- The legacy providers interface, which consisted of the
  :class:`qiskit.providers.BaseBackend`, :class:`qiskit.providers.BaseJob`,
  and :class:`qiskit.providers.BaseProvider` abstract classes, has been
  deprecated and will be removed in a future release. Instead you should use
  the versioned interface, which the current abstract class versions are
  :class:`qiskit.providers.BackendV1`, :class:`qiskit.providers.JobV1`, and
  :class:`qiskit.providers.ProviderV1`. The V1 objects are mostly backwards
  compatible to ease migration from the legacy interface to the versioned
  one. However, expect future versions of the abstract interfaces to diverge
  more. You can refer to the :mod:`qiskit.providers` documentation for
  more high level details about the versioned interface.

- The ``condition`` kwarg to the
  :class:`~qiskit.dagcircuit.DAGDepNode` constructor along with the
  corresponding :attr:`~qiskit.dagcircuit.DAGDepNode.condition` attribute
  of the :class:`~qiskit.dagcircuit.DAGDepNode` have been deprecated and
  will be removed in a future release. Instead, you can access the
  ``condition`` of a ``DAGDepNode`` if the node is of type ``op``, by using
  ``DAGDepNode.op.condition``.

- The :attr:`~qiskit.dagcircuit.DAGNode.condition` attribute of the
  :class:`~qiskit.dagcircuit.DAGNode` class has been deprecated and
  will be removed in a future release. Instead, you can access the
  ``condition`` of a ``DAGNode`` object if the node is of type ``op``, by
  using ``DAGNode.op.condition``.

- The pulse builder (:func:`qiskit.pulse.builder.build`) syntax
  :func:`qiskit.pulse.builder.inline` is deprecated and will be removed in a
  future release. Instead of using this context, you can just remove alignment
  contexts within the inline context.

- The pulse builder (:func:`qiskit.pulse.builder.build`) syntax
  :func:`qiskit.pulse.builder.pad` is deprecated and will be removed in a
  future release. This was done because the :class:`~qiskit.pulse.ScheduleBlock`
  now being returned by the pulse builder  doesn't support the ``.insert`` method
  (and there is no insert syntax in the builder). The use of timeslot placeholders
  to block the insertion of other instructions is no longer necessary.


.. _Release Notes_0.18.0_Bug Fixes:

Bug Fixes
---------

- The :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` and
  :class:`~qiskit.quantum_info.TwoQubitBasisDecomposer` classes for
  one and two qubit gate synthesis have been improved to tighten up
  tolerances, improved repeatability and simplification, and fix
  several global-phase-tracking bugs.

- Fixed an issue in the assignment of the :attr:`~qiskit.circuit.Gate.name`
  attribute to :class:`~qiskit.circuit.Gate` generated by multiple calls to
  the :meth:`~qiskit.circuit.Gate.inverse`` method. Prior to this fix
  when the :meth:`~qiskit.circuit.Gate.inverse`` was called it would
  unconditionally append ``_dg`` on each call to inverse. This has
  been corrected so on a second call of
  :meth:`~qiskit.circuit.Gate.inverse`` the ``_dg`` suffix is now removed.

- Fixes the triviality check conditions of :class:`~qiskit.circuit.library.CZGate`,
  :class:`~qiskit.circuit.library.CRZGate`, :class:`~qiskit.circuit.library.CU1Gate`
  and :class:`~qiskit.circuit.library.MCU1Gate` in the
  :class:`~qiskit.transpiler.passes.HoareOptimizer` pass. Previously, in some cases
  the optimizer would remove these gates breaking the semantic equivalence of
  the transformation.

- Fixed an issue when converting a :class:`~qiskit.opflow.list_ops.ListOp`
  object of :class:`~qiskit.opflow.primitive_ops.PauliSumOp` objects using
  :class:`~qiskit.opflow.expectations.PauliExpectation` or
  :class:`~qiskit.opflow.expectations.AerPauliExpectation`. Previously, it would raise
  a warning about it converting to a Pauli representation which is
  potentially expensive. This has been fixed by instead of internally
  converting the :class:`~qiskit.opflow.list_ops.ListOp` to a
  :class:`~qiskit.opflow.list_ops.SummedOp` of
  :class:`~qiskit.opflow.primitive_ops.PauliOp` objects, it now creates
  a :class:`~qiskit.opflow.primitive_ops.PauliSumOp` which is more
  efficient.
  Fixed `#6159 <https://github.com/Qiskit/qiskit-terra/issues/6159>`__

- Fixed an issue with the :class:`~qiskit.circuit.library.NLocal` class
  in the :mod:`qiskit.circuit.library` module where it wouldn't properly
  raise an exception at object initialization if an invalid type was
  used for the ``reps`` kwarg which would result in an unexpected runtime
  error later. A ``TypeError`` will now be properly raised if the ``reps``
  kwarg is not an ``int`` value.
  Fixed `#6515 <https://github.com/Qiskit/qiskit-terra/issues/6515>`__

- Fixed an issue where the :class:`~qiskit.circuit.library.TwoLocal` class
  in the :mod:`qiskit.circuit.library` module did not accept numpy integer
  types (e.g. ``numpy.int32``, ``numpy.int64``, etc) as a valid input for
  the ``entanglement`` kwarg.
  Fixed `#6455 <https://github.com/Qiskit/qiskit-terra/issues/6455>`__

- When loading an OpenQASM2 file or string with the
  :meth:`~qiskit.circuitQuantumCircuit.from_qasm_file` or
  :meth:`~qiskit.circuitQuantumCircuit.from_qasm_str` constructors for the
  :class:`~qiskit.circuit.QuantumCircuit` class, if the OpenQASM2 circuit
  contains an instruction with the name ``delay`` this will be mapped to
  a :class:`qiskit.circuit.Delay` instruction. For example:

  .. jupyter-execute::

     from qiskit import QuantumCircuit

     qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     opaque delay(time) q;
     qreg q[1];
     delay(172) q[0];
     u3(0.1,0.2,0.3) q[0];
     """
     circuit = QuantumCircuit.from_qasm_str(qasm)
     circuit.draw()

  Fixed `#6510 <https://github.com/Qiskit/qiskit-terra/issues/6510>`__

- Fixed an issue with addition between
  :class:`~qiskit.opflow.primitive_ops.PauliSumOp` objects that had
  :class:`~qiskit.circuit.ParameterExpression` coefficients. Previously
  this would result in a ``QiskitError`` exception being raised because
  the addition of the :class:`~qiskit.circuit.ParameterExpression` was
  not handled correctly. This has been fixed so that addition can be
  performed between :class:`~qiskit.opflow.primitive_ops.PauliSumOp`
  objects with :class:`~qiskit.circuit.ParameterExpression` coefficients.

- Fixed an issue with the initialization of the
  :class:`~qiskit.algorithms.AmplificationProblem` class. The
  ``is_good_state`` kwarg was a required field but incorrectly being treated
  as optional (and documented as such). This has been fixed and also
  updated so unless the input ``oracle`` is a
  :class:`~qiskit.circuit.library.PhaseOracle` object (which provides it's
  on evaluation method) the field is required and will raise a ``TypeError``
  when constructed without ``is_good_state``.

- Fixed an issue where adding a control to a
  :class:`~qiskit.circuit.ControlledGate` with open controls would unset
  the inner open controls.
  Fixes `#5857 <https://github.com/Qiskit/qiskit-terra/issues/5857>`__

- Fixed an issue with the
  :meth:`~qiskit.opflow.expectations.PauliExpectation.convert` method of
  the :class:`~qiskit.opflow.expectations.PauliExpectation` class where
  calling it on an operator that was non-Hermitian would return
  an incorrect result.
  Fixed `#6307 <https://github.com/Qiskit/qiskit-terra/issues/6307>`__

- Fixed an issue with the :func:`qiskit.pulse.transforms.inline_subroutines`
  function which would previously incorrectly not remove all the nested
  components when called on nested schedules.
  Fixed `#6321 <https://github.com/Qiskit/qiskit-terra/issues/6321>`__

- Fixed an issue when passing a partially bound callable created with
  the Python standard library's ``functools.partial()`` function as the
  ``schedule`` kwarg to the
  :meth:`~qiskit.pulse.InstructionScheduleMap.add` method of the
  :class:`~qiskit.pulse.InstructionScheduleMap` class, which would previously
  result in an error.
  Fixed `#6278 <https://github.com/Qiskit/qiskit-terra/issues/6278>`__

- Fixed an issue with the :class:`~qiskit.circuit.library.PiecewiseChebyshev`
  when setting the
  :attr:`~qiskit.circuit.library.PiecewiseChebyshev.breakpoints` to ``None``
  on an existing object was incorrectly being treated as a breakpoint. This
  has been corrected so that when it is set to ``None`` this will switch back
  to the default  behavior of approximating over the full interval.
  Fixed `#6198 <https://github.com/Qiskit/qiskit-terra/issues/6198>`__

- Fixed an issue with the
  :meth:`~qiskit.circuit.QuantumCircuit.num_connected_components` method of
  :class:`~qiskit.circuit.QuantumCircuit` which was returning the incorrect
  number of components when the circuit contains two or more gates conditioned
  on classical registers.
  Fixed `#6477 <https://github.com/Qiskit/qiskit-terra/issues/6477>`__

- Fixed an issue with the :mod:`qiskit.opflow.expectations` module
  where coefficients of a statefunction were not being multiplied correctly.
  This also fixed the calculations
  of Gradients and QFIs when using the
  :class:`~qiskit.opflow.expectations.PauliExpectation` or
  :class:`~qiskit.opflow.expectations.AerPauliExpectation` classes. For
  example, previously::

      from qiskit.opflow import StateFn, I, One

      exp = ~StateFn(I) @ (2 * One)

  evaluated to ``2``
  for :class:`~qiskit.opflow.expectations.AerPauliExpectation` and to ``4``
  for other expectation converters. Since ``~StateFn(I) @ (2 * One)`` is a
  shorthand notation for ``~(2 * One) @ I @ (2 * One)``, the now correct
  coefficient of ``4`` is returned for all expectation converters.
  Fixed `#6497 <https://github.com/Qiskit/qiskit-terra/issues/6497>`__

- Fixed the bug that caused :meth:`~qiskit.opflow.PauliOp.to_circuit` to fail when
  :class:`~qiskit.opflow.PauliOp` had a phase. At the same time, it was made more efficient to
  use :class:`~qiskit.circuit.library.generalized_gates.PauliGate`.

- Fixed an issue where the QASM output generated by the
  :meth:`~qiskit.circuit.QuantumCircuit.qasm` method of
  :class:`~qiskit.circuit.QuantumCircuit` for composite gates such as
  :class:`~qiskit.circuit.library.MCXGate` and its variants (
  :class:`~qiskit.circuit.library.MCXGrayCode`,
  :class:`~qiskit.circuit.library.MCXRecursive`, and
  :class:`~qiskit.circuit.library.MCXVChain`) would be incorrect. Now if a
  :class:`~qiskit.circuit.Gate` in the circuit is not present in
  ``qelib1.inc``, its definition is added to the output QASM string.
  Fixed `#4943 <https://github.com/Qiskit/qiskit-terra/issues/4943>`__ and
  `#3945 <https://github.com/Qiskit/qiskit-terra/issues/3945>`__

- Fixed an issue with the :func:`~qiskit.visualization.circuit_drawer`
  function and :meth:`~qiskit.circuit.QuantumCircuit.draw` method of
  :class:`~qiskit.circuit.QuantumCircuit`. When using the ``mpl`` or ``latex``
  output modes, with the ``cregbundle`` kwarg set to ``False`` and the
  ``reverse_bits`` kwarg set to ``True``, the bits in the classical registers
  displayed in the same order as when ``reverse_bits`` was set to ``False``.

- Fixed an issue when using the :class:`qiskit.extensions.Initialize`
  instruction which was not correctly setting the global phase of the
  synthesized definition when constructed.
  Fixed `#5320 <https://github.com/Qiskit/qiskit-terra/issues/5230>`__

- Fixed an issue where the bit-order in
  :meth:`qiskit.circuit.library.PhaseOracle.evaluate_bitstring` did not agree
  with the order of the measured bitstring. This fix also affects the
  execution of the :class:`~qiskit.algorithms.Grover` algorithm class if the
  oracle is specified as a :class:`~qiskit.circuit.library.PhaseOracle`, which
  now will now correctly identify the correct bitstring.
  Fixed `#6314 <https://github.com/Qiskit/qiskit-terra/issues/6314>`__

- Fixes a bug in :func:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition`
  previously causing certain short sequences of gates to erroneously not be
  rewritten.

- Fixed an issue in the :meth:`qiskit.opflow.gradients.Gradient.gradient_wrapper`
  method with the gradient calculation. Previously, if the operator was
  not diagonal an incorrect result would be returned in some situations.
  This has been fixed by using an expectation converter to ensure the
  result is always correct.

- Fixed an issue with the :func:`~qiskit.visualization.circuit_drawer`
  function and :meth:`~qiskit.circuit.QuantumCircuit.draw` method of
  :class:`~qiskit.circuit.QuantumCircuit` with all output modes
  where it would incorrectly render a custom instruction that includes
  classical bits in some circumstances.
  Fixed `#3201 <https://github.com/Qiskit/qiskit-terra/issues/3201>`__,
  `#3202 <https://github.com/Qiskit/qiskit-terra/issues/3202>`__, and
  `#6178 <https://github.com/Qiskit/qiskit-terra/issues/6178>`__

- Fixed an issue in :func:`~qiskit.visualization.circuit_drawer` and the
  :meth:`~qiskit.circuit.QuantumCircuit.draw` method of the
  :class:`~qiskit.circuit.QuantumCircuit` class when using the ``mpl``
  output mode, controlled-Z Gates were incorrectly drawn as asymmetrical.
  Fixed `#5981 <https://github.com/Qiskit/qiskit-terra/issues/5981>`__

- Fixed an issue with the
  :class:`~qiskit.transpiler.passes.OptimizeSwapBeforeMeasure` transpiler pass
  where in some situations a :class:`~qiskit.circuit.library.SwapGate`
  that that contained a classical condition would be removed.
  Fixed `#6192 <https://github.com/Qiskit/qiskit-terra/issues/6192>`__

- Fixed an issue with the phase of the
  :class:`qiskit.opflow.gradients.QFI` class when the ``qfi_method`` is set
  to ``lin_comb_full`` which caused the incorrect observable to be evaluated.

- Fixed an issue with :class:`~qiskit.algorithms.VQE` algorithm class
  when run with the :class:`~qiskit.algorithms.optimizers.L_BFGS_B`
  or :class:`~qiskit.algorithms.optimizers.P_BFGS` optimizer classes and
  gradients are used, the gradient was incorrectly passed as a numpy array
  instead of the expected list of floats resulting in an error. This has
  been resolved so you can use gradients with :class:`~qiskit.algorithms.VQE`
  and the :class:`~qiskit.algorithms.optimizers.L_BFGS_B` or
  :class:`~qiskit.algorithms.optimizers.P_BFGS` optimizers.


.. _Release Notes_0.18.0_Other Notes:

Other Notes
-----------

- The deprecation of the :meth:`~qiskit.pulse.Instruction.parameters` method
  for the :class:`~qiskit.pulse.Instruction` class has been reversed. This
  method was originally deprecated in the 0.17.0, but it is still necessary
  for several applications, including when running calibration experiments.
  This method will continue to be supported and will **not** be removed.

Aer 0.8.2
=========

No change

Ignis 0.6.0
===========

No change

Aqua 0.9.4
==========

No change

IBM Q Provider 0.15.0
=====================

.. _Release Notes_IBMQ_0.15.0_New Features:

New Features
------------

- Add support for new method :meth:`qiskit.providers.ibmq.runtime.RuntimeJob.error_message`
  which will return a string representing the reason if the job failed.

- The `inputs` parameter to
  :meth:`qiskit.providers.ibmq.runtime.IBMRuntimeService.run`
  method can now be specified as a
  :class:`qiskit.providers.ibmq.runtime.ParameterNamespace` instance which
  supports auto-complete features. You can use
  :meth:`qiskit.providers.ibmq.runtime.RuntimeProgram.parameters` to retrieve
  an ``ParameterNamespace`` instance.

  For example::

      from qiskit import IBMQ

      provider = IBMQ.load_account()

      # Set the "sample-program" program parameters.
      params = provider.runtime.program(program_id="sample-program").parameters()
      params.iterations = 2

      # Configure backend options
      options = {'backend_name': 'ibmq_qasm_simulator'}

      # Execute the circuit using the "circuit-runner" program.
      job = provider.runtime.run(program_id="sample-program",
                                 options=options,
                                 inputs=params)

- The user can now set the visibility (private/public) of a Qiskit Runtime program using
  :meth:`qiskit.providers.ibmq.runtime.IBMRuntimeService.set_program_visibility`.

- An optional boolean parameter `pending` has been added to
  :meth:`qiskit.providers.ibmq.runtime.IBMRuntimeService.jobs`
  and it allows filtering jobs by their status.
  If `pending` is not specified all jobs are returned.
  If `pending` is set to True, 'QUEUED' and 'RUNNING' jobs are returned.
  If `pending` is set to False, 'DONE', 'ERROR' and 'CANCELLED' jobs are returned.

- Add support for the ``use_measure_esp`` flag in the
  :meth:`qiskit.providers.ibmq.IBMQBackend.run` method. If ``True``, the backend will use ESP
  readout for all measurements which are the terminal instruction on that qubit. If used and
  the backend does not support ESP readout, an error is raised.


.. _Release Notes_IBMQ_0.15.0_Upgrade Notes:

Upgrade Notes
-------------

- :meth:`qiskit.providers.ibmq.runtime.RuntimeProgram.parameters` is now a
  method that returns a
  :class:`qiskit.providers.ibmq.runtime.ParameterNamespace` instance, which
  you can use to fill in runtime program parameter values and pass to
  :meth:`qiskit.providers.ibmq.runtime.IBMRuntimeService.run`.

- The ``open_pulse`` flag in backend configuration no longer indicates
  whether a backend supports pulse-level control. As a result,
  :meth:`qiskit.providers.ibmq.IBMQBackend.configuration` may return a
  :class:`~qiskit.providers.models.PulseBackendConfiguration` instance even
  if its ``open_pulse`` flag is ``False``.

- Job share level is no longer supported due to low adoption and the
  corresponding interface will be removed in a future release.
  This means you should no longer pass `share_level` when creating a job or use
  :meth:`qiskit.providers.ibmq.job.IBMQJob.share_level` method to get a job's share level.


.. _Release Notes_IBMQ_0.15.0_Deprecation Notes:

Deprecation Notes
-----------------

- The ``id`` instruction has been deprecated on IBM hardware
  backends. Instead, please use the ``delay`` instruction which
  implements variable-length delays, specified in units of
  ``dt``. When running a circuit containing an ``id`` instruction,
  a warning will be raised on job submission and any ``id``
  instructions in the job will be automatically replaced with their
  equivalent ``delay`` instruction.


*************
Qiskit 0.27.0
*************

Terra 0.17.4
============

No change

Aer 0.8.2
=========

No change

Ignis 0.6.0
===========

No change

Aqua 0.9.2
==========

.. _Release Notes_Aqua_0.9.2_Fixes:

Bug Fixes
---------

- Removed version caps from the requirements list to enable installing with newer
  versions of dependencies.

IBM Q Provider 0.14.0
=====================

.. _Release Notes_IBMQ_0.14.0_New Features:

New Features
------------

- You can now use the :meth:`qiskit.providers.ibmq.runtime.RuntimeJob.logs`
  method to retrieve job logs. Note that logs are only available after the
  job finishes.

- A new backend configuration attribute ``input_allowed`` now tells you the
  types of input supported by the backend. Valid input types are ``job``, which
  means circuit jobs, and ``runtime``, which means Qiskit Runtime.

  You can also use ``input_allowed`` in backend filtering. For example::

    from qiskit import IBMQ

    provider = IBMQ.load_account()
    # Get a list of all backends that support runtime.
    runtime_backends = provider.backends(input_allowed='runtime')


.. _Release Notes_IBMQ_0.14.0_Upgrade Notes:

Upgrade Notes
-------------

- ``qiskit-ibmq-provider`` now uses a new package ``websocket-client`` as its
  websocket client, and packages ``websockets`` and ``nest-asyncio`` are no
  longer required. ``setup.py`` and ``requirements.txt`` have been updated
  accordingly.


.. _Release Notes_IBMQ_0.14.0_Bug Fixes:

Bug Fixes
---------

- Fixes the issue that uses ``shots=1`` instead of the documented default
  when no ``shots`` is specified for
  :meth:`~qiskit.providers.ibmq.AccountProvider.run_circuits`.

- Fixes the issue wherein a ``QiskitBackendNotFoundError`` exception is raised
  when retrieving a runtime job that was submitted using a different provider
  than the one used for retrieval.

- Streaming runtime program interim results with proxies is now supported.
  You can specify the proxies to use when enabling the account as usual,
  for example::

    from qiskit import IBMQ

    proxies = {'urls': {'https://127.0.0.1:8085'}}
    provider = IBMQ.enable_account(API_TOKEN, proxies=proxies)


*************
Qiskit 0.26.1
*************

.. _Release Notes_0.17.4:

Terra 0.17.4
============

.. _Release Notes_0.17.4_Bug Fixes:

Bug Fixes
---------

- Fixed an issue with the :class:`~qiskit.utils.QuantumInstance` with
  :class:`~qiskit.providers.BackendV1` backends with the
  :attr:`~qiskit.providers.models.BackendConfiguration.`max_experiments`
  attribute set to a value less than the number of circuits to run. Previously
  the :class:`~qiskit.utils.QuantumInstance` would not correctly split the
  circuits to run into separate jobs, which has been corrected.

Aer 0.8.2
=========

No change

Ignis 0.6.0
===========

No change

Aqua 0.9.1
==========

No change

IBM Q Provider 0.13.1
=====================

No change

*************
Qiskit 0.26.0
*************

.. _Release Notes_0.17.3:

Terra 0.17.3
============

.. _Release Notes_0.17.3_Prelude:

Prelude
-------

This release includes 2 new classes,
:class:`~qiskit.result.ProbDistribution` and
:class:`~qiskit.result.QuasiDistribution`, which were needed for
compatibility with the recent qiskit-ibmq-provider release's beta support
for the
`qiskit-runtime <https://github.com/Qiskit-Partners/qiskit-runtime>`__.
These were only added for compatibility with that new feature in the
qiskit-ibmq-provider release and the API for these classes is considered
experimental and not considered stable for the 0.17.x release series. The
interface may change when 0.18.0 is released in the future.

.. _Release Notes_0.17.3_Bug Fixes:

Bug Fixes
---------

- Fixed an issue in :func:`~qiskit.visualization.plot_histogram` function where a ``ValueError``
  would be raised when the function run on distributions with unequal lengths.

Aer 0.8.2
=========

No change

Ignis 0.6.0
===========

No change

Aqua 0.9.1
==========

No change

IBM Q Provider 0.13.1
=====================

.. _Release Notes_IBMQ_0.13.0_Prelude:

Prelude
-------

This release introduces a new feature ``Qiskit Runtime Service``.
Qiskit Runtime is a new architecture offered by IBM Quantum that significantly
reduces waiting time during computational iterations. You can execute your
experiments near the quantum hardware, without the interactions of multiple
layers of classical and quantum hardware slowing it down.

Qiskit Runtime allows authorized users to upload their Qiskit quantum programs,
which are Python code that takes certain inputs, performs quantum and maybe
classical computation, and returns the processing results. The same or other
authorized users can then invoke these quantum programs by simply passing in the
required input parameters.

Note that Qiskit Runtime is currently in private beta for select account but
will be released to the public in the near future.

.. _Release Notes_IBMQ_0.13.0_New Features:

New Features
------------

- :class:`qiskit.providers.ibmq.experiment.analysis_result.AnalysisResult` now has an additional
  ``verified`` attribute which identifies if the ``quality`` has been verified by a human.

- :class:`qiskit.providers.ibmq.experiment.Experiment` now has an additional
  ``notes`` attribute which can be used to set notes on an experiment.

- This release introduces a new feature ``Qiskit Runtime Service``.
  Qiskit Runtime is a new architecture that
  significantly reduces waiting time during computational iterations.
  This new service allows authorized users to upload their Qiskit quantum
  programs, which are Python code that takes
  certain inputs, performs quantum and maybe classical computation, and returns
  the processing results. The same or other authorized users can then invoke
  these quantum programs by simply passing in the required input parameters.

  An example of using this new service::

    from qiskit import IBMQ

    provider = IBMQ.load_account()
    # Print all avaiable programs.
    provider.runtime.pprint_programs()

    # Prepare the inputs. See program documentation on input parameters.
    inputs = {...}
    options = {"backend_name": provider.backend.ibmq_montreal.name()}

    job = provider.runtime.run(program_id="runtime-simple",
                               options=options,
                               inputs=inputs)
    # Check job status.
    print(f"job status is {job.status()}")

    # Get job result.
    result = job.result()

.. _Release Notes_IBMQ_0.13.0_Upgrade Notes:

Upgrade Notes
-------------

- The deprecated ``Human Bad``, ``Computer Bad``, ``Computer Good`` and
  ``Human Good`` enum values have been removed from
  :class:`qiskit.providers.ibmq.experiment.constants.ResultQuality`. They
  are replaced with ``Bad`` and ``Good`` values which should be used with
  the ``verified`` attribute on
  :class:`qiskit.providers.ibmq.experiment.analysis_result.AnalysisResult`:

  +---------------+-------------+----------+
  | Old Quality   | New Quality | Verified |
  +===============+=============+==========+
  | Human Bad     | Bad         | True     |
  +---------------+-------------+----------+
  | Computer Bad  | Bad         | False    |
  +---------------+-------------+----------+
  | Computer Good | Good        | False    |
  +---------------+-------------+----------+
  | Human Good    | Good        | True     |
  +---------------+-------------+----------+

  Furthermore, the ``NO_INFORMATION`` enum has been renamed to ``UNKNOWN``.

- The :meth:`qiskit.providers.ibmq.IBMQBackend.defaults` method now always
  returns pulse defaults if they are available, regardless whether open
  pulse is enabled for the provider.

.. _Release Notes_IBMQ_0.13.0_Bug Fixes:

Bug Fixes
---------

- Fixes the issue wherein passing in a noise model when sending a job to
  an IBMQ simulator would raise a ``TypeError``. Fixes
  `#894 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/894>`_

.. _Release Notes_IBMQ_0.13.0_Other Notes:

Other Notes
-----------

- The :class:`qiskit.providers.ibmq.experiment.analysis_result.AnalysisResult`
  ``fit`` attribute is now optional.


*************
Qiskit 0.25.4
*************

.. _Release Notes_0.17.2:

Terra 0.17.2
============

.. _Release Notes_0.17.2_Prelude:

Prelude
-------

This is a bugfix release that fixes several issues from the 0.17.1 release.
Most importantly this release fixes compatibility for the
:class:`~qiskit.utils.QuantumInstance` class when running on backends that are
based on the :class:`~qiskit.providers.BackendV1` abstract class. This fixes
all the algorithms and applications built on :mod:`qiskit.algorithms` or
:mod:`qiskit.opflow` when running on newer backends.

.. _Release Notes_0.17.2_Bug Fixes:

Bug Fixes
---------

- Fixed an issue with the :class:`~qiskit.transpiler.passes.BasisTranslator`
  transpiler pass which in some cases would translate gates already in the
  target basis. This would potentially result in both longer execution time
  and less optimal results.
  Fixed `#6085 <https://github.com/Qiskit/qiskit-terra/issues/6085>`__

- Fixed an issue in the :class:`~qiskit.algorithms.optimisers.SPSA` when
  the optimizer was initialized with a callback function via the ``callback``
  kwarg would potentially cause an error to be raised.

- Fixed an issue in the
  :meth:`qiskit.quantum_info.Statevector.expectation_value`
  and :meth:`qiskit.quantum_info.DensityMatrix.expectation_value`methods
  where the ``qargs`` kwarg was ignored if the operator was a
  :class:`~qiskit.quantum_info.Pauli` or
  :class:`~qiskit.quantum_info.SparsePauliOp` operator object.
  Fixed `#6303 <https://github.com/Qiskit/qiskit-terra/issues/6303>`__

- Fixed an issue in the :meth:`qiskit.quantum_info.Pauli.evolve` method
  which could have resulted in the incorrect Pauli being returned when
  evolving by a :class:`~qiskit.circuit.library.CZGate`,
  :class:`~qiskit.circuit.library.CYGate`, or a
  :class:`~qiskit.circuit.library.SwapGate` gate.

- Fixed an issue in the :meth:`qiskit.opflow.SparseVectorStateFn.to_dict_fn`
  method, which previously had at most one entry for the all zero state due
  to an index error.

- Fixed an issue in the :meth:`qiskit.opflow.SparseVectorStateFn.equals`
  method so that is properly returning ``True`` or ``False`` instead of a
  sparse vector comparison of the single elements.

- Fixes an issue in the :class:`~qiskit.quantum_info.Statevector` and
  :class:`~qiskit.quantum_info.DensityMatrix` probability methods
  :meth:`qiskit.quantum_info.Statevector.probabilities`,
  :meth:`qiskit.quantum_info.Statevector.probabilities_dict`,
  :meth:`qiskit.quantum_info.DensityMatrix.probabilities`,
  :meth:`qiskit.quantum_info.DensityMatrix.probabilities_dict`
  where the returned probabilities could have incorrect ordering
  for certain values of the ``qargs`` kwarg.
  Fixed `#6320 <https://github.com/Qiskit/qiskit-terra/issues/6320>`__

- Fixed an issue where the :class:`~qiskit.opflow.TaperedPauliSumOp` class
  did not support the multiplication with
  :class:`~qiskit.circuit.ParameterExpression` object and also did not have
  a necessary :meth:`~qiskit.opflow.TaperedPauliSumOp.assign_parameters`
  method for working with :class:`~qiskit.circuit.ParameterExpression`
  objects.
  Fixed `#6127 <https://github.com/Qiskit/qiskit-terra/issues/6127>`__

- Fixed compatibility for the :class:`~qiskit.utils.QuantumInstance` class
  when running on backends that are based on the
  :class:`~qiskit.providers.BackendV1` abstract class.
  Fixed `#6280 <https://github.com/Qiskit/qiskit-terra/issues/6280>`__

Aer 0.8.2
=========

No change

Ignis 0.6.0
===========

No change

Aqua 0.9.1
==========

No change

IBM Q Provider 0.12.3
=====================

No change

*************
Qiskit 0.25.3
*************

Terra 0.17.1
============

No change

.. _Release Notes_Aer_0.8.2:

Aer 0.8.2
=========

.. _Release Notes_Aer_0.8.2_Known Issues:

Known Issues
------------

- The :class:`~qiskit.providers.aer.library.SaveExpectationValue` and
  :class:`~qiskit.providers.aer.library.SaveExpectationValueVariance` have
  been disabled for the `extended_stabilizer` method of the
  :class:`~qiskit.providers.aer.QasmSimulator` and
  :class:`~qiskit.providers.aer.AerSimulator` due to returning the
  incorrect value for certain Pauli operator components. Refer to
  `#1227 <https://github.com/Qiskit/qiskit-aer/issues/1227>` for more
  information and examples.


.. _Release Notes_Aer_0.8.2_Bug Fixes:

Bug Fixes
---------

- Fixes performance issue with how the ``basis_gates`` configuration
  attribute was set. Previously there were unintended side-effects to the
  backend class which could cause repeated simulation runtime to
  incrementally increase. Refer to
  `#1229 <https://github.com/Qiskit/qiskit-aer/issues/1229>` for more
  information and examples.

- Fixes a bug with the ``"multiplexer"`` simulator instruction where the
  order of target and control qubits was reversed to the order in the
  Qiskit instruction.

- Fixes a bug introduced in 0.8.0 where GPU simulations would allocate
  unneeded host memory in addition to the GPU memory.

- Fixes a bug in the ``stabilizer`` simulator method of the
  :class:`~qiskit.providers.aer.QasmSimulator` and
  :class:`~qiskit.providers.aer.AerSimulator` where the expectation value
  for the ``save_expectation_value`` and ``snapshot_expectation_value``
  could have the wrong sign for certain ``Y`` Pauli's.


Ignis 0.6.0
===========

No change

Aqua 0.9.1
==========

No change

IBM Q Provider 0.12.3
=====================

No change


*************
Qiskit 0.25.2
*************

Terra 0.17.1
============

No change

Aer 0.8.1
=========

No change

Ignis 0.6.0
===========

No change

Aqua 0.9.1
==========

No change

IBM Q Provider 0.12.3
=====================

.. _Release Notes_IBMQ_0.12.3_Other Notes:

Other Notes
-----------

- The :class:`qiskit.providers.ibmq.experiment.analysis_result.AnalysisResult` ``fit``
  attribute is now optional.

*************
Qiskit 0.25.1
*************

.. _Release Notes_0.17.1:

Terra 0.17.1
============

.. _Release Notes_0.17.1_Prelude:

Prelude
-------

This is a bugfix release that fixes several issues from the 0.17.0 release.
Most importantly this release fixes the incorrectly constructed sdist
package for the 0.17.0 release which was not actually buildable and was
blocking installation on platforms without precompiled binaries available.

.. _Release Notes_0.17.1_Bug Fixes:

Bug Fixes
---------

- Fixed an issue where the :attr:`~qiskit.circuit.QuantumCircuit.global_phase`
  attribute would not be preserved in the output
  :class:`~qiskit.circuit.QuantumCircuit` object when the
  :meth:`qiskit.circuit.QuantumCircuit.reverse_bits` method was called.
  For example::

    import math
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 2, global_phase=math.pi)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.measure(0, 1)
    qc.x(0)
    qc.y(1)

    reversed = qc.reverse_bits()
    print(reversed.global_phase)

  will now correctly print :math:`\pi`.

- Fixed an issue where the transpiler pass
  :class:`~qiskit.transpiler.passes.Unroller` didn't
  preserve global phase in case of nested instructions with one rule in
  their definition.
  Fixed `#6134 <https://github.com/Qiskit/qiskit-terra/issues/6134>`__

- Fixed an issue where the :attr:`~qiskit.circuit.ControlledGate.parameter`
  attribute of a :class:`~qiskit.circuit.ControlledGate` object built from
  a :class:`~qiskit.extensions.UnitaryGate` was not being set to the
  unitary matrix of the :class:`~qiskit.extensions.UnitaryGate` object.
  Previously, :meth:`~qiskit.extensions.UnitaryGate.control` was building a
  :class:`~qiskit.circuit.ControlledGate` with the ``parameter`` attribute
  set to the controlled version of
  :class:`~qiskit.extensions.UnitaryGate` matrix.
  This would lead to a modification of the ``parameter`` of the base
  :class:`~qiskit.extensions.UnitaryGate` object and subsequent calls to
  :meth:`~qiskit.circuit.ControlledGate.inverse` was creating
  the inverse of a double-controlled :class:`~qiskit.extensions.UnitaryGate`.
  Fixed `#5750 <https://github.com/Qiskit/qiskit-terra/issues/5750>`__

- Fixed an issue with the preset pass managers
  :class:`~qiskit.transpiler.preset_passmanagers.level_0_pass_manager` and
  :class:`~qiskit.transpiler.preset_passmanagers.level_1_pass_manager`
  (which corresponds to ``optimization_level`` 0 and 1 for
  :func:`~qiskit.compiler.transpile`) where in some cases they would
  produce circuits not in the requested basis.

- Fix a bug where using :class:`~qiskit.algorithms.optimizers.SPSA` with automatic
  calibration of the learning rate and perturbation (i.e. ``learning_rate`` and
  ``perturbation`` are ``None`` in the initializer), stores the calibration for all
  future optimizations. Instead, the calibration should be done for each new objective
  function.

.. _Aer_Release Notes_0.8.1:

Aer 0.8.1
=========

.. _Aer_Release Notes_0.8.1_Bug Fixes:

Bug Fixes
---------

- Fixed an issue with use of the ``matrix_product_state`` method of the
  :class:`~qiskit.providers.aer.AerSimulator` and
  :class:`~qiskit.providers.aer.QasmSimulator` simulators when running a
  noisy simulation with Kraus errors. Previously, the matrix product state
  simulation method would not propogate changes to neighboring qubits after
  applying the Kraus matrix. This has been fixed so the output from the
  simulation is correct.
  Fixed `#1184 <https://github.com/Qiskit/qiskit-aer/issues/1184>`__ and
  `#1205 <https://github.com/Qiskit/qiskit-aer/issues/1205>`__

- Fixed an issue where the :class:`qiskit.extensions.Initialize` instruction
  would disable measurement sampling optimization for the ``statevector`` and
  ``matrix_product_state`` simulation methods of the
  :class:`~qiskit.providers.aer.AerSimulator` and
  :class:`~qiskit.providers.aer.QasmSimulator` simulators, even when it was
  the first circuit instruction or applied to all qubits and hence
  deterministic.
  Fixed `#1210 <https://github.com/Qiskit/qiskit-aer/issues/1210>`__

- Fix an issue with the :class:`~qiskit.providers.aer.library.SaveStatevector`
  and :class:`~qiskit.providers.aer.extensions.SnapshotStatevector`
  instructions when used with the ``extended_stabilizer`` simulation method
  of the :class:`~qiskit.providers.aer.AerSimulator` and
  :class:`~qiskit.providers.aer.QasmSimulator` simulators where it would
  return an unnormalized statevector.
  Fixed `#1196 <https://github.com/Qiskit/qiskit-aer/issues/1210>`__

- The ``matrix_product_state`` simulation method now has support for it's
  previously missing set state instruction,
  :class:`qiskit.providers.aer.library.SetMatrixProductState`, which enables
  setting the state of a simulation in a circuit.

Ignis 0.6.0
===========

No change

Aqua 0.9.1
==========

IBM Q Provider 0.12.2
=====================

No change

*************
Qiskit 0.25.0
*************

This release officially deprecates the Qiskit Aqua project. Accordingly, in a
future release the ``qiskit-aqua`` package will be removed from the Qiskit
metapackage, which means in that future release ``pip install qiskit`` will no
longer include ``qiskit-aqua``. The application modules that are provided by
qiskit-aqua have been split into several new packages:
``qiskit-optimization``, ``qiskit-nature``, ``qiskit-machine-learning``, and
``qiskit-finance``. These packages can be installed by themselves (via the
standard pip install command, e.g. ``pip install qiskit-nature``) or with the
rest of the Qiskit metapackage as optional extras (e.g.
``pip install 'qiskit[finance,optimization]'`` or ``pip install 'qiskit[all]'``
The core algorithms and the operator flow now exist as part of qiskit-terra at
:mod:`qiskit.algorithms` and :mod:`qiskit.opflow`. Depending on your existing
usage of Aqua you should either use the application packages or the new modules
in Qiskit Terra. For more details on how to migrate from Qiskit Aqua you can
refer to the :ref:`aqua-migration`.

.. _Release Notes_0.17.0:

Terra 0.17.0
============

.. _Release Notes_0.17.0_Prelude:

Prelude
-------

The Qiskit Terra 0.17.0 includes many new features and bug fixes. The major
new feature for this release is the introduction of the
:mod:`qiskit.algorithms` and :mod:`qiskit.opflow` modules which were
migrated and adapted from the :mod:`qiskit.aqua` project.


.. _Release Notes_0.17.0_New Features:

New Features
------------

- The :py:func:`qiskit.pulse.call` function can now take a
  :class:`~qiskit.circuit.Parameter` object along with a parameterized
  subroutine. This enables assigning different values to the
  :class:`~qiskit.circuit.Parameter` objects for each subroutine call.

  For example,

  .. code-block:: python

    from qiskit.circuit import Parameter
    from qiskit import pulse

    amp = Parameter('amp')

    with pulse.build() as subroutine:
        pulse.play(pulse.Gaussian(160, amp, 40), DriveChannel(0))

    with pulse.build() as main_prog:
        pulse.call(subroutine, amp=0.1)
        pulse.call(subroutine, amp=0.3)

- The :class:`qiskit.providers.models.QasmBackendConfiguration` has a new
  field ``processor_type`` which can optionally be used to provide
  information about a backend's processor in the form:
  ``{"family": <str>, "revision": <str>, segment: <str>}``. For example:
  ``{"family": "Canary", "revision": "1.0", segment: "A"}``.

- The :py:class:`qiskit.pulse.Schedule`,
  :py:class:`qiskit.pulse.Instruction`, and :py:class:`qiskit.pulse.Channel`
  classes now have a :attr:`~qiiskit.pulse.Schedule.parameter` property
  which will return any :class:`~qiskit.circuit.Parameter` objects used
  in the object and a :meth:`~qiskit.pulse.Schedule.is_parameterized()`
  method which will return ``True`` if any parameters are used in the
  object.

  For example:

  .. jupyter-execute::

      from qiskit.circuit import Parameter
      from qiskit import pulse

      shift = Parameter('alpha')

      schedule = pulse.Schedule()
      schedule += pulse.SetFrequency(shift, pulse.DriveChannel(0))

      assert schedule.is_parameterized() == True
      print(schedule.parameters)

- Added a :class:`~qiskit.circuit.library.PiecewiseChebyshev` to the
  :mod:`qiskit.circuit.library` for implementing a piecewise Chebyshev
  approximation of an input function. For a given function :math:`f(x)`
  and degree :math:`d`, this class class implements
  a piecewise polynomial Chebyshev approximation on :math:`n` qubits
  to :math:`f(x)` on the given intervals. All the polynomials in the
  approximation are of degree :math:`d`.

  For example:

  .. jupyter-execute::

      import numpy as np
      from qiskit import QuantumCircuit
      from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
      f_x, degree, breakpoints, num_state_qubits = lambda x: np.arcsin(1 / x), 2, [2, 4], 2
      pw_approximation = PiecewiseChebyshev(f_x, degree, breakpoints, num_state_qubits)
      pw_approximation._build()
      qc = QuantumCircuit(pw_approximation.num_qubits)
      qc.h(list(range(num_state_qubits)))
      qc.append(pw_approximation.to_instruction(), qc.qubits)
      qc.draw(output='mpl')

- The :py:class:`~qiskit.providers.models.BackendProperties` class now
  has a :meth:`~qiskit.providers.models.BackendProperties.readout_length`
  method, which returns the readout length [sec] of the given qubit.

- A new class, :py:class:`~qiskit.pulse.ScheduleBlock`, has been added to
  the :class:`qiskit.pulse` module. This class provides a new representation
  of a pulse program. This representation is best suited for the pulse
  builder syntax and is based on relative instruction ordering.

  This representation takes ``alignment_context`` instead of specifying
  starting time ``t0`` for each instruction. The start time of instruction is
  implicitly allocated with the specified transformation and relative
  position of instructions.

  The :py:class:`~qiskit.pulse.ScheduleBlock` allows for lazy instruction
  scheduling, meaning we can assign arbitrary parameters to the duration of
  instructions.

  For example:

  .. code-block:: python

      from qiskit.pulse import ScheduleBlock, DriveChannel, Gaussian
      from qiskit.pulse.instructions import Play, Call
      from qiskit.pulse.transforms import AlignRight
      from qiskit.circuit import Parameter

      dur = Parameter('rabi_duration')

      block = ScheduleBlock(alignment_context=AlignRight())
      block += Play(Gaussian(dur, 0.1, dur/4), DriveChannel(0))
      block += Call(measure_sched)  # subroutine defined elsewhere

  this code defines an experiment scanning a Gaussian pulse's duration
  followed by a measurement ``measure_sched``, i.e. a Rabi experiment.
  You can reuse the ``block`` object for every scanned duration
  by assigning a target duration value.

- Added a new function :func:`~qiskit.visualization.array_to_latex` to
  the :mod:`qiskit.visualization` module that can be used to represent
  and visualize vectors and matrices with LaTeX.

  .. jupyter-execute::

          from qiskit.visualization import array_to_latex
          from numpy import sqrt, exp, pi
          mat = [[0, exp(pi*.75j)],
                 [1/sqrt(8), 0.875]]
          array_to_latex(mat)

- The :class:`~qiskit.quantum_info.Statevector` and
  :class:`~qiskit.quantum_info.DensityMatrix` classes now have
  :meth:`~qiskit.quantum_info.Statevector.draw` methods which allow objects
  to be drawn as either text matrices, IPython Latex objects, Latex source,
  Q-spheres, Bloch spheres and Hinton plots. By default the output type
  is the equivalent output from ``__repr__`` but this default can be changed
  in a user config file by setting the ``state_drawer`` option. For example:

  .. jupyter-execute::

          from qiskit.quantum_info import DensityMatrix
          dm = DensityMatrix.from_label('r0')
          dm.draw('latex')

  .. jupyter-execute::

          from qiskit.quantum_info import Statevector
          sv = Statevector.from_label('+r')
          sv.draw('qsphere')

  Additionally, the :meth:`~qiskit.quantum_info.DensityMatrix.draw` method
  is now used for the ipython display of these classes, so if you change the
  default output type in a user config file then when a
  :class:`~qiskit.quantum_info.Statevector` or a
  :class:`~qiskit.quantum_info.DensityMatrix` object are displayed in
  a jupyter notebook that output type will be used for the object.

- Pulse :class:`qiskit.pulse.Instruction` objects and
  parametric pulse objects (eg :class:`~qiskit.pulse.library.Gaussian` now
  support using :class:`~qiskit.circuit.Parameter` and
  :class:`~qiskit.circuit.ParameterExpression` objects for the ``duration``
  parameter. For example:

  .. code-block:: python

    from qiskit.circuit import Parameter
    from qiskit.pulse import Gaussian

    dur = Parameter('x_pulse_duration')
    double_dur = dur * 2
    rx_pulse = Gaussian(dur, 0.1, dur/4)
    double_rx_pulse = Gaussian(double_dir, 0.1, dur/4)

  Note that while we can create an instruction with a parameterized
  ``duration`` adding an instruction with unbound parameter ``duration``
  to a schedule is supported only by the newly introduced representation
  :class:`~qiskit.pulse.ScheduleBlock`. See the known issues release notes
  section for more details.

- The :meth:`~qiskit.providers.basicaer.QasmSimulatorPy.run` method for the
  :class:`~qiskit.providers.basicaer.QasmSimulatorPy`,
  :class:`~qiskit.providers.basicaer.StatevectorSimulatorPy`, and
  :class:`~qiskit.providers.basicaer.UnitarySimulatorPy` backends now takes a
  :class:`~qiskit.circuit.QuantumCircuit` (or a list of
  :class:`~qiskit.circuit.QuantumCircuit` objects) as its input.
  The previous :class:`~qiskit.qobj.QasmQobj` object is still supported for
  now, but will be deprecated in a future release.

  For an example of how to use this see::

    from qiskit import transpile, QuantumCircuit

    from qiskit.providers.basicaer import BasicAer

    backend = BasicAer.get_backend('qasm_simulator')

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    tqc = transpile(circuit, backend)
    result = backend.run(tqc, shots=4096).result()

- The :class:`~qiskit.transpiler.passes.CommutativeCancellation` transpiler
  pass has a new optional kwarg on the constructor ``basis_gates``, which
  takes the a list of the names of basis gates for the target backend.
  When specified the pass will only use gates in the ``basis_gates`` kwarg.
  Previously, the pass would automatically replace consecutive gates which
  commute with :class:`~qiskit.circuit.library.ZGate` with the
  :class:`~qiskit.circuit.library.U1Gate` unconditionally. The ``basis_gates``
  kwarg enables you to specify which z-rotation gates are present in
  the target basis to avoid this.

- The constructors of the :class:`~qiskit.circuit.Bit` class and subclasses,
  :class:`~qiskit.circuit.Qubit`, :class:`~qiskit.circuit.Clbit`, and
  :class:`~qiskit.circuit.AncillaQubit`, have been updated such that their
  two parameters, ``register`` and ``index`` are now optional. This enables
  the creation of bit objects that are independent of a register.

- A new class,
  :class:`~qiskit.circuit.classicalfunction.BooleanExpression`, has been
  added to the :mod:`qiskit.circuit.classicalfunction` module. This class
  allows for creating an oracle from a Python boolean expression. For example:

  .. jupyter-execute::

    from qiskit.circuit import BooleanExpression, QuantumCircuit

    expression = BooleanExpression('~x & (y | z)')
    circuit = QuantumCircuit(4)
    circuit.append(expression, [0, 1, 2, 3])
    circuit.draw('mpl')

  .. jupyter-execute::

    circuit.decompose().draw('mpl')

  The :class:`~qiskit.circuit.classicalfunction.BooleanExpression` also
  includes a method,
  :meth:`~qiskit.circuit.classicalfunction.BooleanExpression.from_dimacs_file`,
  which allows loading formulas described in the
  `DIMACS-CNF <https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html>`__
  format. For example:

  .. code-block::

    from qiskit.circuit import BooleanExpression, QuantumCircuit

    boolean_exp = BooleanExpression.from_dimacs_file("simple_v3_c2.cnf")
    circuit = QuantumCircuit(boolean_exp.num_qubits)
    circuit.append(boolean_exp, range(boolean_exp.num_qubits))
    circuit.draw('text')

  .. parsed-literal::

         ┌───────────────────┐
    q_0: ┤0                  ├
         │                   │
    q_1: ┤1                  ├
         │  SIMPLE_V3_C2.CNF │
    q_2: ┤2                  ├
         │                   │
    q_3: ┤3                  ├
         └───────────────────┘

  .. code-block::

    circuit.decompose().draw('text')

  .. parsed-literal::

    q_0: ──o────o────────────
           │    │
    q_1: ──■────o────■───────
           │    │    │
    q_2: ──■────┼────o────■──
         ┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐
    q_3: ┤ X ├┤ X ├┤ X ├┤ X ├
         └───┘└───┘└───┘└───┘

- Added a new class, :class:`~qiskit.circuit.library.PhaseOracle`, has been
  added to the :mod:`qiskit.circuit.library` module. This class enables the
  construction of phase oracle circuits from Python boolean expressions.

  .. jupyter-execute::

    from qiskit.circuit.library.phase_oracle import PhaseOracle

    oracle = PhaseOracle('x1 & x2 & (not x3)')
    oracle.draw('mpl')

  These phase oracles can be used as part of a larger algorithm, for example
  with :class:`qiskit.algorithms.AmplificationProblem`:

  .. jupyter-execute::

    from qiskit.algorithms import AmplificationProblem, Grover
    from qiskit import BasicAer

    backend = BasicAer.get_backend('qasm_simulator')

    problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)
    grover = Grover(quantum_instance=backend)
    result = grover.amplify(problem)
    result.top_measurement

  The :class:`~qiskit.circuit.library.PhaseOracle` class also includes a
  :meth:`~qiskit.circuit.library.PhaseOracle.from_dimacs_file` method which
  enables constructing a phase oracle from a file describing a formula in the
  `DIMACS-CNF <https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html>`__
  format.

  .. code-block::

    from qiskit.circuit.library.phase_oracle import PhaseOracle

    oracle = PhaseOracle.from_dimacs_file("simple_v3_c2.cnf")
    oracle.draw('text')

  .. parsed-literal::

     state_0: ─o───────o──────────────
               │ ┌───┐ │ ┌───┐
     state_1: ─■─┤ X ├─■─┤ X ├─■──────
               │ └───┘   └───┘ │ ┌───┐
     state_2: ─■───────────────o─┤ Z ├
                                 └───┘

- All transpiler passes (ie any instances of
  :class:`~qiskit.transpiler.BasePass`) are now directly callable.
  Calling a pass provides a convenient interface for running the pass
  on a :class:`~qiskit.circuit.QuantumCircuit` object.

  For example, running a single transformation pass, such as
  :class:`~qiskit.transpiler.passes.BasisTranslator`, can be done with:

  .. jupyter-execute::

    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import BasisTranslator
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

    circuit = QuantumCircuit(1)
    circuit.h(0)

    pass_instance = BasisTranslator(sel, ['rx', 'rz', 'cx'])
    result = pass_instance(circuit)
    result.draw(output='mpl')

  When running an analysis pass, a property set (as ``dict`` or as
  :class:`~qiskit.transpiler.PropertySet`)
  needs to be added as a parameter and it might be modified "in-place".
  For example:

  .. jupyter-execute::

    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import Depth

    circuit = QuantumCircuit(1)
    circuit.h(0)

    property_set = {}
    pass_instance = Depth()
    pass_instance(circuit, property_set)
    print(property_set)

- The :class:`~qiskit.qobj.QasmQobjConfig` class now has an optional
  kwarg for ``meas_level`` and ``meas_return``. These fields can be used
  to enable generating :class:`~qiskit.qobj.QasmQobj` job payloads that
  support ``meas_level=1`` (kerneled data) for circuit jobs (previously
  this was only exposed for :class:`~qiskit.qobj.PulseQobj` objects).
  The :func:`~qiskit.compiler.assemble` function has been updated
  to set this field for :class:`~qiskit.qobj.QasmQobj` objects it
  generates.

- A new :meth:`~qiskit.circuit.QuantumCircuit.tensor` method has been
  added to the :class:`~qiskit.circuit.QuantumCircuit` class. This
  method enables tensoring another circuit with an existing circuit.
  This method works analogously to
  :meth:`qiskit.quantum_info.Operator.tensor`
  and is consistent with the little-endian convention of Qiskit.

  For example:

  .. jupyter-execute::

    from qiskit import QuantumCircuit
    top = QuantumCircuit(1)
    top.x(0);
    bottom = QuantumCircuit(2)
    bottom.cry(0.2, 0, 1);
    bottom.tensor(top).draw(output='mpl')

- The :class:`qiskit.circuit.QuantumCircuit` class now supports arbitrary
  free form metadata with the :attr:`~qiskit.circuit.QuantumCircuit.metadata`
  attribute. A user (or program built on top of
  :class:`~qiskit.circuit.QuantumCircuit`) can attach metadata to a circuit
  for use in tracking the circuit. For example::

    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2, user_metadata_field_1='my_metadata',
                        user_metadata_field_2='my_other_value')

  or::

    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.metadata = {'user_metadata_field_1': 'my_metadata',
                   'user_metadata_field_2': 'my_other_value'}

  This metadata will **not** be used for influencing the execution of the
  circuit but is just used for tracking the circuit for the lifetime of the
  object. The ``metadata`` attribute will persist between any circuit
  transforms including :func:`~qiskit.compiler.transpile` and
  :func:`~qiskit.compiler.assemble`. The expectation is for providers to
  associate the metadata in the result it returns, so that users can
  filter results based on circuit metadata the same way they can currently
  do with ``QuantumCircuit.name``.

- Add a new operator class :class:`~qiskit.quantum_info.CNOTDihedral` has
  been added to the :mod:`qiskit.quantum_info` module. This class is
  used to represent the CNOT-Dihedral group, which is generated by the
  quantum gates :class:`~qiskit.circuit.library.CXGate`,
  :class:`~qiskit.circuit.library.TGate`,
  and :class:`~qiskit.circuit.library.XGate`.

- Adds a ``&`` (``__and__``) binary operator to ``BaseOperator`` subclasses
  (eg :class:`qiskit.quantum_info.Operator`) in the
  :mod:`qiskit.quantum_info` module. This is shorthand to call the
  classes :meth:`~qiskit.quantum_info.Operator.compose` method
  (ie ``A & B == A.compose(B)``).

  For example:

  .. code:: python

    import qiskit.quantum_info as qi

    qi.Pauli('X') & qi.Pauli('Y')

- Adds a ``&`` (``__and__``) binary operator to
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes. This is shorthand to
  call the classes :meth:`~qiskit.quantum_info.Statevector.evolve` method
  (ie ``psi & U == psi.evolve(U)``).

  For example:

  .. code:: python

    import qiskit.quantum_info as qi

    qi.Statevector.from_label('0') & qi.Pauli('X')

- A new a new 2-qubit gate, :class:`~qiskit.circuit.library.ECRGate`,
  the echo cross-resonance (ECR), has been added to the
  :mod:`qiskit.circuit.library` module along with a corresponding method,
  :meth:`~qiskit.circuit.QuantumCircuit.ecr` for the
  :class:`~qiskit.circuit.QuantumCircuit` class. The ECR gate is two
  :math:`CR(\frac{π}{4})` pulses with an
  :class:`~qiskit.circuit.library.XGate` between them for the echo. This gate
  is locally equivalent to a :class:`~qiskit.circuit.library.CXGate` (can
  convert to a CNOT with local pre- or post-rotation). It is the native gate
  on current IBM hardware and compiling to it allows the pre-/post-rotations
  to be merged into the rest of the circuit.

- A new kwarg ``approximation_degree`` has been added to the
  :func:`~qiskit.compiler.transpile` function for enabling
  approximate compilation. Valid values range from 0 to 1, and higher
  means less approximation. This is a heuristic dial
  to experiment with circuit approximations. The concrete interpretation
  of this number is left to each pass, which may use it to perform
  some approximate version of the pass. Specific examples include
  the :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass or the
  or translators to discrete gate sets. If a pass does not support this
  option, it implies exact transformation.

- Two new transpiler passess, :class:`~qiskit.transpiler.passes.GateDirection`
  and :class:`qiskit.transpiler.passes.CheckGateDirection`, were added to the
  :mod:`qiskit.transpiler.passes` module. These new passes are inteded to
  be more general replacements for
  :class:`~qiskit.transpiler.passes.CXDirection` and
  :class:`~qiskit.transpiler.passes.CheckCXDirection` (which are both now
  deprecated, see the deprecation notes for more details) that perform the
  same function but work with other gates beside just
  :class:`~qiskit.circuit.library.CXGate`.

- When running on Windows, parallel execution with the
  :func:`~qiskit.tools.parallel_map` function can now be enabled (it is
  still disabled by default). To do this you can either set
  ``parallel = True`` in a user config file, or set the ``QISKIT_PARALLEL``
  environment variable to ``TRUE`` (this will also effect
  :func:`~qiskit.compiler.transpile` and :func:`~qiskit.compiler.assemble`
  which both use :func:`~qiskit.tools.parallel_map` internally). It is
  important to note that when enabling parallelism on Windows there are
  limitations around how Python launches processes for Windows, see the
  Known Issues section below for more details on the limitations with
  parallel execution on Windows.

- A new function, :func:`~qiskit.quantum_info.hellinger_distance`, for
  computing the Hellinger distance between two counts distributions has
  been added to the :mod:`qiskit.quantum_info` module.

- The :func:`~qiskit.quantum_info.decompose_clifford` function in the
  :mod:`qiskit.quantum_info` module (which gets used internally by the
  :meth:`qiskit.quantum_info.Clifford.to_circuit` method) has a new kwarg
  ``method`` which enables selecting the synthesis method used by either
  setting it to ``'AG'`` or ``'greedy'``. By default for more than three
  qubits it is set to ``'greedy'`` which uses a non-optimal greedy compilation
  routine for Clifford elements synthesis, by Bravyi et. al., which typically
  yields better CX cost compared to the previously used Aaronson-Gottesman
  method (for more than two qubits). You can use the ``method`` kwarg to revert
  to the previous default Aaronson-Gottesman method by setting ``method='AG'``.

- The :class:`~qiskit.extensions.Initialize` class in the
  :mod:`qiskit.extensions` module can now be constructed using an integer.
  The '1' bits of the integer will insert a :class:`~qiskit.circuit.Reset`
  and an :class:`~qiskit.circuit.library.XGate` into the circuit for the
  corresponding qubit. This will be done using the standard little-endian
  convention is qiskit, ie the rightmost bit of the integer will set qubit
  0. For example, setting the parameter in
  :class:`~qiskit.extensions.Initialize` equal to ``5`` will set qubits 0
  and 2 to value 1.

  .. jupyter-execute::

      from qiskit.extensions import Initialize

      initialize = Initialize(13)
      initialize.definition.draw('mpl')

- The :class:`~qiskit.extensions.Initialize` class in the
  :mod:`qiskit.extensions` module now supports constructing directly from
  a Pauli label (analogous to the
  :meth:`qiskit.quantum_info.Statevector.from_label` method). The Pauli label
  refer to basis states of the Pauli eigenstates Z, X, Y. These labels use
  Qiskit's standard little-endian notation, for example a label of ``'01'``
  would initialize qubit 0 to :math:`|1\rangle` and qubit 1 to
  :math:`|0\rangle`.

  .. jupyter-execute::

      from qiskit.extensions import Initialize

      initialize = Initialize("10+-lr")
      initialize.definition.draw('mpl')

- The kwarg, ``template_list``, for the constructor of the
  :class:`qiskit.transpiler.passes.TemplateOptimization` transpiler pass
  now supports taking in a list of both
  :class:`~qiskit.circuit.QuantumCircuit` and
  :class:`~qiskit.dagcircuit.DAGDependency` objects. Previously, only
  :class:`~qiskit.circuit.QuantumCircuit` were accepted (which were internally
  converted to :class:`~qiskit.dagcircuit.DAGDependency` objects) in the
  input list.

- A new transpiler pass,
  :py:class:`qiskit.transpiler.passes.RZXCalibrationBuilder`, capable
  of generating calibrations and adding them to a quantum circuit has been
  introduced. This pass takes calibrated
  :class:`~qiskit.circuit.library.CXGate` objects and creates the
  calibrations for :class:`qiskit.circuit.library.RZXGate` objects with an
  arbitrary rotation angle. The schedules are created by stretching and
  compressing the :class:`~qiskit.pulse.GaussianSquare` pulses of the
  echoed-cross resonance gates.

- New template circuits for using :class:`qiskit.circuit.library.RZXGate`
  are added to the :mod:`qiskit.circuit.library` module (eg
  :class:`~qiskit.circuit.library.rzx_yz`). This enables pairing
  the :class:`~qiskit.transpiler.passes.TemplateOptimization` pass with the
  :py:class:`qiskit.transpiler.passes.RZXCalibrationBuilder` pass to
  automatically find and replace gate sequences, such as
  ``CNOT - P(theta) - CNOT``, with more efficent circuits based on
  :class:`qiskit.circuit.library.RZXGate` with a calibration.

- The matplotlib output type for the
  :func:`~qiskit.visualization.circuit_drawer` and
  the :meth:`~qiskit.circuit.QuantumCircuit.draw` method for the
  :class:`~qiskit.circuit.QuantumCircuit` class now supports configuration
  files for setting the visualization style. In previous releases, there was
  basic functionality that allowed users to pass in a ``style`` kwarg that
  took in a ``dict`` to customize the colors and other display features of
  the ``mpl`` drawer. This has now been expanded so that these dictionaries
  can be loaded from JSON files directly without needing to pass a dictionary.
  This enables users to create new style files and use that style for
  visualizations by passing the style filename as a string to the ``style``
  kwarg.

  To leverage this feature you must set the ``circuit_mpl_style_path``
  option in a user config file. This option should be set to the path you
  want qiskit to search for style JSON files. If specifying multiple path
  entries they should be separated by ``:``. For example, setting
  ``circuit_mpl_style_path = ~/.qiskit:~/user_styles`` in a user config
  file will look for JSON files in both ``~/.qiskit`` and ``~/user_styles``.

- A new kwarg, ``format_marginal`` has been added to the function
  :func:`~qiskit.result.utils.marginal_counts` which when set to ``True``
  formats the counts output according to the
  :attr:`~qiskit.circuit.QuantumCircuit.cregs` in the circuit and missing
  indices are represented with a ``_``. For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit, execute, BasicAer, result
      from qiskit.result.utils import marginal_counts
      qc = QuantumCircuit(5, 5)
      qc.x(0)
      qc.measure(0, 0)

      result = execute(qc, BasicAer.get_backend('qasm_simulator')).result()
      print(marginal_counts(result.get_counts(), [0, 2, 4], format_marginal=True))

- Improved the performance of
  :meth:`qiskit.quantum_info.Statevector.expectation_value`  and
  :meth:`qiskit.quantum_info.DensityMatrix.expectation_value` when the
  argument operator is a :class:`~qiskit.quantum_info.Pauli`  or
  :class:`~qiskit.quantum_info.SparsePauliOp`  operator.

- The user config file has 2 new configuration options, ``num_processes`` and
  ``parallel``, which are used to control the default behavior of
  :func:`~qiskit.tools.parallel_map`. The ``parallel`` option is a boolean
  that is used to dictate whether :func:`~qiskit.tools.parallel_map` will
  run in multiple processes or not. If it set to ``False`` calls to
  :func:`~qiskit.tools.parallel_map` will be executed serially, while setting
  it to ``True`` will enable parallel execution. The ``num_processes`` option
  takes an integer which sets how many CPUs to use when executing in parallel.
  By default it will use the number of CPU cores on a system.

- There are 2 new environment variables, ``QISKIT_PARALLEL`` and
  ``QISKIT_NUM_PROCS``, that can be used to control the default behavior of
  :func:`~qiskit.tools.parallel_map`. The ``QISKIT_PARALLEL`` option can be
  set to the ``TRUE`` (any capitalization) to set the default to run in
  multiple processes when :func:`~qiskit.tools.parallel_map` is called. If it
  is set to any other
  value :func:`~qiskit.tools.parallel_map` will be executed serially.
  ``QISKIT_NUM_PROCS`` takes an integer (for example ``QISKIT_NUM_PROCS=5``)
  which will be used as the default number of processes to run with. Both
  of these will take precedence over the equivalent option set in the user
  config file.

- A new method, :meth:`~qiskit.circuit.ParameterExpression.gradient`, has
  been added to the :class:`~qiskit.circuit.ParameterExpression` class. This
  method is used to  evaluate the gradient of a
  :class:`~qiskit.circuit.ParameterExpression` object.

- The ``__eq__`` method (ie what is called when the ``==`` operator is used)
  for the :class:`~qiskit.circuit.ParameterExpression` now allows for the
  comparison with a numeric value. Previously, it was only possible
  to compare two instances of
  :class:`~qiskit.circuit.ParameterExpression` with ``==``. For example::

      from qiskit.circuit import Parameter

      x = Parameter("x")
      y = x + 2
      y = y.assign(x, -1)

      assert y == 1

- The :class:`~qiskit.circuit.library.PauliFeatureMap` class in the
  :mod:`qiskit.circuit.library` module now supports adjusting the rotational
  factor, :math:`\alpha`, by either setting using the kwarg ``alpha`` on
  the constructor or setting the
  :attr:`~qiskit.circuit.library.PauliFeatureMap.alpha` attribute after
  creation. Previously this value was fixed at ``2.0``. Adjusting this
  attribute allows for better control of decision boundaries and provides
  additional flexibility handling the input features without needing
  to explicitly scale them in the data set.

- A new :class:`~qiskit.circuit.Gate` class,
  :class:`~qiskit.circuit.library.PauliGate`, has been added
  the :class:`qiskit.circuit.library` module and corresponding method,
  :meth:`~qiskit.circuit.QuantumCircuit.pauli`,  was added to the
  :class:`~qiskit.circuit.QuantumCircuit` class. This new gate class enables
  applying several individual pauli gates to different qubits at the
  simultaneously. This is primarily useful for simulators which can use this
  new gate to more efficiently implement multiple simultaneous Pauli gates.

- Improve the :class:`qiskit.quantum_info.Pauli` operator.
  This class now represents and element from the full N-qubit Pauli group
  including complex coefficients. It now supports the Operator API methods
  including :meth:`~qiskit.quantum_info.Pauli.compose`,
  :meth:`~qiskit.quantum_info.Pauli.dot`,
  :meth:`~qiskit.quantum_info.Pauli.tensor` etc, where compose and dot are
  defined with respect to the full Pauli group.

  This class also allows conversion to and from the string representation
  of Pauli's for convenience.

  For example

  .. jupyter-execute::

    from qiskit.quantum_info import Pauli

    P1 = Pauli('XYZ')
    P2 = Pauli('YZX')
    P1.dot(P2)

  Pauli's can also be directly appended to
  :class:`~qiskit.circuit.QuantumCircuit` objects

  .. jupyter-execute::

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli

    circ = QuantumCircuit(3)
    circ.append(Pauli('XYZ'), [0, 1, 2])
    circ.draw(output='mpl')

  Additional methods allow computing when two Pauli's commute (using the
  :meth:`~qiskit.quantum_info.Pauli.commutes` method) or anticommute
  (using the :meth:`~qiskit.quantum_info.Pauli.anticommutes` method), and
  computing the Pauli resulting from Clifford conjugation
  :math:`P^\prime = C.P.C^\dagger`
  using the  :meth:`~qiskit.quantum_info.Pauli.evolve` method.

  See the API documentation of the :class:`~qiskit.quantum_info.Pauli` class
  for additional information.

- A new function, :func:`~qiskit.quantum_info.random_pauli`, for generating a
  random element of the N-qubit Pauli group has been added to the
  :mod:`qiskit.quantum_info` module.

- A new class,
  :class:`~qiskit.circuit.library.PiecewisePolynomialPauliRotations`, has
  been added to the :mod:`qiskit.circuit.library` module. This circuit library
  element is used for mapping a piecewise polynomial function, :math:`f(x)`,
  which is defined through breakpoints and coefficients, on qubit amplitudes.
  The breakpoints :math:`(x_0, ..., x_J)` are a subset of :math:`[0, 2^n-1]`,
  where :math:`n` is the number of state qubits. The corresponding
  coefficients :math:`[a_{j,1},...,a_{j,d}]`, where :math:`d` is the highest
  degree among all polynomials. Then :math:`f(x)` is defined as:

  .. math::

      f(x) = \begin{cases}
          0, x < x_0 \\
          \sum_{i=0}^{i=d}a_{j,i} x^i, x_j \leq x < x_{j+1}
          \end{cases}

  where we implicitly assume :math:`x_{J+1} = 2^n`. And the mapping applied
  to the amplitudes is given by

  .. math::

      F|x\rangle |0\rangle = \cos(p_j(x))|x\rangle |0\rangle + \sin(p_j(x))|x\rangle |1\rangle

  This mapping is based on controlled Pauli Y-rotations and constructed using
  the :class:`~qiskit.circuit.library.PolynomialPauliRotations`.

- A new module :mod:`qiskit.algorithms` has been introduced. This module
  contains functionality equivalent to what has previously been
  provided by the :mod:`qiskit.aqua.algorithms` module (which is now
  deprecated) and provides the building blocks for constructing quantum
  algorithms. For details on migrating from ``qiskit-aqua`` to this new
  module please refer to the migration guide :ref:`aqua-migration`

- A new module :mod:`qiskit.opflow` has been introduced. This module
  contains functionality equivalent to what has previously been
  provided by the :mod:`qiskit.aqua.operators` module (which is now
  deprecated) and provides the operators and state functions which are
  used to build quantum algorithms. For details on migrating from
  ``qiskit-aqua`` to this new module please refer to the migration guide
  :ref:`aqua-migration`

- This is the first release that includes precompiled binary wheels for
  the for Linux aarch64 systems. If you are running a manylinux2014
  compatible aarch64 Linux system there are now precompiled wheels available
  on PyPI, you are no longer required to build from source to install
  qiskit-terra.

- The :func:`qiskit.quantum_info.process_fidelity` function is now able to be
  used with a non-unitary target channel. In this case the returned value is
  equivalent to the :func:`qiskit.quantum_info.state_fidelity` of the
  normalized :class:`qiskit.quantum_info.Choi` matrices for the channels.

  Note that the :func:`qiskit.quantum_info.average_gate_fidelity` and
  :func:`qiskit.quantum_info.gate_error` functions still require the target
  channel to be unitary and will raise an exception if it is not.

- Added a new pulse builder function, :func:`qiskit.pulse.macro`.
  This enables normal Python functions to be decorated as macros.
  This enables pulse builder functions to be used within the decorated
  function. The builder macro can then be called from within a pulse
  building context, enabling code reuse.

  For Example:

  .. code-block:: python

      from qiskit import pulse

      @pulse.macro
      def measure(qubit: int):
          pulse.play(pulse.GaussianSquare(16384, 256, 15872),
                     pulse.MeasureChannel(qubit))
          mem_slot = pulse.MemorySlot(0)
          pulse.acquire(16384, pulse.AcquireChannel(0), mem_slot)
          return mem_slot

      with pulse.build(backend=backend) as sched:
          mem_slot = measure(0)
          print(f"Qubit measured into {mem_slot}")

      sched.draw()

- A new class, :class:`~qiskit.circuit.library.PauliTwoDesign`, was added
  to the :mod:`qiskit.circuit.library` which implements a particular form
  of a 2-design circuit from https://arxiv.org/pdf/1803.11173.pdf
  For instance, this circuit can look like:

  .. jupyter-execute::

    from qiskit.circuit.library import PauliTwoDesign
    circuit = PauliTwoDesign(4, reps=2, seed=5, insert_barriers=True)
    circuit.decompose().draw(output='mpl')

- A new pulse drawer :func:`qiskit.visualization.pulse_v2.draw`
  (which is aliased as ``qiskit.visualization.pulse_drawer_v2``) is now
  available. This new pulse drawer supports multiple new features not
  present in the original pulse drawer
  (:func:`~qiskit.visualization.pulse_drawer`).

  * Truncation of long pulse instructions.
  * Visualization of parametric pulses.
  * New stylesheets ``IQXStandard``, ``IQXSimple``, ``IQXDebugging``.
  * Visualization of system info (channel frequency, etc...) by specifying
    :class:`qiskit.providers.Backend` objects for visualization.
  * Specifying ``axis`` objects for plotting to allow further extension of
    generated plots, i.e., for publication manipulations.

  New stylesheets can take callback functions that dynamically modify the apperance of
  the output image, for example, reassembling a collection of channels,
  showing details of instructions, updating appearance of pulse envelopes, etc...
  You can create custom callback functions and feed them into a stylesheet instance to
  modify the figure appearance without modifying the drawer code.
  See pulse drawer module docstrings for details.

  Note that file saving is now delegated to Matplotlib.
  To save image files, you need to call ``savefig`` method with returned ``Figure`` object.

- Adds a :meth:`~qiskit.quantum_info.Statevector.reverse_qargs` method to the
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes. This method reverses
  the order of subsystems in the states and is equivalent to the
  :meth:`qiskit.circuit.QuantumCircuit.reverse_bits` method for N-qubit
  states. For example:

    .. jupyter-execute::

      from qiskit.circuit.library import QFT
      from qiskit.quantum_info import Statevector

      circ = QFT(3)

      state1 = Statevector.from_instruction(circ)
      state2 = Statevector.from_instruction(circ.reverse_bits())

      state1.reverse_qargs() == state2

- Adds a :meth:`~qiskit.quantum_info.Operator.reverse_qargs` method to the
  :class:`qiskit.quantum_info.Operator` class. This method reverses
  the order of subsystems in the operator and is equivalent to the
  :meth:`qiskit.circuit.QuantumCircuit.reverse_bits` method for N-qubit
  operators. For example:

    .. jupyter-execute::

      from qiskit.circuit.library import QFT
      from qiskit.quantum_info import Operator

      circ = QFT(3)

      op1 = Operator(circ)
      op2 = Operator(circ.reverse_bits())

      op1.reverse_qargs() == op2

- The ``latex`` output method for the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :meth:`~qiskit.circuit.QuantumCircuit.draw` method now will use a
  user defined label on gates in the output visualization. For example::

    import math

    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(math.pi/2, 0, label='My Special Rotation')

    qc.draw(output='latex')

- The ``routing_method`` kwarg for the :func:`~qiskit.compiler.transpile`
  function now accepts a new option, ``'none'``. When
  ``routing_method='none'`` no routing pass will be run as part of the
  transpilation. If the circuit does not fit coupling map a
  :class:`~qiskit.transpiler.exceptions.TranspilerError` exception will be
  raised.

- A new gate class, :class:`~qiskit.circuit.library.RVGate`, was added to
  the :mod:`qiskit.circuit.library` module along with the corresponding
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.rv`. The
  :class:`~qiskit.circuit.library.RVGate` is a general rotation gate, similar
  to the :class:`~qiskit.circuit.library.UGate`, but instead of specifying
  Euler angles the three components of a rotation vector are specified where
  the direction of the vector specifies the rotation axis and the magnitude
  specifies the rotation angle about the axis in radians. For example::

    import math

    import np

    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    theta = math.pi / 5
    phi = math.pi / 3
    # RGate axis:
    axis = np.array([math.cos(phi), math.sin(phi)])
    rotation_vector = theta * axis
    qc.rv(*rotation_vector, 0)

- Unbound :class:`~qiskit.circuit.Parameter` objects used in a
  :class:`~qiskit.circuit.QuantumCircuit` object will now be sorted
  by name. This will take effect for the parameters returned by the
  :attr:`~qiskit.circuit.QuantumCircuit.parameters` attribute. Additionally,
  the :meth:`qiskit.circuit.QuantumCircuit.bind_parameters` and
  :meth:`qiskit.circuit.QuantumCircuit.assign_parameters` methods can now take
  in a list of a values which will bind/assign them to the parameters in
  name-sorted order. Previously these methods would only take a dictionary of
  parameters and values. For example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit, Parameter

    circuit = QuantumCircuit(1)
    circuit.rx(Parameter('x'), 0)
    circuit.ry(Parameter('y'), 0)

    print(circuit.parameters)

    bound = circuit.bind_parameters([1, 2])
    bound.draw(output='mpl')

- The constructors for the :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes can now take a
  :class:`~qiskit.circuit.QuantumCircuit` object in to build a
  :class:`~qiskit.quantum_info.Statevector` and
  :class:`~qiskit.quantum_info.DensityMatrix` object from that circuit,
  assuming that the qubits are initialized in :math:`|0\rangle`. For example:

  .. jupyter-execute::

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    statevector = Statevector(qc)
    statevector.draw(output='latex')

- New fake backend classes are available under ``qiskit.test.mock``. These
  included mocked versions of ``ibmq_casablanca``, ``ibmq_sydney``,
  ``ibmq_mumbai``, ``ibmq_lima``, ``ibmq_belem``, ``ibmq_quito``. As
  with the other fake backends, these include snapshots of calibration data
  (i.e. ``backend.defaults()``) and error data (i.e. ``backend.properties()``)
  taken from the real system, and can be used for local testing, compilation
  and simulation.


.. _Release Notes_0.17.0_Known Issues:

Known Issues
------------

- Attempting to add an :class:`qiskit.pulse.Instruction` object
  with a parameterized ``duration`` (ie the value of ``duration`` is
  an unbound :class:`~qiskit.circuit.Parameter` or
  :class:`~qiskit.circuit.ParameterExpression` object) to a
  :class:`qiskit.pulse.Schedule` is not supported. Attempting to do
  so will result in ``UnassignedDurationError``
  :class:`~qiskit.pulse.PulseError` being raised. This is a limitation of
  how the :class:`~qiskit.pulse.Instruction` overlap constraints are
  evaluated currently. This is supported by :class:`~qiskit.pulse.ScheduleBlock`,
  in which the overlap constraints are evaluated just before the execution.

- On Windows systems when parallel execution is enabled for
  :func:`~qiskit.tools.parallel_map` parallelism may not work when called
  from a script running outside of a ``if __name__ == '__main__':`` block.
  This is due to how Python launches parallel processes on Windows. If a
  ``RuntimeError`` or ``AttributeError`` are raised by scripts that call
  :func:`~qiskit.tools.parallel_map` (including using functions that use
  ``parallel_map()`` internally like :func:`~qiskit.compiler.transpile`)
  with Windows and parallelism enabled you can try embedding the script
  calls inside ``if __name__ == '__main__':`` to workaround the issue.
  For example::

    from qiskit import QuantumCircuit, QiskitError
    from qiskit import execute, Aer

    qc1 = QuantumCircuit(2, 2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure([0,1], [0,1])
    # making another circuit: superpositions
    qc2 = QuantumCircuit(2, 2)
    qc2.h([0,1])
    qc2.measure([0,1], [0,1])
    execute([qc1, qc2], Aer.get_backend('qasm_simulator'))

  should be changed to::

    from qiskit import QuantumCircuit, QiskitError
    from qiskit import execute, Aer

    def main():
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure([0,1], [0,1])
        # making another circuit: superpositions
        qc2 = QuantumCircuit(2, 2)
        qc2.h([0,1])
        qc2.measure([0,1], [0,1])
        execute([qc1, qc2], Aer.get_backend('qasm_simulator'))

    if __name__ == '__main__':
        main()

  if any errors are encountered with parallelism on Windows.


.. _Release Notes_0.17.0_Upgrade Notes:

Upgrade Notes
-------------

- The preset pass managers
  :class:`~qiskit.transpiler.preset_passmanagers.level_1_pass_manager`,
  :class:`~qiskit.transpiler.preset_passmanagers.level_2_pass_manager`,
  and :class:`~qiskit.transpiler.preset_passmanagers.level_3_pass_manager`
  (which are used for ``optimization_level`` 1, 2, and 3 in the
  :func:`~qiskit.compiler.transpile` and
  :func:`~qiskit.execute_function.execute` functions) now unconditionally
  use the :class:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition`
  pass for 1 qubit gate optimization. Previously, these pass managers would
  use the :class:`~qiskit.transpiler.passes.Optimize1qGates` pass if the basis
  gates contained ``u1``, ``u2``, or ``u3``. If you want to still use
  the old :class:`~qiskit.transpiler.passes.Optimize1qGates` you will need
  to construct a custom :class:`~qiskit.transpiler.PassManager` with the
  pass.

- Following transpilation of a parameterized
  :class:`~qiskit.circuit.QuantumCircuit`, the
  :attr:`~qiskit.circuit.QuantumCircuit.global_phase` attribute of output
  circuit may no longer be returned in a simplified form, if the global phase
  is a :class:`~qiskit.circuit.ParameterExpression`.

  For example::

    qc = QuantumCircuit(1)
    theta = Parameter('theta')

    qc.rz(theta, 0)
    qc.rz(-theta, 0)

    print(transpile(qc, basis_gates=['p']).global_phase)

  previously returned ``0``, but will now return ``-0.5*theta + 0.5*theta``.
  This change was necessary was to avoid a large runtime performance
  penalty as simplifying symbolic expressions can be quite slow, especially
  if there are many :class:`~qiskit.circuit.ParameterExpression` objects
  in a circuit.

- The :class:`~qiskit.providers.basicaer.BasicAerJob` job objects returned
  from BasicAer backends are now synchronous instances of
  :class:`~qiskit.providers.JobV1`. This means that calls to
  the :meth:`~qiskit.providers.basicaer.QasmSimulatorPy.run` will block
  until the simulation finishes executing. If you want to restore the
  previous async behavior you'll need to wrap the
  :meth:`~qiskit.providers.basicaer.QasmSimulatorPy.run` with something that
  will run in a seperate thread or process like ``futures.ThreadPoolExecutor``
  or ``futures.ProcessPoolExecutor``.

- The ``allow_sample_measuring`` option for the
  BasicAer simulator :class:`~qiskit.providers.basicaer.QasmSimulatorPy` has
  changed from a default of ``False`` to ``True``. This was done to better
  reflect the actual default behavior of the simulator, which would use
  sample measuring if the input circuit supported it (even if it was not
  enabled). If you are running a circuit that doesn't support sample
  measurement (ie it has :class:`~qiskit.circuit.Reset` operations or if
  there are operations after a measurement on a qubit) you should make sure
  to explicitly set this option to ``False`` when you call
  :meth:`~qiskit.providers.basicaer.QasmSimulatorPy.run`.

- The :class:`~qiskit.transpiler.passes.CommutativeCancellation` transpiler
  pass is now aware of the target basis gates, which means it will only
  use gates in the specified basis. Previously, the pass would unconditionally
  replace consecutive gates which commute with
  :class:`~qiskit.circuit.library.ZGate` with the
  :class:`~qiskit.circuit.library.U1Gate`. However, now that the pass is
  basis aware and has a kwarg, ``basis_gates``, for specifying the target
  basis there is a potential change in behavior if the kwarg is not set.
  When the ``basis_gates`` kwarg is not used and there are no variable
  z-rotation gates in the circuit then no commutative cancellation will occur.

- :class:`~qiskit.circuit.Register` (which is the parent class for
  :class:`~qiskit.circuit.QuantumRegister` and
  :class:`~qiskit.circuit.ClassicalRegister` and
  :class:`~qiskit.circuit.Bit` (which is the parent class for
  :class:`~qiskit.circuit.Qubit` and :class:`~qiskit.circuit.Clbit`) objects
  are now immutable. In previous releases it was possible to adjust the value
  of a :attr:`~qiskit.circuit.QuantumRegister.size` or
  :attr:`~qiskit.circuit.QuantumRegister.name` attributes of a
  :class:`~qiskit.circuit.Register` object and the
  :attr:`~qiskit.circuit.Qubit.index` or
  :attr:`~qiskit.circuit.Qubit.register` attributes of a
  :class:`~qiskit.circuit.Bit` object after it was initially
  created. However this would lead to unsound behavior that would corrupt
  container structure that rely on a hash (such as a `dict`) since these
  attributes are treated as immutable properties of a register or bit (see
  `#4705 <https://github.com/Qiskit/qiskit-terra/issues/4705>`__ for more
  details). To avoid this unsound behavior this attributes of a
  :class:`~qiskit.circuit.Register` and :class:`~qiskit.circuit.Bit` are
  no longer settable after initial creation. If you were previously adjusting
  the objects at runtime you will now need to create a new ``Register``
  or ``Bit`` object with the new values.

- The ``DAGCircuit.__eq__`` method (which is used by the ``==`` operator),
  which is used to check structural equality of
  :class:`~qiskit.dagcircuit.DAGCircuit` and
  :class:`~qiskit.circuit.QuantumCircuit` instances, will now
  include the :attr:`~qiskit.circuit.QuantumCircuit.global_phase` and
  :attr:`~qiskit.circuit.QuantumCircuit.calibrations` attributes in the
  fields checked for equality. This means that circuits which would have
  evaluated as equal in prior releases may not anymore if the
  ``global_phase`` or ``calibrations`` differ between the circuits. For
  example, in previous releases this would return ``True``::

    import math

    from qiskit import QuantumCircuit

    qc1 = QuantumCircuit(1)
    qc1.x(0)

    qc2 = QuantumCircuit(1, global_phase=math.pi)
    qc2.x(0)

    print(qc2 == qc1)

  However, now because the ``global_phase`` attribute of the circuits differ
  this will now return ``False``.

- The previously deprecated ``qubits()`` and ``clbits()`` methods on the
  :class:`~qiskit.dagcircuit.DAGCircuit` class, which were deprecated in the
  0.15.0 Terra release, have been removed. Instead you should use the
  :attr:`~qiskit.dagcircuit.DAGCircuit.qubits` and
  :attr:`~qiskit.dagcircuit.DAGCircuit.clbits` attributes of the
  :class:`~qiskit.dagcircuit.DAGCircuit` class. For example, if you were
  running::

    from qiskit.dagcircuit import DAGCircuit

    dag = DAGCircuit()
    qubits = dag.qubits()

  That would be replaced by::

    from qiskit.dagcircuit import DAGCircuit

    dag = DAGCircuit()
    qubits = dag.qubits

- The :class:`~qiskit.providers.models.PulseDefaults` returned by the fake
  pulse backends :py:class:`qiskit.test.mock.FakeOpenPulse2Q` and
  :py:class:`qiskit.test.mock.FakeOpenPulse3Q` have been updated to have
  more realistic pulse sequence definitions. If you are using these fake
  backend classes you may need to update your usage because of these changes.

- The default synthesis method used by
  :func:`~qiskit.quantum_info.decompose_clifford` function in the
  :mod:`~qiskit.quantum_info` module (which gets used internally by the
  :meth:`qiskit.quantum_info.Clifford.to_circuit` method) for more than
  3 qubits now uses a non-optimal greedy compilation routine for Clifford
  elements synthesis, by Bravyi et. al., which typically yields better CX
  cost compared to the old default. If you need to revert to the previous
  Aaronson-Gottesman method this can be done by setting ``method='AG'``.

- The previously deprecated module ``qiskit.visualization.interactive``,
  which was deprecated in the 0.15.0 release, has now been removed. Instead
  you should use the matplotlib based visualizations:

  .. list-table::
    :header-rows: 1

    * - Removed Interactive function
      - Equivalent matplotlib function
    * - ``iplot_bloch_multivector``
      - :func:`qiskit.visualization.plot_bloch_multivector`
    * - ``iplot_state_city``
      - :func:`qiskit.visualization.plot_state_city`
    * - ``iplot_state_qsphere``
      - :func:`qiskit.visualization.plot_state_qsphere`
    * - ``iplot_state_hinton``
      - :func:`qiskit.visualization.plot_state_hinton`
    * - ``iplot_histogram``
      - :func:`qiskit.visualization.plot_histogram`
    * - ``iplot_state_paulivec``
      - :func:`qiskit.visualization.plot_state_paulivec`

- The ``qiskit.Aer`` and ``qiskit.IBMQ`` top level attributes are now lazy
  loaded. This means that the objects will now always exist and warnings will
  no longer be raised on import if ``qiskit-aer`` or ``qiskit-ibmq-provider``
  are not installed (or can't be found by Python). If you were checking for
  the presence of ``qiskit-aer`` or ``qiskit-ibmq-provider`` using these
  module attributes and explicitly comparing to ``None`` or looking for the
  absence of the attribute this no longer will work because they are always
  defined as an object now. In other words running something like::

      try:
          from qiskit import Aer
      except ImportError:
          print("Aer not available")

      or::

      try:
          from qiskit import IBMQ
      except ImportError:
          print("IBMQ not available")

  will no longer work. Instead to determine if those providers are present
  you can either explicitly use ``qiskit.providers.aer.Aer`` and
  ``qiskit.providers.ibmq.IBMQ``::

      try:
          from qiskit.providers.aer import Aer
      except ImportError:
          print("Aer not available")

      try:
          from qiskit.providers.ibmq import IBMQ
      except ImportError:
          print("IBMQ not available")

  or check ``bool(qiskit.Aer)`` and ``bool(qiskit.IBMQ)`` instead, for
  example::

      import qiskit

      if not qiskit.Aer:
          print("Aer not available")
      if not qiskit.IBMQ:
          print("IBMQ not available")

  This change was necessary to avoid potential import cycle issues between
  the qiskit packages and also to improve the import time when Aer or IBMQ
  are not being used.

- The user config file option ``suppress_packaging_warnings`` option in the
  user config file and the ``QISKIT_SUPPRESS_PACKAGING_WARNINGS`` environment
  variable no longer has any effect and will be silently ignored. The warnings
  this option controlled have been removed and will no longer be emitted at
  import time from the ``qiskit`` module.

- The previously deprecated ``condition`` kwarg for
  :class:`qiskit.dagcircuit.DAGNode` constructor has been removed.
  It was deprecated in the 0.15.0 release. Instead you should now be setting
  the classical condition on the :class:`~qiskit.circuit.Instruction` object
  passed into the :class:`~qiskit.dagcircuit.DAGNode` constructor when
  creating a new ``op`` node.

- When creating a new :class:`~qiskit.circuit.Register` (which is the parent
  class for :class:`~qiskit.circuit.QuantumRegister` and
  :class:`~qiskit.circuit.ClassicalRegister`) or
  :class:`~qiskit.circuit.QuantumCircuit` object with a number of bits (eg
  ``QuantumCircuit(2)``), it is now required that number of bits are
  specified as an integer or another type which is castable to unambiguous
  integers(e.g. ``2.0``). Non-integer values will now raise an error as the
  intent in those cases was unclear (you can't have fractional bits). For
  more information on why this was changed refer to:
  `#4855 <https://github.com/Qiskit/qiskit-terra/issues/4855>`__

- `networkx <https://networkx.org/>`__ is no longer a requirement for
  qiskit-terra. All the networkx usage inside qiskit-terra has been removed
  with the exception of 3 methods:

  * :class:`qiskit.dagcircuit.DAGCircuit.to_networkx`
  * :class:`qiskit.dagcircuit.DAGCircuit.from_networkx`
  * :class:`qiskit.dagcircuit.DAGDependency.to_networkx`

  If you are using any of these methods you will need to manually install
  networkx in your environment to continue using them.

- By default on macOS with Python >=3.8 :func:`~qiskit.tools.parallel_map`
  will no longer run in multiple processes. This is a change from previous
  releases where the default behavior was that
  :func:`~qiskit.tools.parallel_map` would launch multiple processes. This
  change was made because with newer versions of macOS with Python 3.8 and
  3.9 multiprocessing is either unreliable or adds significant overhead
  because of the change in Python 3.8 to launch new processes with ``spawn``
  instead of ``fork``. To re-enable parallel execution on macOS with
  Python >= 3.8 you can use the user config file ``parallel`` option or set
  the environment variable ``QISKIT_PARALLEL`` to ``True``.

- The previously deprecated kwarg ``callback`` on the constructor for the
  :class:`~qiskit.transpiler.PassManager` class has been removed. This
  kwarg has been deprecated since the 0.13.0 release (April, 9th 2020).
  Instead you can pass the ``callback`` kwarg to the
  :meth:`qiskit.transpiler.PassManager.run` method directly. For example,
  if you were using::

    from qiskit.circuit.random import random_circuit
    from qiskit.transpiler import PassManager

    qc = random_circuit(2, 2)

    def callback(**kwargs)
      print(kwargs['pass_'])

    pm = PassManager(callback=callback)
    pm.run(qc)

  this can be replaced with::

    from qiskit.circuit.random import random_circuit
    from qiskit.transpiler import PassManager

    qc = random_circuit(2, 2)

    def callback(**kwargs)
      print(kwargs['pass_'])

    pm = PassManager()
    pm.run(qc, callback=callback)

- It is now no longer possible to instantiate a base channel without
  a prefix, such as :class:`qiskit.pulse.Channel` or
  :class:`qiskit.pulse.PulseChannel`. These classes are designed to
  classify types of different user facing channel classes, such
  as :class:`qiskit.pulse.DriveChannel`, but do not have a definition as
  a target resource. If you were previously directly instantiating either
  :class:`qiskit.pulse.Channel` or
  :class:`qiskit.pulse.PulseChannel`, this is no longer allowed. Please use
  the appropriate subclass.

- When the ``require_cp`` and/or ``require_tp`` kwargs of
  :func:`qiskit.quantum_info.process_fidelity`,
  :func:`qiskit.quantum_info.average_gate_fidelity`,
  :func:`qiskit.quantum_info.gate_error` are ``True``, they will now only log a
  warning rather than the previous behavior of raising a
  :class:`~qiskit.exceptions.QiskitError` exception if the input channel is
  non-CP or non-TP respectively.

- The :class:`~qiskit.circuit.library.QFT` class in the
  :mod:`qiskit.circuit.library` module now computes the Fourier transform
  using a little-endian representation of tensors, i.e. the state
  :math:`|1\rangle` maps to :math:`|0\rangle - |1\rangle + |2\rangle - ..`
  assuming the computational basis correspond to little-endian bit ordering
  of the integers. :math:`|0\rangle = |000\rangle, |1\rangle = |001\rangle`,
  etc. This was done to make it more consistent with the rest of Qiskit,
  which uses a little-endian convention for bit order. If you were depending
  on the previous bit order you can use the
  :meth:`~qiskit.circuit.library.QFT.reverse_bits` method to revert to the
  previous behavior. For example::

    from qiskit.circuit.library import QFT

    qft = QFT(5).reverse_bits()

- The ``qiskit.__qiskit_version__`` module attribute was previously a ``dict``
  will now return a custom read-only ``Mapping`` object that checks the
  version of qiskit elements at runtime instead of at import time. This was
  done to speed up the import path of qiskit and eliminate a possible import
  cycle by only importing the element packages at runtime if the version
  is needed from the package. This should be fully compatible with the
  ``dict`` previously return and for most normal use cases there will be no
  difference. However, if some applications were relying on either mutating
  the contents or explicitly type checking it may require updates to adapt to
  this change.

- The ``qiskit.execute`` module has been renamed to
  :mod:`qiskit.execute_function`. This was necessary to avoid a potentical
  name conflict between the :func:`~qiskit.execute_function.execute` function
  which is re-exported as ``qiskit.execute``. ``qiskit.execute`` the function
  in some situations could conflict with ``qiskit.execute`` the module which
  would lead to a cryptic error because Python was treating ``qiskit.execute``
  as the module when the intent was to the function or vice versa. The module
  rename was necessary to avoid this conflict. If you're importing
  ``qiskit.execute`` to get the module (typical usage was
  ``from qiskit.execute import execute``) you will need to update this to
  use ``qiskit.execute_function`` instead. ``qiskit.execute`` will now always
  resolve to the function.

- The ``qiskit.compiler.transpile``, ``qiskit.compiler.assemble``,
  ``qiskit.compiler.schedule``, and ``qiskit.compiler.sequence`` modules have
  been renamed to ``qiskit.compiler.transpiler``,
  ``qiskit.compiler.assembler``, ``qiskit.compiler.scheduler``, and
  ``qiskit.compiler.sequence`` respectively. This was necessary to avoid a
  potentical name conflict between the modules and the re-exported function
  paths :func:`qiskit.compiler.transpile`, :func:`qiskit.compiler.assemble`,
  :func:`qiskit.compiler.schedule`, and :func:`qiskit.compiler.sequence`.
  In some situations this name conflict between the module path and
  re-exported function path would lead to a cryptic error because Python was
  treating an import as the module when the intent was to use the function or
  vice versa. The module rename was necessary to avoid this conflict. If
  you were using the imports to get the modules before (typical usage would
  be like``from qiskit.compiler.transpile import transpile``) you will need
  to update this to use the new module paths.
  :func:`qiskit.compiler.transpile`, :func:`qiskit.compiler.assemble`,
  :func:`qiskit.compiler.schedule`, and :func:`qiskit.compiler.sequence`
  will now always resolve to the functions.

- The :class:`qiskit.quantum_info.Quaternion` class was moved from the
  ``qiskit.quantum_info.operator`` submodule to the
  ``qiskit.quantum_info.synthesis`` submodule to better reflect it's purpose.
  No change is required if you were importing it from the root
  :mod:`qiskit.quantum_info` module, but if you were importing from
  ``qiskit.quantum_info.operator`` you will need to update your import path.

- Removed the ``QuantumCircuit.mcmt`` method, which has been
  deprecated since the Qiskit Terra 0.14.0 release in April 2020.
  Instead of using the method, please use the
  :class:`~qiskit.circuit.library.MCMT` class instead to construct
  a multi-control multi-target gate and use the
  :meth:`qiskit.circuit.QuantumCircuit.append` or
  :meth:`qiskit.circuit.QuantumCircuit.compose` to add it to a circuit.

  For example, you can replace::

      circuit.mcmt(ZGate(), [0, 1, 2], [3, 4])

  with::

      from qiskit.circuit.library import MCMT
      mcmt = MCMT(ZGate(), 3, 2)
      circuit.compose(mcmt, range(5))

- Removed the ``QuantumCircuit.diag_gate`` method which has been deprecated since the
  Qiskit Terra 0.14.0 release in April 2020. Instead, use the
  :meth:`~qiskit.circuit.QuantumCircuit.diagonal` method of :class:`~qiskit.circuit.QuantumCircuit`.

- Removed the ``QuantumCircuit.ucy`` method which has been deprecated since the
  Qiskit Terra 0.14.0 release in April 2020. Instead, use the
  :meth:`~qiskit.circuit.QuantumCircuit.ucry` method of :class:`~qiskit.circuit.QuantumCircuit`.

- The previously deprecated ``mirror()`` method for
  :class:`qiskit.circuit.QuantumCircuit` has been removed. It was deprecated
  in the 0.15.0 release. The :meth:`qiskit.circuit.QuantumCircuit.reverse_ops`
  method should be used instead since mirroring could be confused with
  swapping the output qubits of the circuit. The ``reverse_ops()`` method
  only reverses the order of gates that are applied instead of mirroring.

- The previously deprecated support passing a float (for the ``scale`` kwarg
  as the first positional argument to the
  :meth:`qiskit.circuit.QuantumCircuit.draw` has been removed. It was
  deprecated in the 0.12.0 release. The first positional argument to the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method is now the ``output``
  kwarg which does not accept a float. Instead you should be using ``scale``
  as a named kwarg instead of using it positionally.

  For example, if you were previously calling ``draw`` with::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.draw(0.75, output='mpl')

  this would now need to be::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.draw(output='mpl', scale=0.75)

  or::

    qc.draw('mpl', scale=0.75)

- Features of Qiskit Pulse (:mod:`qiskit.pulse`) which were deprecated
  in the 0.15.0 release (August, 2020) have been removed. The full set
  of changes are:

  .. list-table::
    :header-rows: 1

    * - Module
      - Old
      - New
    * - ``qiskit.pulse.library``
      - ``SamplePulse``
      - :class:`~qiskit.pulse.library.Waveform`
    * - ``qiskit.pulse.library``
      - ``ConstantPulse``
      - :class:`~qiskit.pulse.library.Constant`
    * - (module rename)
      - ``pulse.pulse_lib`` Module
      - :mod:`qiskit.pulse.library`

  .. list-table::
    :header-rows: 1

    * - Class
      - Old method
      - New method
    * - :class:`~qiskit.pulse.library.ParametricPulse`
      - ``get_sample_pulse``
      - :class:`~qiskit.pulse.library.ParametricPulse.get_waveform`
    * - :class:`~qiskit.pulse.instructions.Instruction`
      - ``command``
      - N/A. Commands and Instructions have been unified.
        Use :meth:`~qiskit.pulse.instructions.Instruction.operands`
        to get information about the instruction data.
    * - :class:`~qiskit.pulse.instructions.Acquire`
      - ``acquires``, ``mem_slots``, ``reg_slots``
      - :meth:`~qiskit.pulse.instructions.Acquire.acquire`,
        :meth:`~qiskit.pulse.instructions.Acquire.mem_slot`,
        :meth:`~qiskit.pulse.instructions.Acquire.reg_slot`. (The
        :class:`~qiskit.pulse.instructions.Acquire` instruction no
        longer broadcasts across multiple qubits.)

- The dictionary previously held on :class:`~qiskit.dagcircuit.DAGCircuit`
  edges has been removed. Instead, edges now hold the
  :class:`~qiskit.circuit.Bit` instance which had previously been included in
  the dictionary as its ``'wire'`` field. Note that the NetworkX graph
  returned by :meth:`~qiskit.dagcircuit.DAGCircuit.to_networkx` will still
  have a dictionary for its edge attributes, but the ``'name'`` field will no
  longer be populated.

- The :attr:`~qiskit.circuit.QuantumCircuit.parameters` attribute of the
  :class:`~qiskit.circuit.QuantumCircuit` class no longer is returning a
  ``set``. Instead it returns a ``ParameterView`` object which implements
  all the methods that ``set`` offers (albeit deprecated). This was done
  to support a model that preserves name-sorted parameters. It
  should be fully compatible with any previous usage of the ``set`` returned
  by the :attr:`~qiskit.circuit.QuantumCircuit.parameters` attribute, except
  for where explicit type checking of a set was done.

- When running :func:`~qiskit.compiler.transpile` on a
  :class:`~qiskit.circuit.QuantumCircuit` with
  :meth:`~qiskit.circuit.QuantumCircuit.delay` instructions, the units will
  be converted to dt if the value of dt (sample time) is known to
  :func:`~qiskit.compiler.transpile`, either explicitly via the ``dt``
  kwarg or via the :class:`~qiskit.providers.models.BackendConfiguration` for
  a ``Backend`` object passed in via the ``backend`` kwarg.

- The interpretation of ``meas_map`` (which
  is an attribute of a
  :class:`~qiskit.providers.models.PulseBackendConfiguration` object or
  as the corresponding ``meas_map`` kwarg on the
  :func:`~qiskit.compiler.schedule`, :func:`~qiskit.compiler.assemble`,
  :func:`~qiskit.compiler.sequence`, or
  :func:`~qiskit.execute_function.execute` functions) has been updated
  to better match the true constraints of the hardware. The format of this
  data is a list of lists, where the items in the inner list are integers
  specifying qubit labels. For instance::

      [[A, B, C], [D, E, F, G]]

  Previously, the ``meas_map`` constraint was interpreted such that
  if one qubit was acquired (e.g. A), then all other qubits sharing
  a subgroup with that qubit (B and C) would have to be acquired
  at the same time and for the same duration. This constraint has been
  relaxed. One acquisition does not require more acquisitions. (If A is
  acquired, B and C do **not** need to be acquired.) Instead, qubits in the
  same measurement group cannot be acquired in a partially overlapping way
  -- think of the ``meas_map`` as specifying a shared acquisition resource
  (If we acquire A from ``t=1000`` to ``t=2000``, we cannot acquire B
  starting from ``1000<t<2000``). For example:

  .. code-block:: python

      # Good
      meas_map = [[0, 1]]
      # Acquire a subset of [0, 1]
      sched = pulse.Schedule()
      sched = sched.append(pulse.Acquire(10, acq_q0))

      # Acquire 0 and 1 together (same start time, same duration)
      sched = pulse.Schedule()
      sched = sched.append(pulse.Acquire(10, acq_q0))
      sched = sched.append(pulse.Acquire(10, acq_q1))

      # Acquire 0 and 1 disjointly
      sched = pulse.Schedule()
      sched = sched.append(pulse.Acquire(10, acq_q0))
      sched = sched.append(pulse.Acquire(10, acq_q1)) << 10

      # Acquisitions overlap, but 0 and 1 aren't in the same measurement
      # grouping
      meas_map = [[0], [1]]
      sched = pulse.Schedule()
      sched = sched.append(pulse.Acquire(10, acq_q0))
      sched = sched.append(pulse.Acquire(10, acq_q1)) << 1

      # Bad: 0 and 1 are in the same grouping, but acquisitions
      # partially overlap
      meas_map = [[0, 1]]
      sched = pulse.Schedule()
      sched = sched.append(pulse.Acquire(10, acq_q0))
      sched = sched.append(pulse.Acquire(10, acq_q1)) << 1


.. _Release Notes_0.17.0_Deprecation Notes:

Deprecation Notes
-----------------

- Two new arguments have been added to
  :meth:`qiskit.dagcircuit.DAGNode.semantic_eq`, ``bit_indices1`` and
  ``bit_indices2``, which are expected to map the
  :class:`~qiskit.circuit.Bit` instances in each
  :class:`~qiskit.dagcircuit.DAGNode` to their index in ``qubits`` or
  ``clbits`` list of their respective
  :class:`~qiskit.dagcircuit.DAGCircuit`. During the deprecation period,
  these arguments are optional and when **not** specified the mappings will
  be automatically constructed based on the ``register`` and ``index``
  properties of each :class:`~qiskit.circuit.Bit` instance. However, in a
  future release, they will be required arguments and the mapping will need
  to be supplied by the user.

- The :mod:`~qiskit.pulse` builder functions:

  * :py:func:`qiskit.pulse.call_circuit`
  * :py:func:`qiskit.pulse.call_schedule`

  are deprecated and will be removed in a future release.
  These functions are unified into :py:func:`qiskit.pulse.call` which should
  be used instead.

- The :class:`qiskit.pulse.Schedule` method
  :py:meth:`qiskit.pulse.Schedule.flatten` method is deprecated and will
  be removed in a future release. Instead you can use the
  :py:func:`qiskit.pulse.transforms.flatten` function which will perform
  the same operation.

- The :meth:`~qiskit.pulse.channels.Channel.assign_parameters` for the
  following classes:

   * :py:class:`qiskit.pulse.channels.Channel`,
   * :py:class:`qiskit.pulse.library.Pulse`,
   * :py:class:`qiskit.pulse.instructions.Instruction`,

  and all their subclasses is now deprecated and will be removed in a future
  release. This functionality has been subsumed
  :py:class:`~qiskit.pulse.ScheduleBlock` which is the future direction for
  constructing parameterized pulse programs.

- The :attr:`~qiskit.pulse.channels.Channel.parameters` attribute for
  the following clasess:

    * :py:class:`~qiskit.pulse.channels.Channel`
    * :py:class:`~qiskit.pulse.instructions.Instruction`.

  is deprecated and will be removed in a future release. This functionality
  has been subsumed :py:class:`~qiskit.pulse.ScheduleBlock` which is the
  future direction for constructing parameterized pulse programs.

- Python 3.6 support has been deprecated and will be removed in a future
  release. When support is removed you will need to upgrade the Python
  version you're using to Python 3.7 or above.

- Two :class:`~qiskit.circuit.QuantumCircuit` methods
  :meth:`~qiskit.circuit.QuantumCircuit.combine` and
  :meth:`~qiskit.circuit.QuantumCircuit.extend` along with their corresponding
  Python operators ``+`` and ``+=`` are deprecated and will be removed in a
  future release. Instead the :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.compose` should be used.
  The :meth:`~qiskit.circuit.QuantumCircuit.compose` method allows more
  flexibility in composing two circuits that do not have matching registers.
  It does not, however, automatically add qubits/clbits unlike the deprecated
  methods. To add a circuit on new qubits/clbits, the
  :meth:`qiskit.circuit.QuantumCircuit.tensor` method can be used. For
  example::

      from qiskit.circuit import QuantumRegister, QuantumCircuit

      a = QuantumRegister(2, 'a')
      circuit_a = QuantumCircuit(a)
      circuit_a.cx(0, 1)

      b = QuantumRegister(2, 'b')
      circuit_b = QuantumCircuit(b)
      circuit_b.cz(0, 1)

      # same as circuit_a + circuit_b (or combine)
      added_with_different_regs = circuit_b.tensor(circuit_a)

      # same as circuit_a + circuit_a (or combine)
      added_with_same_regs = circuit_a.compose(circuit_a)

      # same as circuit_a += circuit_b (or extend)
      circuit_a = circuit_b.tensor(circuit_a)

      # same as circuit_a += circuit_a (or extend)
      circuit_a.compose(circuit_a, inplace=True)

- Support for passing :class:`~qiskit.circuit.Qubit` instances to the
  ``qubits`` kwarg of the :meth:`qiskit.transpiler.InstructionDurations.get`
  method has been deprecated and will be removed in a future release.
  Instead, you should call the
  :meth:`~qiskit.transpiler.InstructionDurations.get` method with the integer
  indices of the desired qubits.

- Using ``@`` (``__matmul__``) for invoking the ``compose`` method
  of ``BaseOperator`` subclasses (eg :class:`~qiskit.quantum_info.Operator`)
  is deprecated and will be removed in a future release. The
  :meth:`qiskit.quantum_info.Operator.compose` method can be used directly or
  also invoked using the ``&`` (``__and__``) operator.

- Using ``*`` (``__mul__``) for calling the
  :meth:`~qiskit.quantum_info.Operator.dot` method of ``BaseOperator``
  subclasses (eg :class:`qiskit.quantum_info.Operator`) is deprecated and
  will be removed in a future release. Instead you can just call the
  :meth:`~qiskit.quantum_info.Operator.dot` directly.

- Using ``@`` (``__matmul__``) for invoking the
  :meth:`~qiskit.quantum_info.Statevector.evolve` method
  of the :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes is deprecated and will
  be removed in a future release.. The ``evolve`` method can be used directly
  or also invoked using the ``&`` (``__and__``) operator.

- The ``qiskit.pulse.schedule.ParameterizedSchedule`` class has been
  deprecated and will be removed in a future release. Instead you can
  directly parameterize pulse :class:`~qiskit.pulse.Schedule` objects with
  a :class:`~qiskit.circuit.Parameter` object, for example::

    from qiskit.circuit import Parameter
    from qiskit.pulse import Schedule
    from qiskit.pulse import ShiftPhase, DriveChannel

    theta = Parameter('theta')
    target_schedule = Schedule()
    target_schedule.insert(0, ShiftPhase(theta, DriveChannel(0)), inplace=True)

- The :class:`qiskit.pulse.ScheduleComponent` class in the
  :mod:`qiskit.pulse` module has been deprecated and will be removed in a
  future release. Its usage should be replaced either using a
  :class:`qiskit.pulse.Schedule` or :class:`qiskit.pulse.Instruction`
  directly. Additionally, the primary purpose of the ``ScheduleComponent``
  class was as a common base class for both :class:`~qiskit.pulse.Schedule`
  and :class:`~qiskit.pulse.Instruction` for any place that was explicitly
  type checking or documenting accepting a ``ScheduleComponent`` input
  should be updated to accept :class:`~qiskit.pulse.Instruction` or
  :class:`~qiskit.pulse.Schedule`.

- The JSON Schema files and usage for the IBMQ API payloads are deprecated
  and will be removed in a future release. This includes everything under
  the :mod:`qiskit.schemas` module and the :mod:`qiskit.validation` module.
  This also includes the ``validate`` kwargs for
  :meth:`qiskit.qobj.QasmQobj.to_dict` and
  :meth:`qiskit.qobj.QasmQobj.to_dict` along with the module level
  fastjsonschema validators in :mod:`qiskit.qobj` (which do not raise a
  deprecation warning). The schema files have been moved
  to the `Qiskit/ibmq-schemas <https://github.com/Qiskit/ibmq-schemas>`__
  repository and those should be treated as the canonical versions of the
  API schemas. Moving forward only those schemas will recieve updates and
  will be used as the source of truth for the schemas. If you were relying
  on the schemas bundled in qiskit-terra you should update to
  use that repository instead.

- The :mod:`qiskit.util` module has been deprecated and will be removed
  in a future release. It has been replaced by :mod:`qiskit.utils` which
  provides  the same functionality and will be expanded in the future. Note
  that no ``DeprecationWarning`` will be emitted regarding this deprecation
  since it was not feasible on Python 3.6.

- The :class:`~qiskit.transpiler.passes.CXDirection` transpiler pass in the
  :mod:`qiskit.transpiler.passes` module has been deprecated and will be
  removed in a future release. Instead the
  :class:`~qiskit.transpiler.GateDirection` should be used. It behaves
  identically to the :class:`~qiskit.transpiler.passes.CXDirection` except
  that it now also supports transforming a circuit with
  :class:`~qiskit.circuit.library.ECRGate` gates in addition to
  :class:`~qiskit.circuit.library.CXGate` gates.

- The :class:`~qiskit.transpiler.passes.CheckCXDirection` transpiler pass in
  the :mod:`qiskit.transpiler.passes` module has been deprecated and will be
  removed in a future release. Instead the
  :class:`~qiskit.transpiler.CheckGateDirection` pass should be used.
  It behaves identically to the
  :class:`~qiskit.transpiler.passes.CheckCXDirection` except
  that it now also supports checking the direction of all 2-qubit gates, not
  just :class:`~qiskit.circuit.library.CXGate` gates.

- The :class:`~qiskit.circuit.library.WeightedAdder` method
  :meth:`~qiskit.circuit.library.WeightedAdder.num_ancilla_qubits` is
  deprecated and will be removed in a future release. It has been replaced
  with the :attr:`qiskit.circuit.library.WeightedAdder.num_ancillas` attribute
  which is consistent with other circuit libraries' APIs.

- The following legacy methods of the :class:`qiskit.quantum_info.Pauli` class
  have been deprecated. See the method documentation for replacement use in
  the updated Pauli class.

  * :meth:`~qiskit.quantum_info.Pauli.from_label`
  * :meth:`~qiskit.quantum_info.Pauli.sgn_prod`
  * :meth:`~qiskit.quantum_info.Pauli.to_spmatrix`
  * :meth:`~qiskit.quantum_info.Pauli.kron`
  * :meth:`~qiskit.quantum_info.Pauli.update_z`
  * :meth:`~qiskit.quantum_info.Pauli.update_x`
  * :meth:`~qiskit.quantum_info.Pauli.insert_paulis`
  * :meth:`~qiskit.quantum_info.Pauli.append_paulis`
  * :meth:`~qiskit.quantum_info.Pauli.delete_qubits`
  * :meth:`~qiskit.quantum_info.Pauli.pauli_single`
  * :meth:`~qiskit.quantum_info.Pauli.random`

- Using a ``list`` or ``numpy.ndarray`` as the ``channel`` or ``target``
  argument for the :func:`qiskit.quantum_info.process_fidelity`,
  :func:`qiskit.quantum_info.average_gate_fidelity`,
  :func:`qiskit.quantum_info.gate_error`, and
  :func:`qiskit.quantum_info.diamond_norm` functions has been
  deprecated and will not be supported in a future release. The inputs should
  instead be a :class:`~qiskit.circuit.Gate` or a ``BaseOperator`` subclass
  object (eg. :class:`~qiskit.quantum_info.Operator`,
  :class:`~qiskit.quantum_info.Choi`, etc.)

- Accessing references from :class:`~qiskit.circuit.Qubit` and
  :class:`~qiskit.circuit.Clbit` instances to their containing registers
  via the :attr:`~qiskit.circuit.Qubit.register` or
  :attr:`~qiskit.circuit.Qubit.index` properties has been deprecated and will
  be removed in a future release. Instead, :class:`~qiskit.circuit.Register`
  objects can be queried to find the :class:`~qiskit.circuit.Bit` objects
  they contain.

- The current functionality of the :func:`qiskit.visualization.pulse_drawer`
  function is deprecated and will be replaced by
  :func:`qiskit.visualization.pulse_drawer_v2` (which is not backwards
  compatible) in a future release.

- The use of methods inherited from the ``set`` type on the output of the
  :attr:`~qiskit.circuit.QuantumCircuit.parameters` attribute (which used to
  be a ``set``) of the :class:`~qiskit.circuit.QuantumCircuit` class are
  deprecated and will be removed in a future release. This includes the
  methods from the ``add()``, ``difference()``, ``difference_update()``,
  ``discard()``, ``intersection()``, ``intersection_update()``,
  ``issubset()``, ``issuperset()``, ``symmetric_difference()``,
  ``symmetric_difference_update()``, ``union()``, ``update()``,
  ``__isub__()`` (which is the ``-=`` operator), and ``__ixor__()`` (which is
  the ``^=`` operator).

- The name of the first (and only) positional argument for the
  :meth:`qiskit.circuit.QuantumCircuit.bind_parameters` method has changed
  from ``value_dict`` to ``values``. The passing an argument in with the
  name ``values_dict`` is deprecated and will be removed in future release.
  For example, if you were previously calling
  :meth:`~qiskit.circuit.QuantumCircuit.bind_parameters` with a call like:
  ``bind_parameters(values_dict={})`` this is deprecated and should be
  replaced by ``bind_parameters(values={})`` or even better just pass the
  argument positionally ``bind_parameters({})``.

- The name of the first (and only) positional argument for the
  :meth:`qiskit.circuit.QuantumCircuit.assign_parameters` method has changed
  from ``param_dict`` to ``parameters``. Passing an argument in with the name
  ``param_dict`` is deprecated and will be removed in future release. For
  example, if you were previously calling
  :meth:`~qiskit.circuit.QuantumCircuit.assign_parameters` with a call like:
  ``assign_parameters(param_dict={})`` this is deprecated and should be
  replaced by ``assign_parameters(values={})`` or even better just pass the
  argument positionally ``assign_parameters({})``.


.. _Release Notes_0.17.0_Bug Fixes:

Bug Fixes
---------

- Fixed an issue where the :func:`~qiskit.execute_function.execute` function
  would raise :class:`~qiskit.exceptions.QiskitError` exception when a
  :class:`~qiskit.circuit.ParameterVector` object was passed in for the
  ``parameter_bind`` kwarg. parameter. For example, it is now possible to
  call something like::

    execute(circuit, backend, parameter_binds=[{pv1: [...], pv2: [...]}])

  where ``pv1`` and ``pv2`` are :class:`~qiskit.circuit.ParameterVector`
  objects.
  Fixed `#5467 <https://github.com/Qiskit/qiskit-terra/issues/5467>`__

- Fixed an issue with the labels of parametric pulses in the
  :class:`~qiskit.qobj.PulseQobjInstruction` class were not being properly
  set as they are with sampled pulses. This also means that pulse names
  that are imported from the :class:`~qiskit.providers.models.PulseDefaults`
  returned by a :class:`~qiskit.providers.Backend`, such as ``x90``, ``x90m``,
  etc, will properly be set.
  Fixed `#5363 <https://github.com/Qiskit/qiskit-terra/issues/5363>`__

- Fixed an issue where unbound parameters only occurring in
  the :attr:`~qiskit.circuit.QuantumCircuit.global_phase` attribute of
  a :class:`~qiskit.circuit.QuantumCircuit` object would not
  show in the :attr:`~qiskit.circuit.QuantumCircuit.parameters` attribute
  and could not be bound.
  Fixed `#5806 <https://github.com/Qiskit/qiskit-terra/issues/5806>`__

- The :attr:`~qiskit.circuit.QuantumCircuit.calibrations` attribute
  of :class:`~qiskit.circuit.QuantumCircuit` objects are now preserved when
  the ``+=`` (ie the :meth:`~qiskit.circuit.QuantumCircuit.extend`
  method) and the ``+`` (ie the :meth:`~qiskit.circuit.QuantumCircuit.combine`
  method) are used.
  Fixed `#5930 <https://github.com/Qiskit/qiskit-terra/pull/5930>`__ and
  `#5908 <https://github.com/Qiskit/qiskit-terra/issues/5908>`__

- The :attr:`~qiskit.circuit.Register.name` setter method of class
  :class:`~qiskit.circuit.Register` (which is the parent class of
  :class:`~qiskit.circuit.QuantumRegister` and
  :class:`~qiskit.circuit.ClassicalRegister`) previously did not check if
  the assigned string was a valid register name as per the
  `OpenQASM specification <https://arxiv.org/pdf/1707.03429v2.pdf>`__.
  This check was previously only performed when the name was specified in the
  constructor, this has now been fixed so that setting the ``name``
  attribute directly with an invalid value will now also raise an
  exception.
  Fixed `#5461 <https://github.com/Qiskit/qiskit-terra/issues/5461>`__

- Fixed an issue with the :func:`qiskit.visualization.circuit_drawer` function
  and :meth:`qiskit.circuit.QuantumCircuit.draw` method when visualizing a
  :class:`~qiskit.circuit.QuantumCircuit` with a
  :class:`~qiskit.circuit.Gate` that has a classical condition
  after a :class:`~qiskit.circuit.Measure` that used the same
  :class:`~qiskit.circuit.ClassicalRegister`, it was possible
  for the conditional :class:`~qiskit.circuit.Gate` to be displayed to the
  left of the :class:`~qiskit.circuit.Measure`.
  Fixed `#5387 <https://github.com/Qiskit/qiskit-terra/issues/5387>`__

- In the transpiler pass :class:`qiskit.transpiler.passes.CSPLayout` a bias
  towards lower numbered qubits could be observed. This undesireable bias has
  been fixed by shuffling the candidates to randomize the results.
  Furthermore, the usage of the :class:`~qiskit.transpiler.passes.CSPLayout`
  pass in the :mod:`~qiskit.transpiler.preset_passmanagers` (for level 2 and
  3) has been adjusted to use a configured seed if the ``seed_transpiler``
  kwarg is set when :func:`~qiskit.compiler.transpile` is called.
  Fixed `#5990 <https://github.com/Qiskit/qiskit-terra/issues/5990>`__

- Fixes a bug where the ``channels`` field for a
  :class:`~qiskit.providers.models.PulseBackendConfiguration` object was
  not being included in the output of the
  :class:`qiskit.providers.models.PulseBackendConfiguration.to_dict` method.
  Fixed `#5579 <https://github.com/Qiskit/qiskit-terra/issues/5579>`__

- Fixed the ``'circular'`` entanglement in the
  :class:`qiskit.circuit.library.NLocal` circuit class for the edge
  case where the circuit has the same size as the entanglement block (e.g. a two-qubit
  circuit and CZ entanglement gates). In this case there should only be one entanglement
  gate, but there was accidentially added a second one in the inverse direction as the
  first.
  Fixed `Qiskit/qiskit-aqua#1452 <https://github.com/Qiskit/qiskit-aqua/issues/1452>`__

- Fixed the handling of breakpoints in the
  :class:`~qiskit.circuit.library.PiecewisePolynomialPauliRotations` class
  in the :mod:`qiskit.circuit.library`. Now for ``n`` intervals,
  ``n+1`` breakpoints are allowed. This enables specifying another end
  interval other than :math:`2^\text{num qubits}`. This is important because
  from the end of the last interval to :math:`2^\text{num qubits}` the function
  is the identity.

- Fixed an issue in the :class:`qiskit.circuit.library.Permutation` circuit
  class where some permutations would not be properly generated. This issue
  could also effect :class:`qiskit.circuit.library.QuantumVolume` if it were
  called with `classical_permutation=False``.
  Fixed `#5812 <https://github.com/Qiskit/qiskit-terra/issues/5812>`__

- Fixed an issue where generating QASM output with the
  :meth:`~qiskit.circuit.QuantumCircuit.qasm` method for a
  :class:`~qiskit.circuit.QuantumCircuit` object that has a
  :class:`~qiskit.circuit.ControlledGate` with an open control the output
  would be as if all controls were closed independent of the specified
  control state. This would result in a different circuit being created
  from :meth:`~qiskit.circuit.QuantumCircuit.from_qasm_str` if
  parsing the generated QASM.

  This was fixed by updating the QASM output from
  :meth:`~qiskit.circuit.QuantumCircuit.qasm` by defining a composite gate
  which uses :class:`~qiskit.circuit.XGate` to implement the open controls.
  The composite gate is named like ``<original_gate_name>_o<ctrl_state>``
  where ``o`` stands for open control and ``ctrl_state`` is the integer value
  of the control state.
  Fixed `#5443 <https://github.com/Qiskit/qiskit-terra/issues/5443>`__

- Fixed an issue where binding :class:`~qiskit.circuit.Parameter` objects
  in a :class:`~qiskit.circuit.QuantumCircuit` with the ``parameter_binds``
  in the :class:`~qiskit.execute_function.execute` function would cause all
  the bound :class:`~qiskit.circuit.QuantumCircuit` objects would have the
  same :attr:`~qiskit.circuit.QuantumCircuit.name`, which meant the
  result names were also not unique. This fix causes
  the :meth:`~qiskit.circuit.QuantumCircuit.bind_parameters` and
  :meth:`~qiskit.circuit.QuantumCircuit.assign_parameters` to assign a unique
  circuit name when ``inplace=False`` as::

     <base name>-<class instance no.>[-<pid name>]

  where ``<base name>`` is the name supplied by the "name" kwarg,
  otherwise it defaults to "circuit". The class instance number gets
  incremented every time an instance of the class is generated. ``<pid name>``
  is appended if called outside the main process.
  Fixed `#5185 <https://github.com/Qiskit/qiskit-terra/issues/5185>`__

- Fixed an issue with the :func:`~qiskit.compiler.scheduler` function where
  it would raise an exception if an input circuit contained an unbound
  :class:`~qiskit.circuit.QuantumCircuit` object.
  Fixed `#5304 <https://github.com/Qiskit/qiskit-terra/issues/5304>`__

- Fixed an issue in the :class:`qiskit.transpiler.passes.TemplateOptimization`
  transpiler passes where template circuits that contained unbound
  :class:`~qiskit.circuit.Parameter` objects would crash under some scenarios
  if the parameters could not be bound during the template matching.
  Now, if the :class:`~qiskit.circuit.Parameter` objects can not be bound
  templates with unbound :class:`~qiskit.circuit.Parameter` are discarded and
  ignored by the :class:`~qiskit.transpiler.passes.TemplateOptimization` pass.
  Fixed `#5533 <https://github.com/Qiskit/qiskit-terra/issues/5533>`__

- Fixed an issue with the :func:`qiskit.visualization.timeline_drawer`
  function where classical bits were inproperly handled.
  Fixed `#5361 <https://github.com/Qiskit/qiskit-terra/issues/5361>`__

- Fixed an issue in the :func:`qiskit.visualization.circuit_drawer` function
  and the :meth:`qiskit.circuit.QuantumCircuit.draw` method where
  :class:`~qiskit.circuit.Delay` instructions in a
  :class:`~qiskit.circuit.QuantumCircuit` object were not being correctly
  treated as idle time. So when the ``idle_wires`` kwarg was set to
  ``False`` the wires with the :class:`~qiskit.circuit.Delay` objects would
  still be shown. This has been fixed so that the idle wires are removed from
  the visualization if there are only :class:`~qiskit.circuit.Delay` objects
  on a wire.

- Previously, when the option ``layout_method`` kwarg was provided to
  the :func:`~qiskit.compiler.transpile` function and the
  ``optimization_level`` kwarg was set to >= 2 so that the pass
  :class:`qiskit.transpiler.passes.CSPLayout` would run, if
  :class:`~qiskit.transpiler.passes.CSPLayout` found a solution then
  the method in ``layout_method`` was not executed. This has been fixed so
  that if specified, the ``layout_method`` is always honored.
  Fixed `#5409 <https://github.com/Qiskit/qiskit-terra/issues/5409>`__

- When the argument ``coupling_map=None`` (either set explicitly, set
  implicitly as the default value, or via the ``backend`` kwarg), the
  transpiling process was not "embedding" the circuit. That is, even when an
  ``initial_layout`` was specified, the virtual qubits were not assigned to
  physical qubits. This has been fixed so that now, the
  :func:`qiskit.compiler.transpile` function honors the ``initial_layout``
  argument by embedding the circuit:

  .. jupyter-execute::

      from qiskit import QuantumCircuit, QuantumRegister
      from qiskit.compiler import transpile

      qr = QuantumRegister(2, name='qr')
      circ = QuantumCircuit(qr)
      circ.h(qr[0])
      circ.cx(qr[0], qr[1])

      transpile(circ, initial_layout=[1, 0]).draw(output='mpl')


  If the ``initial_layout`` refers to more qubits than in the circuit, the
  transpiling process will extended the circuit with ancillas.

  .. jupyter-execute::

      from qiskit import QuantumCircuit, QuantumRegister
      from qiskit.compiler import transpile

      qr = QuantumRegister(2, name='qr')
      circ = QuantumCircuit(qr)
      circ.h(qr[0])
      circ.cx(qr[0], qr[1])

      transpile(circ, initial_layout=[4, 2], coupling_map=None).draw()

  Fixed `#5345 <https://github.com/Qiskit/qiskit-terra/issues/5345>`__

- A new kwarg, ``user_cost_dict`` has been added to the constructor for the
  :class:`qiskit.transpiler.passes.TemplateOptimization` transpiler pass.
  This enables users to provide a custom cost dictionary for the gates to
  the underlying template matching algorithm. For example::

    from qiskit.transpiler.passes import TemplateOptimization

    cost_dict = {'id': 0, 'x': 1, 'y': 1, 'z': 1, 'h': 1, 't': 1}
    pass = TemplateOptimization(user_cost_dict=cost_dict)

- An issue when passing the :class:`~qiskit.result.Counts` object
  returned by :meth:`~qiskit.result.Result.get_counts` to
  :func:`~qiskit.result.marginal_counts` would produce an improperly
  formatted :class:`~qiskit.result.Counts` object with certain inputs has
  been fixed. Fixes
  `#5424 <https://github.com/Qiskit/qiskit-terra/issues/5424>`__

- Improved the allocation of helper qubits in
  :class:`~qiskit.circuit.library.PolynomialPauliRotations` and
  :class:`~qiskit.circuit.library.PiecewiseLinearPauliRotations` which makes
  the implementation of these circuit more efficient.
  Fixed `#5320 <https://github.com/Qiskit/qiskit-terra/issues/5320>`__ and
  `#5322 <https://github.com/Qiskit/qiskit-terra/issues/5322>`__

- Fix the usage of the allocated helper qubits in the
  :class:`~qiskit.circuit.library.MCXGate` in the
  :class:`~qiskit.circuit.library.WeightedAdder` class. These were previously
  allocated but not used prior to this fix.
  Fixed `#5321 <https://github.com/Qiskit/qiskit-terra/issues/5321>`__

- In a number of cases, the ``latex`` output method for the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :meth:`~qiskit.circuit.QuantumCircuit.draw` method did not display the
  gate name correctly, and in other cases, did not include gate parameters
  where they should be. Now the gate names will be displayed the same way
  as they are displayed with the ``mpl`` output method, and parameters will
  display for all the gates that have them. In addition, some of the gates
  did not display in the correct form, and these have been fixed. Fixes
  `#5605 <https://github.com/Qiskit/qiskit-terra/issues/5605>`__,
  `#4938 <https://github.com/Qiskit/qiskit-terra/issues/4938>`__, and
  `#3765 <https://github.com/Qiskit/qiskit-terra/issues/3765>`__

- Fixed an issue where, if the
  :meth:`qiskit.circuit.Instruction.to_instruction` method was used on a subcircuit which
  contained classical registers and that
  :class:`~qiskit.circuit.Instruction` object was then added to a
  :class:`~qiskit.circuit.QuantumCircuit` object, then the output from the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method would in some instances
  display the subcircuit to the left of a measure when it should have been
  displayed to the right.
  Fixed `#5947 <https://github.com/Qiskit/qiskit-terra/issues/5947>`__

- Fixed an issue with :class:`~qiskit.circuit.Delay` objects in a
  :class:`~qiskit.circuit.QuantumCircuit` where
  :func:`qiskit.compiler.transpile` would not be convert the units of
  the :class:`~qiskit.circuit.Delay` to the units of the
  :class:`~qiskit.providers.Backend`, if the ``backend`` kwarg is set on
  :func:`~qiskit.circuit.transpile`. This could result in the wrong behavior
  because of a unit mismatch, for example running::

    from qiskit import transpile, execute
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.delay(100, [0], unit='us')

    qc = transpile(qc, backend)
    job = execute(qc, backend)

  would previously have resulted in the backend delay for 100 timesteps (each
  of duration dt) rather than expected (100e-6 / dt) timesteps. This has been
  corrected so the :func:`qiskit.compiler.transpile` function properly
  converts the units.


.. _Release Notes_0.17.0_Other Notes:

Other Notes
-----------

- The snapshots of all the fake/mock backends in ``qiskit.test.mock`` have
  been updated to reflect recent device changes. This includes a change in
  the :attr:`~qiskit.providers.models.QasmBackendConfiguration.basis_gates`
  attribute for the :class:`~qiskit.providers.models.BackendConfiguration`
  to ``['cx', 'rz', 'sx', 'x', 'id']``, the addition of a ``readout_length``
  property to the qubit properties in the
  :class:`~qiskit.providers.models.BackendProperties`, and updating the
  :class:`~qiskit.providers.models.PulseDefaults` so that all the mock
  backends support parametric pulse based
  :class:`~qiskit.pulse.InstructionScheduleMap` instances.

.. _Aer_Release Notes_0.8.0:

Aer 0.8.0
============

.. _Aer_Release Notes_0.8.0_Prelude:

Prelude
-------

The 0.8 release includes several new features and bug fixes. The
highlights for this release are: the introduction of a unified
:class:`~qiskit.providers.aer.AerSimulator` backend for running circuit
simulations using any of the supported simulation methods; a simulator
instruction library (:mod:`qiskit.providers.aer.library`)
which includes custom instructions for saving various kinds of simulator
data; MPI support for running large simulations on a distributed
computing environment.


.. _Aer_Release Notes_0.8.0_New Features:

New Features
------------

- Python 3.9 support has been added in this release. You can now run Qiskit
  Aer using Python 3.9 without building from source.

- Add the CMake flag ``DISABLE_CONAN`` (default=``OFF``)s. When installing from source,
  setting this to ``ON`` allows bypassing the Conan package manager to find libraries
  that are already installed on your system. This is also available as an environment
  variable ``DISABLE_CONAN``, which takes precedence over the CMake flag.
  This is not the official procedure to build AER. Thus, the user is responsible
  of providing all needed libraries and corresponding files to make them findable to CMake.

- This release includes support for building qiskit-aer with MPI support to
  run large simulations on a distributed computing environment. See the
  `contributing guide <https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md#building-with-mpi-support>`__
  for instructions on building and running in an MPI environment.

- It is now possible to build qiskit-aer with CUDA enabled in Windows.
  See the
  `contributing guide <https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md#building-with-gpu-support>`__
  for instructions on building from source with GPU support.

- When building the qiskit-aer Python extension from source several build
  dependencies need to be pre-installed to enable C++ compilation. As a
  user convenience when building the extension any of these build
  dependencies which were missing would be automatically installed using
  ``pip`` prior to the normal ``setuptools`` installation steps, however it was
  previously was not possible to avoid this automatic installation. To solve
  this issue a new environment variable ``DISABLE_DEPENDENCY_INSTALL``
  has been added. If it is set to ``1`` or ``ON`` when building the python
  extension from source this will disable the automatic installation of these
  missing build dependencies.

- Adds support for optimized N-qubit Pauli gate (
  :class:`qiskit.circuit.library.PauliGate`) to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  statevector and density matrix methods of the
  :class:`~qiskit.providers.aer.QasmSimulator` and
  :class:`~qiskit.providers.aer.AerSimulator`.

- The :meth:`~qiskit.providers.aer.AerSimulator.run` method for the
  :class:`~qiskit.providers.aer.AerSimulator`,
  :class:`~qiskit.providers.aer.QasmSimulator`,
  :class:`~qiskit.providers.aer.StatevectorSimulator`, and
  :class:`~qiskit.providers.aer.UnitarySimulator` backends now takes a
  :class:`~qiskit.circuit.QuantumCircuit` (or a list of
  :class:`~qiskit.circuit.QuantumCircuit` objects) as it's input.
  The previous :class:`~qiskit.qobj.QasmQobj` object is still supported for
  now, but will be deprecated in a future release.

  For an example of how to use this see::

    from qiskit import transpile, QuantumCircuit

    from qiskit.providers.aer import Aer

    backend = Aer.get_backend('aer_simulator')

    circuit = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    tqc = transpile(circuit, backend)
    result = backend.run(tqc, shots=4096).result()

- The :meth:`~qiskit.providers.aer.PulseSimulator.run` method for the
  :class:`~qiskit.providers.aer.PulseSimulator` backend now takes a
  :class:`~qiskit.pulse.Schedule` (or a list of
  :class:`~qiskit.pulse.Schedule` objects) as it's input.
  The previous :class:`~qiskit.qobj.PulseQobj` object is still supported for
  now, but will be deprecated in a future release.

- Adds the new :class:`~qiskit.provider.aer.AerSimulator` simulator backend
  supporting the following simulation methods

    * ``automatic``
    * ``statevector``
    * ``stabilizer``
    * ``density_matrix``
    * ``matrix_product_state``
    * ``unitary``
    * ``superop``

  The default `automatic` method will automatically choose a simulation
  method separately for each run circuit based on the circuit instructions
  and noise model (if any). Initializing a simulator with a specific
  method can be done using the `method` option.

  .. code::python

    from qiskit.providers.aer import AerSimulator

    # Create a MPS simulator backend
    backend = AerSimulator(method='matrix_product_state')

  GPU simulation for the statevector, density matrix and unitary methods
  can be enabled by setting the ``device='GPU'`` backend option.

  .. code::python

    from qiskit.providers.aer import AerSimulator

    # Create a GPU statevector backend
    backend = AerSimulator(method='statevector', device='GPU')

  Note that the ``unitary`` and ``superop`` methods do not support measurement
  as they simulate the unitary matrix or superoperator matrix of the run
  circuit so one of the new :func:`~qiskit.providers.aer.library.save_unitary`,
  :func:`~qiskit.providers.aer.library.save_superop`, or
  :func:`~qiskit.providers.aer.library.save_state` instructions must
  be used to save the simulator state to the returned results. Similarly
  state of the other simulations methods can be saved using the
  appropriate instructions. See the :mod:`qiskit.providers.aer.library`
  API documents for more details.

  Note that the :class:`~qiskit.providers.aer.AerSimulator` simulator
  superceds the :class:`~qiskit.providers.aer.QasmSimulator`,
  :class:`~qiskit.providers.aer.StatevectorSimulator`, and
  :class:`~qiskit.providers.aer.UnitarySimulator` backends which will
  be deprecated in a future release.

- Updates the :class:`~qiskit.providers.aer.AerProvider` class to include
  multiple :class:`~qiskit.providers.aer.AerSimulator` backends preconfigured
  for all available simulation methods and simulation devices. The new
  backends can be accessed through the provider interface using the names

    * ``"aer_simulator"``
    * ``"aer_simulator_statevector"``
    * ``"aer_simulator_stabilizer"``
    * ``"aer_simulator_density_matrix"``
    * ``"aer_simulator_matrix_product_state"``
    * ``"aer_simulator_extended_stabilizer"``
    * ``"aer_simulator_unitary"``
    * ``"aer_simulator_superop"``

  Additional if Aer was installed with GPU support on a compatible system
  the following GPU backends will also be available

    * ``"aer_simulator_statevector_gpu"``
    * ``"aer_simulator_density_matrix_gpu"``
    * ``"aer_simulator_unitary_gpu"``

  For example::

    from qiskit import Aer

    # Get the GPU statevector simulator backend
    backend = Aer.get_backend('aer_simulator_statevector_gpu')

- Added a new ``norm estimation`` method for performing measurements when using
  the ``"extended_stabilizer"`` simulation method. This norm estimation method
  can be used by passing the following options to the
  :class:`~qiskit.providers.aer.AerSimulator` and
  :class:`~qiskit.providers.aer.QasmSimulator` backends

  .. code-block:: python

    simulator = QasmSimulator(
        method='extended_stabilizer',
        extended_stabilizer_sampling_method='norm_estimation')

  The norm estimation method is slower than the alternative ``metropolis``
  or ``resampled_metropolis`` options, but gives better performance on circuits
  with sparse output distributions. See the documentation of the
  :class:`~qiskit.providers.aer.QasmSimulator` for more information.

- Adds instructions for saving the state of the simulator in various
  formats. These instructions are

  * :class:`qiskit.providers.aer.library.SaveDensityMatrix`
  * :class:`qiskit.providers.aer.library.SaveMatrixProductState`
  * :class:`qiskit.providers.aer.library.SaveStabilizer`
  * :class:`qiskit.providers.aer.library.SaveState`
  * :class:`qiskit.providers.aer.library.SaveStatevector`
  * :class:`qiskit.providers.aer.library.SaveStatevectorDict`
  * :class:`qiskit.providers.aer.library.SaveUnitary`

  These instructions can be appended to a quantum circuit by using the
  :class:`~qiskit.providers.aer.library.save_density_matrix`,
  :class:`~qiskit.providers.aer.library.save_matrix_product_state`,
  :class:`~qiskit.providers.aer.library.save_stabilizer`,
  :class:`~qiskit.providers.aer.library.save_state`,
  :class:`~qiskit.providers.aer.library.save_statevector`,
  :class:`~qiskit.providers.aer.library.save_statevector_dict`,
  :class:`~qiskit.providers.aer.library.save_unitary`
  circuit methods which are added to ``QuantumCircuit`` when importing Aer.

  See the :mod:`qiskit.providers.aer.library` API documentation
  for details on method compatibility for each instruction.

  Note that the snapshot instructions
  :class:`~qiskit.providers.aer.extensions.SnapshotStatevector`,
  :class:`~qiskit.providers.aer.extensions.SnapshotDensityMatrix`,
  :class:`~qiskit.providers.aer.extensions.SnapshotStabilizer` are
  still supported but will be deprecated in a future release.

- Adds :class:`qiskit.providers.aer.library.SaveExpectationValue` and
  :class:`qiskit.providers.aer.library.SaveExpectationValueVariance`
  quantum circuit instructions for saving the expectation value
  :math:`\langle H\rangle = Tr[H\rho]`, or expectation value and variance
  :math:`Var(H) = \langle H^2\rangle - \langle H\rangle^2`,
  of a Hermitian operator :math:`H` for the simulator state :math:`\rho`.
  These instruction can be appended to a quantum circuit by using the
  :class:`~qiskit.providers.aer.library.save_expectation_value` and
  :class:`~qiskit.providers.aer.library.save_expectation_value_variance`
  circuit methods which is added to ``QuantumCircuit`` when importing Aer.

  Note that the snapshot instruction
  :class:`~qiskit.providers.aer.extensions.SnapshotExpectationValue`,
  is still supported but will be deprecated in a future release.

- Adds :class:`qiskit.providers.aer.library.SaveProbabilities` and
  :class:`qiskit.providers.aer.library.SaveProbabilitiesDict` quantum
  circuit instruction for saving all measurement outcome probabilities for
  Z-basis measurements of the simualtor state. These instruction can be
  appended to a quantum circuit by using the
  :class:`~qiskit.providers.aer.library.save_probabilities` and
  :class:`~qiskit.providers.aer.library.save_probabilities_dict` circuit
  methods which is added to ``QuantumCircuit`` when importing Aer.

  Note that the snapshot instruction
  :class:`~qiskit.providers.aer.extensions.SnapshotProbabilities`,
  is still supported but will be deprecated in a future release.

- Adds :class:`qiskit.providers.aer.library.SaveAmplitudes` and
  :class:`qiskit.providers.aer.library.SaveAmplitudesSquared`
  circuit instructions for saving select complex statevector amplitudes,
  or select probabilities (amplitudes squared) for supported simulation
  methods. These instructions can be appended to a quantum circuit by using the
  :class:`~qiskit.providers.aer.library.save_amplitudes` and
  :class:`~qiskit.providers.aer.library.save_amplitudes_squared` circuit
  methods which is added to ``QuantumCircuit`` when importing Aer.

- Adds instructions for setting the state of the simulators. These
  instructions must be defined on the full number of qubits in the circuit.
  They can be applied at any point in a circuit and will override the
  simulator state with the one specified. Added instructions are

  * :class:`qiskit.providers.aer.library.SetDensityMatrix`
  * :class:`qiskit.providers.aer.library.SetStabilizer`
  * :class:`qiskit.providers.aer.library.SetStatevector`
  * :class:`qiskit.providers.aer.library.SetUnitary`

  These instruction can be appended to a quantum circuit by using the
  :class:`~qiskit.providers.aer.library.set_density_matrix`,
  :class:`~qiskit.providers.aer.library.set_stabilizer`,
  :class:`~qiskit.providers.aer.library.set_statevector`,
  :class:`~qiskit.providers.aer.library.set_unitary`
  circuit methods which are added to ``QuantumCircuit`` when importing Aer.

  See the :mod:`qiskit.providers.aer.library` API documentation
  for details on method compatibility for each instruction.

- Added support for diagonal gates to the ``"matrix_product_state"`` simulation
  method.

- Added support for the ``initialize`` instruction to the
  ``"matrix_product_state"`` simulation method.


.. _Aer_Release Notes_0.8.0_Known Issues:

Known Issues
------------

- There is a known issue where the simulation of certain circuits with a Kraus
  noise model using the ``"matrix_product_state"`` simulation method can cause
  the simulator to crash. Refer to
  `#306 <https://github.com/Qiskit/qiskit-aer/issues/1184>`__ for more
  information.


.. _Aer_Release Notes_0.8.0_Upgrade Notes:

Upgrade Notes
-------------

- The minimum version of `Conan <https://conan.io/>`__ has been increased to 1.31.2.
  This was necessary to fix a compatibility issue with newer versions of the
  `urllib3 <https://pypi.org/project/urllib3/>`__ (which is a dependency of Conan).
  It also adds native support for AppleClang 12 which is useful for users with
  new Apple computers.

- ``pybind11`` minimum version required is 2.6 instead of 2.4. This is needed
  in order to support CUDA enabled compilation in Windows.

- Cython has been removed as a build dependency.

- Removed x90 gate decomposition from noise models that was deprecated
  in qiskit-aer 0.7. This decomposition is now done by using regular
  noise model basis gates and the qiskit transpiler.

- The following options for the ``"extended_stabilizer"`` simulation method
  have changed.

    + ``extended_stabilizer_measure_sampling``: This option has been replaced
      by the options ``extended_stabilizer_sampling_method``, which controls
      how we simulate qubit measurement.

    + ``extended_stabilizer_mixing_time``: This option has been renamed as
      ``extended_stabilizer_metropolis_mixing_time`` to clarify it only applies
      to the ``metropolis`` and ``resampled_metropolis`` sampling methods.

    + ``extended_stabilizer_norm_estimation_samples``: This option has been renamed
      to ``extended_stabilizer_norm_estimation_default_samples``.

  One additional option, ``extended_stabilizer_norm_estimation_repetitions`` has been
  added, whih controls part of the behaviour of the norm estimation sampling method.


.. _Aer_Release Notes_0.8.0_Deprecation Notes:

Deprecation Notes
-----------------

- Python 3.6 support has been deprecated and will be removed in a future
  release. When support is removed you will need to upgrade the Python
  version you're using to Python 3.7 or above.


.. _Aer_Release Notes_0.8.0_Bug Fixes:

Bug Fixes
---------

- Fixes bug with :class:`~qiskit.providers.aer.AerProvider` where options set
  on the returned backends using
  :meth:`~qiskit.providers.aer.QasmSimulator.set_options` were stored in the
  provider and would persist for subsequent calls to
  :meth:`~qiskit.providers.aer.AerProvider.get_backend` for the same named
  backend. Now every call to
  and :meth:`~qiskit.providers.aer.AerProvider.backends` returns a new
  instance of the simulator backend that can be configured.

- Fixes bug in the error message returned when a circuit contains unsupported
  simulator instructions. Previously some supported instructions were also
  being listed in the error message along with the unsupported instructions.

- Fixes issue with setting :class:`~qiskit.providers.aer.QasmSimulator`
  basis gates when using ``"method"`` and ``"noise_model"`` options
  together, and when using them with a simulator constructed using
  :meth:`~qiskit.providers.aer.QasmSimulator.from_backend`. Now the
  listed basis gates will be the intersection of gates supported by
  the backend configuration, simulation method, and noise model basis
  gates. If the intersection of the noise model basis gates and
  simulator basis gates is empty a warning will be logged.

- Fix bug where the ``"sx"``` gate :class:`~qiskit.circuit.library.SXGate` was
  not listed as a supported gate in the C++ code, in ``StateOpSet`` of
  ``matrix_product_state.hp``.

- Fix bug where ``"csx"``, ``"cu2"``, ``"cu3"`` were incorrectly listed as
  supported basis gates for the ``"density_matrix"`` method of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Fix bug where parameters were passed incorrectly between functions in
  ``matrix_product_state_internal.cpp``, causing wrong simulation, as well
  as reaching invalid states, which in turn caused an infinite loop.

- Fixes a bug that resulted in ``c_if`` not working when the
  width of the conditional register was greater than 64. See
  `#1077 <https://github.com/Qiskit/qiskit-aer/issues/1077>`__.

- Fixes a bug `#1153 <https://github.com/Qiskit/qiskit-aer/issues/1153>`__)
  where noise on conditional gates was always being applied regardless of
  whether the conditional gate was actually applied based on the classical
  register value. Now noise on a conditional gate will only be applied in
  the case where the conditional gate is applied.

- Fixes a bug with nested OpenMP flag was being set to true when it
  shouldn't be.

- Fixes a bug when applying truncation in the matrix product state method of the QasmSimulator.

- Fixed issue `#1126 <https://github.com/Qiskit/qiskit-aer/issues/1126>`__:
  bug in reporting measurement of a single qubit. The bug occured when copying
  the measured value to the output data structure.

- In MPS, apply_kraus was operating directly on the input bits in the
  parameter qubits, instead of on the internal qubits. In the MPS algorithm,
  the qubits are constantly moving around so all operations should be applied
  to the internal qubits.

- When invoking MPS::sample_measure, we need to first sort the qubits to the
  default ordering because this is the assumption in qasm_controller.This is
  done by invoking the method move_all_qubits_to_sorted_ordering. It was
  correct in sample_measure_using_apply_measure, but missing in
  sample_measure_using_probabilities.

- Fixes bug with the :meth:`~qiskit.providers.aer.QasmSimulator.from_backend`
  method of the :class:`~qiskit.provider.aer.QasmSimulator` that would set the
  ``local`` attribute of the configuration to the backend value rather than
  always being set to ``True``.

- Fixes bug in
  :meth:`~qiskit.providers.aer.noise.NoiseModel.from_backend` and
  :meth:`~qiskit.providers.aer.QasmSimulator.from_backend` where
  :attr:`~qiskit.providers.aer.noise.NoiseModel.basis_gates` was set
  incorrectly for IBMQ devices with basis gate set
  ``['id', 'rz', 'sx', 'x', 'cx']``. Now the noise model will always
  have the same basis gates as the backend basis gates regardless of
  whether those instructions have errors in the noise model or not.

- Fixes an issue where the Extended `"extended_stabilizer"` simulation method
  would give incorrect results on quantum circuits with sparse output
  distributions. Refer to
  `#306 <https://github.com/Qiskit/qiskit-aer/issues/306>`__ for more
  information and examples.

Ignis 0.6.0
===========

.. _Ignis_Release Notes_0.6.0_New Features:

New Features
------------

- The :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits` function
  has been improved so that the number of circuits generated by the function
  used for calibration by the CTMP method are reduced from :math:`O(n)` to
  :math:`O(\log{n})` (where :math:`n` is the number of qubits).


.. _Ignis_Release Notes_0.6.0_Upgrade Notes:

Upgrade Notes
-------------

- The :func:`qiskit.ignis.verification.randomized_benchmarking_seq`
  function is now using the upgraded CNOTDihedral class,
  :class:`qiskit.ignis.verification.CNOTDihedral`, which enables performing
  CNOT-Dihedral Randomized Benchmarking on more than two qubits.

- The python package ``retworkx`` is now a requirement for installing
  qiskit-ignis. It replaces the previous usage of ``networkx`` (which is
  no longer a requirement) to get better performance.

- The ``scikit-learn`` dependency is no longer required and is now an optional
  requirement. If you're using the IQ measurement discriminators
  (:class:`~qiskit.ignis.measurement.IQDiscriminationFitter`,
  :class:`~qiskit.ignis.measurement.LinearIQDiscriminationFitter`,
  :class:`~qiskit.ignis.measurement.QuadraticIQDiscriminationFitter`,
  or :class:`~qiskit.ignis.measurement.SklearnIQDiscriminator`) you will
  now need to manually install scikit-learn, either by running
  ``pip install scikit-learn`` or when you're also installing
  qiskit-ignis with ``pip install qiskit-ignis[iq]``.


.. _Ignis_Release Notes_0.6.0_Bug Fixes:

Bug Fixes
---------

- Fixed an issue in the expectation value method
  :meth:`~qiskit.ignis.mitigation.TensoredExpvalMeasMitigator.expectation_value`,
  for the error mitigation classes
  :class:`~qiskit.ignis.mitigation.TensoredExpvalMeasMitigator` and
  :class:`~qiskit.ignis.mitigation.CTMPExpvalMeasMitigator` if the
  ``qubits`` kwarg was not specified it would incorrectly use the
  total number of qubits of the mitigator, rather than the number of
  classical bits in the count dictionary leading to greatly reduced
  performance.
  Fixed `#561 <https://github.com/Qiskit/qiskit-ignis/issues/561>`__

- Fix the ``"auto"`` method of the
  :class:`~qiskit.ignis.verification.tomography.TomographyFitter`,
  :class:`~qiskit.ignis.verification.tomography.StateTomographyFitter`, and
  :class:`~qiskit.ignis.verification.tomography.ProcessTomographyFitter` to
  only use ``"cvx"`` if CVXPY is installed *and* a third-party SDP solver
  other than SCS is available. This is because the SCS solver has lower
  accuracy than other solver methods and often returns a density matrix or
  Choi-matrix that is not completely-positive and fails validation when used
  with the :func:`qiskit.quantum_info.state_fidelity` or
  :func:`qiskit.quantum_info.process_fidelity` functions.

Aqua 0.9.0
==========

This release officially deprecates the Qiskit Aqua project, in the future
(no sooner than 3 months from this release) the Aqua project will have it's
final release and be archived. All the functionality that qiskit-aqua provides
has been migrated to either new packages or to other qiskit packages. The
application modules that are provided by qiskit-aqua have been split into
several new packages: ``qiskit-optimization``, ``qiskit-nature``,
``qiskit-machine-learning``, and ``qiskit-finance``. These packages can be
installed by themselves (via the standard pip install command,
ie ``pip install qiskit-nature``) or with the rest of the Qiskit metapackage as
optional extras (ie, ``pip install 'qiskit[finance,optimization]'`` or
``pip install 'qiskit[all]'``. The core building blocks for algorithms and the
operator flow now exist as part of qiskit-terra at :mod:`qiskit.algorithms` and
:mod:`qiskit.opflow`. Depending on your existing usage of Aqua you should either
use the application packages or the new modules in Qiskit Terra.

For more details on how to migrate from using Qiskit Aqua you can refer to the
:ref:`aqua-migration`.

IBM Q Provider 0.12.2
=====================

No change

*************
Qiskit 0.24.1
*************

Terra 0.16.4
============

No change

Aer 0.7.6
=========

No change

Ignis 0.5.2
===========

No change

Aqua 0.8.2
==========

No change

IBM Q Provider 0.12.2
=====================

.. _Release Notes_IBMQ_0.12.2_New Features:

Upgrade Notes
-------------

- :meth:`qiskit.providers.ibmq.IBMQBackend.defaults` now returns the pulse defaults for
  the backend if the backend supports pulse. However, your provider may not support pulse
  even if the backend does. The ``open_pulse`` flag in backend configuration indicates
  whether the provider supports it.

*************
Qiskit 0.24.0
*************

Terra 0.16.4
============

No change

Aer 0.7.6
=========

.. _Release Notes_Aer_0.7.6_New Features:

New Features
-------------

- This is the first release of qiskit-aer that publishes precompiled binaries
  to PyPI for Linux on aarch64 (arm64). From this release onwards Linux aarch64
  packages will be published and supported.


.. _Release Notes_Aer_0.7.6_Bug Fixes:

Bug Fixes
---------

- Fixes a bug `#1153 <https://github.com/Qiskit/qiskit-aer/issues/1153>`__
  where noise on conditional gates was always being applied regardless of
  whether the conditional gate was actually applied based on the classical
  register value. Now noise on a conditional gate will only be applied in
  the case where the conditional gate is applied.

- Fixed issue `#1126 <https://github.com/Qiskit/qiskit-aer/issues/1126>`__:
  bug in reporting measurement of a single qubit. The bug occured when
  copying the measured value to the output data structure.

- There was previously a mismatch between the default reported number of qubits
  the Aer backend objects would say were supported and the the maximum number
  of qubits the simulator would actually run. This was due to a mismatch
  between the Python code used for calculating the max number of qubits and
  the C++ code used for a runtime check for the max number of qubits based on
  the available memory. This has been correct so by default now Aer backends
  will allow running circuits that can fit in all the available system memory.
  Fixes `#1114 <https://github.com/Qiskit/qiskit-aer/issues/1126>`__


No change

Ignis 0.5.2
===========

No change

Aqua 0.8.2
==========

No change

IBM Q Provider 0.12.0
=====================

.. _Release Notes_IBMQ_0.12.0_Prelude:

Prelude
-------

- :meth:`qiskit.providers.ibmq.IBMQBackend.run` method now takes one or more
  :class:`~qiskit.circuit.QuantumCircuit` or :class:`~qiskit.pulse.Schedule`.
  Use of :class:`~qiskit.qobj.QasmQobj` and :class:`~qiskit.qobj.PulseQobj` is
  now deprecated. Runtime configuration options, such as the number of shots,
  can be set via either the :meth:`~qiskit.providers.ibmq.IBMQBackend.run`
  method, or the :meth:`qiskit.providers.ibmq.IBMQBackend.set_options` method.
  The former is used as a one-time setting for the job, and the latter for all
  jobs sent to the backend. If an option is set in both places, the value set
  in :meth:`~qiskit.providers.ibmq.IBMQBackend.run` takes precedence.

- IBM Quantum credentials are now loaded only from sections of the ``qiskitrc``
  file that start with 'ibmq'.

.. _Release Notes_IBMQ_0.12.0_New Features:

New Features
------------

- Python 3.9 support has been added in this release. You can now run Qiskit
  IBMQ provider using Python 3.9.

- :meth:`qiskit.providers.ibmq.AccountProvider.backends` now has a new
  parameter `min_num_qubits` that allows you to filter by the minimum number
  of qubits.

- :meth:`qiskit.providers.ibmq.IBMQBackend.run` method now takes one or more
  :class:`~qiskit.circuit.QuantumCircuit` or :class:`~qiskit.pulse.Schedule`.
  Runtime configuration options, such as the number of shots, can be set via
  either the :meth:`~qiskit.providers.ibmq.IBMQBackend.run` method, or
  the :meth:`qiskit.providers.ibmq.IBMQBackend.set_options` method. The former
  is used as a one-time setting for the job, and the latter for all jobs
  sent to the backend. If an option is set in both places, the value set
  in :meth:`~qiskit.providers.ibmq.IBMQBackend.run` takes precedence. For
  example:

  .. code-block:: python

      from qiskit import IBMQ, transpile
      from qiskit.test.reference_circuits import ReferenceCircuits

      provider = IBMQ.load_account()
      backend = provider.get_backend('ibmq_vigo')
      circuits = transpile(ReferenceCircuits.bell(), backend=backend)
      default_shots = backend.options.shots  # Returns the backend default of 1024 shots.
      backend.set_options(shots=2048)        # All jobs will now have use 2048 shots.
      backend.run(circuits)                  # This runs with 2048 shots.
      backend.run(circuits, shots=8192)      # This runs with 8192 shots.
      backend.run(circuits)                  # This again runs with 2048 shots.


- :class:`qiskit.providers.ibmq.experiment.Experiment` now has three
  additional attributes, `hub`, `group`, and `project`, that identify
  the provider used to create the experiment.

- You can now assign an ``experiment_id`` to a job when submitting it using
  :meth:`qiskit.providers.ibmq.IBMQBackend.run`. You can use this new field
  to group together a collection of jobs that belong to the same experiment.
  The :meth:`qiskit.providers.ibmq.IBMQBackendService.jobs` method was also
  updated to allow filtering by ``experiment_id``.

- :class:`qiskit.providers.ibmq.experiment.Experiment` now has two
  additional attributes:

  * share_level: The level at which the experiment is shared which determines
    who can see it when listing experiments. This can be updated.
  * owner: The ID of the user that uploaded the experiment. This is set by
    the server and cannot be updated.

- The method
  :meth:`qiskit.providers.ibmq.experimentservice.ExperimentService.experiments`
  now accepts ``hub``, ``group``, and ``project`` as filtering keywords.

- Methods
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.experiments` and
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.analysis_results`
  now support a ``limit`` parameter that allows you to limit the number of
  experiments and analysis results returned.

- The method
  :meth:`qiskit.providers.ibmq.experimentservice.ExperimentService.experiments`
  now accepts ``exclude_mine`` and ``mine_only`` as filtering keywords.

- The method
  :meth:`qiskit.providers.ibmq.experimentservice.ExperimentService.experiments`
  now accepts ``exclude_public`` and ``public_only`` as filtering keywords.

- :meth:`qiskit.providers.ibmq.managed.IBMQJobManager.run` now accepts a
  single :class:`~qiskit.circuit.QuantumCircuit` or
  :class:`~qiskit.pulse.Schedule` in addition to a list of them.

- The :func:`~qiskit.providers.ibmq.least_busy` function now skips backends
  that are operational but paused, meaning they are accepting but not
  processing jobs.

- You can now pickle an :class:`~qiskit.providers.ibmq.job.IBMQJob` instance,
  as long as it doesn't contain custom data that is not picklable (e.g.
  in Qobj header).

- You can now use the two new methods,
  :meth:`qiskit.providers.ibmq.AccountProvider.services` and
  :meth:`qiskit.providers.ibmq.AccountProvider.service` to find out what
  services are available to your account and get an instance of a
  particular service.

- The :meth:`qiskit.providers.ibmq.IBMQBackend.reservations` method
  now always returns the reservation scheduling modes even for
  reservations that you don't own.


.. _Release Notes_IBMQ_0.12.0_Upgrade Notes:

Upgrade Notes
-------------

- A number of previously deprecated methods and features have been removed,
  including:

    * :meth:`qiskit.providers.ibmq.job.IBMQJob.to_dict`
    * :meth:`qiskit.providers.ibmq.job.IBMQJob.from_dict`
    * `Qconfig.py` support
    * Use of proxy URLs that do not include protocols

- A new parameter, ``limit`` is now the first parameter for both
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.experiments` and
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.analysis_results`
  methods. This ``limit`` has a default value of 10, meaning by deafult only
  10 experiments and analysis results will be returned.

- IBM Quantum credentials are now loaded only from sections of the ``qiskitrc``
  file that start with 'ibmq'.
  This allows the ``qiskitrc`` file to be used for other functionality.


.. _Release Notes_IBMQ_0.12.0_Deprecation Notes:

Deprecation Notes
-----------------

- Use of :class:`~qiskit.qobj.QasmQobj` and :class:`~qiskit.qobj.PulseQobj` in
  the :meth:`qiskit.providers.ibmq.IBMQBackend.run` method is now deprecated.
  :class:`~qiskit.circuit.QuantumCircuit` and :class:`~qiskit.pulse.Schedule`
  should now be used instead.

- The ``backends`` attribute of :class:`qiskit.providers.ibmq.AccountProvider`
  has been renamed to ``backend`` (sigular). For backward compatibility, you
  can continue to use ``backends``, but it is deprecated and will be removed
  in a future release. The :meth:`qiskit.providers.ibmq.AccountProvider.backends`
  method remains unchanged. For example:

  .. code-block:: python

      backend = provider.backend.ibmq_vigo   # This is the new syntax.
      backend = provider.backends.ibmq_vigo  # This is deprecated.
      backends = provider.backends()         # This continues to work as before.

- Setting of the :class:`~qiskit.providers.ibmq.job.IBMQJob`
  ``client_version`` attribute has been deprecated. You can, however, continue
  to read the value of attribute.

- "The ``validate_qobj`` keyword in :meth:`qiskit.providers.ibmq.IBMQBackend.run`
  is deprecated and will be removed in a future release.
  If you're relying on this schema validation you should pull the schemas
  from the `Qiskit/ibmq-schemas <https://github.com/Qiskit/ibm-quantum-schemas>`_
  and directly validate your payloads with that.


.. _Release Notes_IBMQ_0.12.0_Bug Fixes:

Bug Fixes
---------

- Fixes the issue wherein a job could be left in the ``CREATING`` state if
  job submit fails half-way through.

- Fixes the issue wherein using Jupyter backend widget would fail if the
  backend's basis gates do not include the traditional u1, u2, and u3.
  Fixes `#844 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/844>`_

- Fixes the infinite loop raised when passing an ``IBMQRandomService`` instance
  to a child process.

- Fixes the issue wherein a ``TypeError`` is raised if the server returns
  an error code but the response data is not in the expected format.

*************
Qiskit 0.23.6
*************

Terra 0.16.4
============

No change

Aer 0.7.5
=========

.. _Release Notes_Aer_0.7.5_Prelude:

Prelude
-------

This release is a bugfix release that fixes compatibility in the precompiled
binary wheel packages with numpy versions < 1.20.0. The previous release 0.7.4
was building the binaries in a way that would require numpy 1.20.0 which has
been resolved now, so the precompiled binary wheel packages will work with any
numpy compatible version.

Ignis 0.5.2
===========

No change

Aqua 0.8.2
==========

No change

IBM Q Provider 0.11.1
=====================

No change

*************
Qiskit 0.23.5
*************

Terra 0.16.4
============

.. _Release Notes_0.16.4_Prelude:

Prelude
-------

This release is a bugfix release that primarily fixes compatibility with numpy
1.20.0. This numpy release deprecated their local aliases for Python's numeric
types (``np.int`` -> ``int``, ``np.float`` -> ``float``, etc.) and the usage of
these aliases in Qiskit resulted in a large number of deprecation warnings being
emitted. This release fixes this so you can run Qiskit with numpy 1.20.0 without
those deprecation warnings.

Aer 0.7.4
=========

.. _Release Notes_Aer_0.7.4_Bug Fixes:

Bug Fixes
----------

Fixes compatibility with numpy 1.20.0. This numpy release deprecated their local
aliases for Python's numeric types (``np.int`` -> ``int``,
``np.float`` -> ``float``, etc.) and the usage of these aliases in Qiskit Aer
resulted in a large number of deprecation warnings being emitted. This release
fixes this so you can run Qiskit Aer with numpy 1.20.0 without those deprecation
warnings.

Ignis 0.5.2
===========

.. _Release Notes_Ignis_0.5.2_Prelude:

Prelude
-------

This release is a bugfix release that primarily fixes compatibility with numpy
1.20.0. It is also the first release to include support for Python 3.9. Earlier
releases (including 0.5.0 and 0.5.1) worked with Python 3.9 but did not
indicate this in the package metadata, and there was no upstream testing for
those releases. This release fixes that and was tested on Python 3.9 (in
addition to 3.6, 3.7, and 3.8).

.. _Release Notes_Ignis_0.5.2_Bug Fixes:

Bug Fixes
---------

- `networkx <https://networkx.org/>`__ is explicitly listed as a dependency
  now. It previously was an implicit dependency as it was required for the
  :mod:`qiskit.ignis.verification.topological_codes` module but was not
  correctly listed as a depdendency as qiskit-terra also requires networkx
  and is also a depdency of ignis so it would always be installed in practice.
  However, it is necessary to list it as a requirement for future releases
  of qiskit-terra that will not require networkx. It's also important to
  correctly list the dependencies of ignis in case there were a future
  incompatibility between version requirements.

Aqua 0.8.2
==========


IBM Q Provider 0.11.1
=====================

No change

*************
Qiskit 0.23.4
*************

Terra 0.16.3
============

.. _Release Notes_0.16.3_Bug Fixes:

Bug Fixes
---------

- Fixed an issue introduced in 0.16.2 that would cause errors when running
  :func:`~qiskit.compiler.transpile` on a circuit with a series of 1 qubit
  gates and a non-gate instruction that only operates on a qubit (e.g.
  :class:`~qiskit.circuit.Reset`). Fixes
  `#5736 <https://github.com/Qiskit/qiskit-terra/issues/5736>`__

Aer 0.7.3
=========

No change

Ignis 0.5.1
===========

No change

Aqua 0.8.1
==========

No change

IBM Q Provider 0.11.1
=====================

No change

*************
Qiskit 0.23.3
*************

Terra 0.16.2
============

.. _Release Notes_0.16.2_New Features:

New Features
------------

- Python 3.9 support has been added in this release. You can now run Qiskit
  Terra using Python 3.9.


.. _Release Notes_0.16.2_Upgrade Notes:

Upgrade Notes
-------------

- The class :class:`~qiskit.library.standard_gates.x.MCXGrayCode` will now create
  a ``C3XGate`` if ``num_ctrl_qubits`` is 3 and a ``C4XGate`` if ``num_ctrl_qubits``
  is 4. This is in addition to the previous functionality where for any of the
  modes of the :class:'qiskit.library.standard_gates.x.MCXGate`, if ``num_ctrl_bits``
  is 1, a ``CXGate`` is created, and if 2, a ``CCXGate`` is created.


.. _Release Notes_0.16.2_Bug Fixes:

Bug Fixes
---------

- Pulse :py:class:`~qiskit.pulse.instructions.Delay` instructions are now
  explicitly assembled as :class:`~qiskit.qobj.PulseQobjInstruction` objects
  included in the :class:`~qiskit.qobj.PulseQobj` output from
  :func:`~qiskit.compiler.assemble`.

  Previously, we could ignore :py:class:`~qiskit.pulse.instructions.Delay`
  instructions in a :class:`~qiskit.pulse.Schedule` as part of
  :func:`~qiskit.compiler.assemble` as the time was explicit in the
  :class:`~qiskit.qobj.PulseQobj` objects. But, now with pulse gates, there
  are situations where we can schedule ONLY a delay, and not including the
  delay itself would remove the delay.

- Circuits with custom gate calibrations can now be scheduled with the
  transpiler without explicitly providing the durations of each circuit
  calibration.

- The :class:`~qiskit.transpiler.passes.BasisTranslator` and
  :class:`~qiskit.transpiler.passes.Unroller` passes, in some cases, had not been
  preserving the global phase of the circuit under transpilation. This has
  been fixed.

- A bug in :func:`qiskit.pulse.builder.frequency_offset` where when
  ``compensate_phase`` was set a factor of :math:`2\pi`
  was missing from the appended phase.

- Fix the global phase of the output of the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.repeat`. If a circuit with global
  phase is appended to another circuit, the global phase is currently not
  propagated. Simulators rely on this, since the phase otherwise gets
  applied multiple times. This sets the global phase of
  :meth:`~qiskit.circuit.QuantumCircuit.repeat` to 0 before appending the
  repeated circuit instead of multiplying the existing phase times the
  number of repetitions.

- Fixes bug in :class:`~qiskit.quantum_info.SparsePauliOp` where multiplying
  by a certain non Python builtin Numpy scalar types returned incorrect values.
  Fixes `#5408 <https://github.com/Qiskit/qiskit-terra/issues/5408>`__

- The definition of the Hellinger fidelity from has been corrected from the
  previous defition of :math:`1-H(P,Q)` to :math:`[1-H(P,Q)^2]^2` so that it
  is equal to the quantum state fidelity of P, Q as diagonal density
  matrices.

- Reduce the number of CX gates in the decomposition of the 3-controlled
  X gate, :class:`~qiskit.circuit.library.C3XGate`. Compiled and optimized
  in the `U CX` basis, now only 14 CX and 16 U gates are used instead of
  20 and 22, respectively.

- Fixes the issue wherein using Jupyter backend widget or
  :meth:`qiskit.tools.backend_monitor` would fail if the
  backend's basis gates do not include the traditional u1, u2, and u3.

- When running :func:`qiskit.compiler.transpile` on a list of circuits with a
  single element, the function used to return a circuit instead of a list. Now,
  when :func:`qiskit.compiler.transpile` is called with a list, it will return a
  list even if that list has a single element. See
  `#5260 <https://github.com/Qiskit/qiskit-terra/issues/5260>`__.

  .. code-block:: python

    from qiskit import *

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    transpiled = transpile([qc])
    print(type(transpiled), len(transpiled))

  .. parsed-literal::
   <class 'list'> 1

Aer 0.7.3
==========

.. _Release Notes_Aer_0.7.3_New Features:

New Features
------------

- Python 3.9 support has been added in this release. You can now run Qiskit
  Aer using Python 3.9 without building from source.


.. _Release Notes_Aer_0.7.3_Bug Fixes:

Bug Fixes
---------

- Fixes issue with setting :class:`~qiskit.providers.aer.QasmSimulator`
  basis gates when using ``"method"`` and ``"noise_model"`` options
  together, and when using them with a simulator constructed using
  :meth:`~qiskit.providers.aer.QasmSimulator.from_backend`. Now the
  listed basis gates will be the intersection of gates supported by
  the backend configuration, simulation method, and noise model basis
  gates. If the intersection of the noise model basis gates and
  simulator basis gates is empty a warning will be logged.

- Fixes a bug that resulted in `c_if` not working when the
  width of the conditional register was greater than 64. See
  `#1077 <https://github.com/Qiskit/qiskit-aer/issues/1077>`__.

- Fixes bug in
  :meth:`~qiskit.providers.aer.noise.NoiseModel.from_backend` and
  :meth:`~qiskit.providers.aer.QasmSimulator.from_backend` where
  :attr:`~qiskit.providers.aer.noise.NoiseModel.basis_gates` was set
  incorrectly for IBMQ devices with basis gate set
  ``['id', 'rz', 'sx', 'x', 'cx']``. Now the noise model will always
  have the same basis gates as the backend basis gates regardless of
  whether those instructions have errors in the noise model or not.

- Fixes a bug when applying truncation in the matrix product state method of the QasmSimulator.

Ignis 0.5.1
===========

No change

Aqua 0.8.1
==========

No change

IBM Q Provider 0.11.1
=====================

No change

*************
Qiskit 0.23.2
*************

Terra 0.16.1
============

No change

Aer 0.7.2
==========

.. _Release Notes_0.7.2_New Features:

New Features
------------

- Add the CMake flag ``DISABLE_CONAN`` (default=``OFF``)s. When installing from source,
  setting this to ``ON`` allows bypassing the Conan package manager to find libraries
  that are already installed on your system. This is also available as an environment
  variable ``DISABLE_CONAN``, which takes precedence over the CMake flag.
  This is not the official procedure to build AER. Thus, the user is responsible
  of providing all needed libraries and corresponding files to make them findable to CMake.


.. _Release Notes_0.7.2_Bug Fixes:

Bug Fixes
---------

- Fixes a bug with nested OpenMP flag was being set to true when it
  shouldn't be.

Ignis 0.5.1
===========

No change

Aqua 0.8.1
==========

No change

IBM Q Provider 0.11.1
=====================

No change


*************
Qiskit 0.23.1
*************

.. _Release Notes_0.16.1:

Terra 0.16.1
============

.. _Release Notes_0.16.1_Bug Fixes:

Bug Fixes
---------

- Fixed an issue where an error was thrown in execute for valid circuits
  built with delays.

- The QASM definition of 'c4x' in qelib1.inc has been corrected to match
  the standard library definition for C4XGate.

- Fixes a bug in subtraction for quantum channels :math:`A - B` where :math:`B`
  was an :class:`~qiskit.quantum_info.Operator` object. Negation was being
  applied to the matrix in the Operator representation which is not equivalent
  to negation in the quantum channel representation.

- Changes the way
  :meth:`~qiskit.quantum_info.states.statevector.Statevector._evolve_instruction`
  access qubits to handle the case of an instruction with multiple registers.

.. _Release Notes_Aer_0.7.1:

Aer 0.7.1
=========

.. _Release Notes_Aer_0.7.1_Upgrade Notes:

Upgrade Notes
-------------

- The minimum cmake version to build qiskit-aer has increased from 3.6 to
  3.8. This change was necessary to enable fixing GPU version builds that
  support running on x86_64 CPUs lacking AVX2 instructions.


.. _Release Notes_Aer_0.7.1_Bug Fixes:

Bug Fixes
---------

- qiskit-aer with GPU support will now work on systems with x86_64 CPUs
  lacking AVX2 instructions. Previously, the GPU package would only run if
  the AVX2 instructions were available. Fixes
  `#1023 <https://github.com/Qiskit/qiskit-aer/issues/1023>`__

- Fixes bug with :class:`~qiskit.providers.aer.AerProvider` where options set
  on the returned backends using
  :meth:`~qiskit.providers.aer.QasmSimulator.set_options` were stored in the
  provider and would persist for subsequent calls to
  :meth:`~qiskit.providers.aer.AerProvider.get_backend` for the same named
  backend. Now every call to
  and :meth:`~qiskit.providers.aer.AerProvider.backends` returns a new
  instance of the simulator backend that can be configured.

- Fixes bug in the error message returned when a circuit contains unsupported
  simulator instructions. Previously some supported instructions were also
  being listed in the error message along with the unsupported instructions.

- Fix bug where the `"sx"`` gate :class:`~qiskit.circuit.library.SXGate` was
  not listed as a supported gate in the C++ code, in `StateOpSet` of
  `matrix_product_state.hp`.

- Fix bug where ``"csx"``, ``"cu2"``, ``"cu3"`` were incorrectly listed as
  supported basis gates for the ``"density_matrix"`` method of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- In MPS, apply_kraus was operating directly on the input bits in the
  parameter qubits, instead of on the internal qubits. In the MPS algorithm,
  the qubits are constantly moving around so all operations should be applied
  to the internal qubits.

- When invoking MPS::sample_measure, we need to first sort the qubits to the
  default ordering because this is the assumption in qasm_controller.This is
  done by invoking the method move_all_qubits_to_sorted_ordering. It was
  correct in sample_measure_using_apply_measure, but missing in
  sample_measure_using_probabilities.


.. _Release Notes_Ignis_0.5.1:

Ignis 0.5.1
===========

.. _Release Notes_Ignis_0.5.1_Bug Fixes:

Bug Fixes
---------

- Fix the ``"auto"`` method of the
  :class:`~qiskit.ignis.verification.tomography.TomographyFitter`,
  :class:`~qiskit.ignis.verification.tomography.StateTomographyFitter`, and
  :class:`~qiskit.ignis.verification.tomography.ProcessTomographyFitter` to
  only use ``"cvx"`` if CVXPY is installed *and* a third-party SDP solver
  other than SCS is available. This is because the SCS solver has lower
  accuracy than other solver methods and often returns a density matrix or
  Choi-matrix that is not completely-positive and fails validation when used
  with the :func:`qiskit.quantum_info.state_fidelity` or
  :func:`qiskit.quantum_info.process_fidelity` functions.

.. _Release Notes_Aqua_0.8.1:

Aqua 0.8.1
==========

0.8.1
=====

.. _Release Notes_Aqua_0.8.1_New Features:

New Features
------------

- A new algorithm has been added: the Born Openheimer Potential Energy surface for the
  calculation of potential energy surface along different degrees of freedom of the molecule.
  The algorithm is called ``BOPESSampler``. It further provides functionalities of fitting the
  potential energy surface to an analytic function of predefined potentials.some details.


.. _Release Notes_Aqua_0.8.1_Critical Issues:

Critical Issues
---------------

- Be aware that ``initial_state`` parameter in ``QAOA`` has now different implementation
  as a result of a bug fix. The previous implementation wrongly mixed the user provided
  ``initial_state`` with Hadamard gates. The issue is fixed now. No attention needed if
  your code does not make use of the user provided ``initial_state`` parameter.


.. _Release Notes_Aqua_0.8.1_Bug Fixes:

Bug Fixes
---------

- optimize_svm method of qp_solver would sometimes fail resulting in an error like this
  `ValueError: cannot reshape array of size 1 into shape (200,1)` This addresses the issue
  by adding an L2 norm parameter, lambda2, which defaults to 0.001 but can be changed via
  the QSVM algorithm, as needed, to facilitate convergence.

- A method ``one_letter_symbol`` has been removed from the ``VarType`` in the latest
  build of DOCplex making Aqua incompatible with this version. So instead of using this method
  an explicit type check of variable types has been introduced in the Aqua optimization module.

- :meth`~qiskit.aqua.operators.state_fns.DictStateFn.sample()` could only handle
  real amplitudes, but it is fixed to handle complex amplitudes.
  `#1311 <https://github.com/Qiskit/qiskit-aqua/issues/1311>` for more details.

- Trotter class did not use the reps argument in constructor.
  `#1317 <https://github.com/Qiskit/qiskit-aqua/issues/1317>` for more details.

- Raise an `AquaError` if :class`qiskit.aqua.operators.converters.CircuitSampler`
  samples an empty operator.
  `#1321 <https://github.com/Qiskit/qiskit-aqua/issues/1321>` for more details.

- :meth:`~qiskit.aqua.operators.legacy.WeightedPauliOperator.to_opflow()`
  returns a correct operator when coefficients are complex numbers.
  `#1381 <https://github.com/Qiskit/qiskit-aqua/issues/1381>` for more details.

- Let backend simulators validate NoiseModel support instead of restricting to Aer only
  in QuantumInstance.

- Correctly handle PassManager on QuantumInstance ``transpile`` method by
  calling its ``run`` method if it exists.

- A bug that mixes custom ``initial_state`` in ``QAOA`` with Hadamard gates has been fixed.
  This doesn't change functionality of QAOA if no initial_state is provided by the user.
  Attention should be taken if your implementation uses QAOA with cusom ``initial_state``
  parameter as the optimization results might differ.

- Previously, setting `seed_simulator=0` in the `QuantumInstance` did not set
  any seed. This was only affecting the value 0. This has been fixed.


 .. _Release Notes_IBMQ_0.11.1:

IBM Q Provider 0.11.1
=====================

 .. _Release Notes_IBMQ_0.11.1_New Features:

New Features
------------

- :class:`qiskit.providers.ibmq.experiment.Experiment` now has three
  additional attributes, `hub`, `group`, and `project`, that identify
  the provider used to create the experiment.

- Methods
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.experiments` and
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.analysis_results`
  now support a ``limit`` parameter that allows you to limit the number of
  experiments and analysis results returned.


.. _Release Notes_IBMQ_0.11.1_Upgrade Notes:

Upgrade Notes
-------------

- A new parameter, ``limit`` is now the first parameter for both
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.experiments` and
  :meth:`qiskit.providers.ibmq.experiment.ExperimentService.analysis_results`
  methods. This ``limit`` has a default value of 10, meaning by deafult only
  10 experiments and analysis results will be returned.


.. _Release Notes_IBMQ_0.11.1_Bug Fixes:

Bug Fixes
---------

- Fixes the issue wherein a job could be left in the ``CREATING`` state if
  job submit fails half-way through.

- Fixes the infinite loop raised when passing an ``IBMQRandomService`` instance
  to a child process.


*************
Qiskit 0.23.0
*************

Terra 0.16.0
============

.. _Release Notes_0.16.0_Prelude:

Prelude
-------

The 0.16.0 release includes several new features and bug fixes. The
major features in this release are the following:

* Introduction of scheduled circuits, where delays can be used to control
  the timing and alignment of operations in the circuit.
* Compilation of quantum circuits from classical functions, such as
  oracles.
* Ability to compile and optimize single qubit rotations over different
  Euler basis as well as the phase + square-root(X) basis (i.e.
  ``['p', 'sx']``), which will replace the older IBM Quantum basis of
  ``['u1', 'u2', 'u3']``.
* Tracking of :meth:`~qiskit.circuit.QuantumCircuit.global_phase` on the
  :class:`~qiskit.circuit.QuantumCircuit` class has been extended through
  the :mod:`~qiskit.transpiler`, :mod:`~qiskit.quantum_info`, and
  :mod:`~qiskit.assembler` modules, as well as the BasicAer and Aer
  simulators. Unitary and state vector simulations will now return global
  phase-correct unitary matrices and state vectors.

Also of particular importance for this release is that Python 3.5 is no
longer supported. If you are using Qiskit Terra with Python 3.5, the
0.15.2 release is that last version which will work.


.. _Release Notes_0.16.0_New Features:

New Features
------------

- Global R gates have been added to :mod:`qiskit.circuit.library`. This
  includes the global R gate (:class:`~qiskit.circuit.library.GR`),
  global Rx (:class:`~qiskit.circuit.library.GRX`) and global Ry
  (:class:`~qiskit.circuit.library.GRY`) gates which are derived from the
  :class:`~qiskit.circuit.library.GR` gate, and global Rz (
  :class:`~qiskit.circuit.library.GRZ`) that is defined in a similar way
  to the :class:`~qiskit.circuit.library.GR` gates. The global R gates are
  defined on a number of qubits simultaneously, and act as a direct sum of
  R gates on each qubit.

  For example:

  .. code-block :: python

    from qiskit import QuantumCircuit, QuantumRegister
    import numpy as np

    num_qubits = 3
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    qc.compose(GR(num_qubits, theta=np.pi/3, phi=2*np.pi/3), inplace=True)

  will create a :class:`~qiskit.circuit.QuantumCircuit` on a
  :class:`~qiskit.circuit.QuantumRegister` of 3 qubits and perform a
  :class:`~qiskit.circuit.library.RGate` of an angle
  :math:`\theta = \frac{\pi}{3}` about an axis in the xy-plane of the Bloch
  spheres that makes an angle of :math:`\phi = \frac{2\pi}{3}` with the x-axis
  on each qubit.

- A new color scheme, ``iqx``, has been added to the ``mpl`` backend for the
  circuit drawer :func:`qiskit.visualization.circuit_drawer` and
  :meth:`qiskit.circuit.QuantumCircuit.draw`. This uses the same color scheme
  as the Circuit Composer on the IBM Quantum Experience website. There are
  now 3 available color schemes - ``default``, ``iqx``, and ``bw``.

  There are two ways to select a color scheme. The first is to use a user
  config file, by default in the ``~/.qiskit`` directory, in the
  file ``settings.conf`` under the ``[Default]`` heading, a user can enter
  ``circuit_mpl_style = iqx`` to select the ``iqx`` color scheme.

  The second way is to add ``{'name': 'iqx'}`` to the ``style`` kwarg to the
  ``QuantumCircuit.draw`` method or to the ``circuit_drawer`` function. The
  second way will override the setting in the settings.conf file. For example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    circuit.draw('mpl', style={'name': 'iqx'})

- In the ``style`` kwarg for the the circuit drawer
  :func:`qiskit.visualization.circuit_drawer` and
  :meth:`qiskit.circuit.QuantumCircuit.draw` the ``displaycolor`` field with
  the ``mpl`` backend now allows for entering both the gate color and the text
  color for each gate type in the form ``(gate_color, text_color)``.  This
  allows the use of light and dark gate colors with contrasting text colors.
  Users can still set only the gate color, in which case the ``gatetextcolor``
  field will be used. Gate colors can be set in the ``style`` dict for any
  number of gate types, from one to the entire ``displaycolor`` dict. For
  example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    circuit = QuantumCircuit(1)
    circuit.h(0)

    style_dict = {'displaycolor': {'h': ('#FA74A6', '#000000')}}
    circuit.draw('mpl', style=style_dict)

  or

  .. jupyter-execute::

    style_dict = {'displaycolor': {'h': '#FA74A6'}}
    circuit.draw('mpl', style=style_dict)

- Two alignment contexts are added to the pulse builder
  (:mod:`qiskit.pulse.builder`) to facilitate writing a repeated pulse
  sequence with delays.

  * :func:`qiskit.pulse.builder.align_equispaced` inserts delays with
    equivalent length in between pulse schedules within the context.
  * :func:`qiskit.pulse.builder.align_func` offers more advanced control of
    pulse position. This context takes a callable that calculates a fractional
    coordinate of i-th pulse and aligns pulses within the context. This makes
    coding of dynamical decoupling easy.

- A ``rep_delay`` parameter has been added to the
  :class:`~qiskit.qobj.QasmQobj` class under the run configuration,
  :class:`~qiskit.qobj.QasmQobjConfig`. This parameter is used to denote the
  time between program executions. It must be chosen from the backend range
  given by the :class:`~qiskit.providers.models.BackendConfiguration`
  method
  :meth:`~qiskit.providers.models.BackendConfiguration.rep_delay_range`. If a
  value is not provided a backend default,
  :attr:`qiskit.providers.models.BackendConfiguration.default_rep_delay`,
  will be used. ``rep_delay`` will only work on backends which allow for
  dynamic repetition time. This is can be checked with the
  :class:`~qiskit.providers.models.BackendConfiguration` property
  :attr:`~qiskit.providers.models.BackendConfiguration.dynamic_reprate_enabled`.

- The ``qobj_schema.json`` JSON Schema file in :mod:`qiskit.schemas` has
  been updated to include the ``rep_delay`` as an optional configuration
  property for QASM Qobjs.

- The ``backend_configuration_schema.json`` JSON Schema file in
  :mod:`qiskit.schemas` has been updated to include ``dynamic_reprate_enabled``,
  ``rep_delay_range`` and ``default_rep_delay`` as optional properties for a QASM
  backend configuration payload.

- A new optimization pass,
  :class:`qiskit.transpiler.passes.TemplateOptimization` has been added to
  the transpiler. This pass applies a template matching algorithm described
  in `arXiv:1909.05270 <https://arxiv.org/pdf/1909.05270.pdf>`__ that
  replaces all compatible maximal matches in the circuit.

  To implement this new transpiler pass a new module, ``template_circuits``,
  was added to the circuit library (:mod:`qiskit.circuit.library`). This new
  module contains all the Toffoli circuit templates used in the
  :class:`~qiskit.transpiler.passes.TemplateOptimization`.

  This new pass is **not** currently included in the preset pass managers
  (:mod:`qiskit.transpiler.preset_passmanagers`), to use it you will need
  to create a custom :class:`~qiskit.transpiler.PassManager`.

- A new version of the providers interface has been added. This new interface,
  which can be found in :mod:`qiskit.providers`, provides a new versioning
  mechanism that will enable changes to the interface to happen in a
  compatible manner over time. The new interface should be simple to migrate
  existing providers, as it is mostly identical except for the explicit
  versioning.

  Besides having explicitly versioned abstract classes the key changes for
  the new interface are that the :class:`~qiskit.providers.BackendV1`
  method :meth:`~qiskit.providers.BackendV1.run` can now
  take a :class:`~qiskit.circuits.QuantumCircuit` or
  :class:`~qiskit.pulse.Schedule` object as inputs instead of ``Qobj``
  objects. To go along with that options are now part of a backend class
  so that users can configure run time options when running with a circuit.
  The final change is that :class:`qiskit.providers.JobV1` can now be
  synchronous or asynchronous, the exact configuration and method for
  configuring this is up to the provider, but there are interface hook
  points to make it explicit which execution model a job is running under
  in the ``JobV1`` abstract class.

- A new kwarg, ``inplace``, has been added to the function
  :func:`qiskit.result.marginal_counts`. This kwarg is used to control whether
  the contents are marginalized in place or a new copy is returned, for
  :class:`~qiskit.result.Result` object input. This parameter does not have
  any effect for an input ``dict`` or :class:`~qiskit.result.Counts` object.

- An initial version of a classical function compiler,
  :mod:`qiskit.circuit.classicalfunction`, has been added. This
  enables compiling typed python functions (operating only on bits of type
  ``Int1`` at the moment) into :class:`~qiskit.circuit.QuantumCircuit`
  objects. For example:

  .. jupyter-execute::

    from qiskit.circuit import classical_function, Int1

    @classical_function
    def grover_oracle(a: Int1, b: Int1, c: Int1, d: Int1) -> Int1:
         x = not a and b
         y = d and not c
         z = not x or y
         return z

    quantum_circuit = grover_oracle.synth()
    quantum_circuit.draw()

  The parameter ``registerless=False`` in the
  :class:`qiskit.circuit.classicalfunction.ClassicalFunction` method
  :meth:`~qiskit.circuit.classicalfunction.ClassicalFunction.synth` creates a
  circuit with registers refering to the parameter names. For example:

  .. jupyter-execute::

    quantum_circuit = grover_oracle.synth(registerless=False)
    quantum_circuit.draw()

  A decorated classical function can be used the same way as any other
  quantum gate when appending it to a circuit.

  .. jupyter-execute::

    circuit = QuantumCircuit(5)
    circuit.append(grover_oracle, range(5))
    circuit.draw()

  The ``GROVER_ORACLE`` gate is synthesized when its decomposition is required.

  .. jupyter-execute::

    circuit.decompose().draw()

  The feature requires ``tweedledum``, a library for synthesizing quantum
  circuits, that can be installed via pip with ``pip install tweedledum``.

- A new class :class:`qiskit.circuit.Delay` for representing a delay
  instruction in a circuit has been added. A new method
  :meth:`~qiskit.circuit.QuantumCircuit.delay` is now available for easily
  appending delays to circuits. This makes it possible to describe
  timing-sensitive experiments (e.g. T1/T2 experiment) in the circuit level.

  .. jupyter-execute::

      from qiskit import QuantumCircuit

      qc = QuantumCircuit(1, 1)
      qc.delay(500, 0, unit='ns')
      qc.measure(0, 0)

      qc.draw()

- A new argument ``scheduling_method`` for
  :func:`qiskit.compiler.transpile` has been added. It is required when
  transpiling circuits with delays.  If ``scheduling_method`` is specified,
  the transpiler returns a scheduled circuit such that all idle times in it
  are padded with delays (i.e. start time of each instruction is uniquely
  determined). This makes it possible to see how scheduled instructions
  (gates) look in the circuit level.

  .. jupyter-execute::

      from qiskit import QuantumCircuit, transpile
      from qiskit.test.mock.backends import FakeAthens

      qc = QuantumCircuit(2)
      qc.h(0)
      qc.cx(0, 1)

      scheduled_circuit = transpile(qc, backend=FakeAthens(), scheduling_method="alap")
      print("Duration in dt:", scheduled_circuit.duration)
      scheduled_circuit.draw(idle_wires=False)

  See also :func:`~qiskit.visualization.timeline_drawer` for the best visualization
  of scheduled circuits.

- A new fuction :func:`qiskit.compiler.sequence` has been also added so that
  we can convert a scheduled circuit into a :class:`~qiskit.pulse.Schedule`
  to make it executable on a pulse-enabled backend.

  .. code-block:: python

      from qiskit.compiler import sequence

      sched = sequence(scheduled_circuit, pulse_enabled_backend)

- The :func:`~qiskit.compiler.schedule` has been updated so that it can
  schedule circuits with delays. Now there are two paths to schedule a
  circuit with delay:

  .. code-block:: python

      qc = QuantumCircuit(1, 1)
      qc.h(0)
      qc.delay(500, 0, unit='ns')
      qc.h(0)
      qc.measure(0, 0)

      sched_path1 = schedule(qc.decompose(), backend)
      sched_path2 = sequence(transpile(qc, backend, scheduling_method='alap'), backend)
      assert pad(sched_path1) == sched_path2

  Refer to the release notes and documentation for
  :func:`~qiskit.compiler.transpile` and :func:`~qiskit.compiler.sequence`
  for the details on the other path.

- Added the :class:`~qiskit.circuit.library.GroverOperator` to the circuit
  library (:mod:`qiskit.circuit.library`) to construct the Grover operator
  used in Grover's search algorithm and Quantum Amplitude
  Amplification/Estimation. Provided with an oracle in form of a circuit,
  ``GroverOperator`` creates the textbook Grover operator. To generalize
  this for amplitude amplification and use a generic operator instead of
  Hadamard gates as state preparation, the ``state_in`` argument can be
  used.

- The :class:`~qiskit.pulse.InstructionScheduleMap` methods
  :meth:`~qiskit.pulse.InstructionScheduleMap.get` and
  :meth:`~qiskit.pulse.InstructionScheduleMap.pop` methods now take
  :class:`~qiskit.circuit.ParameterExpression` instances
  in addition to numerical values for schedule generator parameters. If the
  generator is a function, expressions may be bound before or within the
  function call. If the generator is a
  :class:`~qiskit.pulse.ParametrizedSchedule`, expressions must be
  bound before the schedule itself is bound/called.

- A new class :class:`~qiskit.circuit.library.LinearAmplitudeFunction` was
  added to the circuit library (:mod:`qiskit.circuit.library`) for mapping
  (piecewise) linear functions on qubit amplitudes,

  .. math::

      F|x\rangle |0\rangle = \sqrt{1 - f(x)}|x\rangle |0\rangle + \sqrt{f(x)}|x\rangle |1\rangle


  The mapping is based on a controlled Pauli Y-rotations and
  a Taylor approximation, as described in https://arxiv.org/abs/1806.06893.
  This circuit can be used to compute expectation values of linear
  functions using the quantum amplitude estimation algorithm.

- The new jupyter magic ``monospaced_output`` has been added to the
  :mod:`qiskit.tools.jupyter` module. This magic sets the Jupyter notebook
  output font to "Courier New", when possible. When used this fonts returns
  text circuit drawings that are better aligned.

  .. code-block:: python

    import qiskit.tools.jupyter
    %monospaced_output

- A new transpiler pass,
  :class:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition`,
  has been added. This transpiler pass is an alternative to the existing
  :class:`~qiskit.transpiler.passes.Optimize1qGates` that uses the
  :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` class to decompose
  and simplify a chain of single qubit gates. This method is compatible with
  any basis set, while :class:`~qiskit.transpiler.passes.Optimize1qGates`
  only works for u1, u2, and u3. The default pass managers for
  ``optimization_level`` 1, 2, and 3 have been updated to use this new pass
  if the basis set doesn't include u1, u2, or u3.

- The :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` now supports
  two new basis, ``'PSX'`` and ``'U'``. These can be specified with the
  ``basis`` kwarg on the constructor. This will decompose the matrix into a
  circuit using :class:`~qiskit.circuit.library.PGate` and
  :class:`~qiskit.circuit.library.SXGate` for ``'PSX'``, and
  :class:`~qiskit.circuit.library.UGate` for ``'U'``.

- A new method :meth:`~qiskit.transpiler.PassManager.remove` has been added
  to the :class:`qiskit.transpiler.PassManager` class. This method enables
  removing a pass from a :class:`~qiskit.transpiler.PassManager` instance.
  It works on indexes, similar to
  :meth:`~qiskit.transpiler.PassManager.replace`. For example, to
  remove the :class:`~qiskit.transpiler.passes.RemoveResetInZeroState` pass
  from the pass manager used at optimization level 1:

  .. code-block:: python

    from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
    from qiskit.transpiler.passmanager_config import PassManagerConfig

    pm = level_1_pass_manager(PassManagerConfig())
    pm.draw()

  .. code-block::

    [0] FlowLinear: UnrollCustomDefinitions, BasisTranslator
    [1] FlowLinear: RemoveResetInZeroState
    [2] DoWhile: Depth, FixedPoint, Optimize1qGates, CXCancellation

  The stage ``[1]`` with ``RemoveResetInZeroState`` can be removed like this:

  .. code-block:: python

    pass_manager.remove(1)
    pass_manager.draw()

  .. code-block::

    [0] FlowLinear: UnrollCustomDefinitions, BasisTranslator
    [1] DoWhile: Depth, FixedPoint, Optimize1qGates, CXCancellation

- Several classes to load probability distributions into qubit amplitudes;
  :class:`~qiskit.circuit.library.UniformDistribution`,
  :class:`~qiskit.circuit.library.NormalDistribution`, and
  :class:`~qiskit.circuit.library.LogNormalDistribution` were added to the
  circuit library (:mod:`qiskit.circuit.library`). The normal and
  log-normal distribution support both univariate and multivariate
  distributions. These circuits are central to applications in finance
  where quantum amplitude estimation is used.

- Support for pulse gates has been added to the
  :class:`~qiskit.circuit.QuantumCircuit` class. This enables a
  :class:`~qiskit.circuit.QuantumCircuit` to override (for basis gates) or
  specify (for standard and custom gates) a definition of a
  :class:`~qiskit.circuit.Gate` operation in terms of time-ordered signals
  across hardware channels. In other words, it enables the option to provide
  pulse-level custom gate calibrations.

  The circuits are built exactly as before. For example::

      from qiskit import pulse
      from qiskit.circuit import QuantumCircuit, Gate

      class RxGate(Gate):
          def __init__(self, theta):
              super().__init__('rxtheta', 1, [theta])

      circ = QuantumCircuit(1)
      circ.h(0)
      circ.append(RxGate(3.14), [0])

  Then, the calibration for the gate can be registered using the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.add_calibration` which takes a
  :class:`~qiskit.pulse.Schedule` definition as well as the qubits and
  parameters that it is defined for::

      # Define the gate implementation as a schedule
      with pulse.build() as custom_h_schedule:
          pulse.play(pulse.library.Drag(...), pulse.DriveChannel(0))

      with pulse.build() as q1_x180:
          pulse.play(pulse.library.Gaussian(...), pulse.DriveChannel(1))

      # Register the schedule to the gate
      circ.add_calibration('h', [0], custom_h_schedule)  # or gate.name string to register
      circ.add_calibration(RxGate(3.14), [0], q1_x180)   # Can accept gate

  Previously, this functionality could only be used through complete Pulse
  Schedules. Additionally, circuits can now be submitted to backends with
  your custom definitions (dependent on backend support).

  Circuits with pulse gates can still be lowered to a
  :class:`~qiskit.pulse.Schedule` by using the
  :func:`~qiskit.compiler.schedule` function.

  The calibrated gate can also be transpiled using the regular transpilation
  process::

      transpiled_circuit = transpile(circ, backend)

  The transpiled circuit will leave the calibrated gates on the same qubit as
  the original circuit and will not unroll them to the basis gates.

- Support for disassembly of :class:`~qiskit.qobj.PulseQobj` objects has
  been added to the :func:`qiskit.assembler.disassemble` function.
  For example:

  .. code-block::

    from qiskit import pulse
    from qiskit.assembler.disassemble import disassemble
    from qiskit.compiler.assemble import assemble
    from qiskit.test.mock import FakeOpenPulse2Q

    backend = FakeOpenPulse2Q()

    d0 = pulse.DriveChannel(0)
    d1 = pulse.DriveChannel(1)
    with pulse.build(backend) as sched:
        with pulse.align_right():
            pulse.play(pulse.library.Constant(10, 1.0), d0)
            pulse.shift_phase(3.11, d0)
            pulse.measure_all()

    qobj = assemble(sched, backend=backend, shots=512)
    scheds, run_config, header = disassemble(qobj)

- A new kwarg, ``coord_type`` has been added to
  :func:`qiskit.visualization.plot_bloch_vector`. This kwarg enables
  changing the coordinate system used for the input parameter that
  describes the positioning of the vector on the Bloch sphere in the
  generated visualization. There are 2 supported values for this new kwarg,
  ``'cartesian'`` (the default value) and ``'spherical'``. If the
  ``coord_type`` kwarg is set to ``'spherical'`` the list of parameters
  taken in are of the form ``[r, theta,  phi]`` where ``r`` is the
  radius, ``theta`` is the inclination from +z direction, and ``phi`` is
  the azimuth from +x direction. For example:

  .. jupyter-execute::

    from numpy import pi

    from qiskit.visualization import plot_bloch_vector

    x = 0
    y = 0
    z = 1
    r = 1
    theta = pi
    phi = 0


    # Cartesian coordinates, where (x,y,z) are cartesian coordinates
    # for bloch vector
    plot_bloch_vector([x,y,z])

  .. jupyter-execute::

    plot_bloch_vector([x,y,z], coord_type="cartesian")  # Same as line above

  .. jupyter-execute::

    # Spherical coordinates, where (r,theta,phi) are spherical coordinates
    # for bloch vector
    plot_bloch_vector([r, theta, phi], coord_type="spherical")

- Pulse :py:class:`~qiskit.pulse.Schedule` objects now support
  using :py:class:`~qiskit.circuit.ParameterExpression` objects
  for parameters.

  For example::

      from qiskit.circuit import Parameter
      from qiskit import pulse

      alpha = Parameter('⍺')
      phi = Parameter('ϕ')
      qubit = Parameter('q')
      amp = Parameter('amp')

      schedule = pulse.Schedule()
      schedule += SetFrequency(alpha, DriveChannel(qubit))
      schedule += ShiftPhase(phi, DriveChannel(qubit))
      schedule += Play(Gaussian(duration=128, sigma=4, amp=amp),
                       DriveChannel(qubit))
      schedule += ShiftPhase(-phi, DriveChannel(qubit))

  Parameter assignment is done via the
  :meth:`~qiskit.pulse.Schedule.assign_parameters` method::

      schedule.assign_parameters({alpha: 4.5e9, phi: 1.57,
                                  qubit: 0, amp: 0.2})

  Expressions and partial assignment also work, such as::

      beta = Parameter('b')
      schedule += SetFrequency(alpha + beta, DriveChannel(0))
      schedule.assign_parameters({alpha: 4.5e9})
      schedule.assign_parameters({beta: phi / 6.28})

- A new visualization function :func:`~qiskit.visualization.timeline_drawer`
  was added to the :mod:`qiskit.visualization` module.

  For example:

  .. jupyter-execute::

    from qiskit.visualization import timeline_drawer
    from qiskit import QuantumCircuit, transpile
    from qiskit.test.mock import FakeAthens

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    timeline_drawer(transpile(qc, FakeAthens(), scheduling_method='alap'))


.. _Release Notes_0.16.0_Upgrade Notes:

Upgrade Notes
-------------

- Type checking for the ``params`` kwarg of the constructor for the
  :class:`~qiskit.circuit.Gate` class and its subclasses has been changed.
  Previously all :class:`~qiskit.circuit.Gate` parameters had to be
  in a set of allowed types defined in the
  :class:`~qiskit.circuit.Instruction` class. Now a new method,
  :meth:`~qiskit.circuit.Gate.validate_parameter` is used to determine
  if a parameter type is valid or not. The definition of this method in
  a subclass will take priority over its parent. For example,
  :class:`~qiskit.extensions.UnitaryGate` accepts a parameter of the type
  ``numpy.ndarray`` and defines a custom
  :meth:`~qiskit.extensionst.UnitaryGate.validate_parameter` method that
  returns the parameter if it's an ``numpy.ndarray``. This takes priority
  over the function defined in its parent class :class:`~qiskit.circuit.Gate`.
  If :class:`~qiskit.extensions.UnitaryGate` were to be used as parent
  for a new class, this ``validate_parameter`` method would be used unless
  the new child class defines its own method.

- The previously deprecated methods, arguments, and properties named
  ``n_qubits`` and ``numberofqubits``  have been removed. These were
  deprecated in the 0.13.0 release. The full set of changes are:

  .. list-table::
    :header-rows: 1

    * - Class
      - Old
      - New
    * - :class:`~qiskit.circuit.QuantumCircuit`
      - ``n_qubits``
      - :class:`~qiskit.circuit.QuantumCircuit.num_qubits`
    * - :class:`~qiskit.quantum_info.Pauli`
      - ``numberofqubits``
      - :attr:`~qiskit.quantum_info.Pauli.num_qubits`

  .. list-table::
    :header-rows: 1

    * - Function
      - Old Argument
      - New Argument
    * - :func:`qiskit.circuit.random.random_circuit`
      - ``n_qubits``
      - ``num_qubits``
    * - :class:`qiskit.circuit.library.MSGate`
      - ``n_qubits``
      - ``num_qubits``

- Inserting a parameterized :class:`~qiskit.circuit.Gate` instance into
  a :class:`~qiskit.circuit.QuantumCircuit` now creates a copy of that
  gate which is used in the circuit. If changes are made to the instance
  inserted into the circuit it will no longer be reflected in the gate in
  the circuit. This change was made to fix an issue when inserting a single
  parameterized :class:`~qiskit.circuit.Gate` object into multiple circuits.

- The function :func:`qiskit.result.marginal_counts` now, by default,
  does not modify the :class:`qiskit.result.Result` instance
  parameter. Previously, the ``Result`` object was always modified in place.
  A new kwarg ``inplace`` has been added
  :func:`~qiskit.result.marginal_counts` which enables using the previous
  behavior when ``inplace=True`` is set.

- The :class:`~qiskit.circuit.library.U3Gate` definition has been changed to
  be in terms of the :class:`~qiskit.circuit.library.UGate` class. The
  :class:`~qiskit.circuit.library.UGate` class has no definition. It is
  therefore not possible to unroll **every** circuit in terms of U3
  and CX anymore. Instead, U and CX can be used for **every** circuit.

- The deprecated support for running Qiskit Terra with Python 3.5 has been
  removed. To use Qiskit Terra from this release onward you will now need to
  use at least Python 3.6. If you are using Python 3.5 the last version which
  will work is Qiskit Terra 0.15.2.

- In the :class:`~qiskit.providers.models.PulseBackendConfiguration`
  in the ``hamiltonian`` attributes the ``vars`` field  is now returned
  in a unit of Hz instead of the previously used GHz. This change was made
  to be consistent with the units used with the other attributes in the
  class.

- The previously deprecated support for passing in a dictionary as the
  first positional argument to :class:`~qiskit.dagcircuit.DAGNode` constructor
  has been removed. Using a dictonary for the first positional argument
  was deprecated in the 0.13.0 release. To create a
  :class:`~qiskit.dagcircuit.DAGNode` object now you should directly
  pass the attributes as kwargs on the constructor.

- The keyword arguments for the circuit gate methods (for example:
  :class:`qiskit.circuit.QuantumCircuit.cx`) ``q``, ``ctl*``, and
  ``tgt*``, which were deprecated in the 0.12.0 release, have been removed.
  Instead, only  ``qubit``, ``control_qubit*`` and ``target_qubit*`` can be
  used as named arguments for these methods.

- The previously deprecated module ``qiskit.extensions.standard`` has been
  removed. This module has been deprecated since the 0.14.0 release.
  The :mod:`qiskit.circuit.library` can be used instead.
  Additionally, all the gate classes previously in
  ``qiskit.extensions.standard`` are still importable from
  :mod:`qiskit.extensions`.

- The previously deprecated gates in the module
  ``qiskit.extensions.quantum_initializer``:
  ``DiagGate``, `UCG``, ``UCPauliRotGate``, ``UCRot``, ``UCRXGate``, ``UCX``,
  ``UCRYGate``, ``UCY``, ``UCRZGate``, ``UCZ`` have been removed. These were
  all deprecated in the 0.14.0 release and have alternatives available in
  the circuit library (:mod:`qiskit.circuit.library`).

- The previously deprecated :class:`qiskit.circuit.QuantumCircuit` gate method
  :meth:`~qiskit.circuit.QuantumCircuit.iden` has been removed. This was
  deprecated in the 0.13.0 release and
  :meth:`~qiskit.circuit.QuantumCircuit.i` or
  :meth:`~qiskit.circuit.QuantumCircuit.id` can be used instead.


Deprecation Notes
-----------------

- The use of a ``numpy.ndarray`` for a parameter in the ``params`` kwarg
  for the constructor of the :class:`~qiskit.circuit.Gate` class and
  subclasses has been deprecated and will be removed in future releases. This
  was done as part of the refactoring of how ``parms`` type checking is
  handled for the :class:`~qiskit.circuit.Gate` class. If you have a custom
  gate class which is a subclass of :class:`~qiskit.circuit.Gate` directly
  (or via a different parent in the hierarchy) that accepts an ``ndarray``
  parameter, you should define a custom
  :meth:`~qiskit.circuit.Gate.validate_parameter` method for your class
  that will return the allowed parameter type. For example::

    def validate_parameter(self, parameter):
        """Custom gate parameter has to be an ndarray."""
        if isinstance(parameter, numpy.ndarray):
            return parameter
        else:
            raise CircuitError("invalid param type {0} in gate "
                               "{1}".format(type(parameter), self.name))

- The
  :attr:`~qiskit.circuit.library.PiecewiseLinearPauliRotations.num_ancilla_qubits`
  property of the :class:`~qiskit.circuit.library.PiecewiseLinearPauliRotations`
  and :class:`~qiskit.circuit.library.PolynomialPauliRotations` classes has been
  deprecated and will be removed in a future release. Instead the property
  :attr:`~qiskit.circuit.library.PolynomialPauliRotations.num_ancillas` should
  be used instead. This was done to make it consistent with the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.num_ancillas`.

- The :class:`qiskit.circuit.library.MSGate` class has been
  deprecated, but will remain in place to allow loading of old jobs. It has been replaced
  with the :class:`qiskit.circuit.library.GMS` class which should be used
  instead.

- The :class:`~qiskit.transpiler.passes.MSBasisDecomposer` transpiler pass
  has been deprecated and will be removed in a future release.
  The :class:`qiskit.transpiler.passes.BasisTranslator` pass can be used
  instead.

- The :class:`~qiskit.circuit.QuantumCircuit` methods ``u1``, ``u2`` and
  ``u3`` are now deprecated. Instead the following replacements can be
  used.

  .. code-block::

      u1(theta) = p(theta) = u(0, 0, theta)
      u2(phi, lam) = u(pi/2, phi, lam) = p(pi/2 + phi) sx p(pi/2 lam)
      u3(theta, phi, lam) = u(theta, phi, lam) = p(phi + pi) sx p(theta + pi) sx p(lam)

  The gate classes themselves, :class:`~qiskit.circuit.library.U1Gate`,
  :class:`~qiskit.circuit.library.U2Gate` and :class:`~qiskit.circuit.library.U3Gate`
  remain, to allow loading of old jobs.


.. _Release Notes_0.16.0_Bug Fixes:

Bug Fixes
---------

- The :class:`~qiskit.result.Result` class's methods
  :meth:`~qiskit.result.Result.data`, :meth:`~qiskit.result.Result.get_memory`,
  :meth:`~qiskit.result.Result.get_counts`,  :meth:`~qiskit.result.Result.get_unitary`,
  and :meth:`~qiskit.result.Result.get_statevector ` will now emit a warning
  when the ``experiment`` kwarg is specified for attempting to fetch
  results using either a :class:`~qiskit.circuit.QuantumCircuit` or
  :class:`~qiskit.pulse.Schedule` instance, when more than one entry matching
  the instance name is present in the ``Result`` object. Note that only the
  first entry matching this name will be returned. Fixes
  `#3207 <https://github.com/Qiskit/qiskit-terra/issues/3207>`__

- The :class:`qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.append` can now be used to insert one
  parameterized gate instance into multiple circuits. This fixes a previous
  issue where inserting a single parameterized
  :class:`~qiskit.circuit.Gate` object into multiple circuits would
  cause failures when one circuit had a parameter assigned.
  Fixes `#4697 <https://github.com/Qiskit/qiskit-terra/issues/4697>`__

- Previously the :func:`qiskit.execute.execute` function would incorrectly
  disallow both the ``backend`` and ``pass_manager`` kwargs to be
  specified at the same time. This has been fixed so that both
  ``backend`` and ``pass_manager`` can be used together on calls to
  :func:`~qiskit.execute.execute`.
  Fixes `#5037 <https://github.com/Qiskit/qiskit-terra/issues/5037>`__

- The :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.unitary` method has been fixed
  to accept a single integer for the ``qarg`` argument (when adding a
  1-qubit unitary). The allowed types for the ``qargs`` argument are now
  ``int``, :class:`~qiskit.circuit.Qubit`, or a list of integers.
  Fixes `#4944 <https://github.com/Qiskit/qiskit-terra/issues/4944>`__

- Previously, calling :meth:`~qiskit.circuit.library.BlueprintCircuit.inverse`
  on a :class:`~qiskit.circuit.library.BlueprintCircuit` object
  could fail if its internal data property was not yet populated. This has
  been fixed so that the calling
  :meth:`~qiskit.circuit.library.BlueprintCircuit.inverse` will populate
  the internal data before generating the inverse of the circuit.
  Fixes `#5140 <https://github.com/Qiskit/qiskit-terra/issues/5140>`__

- Fixed an issue when creating a :class:`qiskit.result.Counts` object from an
  empty data dictionary. Now this will create an empty
  :class:`~qiskit.result.Counts` object. The
  :meth:`~qiskit.result.Counts.most_frequent` method is also updated to raise
  a more descriptive exception when the object is empty. Fixes
  `#5017 <https://github.com/Qiskit/qiskit-terra/issues/5017>`__

- Fixes a bug where setting ``ctrl_state`` of a
  :class:`~qiskit.extensions.UnitaryGate` would be applied twice; once
  in the creation of the matrix for the controlled unitary and again
  when calling the :meth:`~qiskit.circuit.ControlledGate.definition` method of
  the :class:`qiskit.circuit.ControlledGate` class. This would give the
  appearence that setting ``ctrl_state`` had no effect.

- Previously the :class:`~qiskit.circuit.ControlledGate` method
  :meth:`~qiskit.circuit.ControlledGate.inverse` would not preserve the
  ``ctrl_state`` parameter in some cases. This has been fixed so that
  calling :meth:`~qiskit.circuit.ControlledGate.inverse` will preserve
  the value ``ctrl_state`` in its output.

- Fixed a bug in the ``mpl`` output backend of the circuit drawer
  :meth:`qiskit.circuit.QuantumCircuit.draw` and
  :func:`qiskit.visualization.circuit_drawer` that would
  cause the drawer to fail if the ``style`` kwarg was set to a string.
  The correct behavior would be to treat that string as a path to
  a JSON file containing the style sheet for the visualization. This has
  been fixed, and warnings are raised if the JSON file for the style
  sheet can't be loaded.

- Fixed an error where loading a QASM file via
  :meth:`~qiskit.circuit.QuantumCircuit.from_qasm_file` or
  :meth:`~qiskit.circuit.QuantumCircuit.from_qasm_str` would fail
  if a ``u``, ``phase(p)``, ``sx``, or ``sxdg`` gate were present in
  the QASM file.
  Fixes `#5156 <https://github.com/Qiskit/qiskit-terra/issues/5151>`__

- Fixed a bug that would potentially cause registers to be mismapped when
  unrolling/decomposing a gate defined with only one 2-qubit operation.

Aer 0.7.0
=========

.. _Release Notes_Aer_0.7.0_Prelude:

Prelude
-------

This 0.7.0 release includes numerous performance improvements and significant
enhancements to the simulator interface, and drops support for Python 3.5. The
main interface changes are configurable simulator backends, and constructing
preconfigured simulators from IBMQ backends. Noise model an basis gate support
has also been extended for most of the Qiskit circuit library standard gates,
including new support for 1 and 2-qubit rotation gates. Performance
improvements include adding SIMD support to the density matrix and unitary
simulation methods, reducing the used memory and improving the performance of
circuits using statevector and density matrix snapshots, and adding support
for Kraus instructions to the gate fusion circuit optimization for greatly
improving the performance of noisy statevector simulations.

.. _Release Notes_Aer_0.7.0_New Features:

New Features
------------

- Adds basis gate support for the :class:`qiskit.circuit.Delay`
  instruction to the :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and
  :class:`~qiskit.providers.aer.QasmSimulator`.
  Note that this gate is treated as an identity gate during simulation
  and the delay length parameter is ignored.

- Adds basis gate support for the single-qubit gate
  :class:`qiskit.circuit.library.UGate` to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  ``"statevector"``, ``"density_matrix"``, ``"matrix_product_state"``,
  and ``"extended_stabilizer"`` methods of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds basis gate support for the phase gate
  :class:`qiskit.circuit.library.PhaseGate` to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  ``"statevector"``, ``"density_matrix"``, ``"matrix_product_state"``,
  and ``"extended_stabilizer"`` methods of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds basis gate support for the controlled-phase gate
  :class:`qiskit.circuit.library.CPhaseGate` to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  ``"statevector"``, ``"density_matrix"``, and
  ``"matrix_product_state"`` methods of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds support for the multi-controlled phase gate
  :class:`qiskit.circuit.library.MCPhaseGate` to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  ``"statevector"`` method of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds support for the :math:`\sqrt(X)` gate
  :class:`qiskit.circuit.library.SXGate` to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds support for 1 and 2-qubit Qiskit circuit library rotation gates
  :class:`~qiskit.circuit.library.RXGate`, :class:`~qiskit.circuit.library.RYGate`,
  :class:`~qiskit.circuit.library.RZGate`, :class:`~qiskit.circuit.library.RGate`,
  :class:`~qiskit.circuit.library.RXXGate`, :class:`~qiskit.circuit.library.RYYGate`,
  :class:`~qiskit.circuit.library.RZZGate`, :class:`~qiskit.circuit.library.RZXGate`
  to the :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  ``"statevector"`` and ``"density_matrix"`` methods of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds support for multi-controlled rotation gates ``"mcr"``, ``"mcrx"``,
  ``"mcry"``, ``"mcrz"``
  to the :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and the
  ``"statevector"`` method of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Make simulator backends configurable. This allows setting persistant options
  such as simulation method and noise model for each simulator backend object.

  The :class:`~qiskit.providers.aer.QasmSimulator` and
  :class:`~qiskit.providers.aer.PulseSimulator` can also be configured from
  an :class:`~qiskit.providers.ibmq.IBMQBackend` backend object using the
  `:meth:`~qiskit.providers.aer.QasmSimulator.from_backend` method.
  For the :class:`~qiskit.providers.aer.QasmSimulator` this will configure the coupling map,
  basis gates, and basic device noise model based on the backend configuration and
  properties. For the :class:`~qiskit.providers.aer.PulseSimulator` the system model
  and defaults will be configured automatically from the backend configuration, properties and
  defaults.

  For example a noisy density matrix simulator backend can be constructed as
  ``QasmSimulator(method='density_matrix', noise_model=noise_model)``, or an ideal
  matrix product state simulator as ``QasmSimulator(method='matrix_product_state')``.

  A benefit is that a :class:`~qiskit.providers.aer.PulseSimulator` instance configured from
  a backend better serves as a drop-in replacement to the original backend, making it easier to
  swap in and out a simulator and real backend, e.g. when testing code on a simulator before
  using a real backend.
  For example, in the following code-block, the :class:`~qiskit.providers.aer.PulseSimulator` is
  instantiated from the ``FakeArmonk()`` backend. All configuration and default data is copied
  into the simulator instance, and so when it is passed as an argument to ``assemble``,
  it behaves as if the original backend was supplied (e.g. defaults from ``FakeArmonk`` will be
  present and used by ``assemble``).

  .. code-block:: python

      armonk_sim = qiskit.providers.aer.PulseSimulator.from_backend(FakeArmonk())
      pulse_qobj = assemble(schedules, backend=armonk_sim)
      armonk_sim.run(pulse_qobj)

  While the above example is small, the demonstrated 'drop-in replacement' behavior should
  greatly improve the usability in more complicated work-flows, e.g. when calibration experiments
  are constructed using backend attributes.

- Adds support for qobj global phase to the
  :class:`~qiskit.providers.aer.StatevectorSimulator`,
  :class:`~qiskit.providers.aer.UnitarySimulator`, and statevector
  methods of the :class:`~qiskit.providers.aer.QasmSimulator`.

- Improves general noisy statevector simulation performance by adding a Kraus
  method to the gate fusion circuit optimization that allows applying gate
  fusion to noisy statevector simulations with general Kraus noise.

- Use move semantics for statevector and density matrix snapshots for the
  `"statevector"` and `"density_matrix"` methods of the
  :class:`~qiskit.providers.aer.QasmSimulator` if they are the final
  instruction in a circuit. This reduces the memory usage of the
  simulator improves the performance by avoiding copying a large array in
  the results.

- Adds support for general Kraus
  :class:`~qiskit.providers.aer.noise.QauntumError` gate errors in the
  :class:`~qiskit.providers.aer.noise.NoiseModel` to the
  ``"matrix_product_state"`` method of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Adds support for density matrix snapshot instruction
  :class:`qiskit.providers.aer.extensions.SnapshotDensityMatrix` to the
  ``"matrix_product_state"`` method of the
  :class:`~qiskit.providers.aer.QasmSimulator`.

- Extends the SIMD vectorization of the statevector simulation method to the
  unitary matrix, superoperator matrix, and density matrix simulation methods.
  This gives roughtly a 2x performance increase general simulation using the
  :class:`~qiskit.providers.aer.UnitarySimulator`, the ``"density_matrix"``
  method of the :class:`~qiskit.providers.aer.QasmSimulator`, gate
  fusion, and noise simulation.

- Adds a custom vector class to C++ code that has better integration with
  Pybind11. This haves the memory requirement of the
  :class:`~qiskit.providers.aer.StatevectorSimulator` by avoiding an
  memory copy during Python binding of the final simulator state.


.. _Release Notes_Aer_0.7.0_Upgrade Notes:

Upgrade Notes
-------------

- AER now uses Lapack to perform some matrix related computations.
  It uses the Lapack library bundled with OpenBlas (already available
  in Linux and Macos typical OpenBlas dsitributions; Windows version
  distributed with AER) or with the accelerate framework in MacOS.

- The deprecated support for running qiskit-aer with Python 3.5 has
  been removed. To use qiskit-aer >=0.7.0 you will now need at
  least Python 3.6. If you are using Python 3.5 the last version which will
  work is qiskit-aer 0.6.x.

- Updates gate fusion default thresholds so that gate fusion will be applied
  to circuits with of more than 14 qubits for statevector simulations on the
  :class:`~qiskit.providers.aer.StatevectorSimulator` and
  :class:`~qiskit.providers.aer.QasmSimulator`.

  For the ``"density_matrix"``
  method of the :class:`~qiskit.providers.aer.QasmSimulator` and for the
  :class:`~qiskit.providers.aer.UnitarySimulator` gate fusion will be applied
  to circuits with more than 7 qubits.

  Custom qubit threshold values can be set using the ``fusion_threshold``
  backend option ie ``backend.set_options(fusion_threshold=10)``

- Changes ``fusion_threshold`` backend option to apply fusion when the
  number of qubits is above the threshold, not equal or above the threshold,
  to match the behavior of the OpenMP qubit threshold parameter.


.. _Release Notes_Aer_0.7.0_Deprecation Notes:

Deprecation Notes
-----------------

- :meth:`qiskit.providers.aer.noise.NoiseModel.set_x90_single_qubit_gates` has
  been deprecated as unrolling to custom basis gates has been added to the
  qiskit transpiler. The correct way to use an X90 based noise model is to
  define noise on the Sqrt(X) ``"sx"`` or ``"rx"`` gate and one of the single-qubit
  phase gates ``"u1"``, ``"rx"``, or ``"p"`` in the noise model.

- The ``variance`` kwarg of Snapshot instructions has been deprecated. This
  function computed the sample variance in the snapshot due to noise model
  sampling, not the variance due to measurement statistics so was often
  being used incorrectly. If noise modeling variance is required single shot
  snapshots should be used so variance can be computed manually in
  post-processing.


.. _Release Notes_Aer_0.7.0_Bug Fixes:

Bug Fixes
---------

- Fixes bug in the :class:`~qiskit.providers.aer.StatevectorSimulator` that
  caused it to always run as CPU with double-precision without SIMD/AVX2
  support even on systems with AVX2, or when single-precision or the GPU
  method was specified in the backend options.

- Fixes some for-loops in C++ code that were iterating over copies
  rather than references of container elements.

- Fixes a bug where snapshot data was always copied from C++ to Python rather
  than moved where possible. This will halve memory usage and improve simulation
  time when using large statevector or density matrix snapshots.

- Fix `State::snapshot_pauli_expval` to return correct Y
  expectation value in stabilizer simulator. Refer to
  `#895 <https://github.com/Qiskit/qiskit-aer/issues/895>`
  for more details.

- The controller_execute wrappers have been adjusted to be functors (objects)
  rather than free functions. Among other things, this allows them to be used
  in multiprocessing.pool.map calls.

- Add missing available memory checks for the
  :class:`~qiskit.providers.aer.StatevectorSimulator` and
  :class:`~qiskit.providers.aer.UnitarySimulator`. This throws an exception if
  the memory required to simulate the number of qubits in a circuit exceeds the
  available memory of the system.


.. _Release Notes_Ignis_0.5.0:

Ignis 0.5.0
===========

.. _Release Notes_Ignis_0.5.0_Prelude:

Prelude
-------

This release includes a new module for expectation value measurement error
mitigation, improved plotting functionality for quantum volume experiments,
several bug fixes, and drops support for Python 3.5.


.. _Release Notes_Ignis_0.5.0_New Features:

New Features
------------

- The :func:`qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
  function allows an optional input of gate objects as `interleaved_elem`.
  In addition, the CNOT-Dihedral class
  :class:`qiskit.ignis.verification.randomized_benchmarking.CNOTDihedral`
  has a new method `to_instruction`, and the existing `from_circuit` method has
  an optional input of an `Instruction` (in addition to `QuantumCircuit`).

- The :class:`qiskit.ignis.verification.randomized_benchmarking.CNOTDihedral`
  now contains the following new features.
  Initialization from various types of objects:
  `CNOTDihedral`, `ScalarOp`, `QuantumCircuit`, `Instruction` and `Pauli`.
  Converting to a matrix using `to_matrix` and to an operator using `to_operator`.
  Tensor product methods `tensor` and `expand`.
  Calculation of the adjoint, conjugate and transpose using `conjugate`, `adjoint`
  and `transpose` methods.
  Verify that an element is CNOTDihedral using `is_cnotdihedral` method.
  Decomposition method `to_circuit` of a CNOTDihedral element into a circuit
  was extended to allow any number of qubits, based on the function
  `decompose_cnotdihedral_general`.

- Adds expectation value measurement error mitigation to the mitigation module.
  This supports using *complete* N-qubit assignment matrix, single-qubit
  *tensored* assignment matrix, or *continuous time Markov process (CTMP)* [1]
  measurement error mitigation when computing expectation values of diagonal
  operators from counts dictionaries. Expectation values are computed using
  the using the :func:`qiskit.ignis.mitigation.expectation_value` function.

  Calibration circuits for calibrating a measurement error mitigator are
  generated using the :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits`
  function, and the result fitted using the
  :class:`qiskit.ignis.mitigation.ExpvalMeasMitigatorFitter` class. The
  fitter returns a mitigator object can the be supplied as an argument to the
  :func:`~qiskit.ignis.mitigation.expectation_value` function to apply mitigation.

  [1] S Bravyi, S Sheldon, A Kandala, DC Mckay, JM Gambetta,
      *Mitigating measurement errors in multi-qubit experiments*,
      arXiv:2006.14044 [quant-ph].

  Example:

      The following example shows calibrating a 5-qubit expectation value
      measurement error mitigator using the ``'tensored'`` method.

      .. jupyter-execute::

          from qiskit import execute
          from qiskit.test.mock import FakeVigo
          import qiskit.ignis.mitigation as mit

          backend = FakeVigo()
          num_qubits = backend.configuration().num_qubits

          # Generate calibration circuits
          circuits, metadata = mit.expval_meas_mitigator_circuits(
              num_qubits, method='tensored')
          result = execute(circuits, backend, shots=8192).result()

          # Fit mitigator
          mitigator = mit.ExpvalMeasMitigatorFitter(result, metadata).fit()

          # Plot fitted N-qubit assignment matrix
          mitigator.plot_assignment_matrix()

      The following shows how to use the above mitigator to apply measurement
      error mitigation to expectation value computations

      .. jupyter-execute::

          from qiskit import QuantumCircuit

          # Test Circuit with expectation value -1.
          qc = QuantumCircuit(num_qubits)
          qc.x(range(num_qubits))
          qc.measure_all()

          # Execute
          shots = 8192
          seed_simulator = 1999
          result = execute(qc, backend, shots=8192, seed_simulator=1999).result()
          counts = result.get_counts(0)

          # Expectation value of Z^N without mitigation
          expval_nomit, error_nomit = mit.expectation_value(counts)
          print('Expval (no mitigation): {:.2f} \u00B1 {:.2f}'.format(
              expval_nomit, error_nomit))

          # Expectation value of Z^N with mitigation
          expval_mit, error_mit = mit.expectation_value(counts,
              meas_mitigator=mitigator)
          print('Expval (with mitigation): {:.2f} \u00B1 {:.2f}'.format(
              expval_mit, error_mit))


- Adds Numba as an optional dependency. Numba is used to significantly increase
  the performance of the :class:`qiskit.ignis.mitigation.CTMPExpvalMeasMitigator`
  class used for expectation value measurement error mitigation with the CTMP
  method.


- Add two methods to :class:`qiskit.ignis.verification.quantum_volume.QVFitter`.

  * :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.calc_z_value` to
    calculate z value in standard normal distribution using mean and standard
    deviation sigma. If sigma = 0, it raises a warning and assigns a small
    value (1e-10) for sigma so that the code still runs.
  * :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.calc_confidence_level`
    to calculate confidence level using z value.


- Store confidence level even when hmean < 2/3 in
  :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.qv_success`.

- Add explanations for how to calculate statistics based on binomial
  distribution in
  :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.calc_statistics`.

- The :class:`qiskit.ignis.verification.QVFitter` method
  :meth:`~qiskit.ignis.verification.QVFitter.plot_qv_data` has been updated to return a
  ``matplotlib.Figure`` object. Previously, it would not return anything. By returning a figure
  this makes it easier to integrate the visualizations into a larger ``matplotlib`` workflow.

- The error bars in the figure produced by the
  :class:`qiskit.ignis.verification.QVFitter` method
  :meth:`qiskit.ignis.verification.QVFitter.plot_qv_data` has been updated to represent
  two-sigma confidence intervals. Previously, the error bars represent one-sigma confidence
  intervals. The success criteria of Quantum Volume benchmarking requires heavy output
  probability > 2/3 with one-sided two-sigma confidence (~97.7%). Changing error bars to
  represent two-sigma confidence intervals allows easily identification of success in the
  figure.

- A new kwarg, ``figsize`` has been added to the
  :class:`qiskit.ignis.verification.QVFitter` method
  :meth:`qiskit.ignis.verification.QVFitter.plot_qv_data`. This kwarg takes in a tuple of the
  form ``(x, y)`` where ``x`` and ``y`` are the dimension in inches to make the generated
  plot.

- The :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.plot_hop_accumulative` method
  has been added to plot heavy output probability (HOP) vs number of trials similar to
  Figure 2a of Quantum Volume 64 paper (`arXiv:2008.08571 <https://arxiv.org/abs/2008.08571>`_).
  HOP of individual trials are plotted as scatters and cummulative HOP are plotted in red line.
  Two-sigma confidence intervals are plotted as shaded area and 2/3 success threshold is plotted
  as dashed line.

- The :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.plot_qv_trial` method
  has been added to plot individual trials, leveraging on the
  :meth:`qiskit.visualization.plot_histogram` method from Qiskit Terra.
  Bitstring counts are plotted as overlapping histograms for ideal (hollow) and experimental
  (filled) values.
  Experimental heavy output probability are shown on the legend.
  Median probability is plotted as red dashed line.


.. _Release Notes_Ignis_0.5.0_Upgrade Notes:

Upgrade Notes
-------------

- The deprecated support for running qiskit-ignis with Python 3.5 has
  been removed. To use qiskit-ignis >=0.5.0 you will now need at
  least Python 3.6. If you are using Python 3.5 the last version which will
  work is qiskit-ignis 0.4.x.


.. _Release Notes_Ignis_0.5.0_Bug Fixes:

Bug Fixes
---------


- Fixing a bug in the class
  :class:`qiskit.ignis.verification.randomized_benchmarking.CNOTDihedral`
  for elements with more than 5 quits.

- Fix the confidence level threshold for
  :meth:`qiskit.ignis.verification.quantum_volume.QVFitter.qv_success` to 0.977
  corresponding to z = 2 as defined by the QV paper Algorithm 1.

- Fix a bug at
  :func:`qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
  which caused all the subsystems with the same size in the given rb_pattern to
  have the same gates when a 'rand_seed' parameter was given to the function.

Aqua 0.8.0
==========

.. _Release Notes_Aqua_0.8.0_Prelude:

Prelude
-------

This release introduces an interface for running the available methods for
Bosonic problems. In particular we introduced a full interface for running
vibronic structure calculations.

This release introduces an interface for excited states calculations. It is
now easier for the user to create a general excited states calculation.
This calculation is based on a Driver which provides the relevant information
about the molecule, a Transformation which provides the information about the
mapping of the problem into a qubit Hamiltonian, and finally a Solver.
The Solver is the specific way which the excited states calculation is done
(the algorithm). This structure follows the one of the ground state
calculations. The results are modified to take lists of expectation values
instead of a single one. The QEOM and NumpyEigensolver are adapted to the new
structure. A factory is introduced to run a numpy eigensolver with a specific
filter  (to target states of specific symmetries).

VQE expectation computation with Aer qasm_simulator now defaults to a
computation that has the expected shot noise behavior.


.. _Release Notes_Aqua_0.8.0_New Features:

New Features
------------

- Introduced an option `warm_start` that should be used when tuning other options does not help.
  When this option is enabled, a relaxed problem (all variables are continuous) is solved first
  and the solution is used to initialize the state of the optimizer before it starts the
  iterative process in the `solve` method.

- The amplitude estimation algorithms now use ``QuantumCircuit`` objects as
  inputs to specify the A- and Q operators. This change goes along with the
  introduction of the ``GroverOperator`` in the circuit library, which allows
  an intuitive and fast construction of different Q operators.
  For example, a Bernoulli-experiment can now be constructed as

  .. code-block:: python

    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.aqua.algorithms import AmplitudeEstimation

    probability = 0.5
    angle = 2 * np.sqrt(np.arcsin(probability))
    a_operator = QuantumCircuit(1)
    a_operator.ry(angle, 0)

    # construct directly
    q_operator = QuantumCircuit(1)
    q_operator.ry(2 * angle, 0)

    # construct via Grover operator
    from qiskit.circuit.library import GroverOperator
    oracle = QuantumCircuit(1)
    oracle.z(0)  # good state = the qubit is in state |1>
    q_operator = GroverOperator(oracle, state_preparation=a_operator)

    # use default construction in QAE
    q_operator = None

    ae = AmplitudeEstimation(a_operator, q_operator)

- Add the possibility to compute Conditional Value at Risk (CVaR) expectation
  values.

  Given a diagonal observable H, often corresponding to the objective function
  of an optimization problem, we are often not as interested in minimizing the
  average energy of our observed measurements. In this context, we are
  satisfied if at least some of our measurements achieve low energy. (Note that
  this is emphatically not the case for chemistry problems).

  To this end, one might consider using the best observed sample as a cost
  function during variational optimization. The issue here, is that this can
  result in a non-smooth optimization surface. To resolve this issue, we can
  smooth the optimization surface by using not just the best observed sample,
  but instead average over some fraction of best observed samples. This is
  exactly what the CVaR estimator accomplishes [1].

  Let :math:`\alpha` be a real number in :math:`[0,1]` which specifies the
  fraction of best observed samples which are used to compute the objective
  function. Observe that if :math:`\alpha = 1`, CVaR is equivalent to a
  standard expectation value. Similarly, if :math:`\alpha = 0`, then CVaR
  corresponds to using the best observed sample. Intermediate values of
  :math:`\alpha` interpolate between these two objective functions.

  The functionality to use CVaR is included into the operator flow through a
  new subclass of OperatorStateFn called CVaRMeasurement. This new StateFn
  object is instantied in the same way as an OperatorMeasurement with the
  exception that it also accepts an `alpha` parameter and that it automatically
  enforces the  `is_measurement` attribute to be True. Observe that it is
  unclear what a CVaRStateFn would represent were it not a measurement.

  Examples::

          qc = QuantumCircuit(1)
          qc.h(0)
          op = CVaRMeasurement(Z, alpha=0.5) @ CircuitStateFn(primitive=qc, coeff=1.0)
          result = op.eval()


  Similarly, an operator corresponding to a standard expectation value can be
  converted into a CVaR expectation using the CVaRExpectation converter.

  Examples::

          qc = QuantumCircuit(1)
          qc.h(0)
          op = ~StateFn(Z) @ CircuitStateFn(primitive=qc, coeff=1.0)
          cvar_expecation = CVaRExpectation(alpha=0.1).convert(op)
          result = cvar_expecation.eval()

  See [1] for additional details regarding this technique and it's empircal
  performance.

  References:

      [1]: Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., and Woerner, S.,
           "Improving Variational Quantum Optimization using CVaR"
           `arXiv:1907.04769 <https://arxiv.org/abs/1907.04769>`_

- New  interface ``Eigensolver`` for Eigensolver algorithms.

- An interface for excited states calculation has been added to the chemistry module.
  It is now easier for the user to create a general excited states calculation.
  This calculation is based on a ``Driver`` which provides the relevant information
  about the molecule, a ``Transformation`` which provides the information about the
  mapping of the problem into a qubit Hamiltonian, and finally a Solver.
  The Solver is the specific way which the excited states calculation is done
  (the algorithm). This structure follows the one of the ground state calculations.
  The results are modified to take lists of expectation values instead of a single one.
  The ``QEOM`` and ``NumpyEigensolver`` are adapted to the new structure.
  A factory is introduced to run a numpy eigensolver with a specific filter
  (to target states of specific symmetries).

- In addition to the workflows for solving Fermionic problems, interfaces for calculating
  Bosonic ground and excited states have been added. In particular we introduced a full
  interface for running vibronic structure calculations.

- The ``OrbitalOptimizationVQE`` has been added as new ground state solver in the chemistry
  module. This solver allows for the simulatneous optimization of the variational parameters
  and the orbitals of the molecule. The algorithm is introduced in Sokolov et al.,
  The Journal of Chemical Physics 152 (12).

- A new algorithm has been added: the Born Openheimer Potential Energy surface for the calculation
  of potential energy surface along different degrees of freedom of the molecule. The algorithm
  is called ``BOPESSampler``. It further provides functionalities of fitting the potential energy
  surface to an analytic function of predefined potentials.

- A feasibility check of the obtained solution has been added to all optimizers in the
  optimization stack. This has been implemented by adding two new methods to ``QuadraticProgram``:
  * ``get_feasibility_info(self, x: Union[List[float], np.ndarray])`` accepts an array and returns
  whether this solution is feasible and a list of violated variables(violated bounds) and
  a list of violated constraints.
  * ``is_feasible(self, x: Union[List[float], np.ndarray])`` accepts an array and returns whether
  this solution is feasible or not.

- Add circuit-based versions of ``FixedIncomeExpectedValue``, ``EuropeanCallDelta``,
  ``GaussianConditionalIndependenceModel`` and ``EuropeanCallExpectedValue`` to
  ``qiskit.finance.applications``.

- Gradient Framework.
  :class:`qiskit.operators.gradients`
  Given an operator that represents either a quantum state resp. an expectation
  value, the gradient framework enables the evaluation of gradients, natural
  gradients, Hessians, as well as the Quantum Fisher Information.

  Suppose a parameterized quantum state `|ψ(θ)〉 = V(θ)|ψ〉` with input state
  `|ψ〉` and parametrized Ansatz `V(θ)`, and an Operator `O(ω)`.

  Gradients: We want to compute :math:`d⟨ψ(θ)|O(ω)|ψ(θ)〉/ dω`
  resp. :math:`d⟨ψ(θ)|O(ω)|ψ(θ)〉/ dθ`
  resp. :math:`d⟨ψ(θ)|i〉⟨i|ψ(θ)〉/ dθ`.

  The last case corresponds to the gradient w.r.t. the sampling probabilities
  of `|ψ(θ)`. These gradients can be computed with different methods, i.e. a
  parameter shift, a linear combination of unitaries and a finite difference
  method.

  Examples::

    x = Parameter('x')
    ham = x * X
    a = Parameter('a')

    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.h(q)
    qc.p(params[0], q[0])
    op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.)

    value_dict = {x: 0.1, a: np.pi / 4}

    ham_grad = Gradient(grad_method='param_shift').convert(operator=op, params=[x])
    ham_grad.assign_parameters(value_dict).eval()

    state_grad = Gradient(grad_method='lin_comb').convert(operator=op, params=[a])
    state_grad.assign_parameters(value_dict).eval()

    prob_grad = Gradient(grad_method='fin_diff').convert(operator=CircuitStateFn(primitive=qc, coeff=1.),
                                                         params=[a])
    prob_grad.assign_parameters(value_dict).eval()

  Hessians: We want to compute :math:`d^2⟨ψ(θ)|O(ω)|ψ(θ)〉/ dω^2`
  resp. :math:`d^2⟨ψ(θ)|O(ω)|ψ(θ)〉/ dθ^2`
  resp. :math:`d^2⟨ψ(θ)|O(ω)|ψ(θ)〉/ dθdω`
  resp. :math:`d^2⟨ψ(θ)|i〉⟨i|ψ(θ)〉/ dθ^2`.

  The last case corresponds to the Hessian w.r.t. the sampling probabilities of `|ψ(θ)`.
  Just as the first order gradients, the Hessians can be evaluated with
  different methods, i.e. a parameter shift, a linear combination of unitaries
  and a finite difference method. Given a tuple of parameters
  ``Hessian().convert(op, param_tuple)`` returns the value for the second order
  derivative. If a list of parameters is given ``Hessian().convert(op, param_list)``
  returns the full Hessian for all the given parameters according to the given
  parameter order.

  QFI: The Quantum Fisher Information `QFI` is a metric tensor which is
  representative for the representation capacity of a parameterized quantum
  state `|ψ(θ)〉 = V(θ)|ψ〉` generated by an input state `|ψ〉` and a
  parametrized Ansatz `V(θ)`. The entries of the `QFI` for a pure state read
  :math:`[QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4`.

  Just as for the previous derivative types, the QFI can be computed using
  different methods: a full representation based on a linear combination of
  unitaries implementation, a block-diagonal and a diagonal representation
  based on an overlap method.

  Examples::

    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.h(q)
    qc.p(params[0], q[0])
    op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.)

    value_dict = {x: 0.1, a: np.pi / 4}
    qfi = QFI('lin_comb_full').convert(operator=CircuitStateFn(primitive=qc, coeff=1.), params=[a])
    qfi.assign_parameters(value_dict).eval()


  The combination of the QFI and the gradient lead to a special form of a
  gradient, namely

  NaturalGradients: The natural gradient is a special gradient method which
  rescales a gradient w.r.t. a state parameter with the inverse of the
  corresponding Quantum Fisher Information (QFI)
  :math:`QFI^-1 d⟨ψ(θ)|O(ω)|ψ(θ)〉/ dθ`.
  Hereby, we can choose a gradient as well as a QFI method and a
  regularization method which is used together with a least square solver
  instead of exact invertion of the QFI:

  Examples::

    op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.)
    nat_grad = NaturalGradient(grad_method='lin_comb, qfi_method='lin_comb_full', \
                               regularization='ridge').convert(operator=op, params=params)

  The gradient framework is also compatible with the optimizers from
  `qiskit.aqua.components.optimizers`. The derivative classes come with a
  `gradient_wrapper()` function which returns the corresponding callable.

- Introduces ``transformations`` for the fermionic and bosonic transformation of a problem
  instance. Transforms the fermionic operator to qubit operator. Respective class for the
  transformation is ``fermionic_transformation``
  Introduces in algorithms ``ground_state_solvers`` for the calculation of ground state
  properties. The calculation can be done either using an ``MinimumEigensolver`` or using
  ``AdaptVQE``
  Introduces ``chemistry/results`` where the eigenstate_result and the
  electronic_structure_result are also used for the algorithms.
  Introduces Minimum Eigensolver factories ``minimum_eigensolver_factories`` where chemistry
  specific minimum eigensolvers can be initialized Introduces orbital optimization vqe
  ``oovqe`` as a ground state solver for chemistry applications

- New Algorithm result classes:

  :class:`~qiskit.aqua.algorithms.Grover` method
  :meth:`~qiskit.aqua.algorithms.Grover._run`
  returns class :class:`~qiskit.aqua.algorithms.GroverResult`.
  :class:`~qiskit.aqua.algorithms.AmplitudeEstimation` method
  :meth:`~qiskit.aqua.algorithms.AmplitudeEstimation._run`
  returns class :class:`~qiskit.aqua.algorithms.AmplitudeEstimationResult`.
  :class:`~qiskit.aqua.algorithms.IterativeAmplitudeEstimation` method
  :meth:`~qiskit.aqua.algorithms.IterativeAmplitudeEstimation._run`
  returns class :class:`~qiskit.aqua.algorithms.IterativeAmplitudeEstimationResult`.
  :class:`~qiskit.aqua.algorithms.MaximumLikelihoodAmplitudeEstimation` method
  :meth:`~qiskit.aqua.algorithms.MaximumLikelihoodAmplitudeEstimation._run`
  returns class :class:`~qiskit.aqua.algorithms.MaximumLikelihoodAmplitudeEstimationResult`.

  All new result classes are backwards compatible with previous result dictionary.

- New Linear Solver result classes:

  :class:`~qiskit.aqua.algorithms.HHL` method
  :meth:`~qiskit.aqua.algorithms.HHL._run`
  returns class :class:`~qiskit.aqua.algorithms.HHLResult`.
  :class:`~qiskit.aqua.algorithms.NumPyLSsolver` method
  :meth:`~qiskit.aqua.algorithms.NumPyLSsolver._run`
  returns class :class:`~qiskit.aqua.algorithms.NumPyLSsolverResult`.

  All new result classes are backwards compatible with previous result dictionary.

- ``MinimumEigenOptimizationResult`` now exposes properties: ``samples`` and
  ``eigensolver_result``. The latter is obtained from the underlying algorithm used by the
  optimizer and specific to the algorithm.
  ``RecursiveMinimumEigenOptimizer`` now returns an instance of the result class
  ``RecursiveMinimumEigenOptimizationResult`` which in turn may contains intermediate results
  obtained from the underlying algorithms. The dedicated result class exposes properties
  ``replacements`` and ``history`` that are specific to this optimizer. The depth of the history
  is managed by the ``history`` parameter of the optimizer.

- ``GroverOptimizer`` now returns an instance of ``GroverOptimizationResult`` and this result
  class exposes properties ``operation_counts``, ``n_input_qubits``, and ``n_output_qubits``
  directly. These properties are not available in the ``raw_results`` dictionary anymore.

- ``SlsqpOptimizer`` now returns an instance of ``SlsqpOptimizationResult`` and this result class
  exposes additional properties specific to the SLSQP implementation.

- Support passing ``QuantumCircuit`` objects as generator circuits into
  the ``QuantumGenerator``.

- Removes the restriction to real input vectors in CircuitStateFn.from_vector.
  The method calls extensions.Initialize. The latter explicitly supports (in API
  and documentation) complex input vectors. So this restriction seems unnecessary.

- Simplified `AbelianGrouper` using a graph coloring algorithm of retworkx.
  It is faster than the numpy-based coloring algorithm.

- Allow calling ``eval`` on state function objects with no argument, which returns the
  ``VectorStateFn`` representation of the state function.
  This is consistent behavior with ``OperatorBase.eval``, which returns the
  ``MatrixOp`` representation, if no argument is passed.

- Adds ``max_iterations`` to the ``VQEAdapt`` class in order to allow
  limiting the maximum number of iterations performed by the algorithm.

- VQE expectation computation with Aer qasm_simulator now defaults to a
  computation that has the expected shot noise behavior. The special Aer
  snapshot based computation, that is much faster, with the ideal output
  similar to state vector simulator, may still be chosen but like before
  Aqua 0.7 it now no longer defaults to this but can be chosen.


.. _Release Notes_Aqua_0.8.0_Upgrade Notes:

Upgrade Notes
-------------

- Extension of the previous Analytic Quantum Gradient Descent (AQGD) classical
  optimizer with the AQGD with Epochs. Now AQGD performs the gradient descent
  optimization with a momentum term, analytic gradients, and an added customized
  step length schedule for parametrized quantum gates. Gradients are computed
  "analytically" using the quantum circuit when evaluating the objective function.


- The deprecated support for running qiskit-aqua with Python 3.5 has
  been removed. To use qiskit-aqua >=0.8.0 you will now need at
  least Python 3.6. If you are using Python 3.5 the last version which will
  work is qiskit-aqua 0.7.x.

- Added retworkx as a new dependency.


.. _Release Notes_Aqua_0.8.0_Deprecation Notes:

Deprecation Notes
-----------------

- The ``i_objective`` argument of the amplitude estimation algorithms has been
  renamed to ``objective_qubits``.

- TransformationType

- QubitMappingType

- Deprecate the ``CircuitFactory`` and derived types. The ``CircuitFactory`` has
  been introduced as temporary class when the ``QuantumCircuit`` missed some
  features necessary for applications in Aqua. Now that the circuit has all required
  functionality, the circuit factory can be removed.
  The replacements are shown in the following table.

  .. code-block::

      Circuit factory class               | Replacement
      ------------------------------------+-----------------------------------------------
      CircuitFactory                      | use QuantumCircuit
                                          |
      UncertaintyModel                    | -
      UnivariateDistribution              | -
      MultivariateDistribution            | -
      NormalDistribution                  | qiskit.circuit.library.NormalDistribution
      MultivariateNormalDistribution      | qiskit.circuit.library.NormalDistribution
      LogNormalDistribution               | qiskit.circuit.library.LogNormalDistribution
      MultivariateLogNormalDistribution   | qiskit.circuit.library.LogNormalDistribution
      UniformDistribution                 | qiskit.circuit.library.UniformDistribution
      MultivariateUniformDistribution     | qiskit.circuit.library.UniformDistribution
      UnivariateVariationalDistribution   | use parameterized QuantumCircuit
      MultivariateVariationalDistribution | use parameterized QuantumCircuit
                                          |
      UncertaintyProblem                  | -
      UnivariateProblem                   | -
      MultivariateProblem                 | -
      UnivariatePiecewiseLinearObjective  | qiskit.circuit.library.LinearAmplitudeFunction

- The ising convert classes
  :class:`qiskit.optimization.converters.QuadraticProgramToIsing` and
  :class:`qiskit.optimization.converters.IsingToQuadraticProgram` have
  been deprecated and will be removed in a future release. Instead the
  :class:`qiskit.optimization.QuadraticProgram` methods
  :meth:`~qiskit.optimization.QuadraticProgram.to_ising` and
  :meth:`~qiskit.optimization.QuadraticPrgraom.from_ising` should be used
  instead.

- Deprecate the ``WeightedSumOperator`` which has been ported to the circuit library as
  ``WeightedAdder`` in ``qiskit.circuit.library``.

- ``Core Hamiltonian`` class is deprecated in favor of the ``FermionicTransformation``
  ``Chemistry Operator`` class is deprecated in favor of the ``tranformations``
  ``minimum_eigen_solvers/vqe_adapt`` is also deprecated and moved as an implementation
  of the ground_state_solver interface
  ``applications/molecular_ground_state_energy`` is deprecated in favor of ``ground_state_solver``

- ``Optimizer.SupportLevel`` nested enum is replaced by ``OptimizerSupportLevel``
  and ``Optimizer.SupportLevel`` was removed. Use, for example,
  ``OptimizerSupportLevel.required`` instead of ``Optimizer.SupportLevel.required``.

- Deprecate the ``UnivariateVariationalDistribution`` and
  ``MultivariateVariationalDistribution`` as input
  to the ``QuantumGenerator``. Instead, plain ``QuantumCircuit`` objects can
  be used.

- Ignored `fast` and `use_nx` options of `AbelianGrouper.group_subops` to be removed in the
  future release.

- GSLS optimizer class deprecated ``__init__`` parameter ``max_iter`` in favor of ``maxiter``.
  SPSA optimizer class deprecated ``__init__`` parameter ``max_trials`` in favor of ``maxiter``.
  optimize_svm function deprecated ``max_iters`` parameter in favor of ``maxiter``.
  ADMMParameters class deprecated ``__init__`` parameter ``max_iter`` in favor of ``maxiter``.


.. _Release Notes_Aqua_0.8.0_Bug Fixes:

Bug Fixes
---------


- The UCCSD excitation list, comprising single and double excitations, was not being
  generated correctly when an active space was explicitly provided to UCSSD via the
  active_(un)occupied parameters.

- For the amplitude estimation algorithms, we define the number of oracle queries
  as number of times the Q operator/Grover operator is applied. This includes
  the number of shots. That factor has been included in MLAE and IQAE but
  was missing in the 'standard' QAE.

- Fix CircuitSampler.convert, so that the ``is_measurement`` property is
  propagated to converted StateFns.

- Fix double calculation of coefficients in
  :meth`~qiskit.aqua.operators.VectorStateFn.to_circuit_op`.

- Calling PauliTrotterEvolution.convert on an operator including a term that
  is a scalar multiple of the identity gave an incorrect circuit, one that
  ignored the scalar coefficient. This fix includes the effect of the
  coefficient in the global_phase property of the circuit.

- Make ListOp.num_qubits check that all ops in list have the same num_qubits
  Previously, the number of qubits in the first operator in the ListOp
  was returned. With this change, an additional check is made that all
  other operators also have the same number of qubits.

- Make PauliOp.exp_i() generate the correct matrix with the following changes.
  1) There was previously an error in the phase of a factor of 2.
  2) The global phase was ignored when converting the circuit
  to a matrix. We now use qiskit.quantum_info.Operator, which is
  generally useful for converting a circuit to a unitary matrix,
  when possible.

- Fixes the cyclicity detection as reported buggy in
  https://github.com/Qiskit/qiskit-aqua/issues/1184.


IBM Q Provider 0.11.0
=====================

.. _Release Notes_0.11.0_IBMQ_Upgrade Notes:

Upgrade Notes
-------------

- The deprecated support for running qiskit-ibmq-provider with Python 3.5 has
  been removed. To use qiskit-ibmq-provider >=0.11.0 you will now need at
  least Python 3.6. If you are using Python 3.5 the last version which will
  work is qiskit-ibmq-provider 0.10.x.

- Prior to this release, ``websockets`` 7.0 was used for Python 3.6.
  With this release, ``websockets`` 8.0 or above is required for all Python versions.
  The package requirements have been updated to reflect this.


*************
Qiskit 0.22.0
*************

Terra 0.15.2
============

No change

Aer 0.6.1
=========

No change

Ignis 0.4.0
===========

No change

Aqua 0.7.5
==========

No change

IBM Q Provider 0.10.0
=====================

.. _Release Notes_IBMQ_provider_0.10.0_New Features:

New Features
------------

- CQC randomness extractors can now be invoked asynchronously, using methods
  :meth:`~qiskit.providers.ibmq.random.CQCExtractor.run_async_ext1` and
  :meth:`~qiskit.providers.ibmq.random.CQCExtractor.run_async_ext2`. Each of
  these methods returns a :class:`~qiskit.providers.ibmq.random.CQCExtractorJob`
  instance that allows you to check on the job status (using
  :meth:`~qiskit.providers.ibmq.random.CQCExtractorJob.status`) and wait for
  its result (using
  :meth:`~qiskit.providers.ibmq.random.CQCExtractorJob.block_until_ready`).
  The :meth:`qiskit.provider.ibmq.random.CQCExtractor.run` method remains
  synchronous.

- You can now use the new IBMQ experiment service to query, retrieve, and
  download experiment related data. Interface to this service is located
  in the new :mod:`qiskit.providers.ibmq.experiment` package.
  Note that this feature is still in
  beta, and not all accounts have access to it. It is also subject to heavy
  modification in both functionality and API without backward compatibility.

- Two Jupyter magic functions, the IQX dashboard and the backend widget, are
  updated to display backend reservations. If a backend has reservations
  scheduled in the next 24 hours, time to the next one and its duration
  are displayed (e.g. ``Reservation: in 6 hrs 30 min (60m)``). If there is
  a reservation and the backend is active, the backend status is displayed
  as ``active [R]``.


.. _Release Notes_IBMQ_provider_0.10.0_Upgrade Notes:

Upgrade Notes
-------------

- Starting from this release, the `basis_gates` returned by
  :meth:`qiskit.providers.ibmq.IBMQBackend.configuration` may differ for each backend.
  You should update your program if it relies on the basis gates being
  ``['id','u1','u2','u3','cx']``. We recommend always using the
  :meth:`~qiskit.providers.ibmq.IBMQBackend.configuration` method to find backend
  configuration values instead of hard coding them.

- ``qiskit-ibmq-provider`` release 0.10 requires ``qiskit-terra``
  release 0.15 or above. The package metadata has been updated to reflect
  the new dependency.

*************
Qiskit 0.21.0
*************

Terra 0.15.2
============

No change

Aer 0.6.1
=========

No change

Ignis 0.4.0
===========

No change

Aqua 0.7.5
==========

No change

IBM Q Provider 0.9.0
====================

.. _Release Notes_IBMQ_provider_0.9.0_New Features:

New Features
------------

- You can now access the IBMQ random number services, such as the CQC
  randomness extractor, using the new package
  :mod:`qiskit.providers.ibmq.random`. Note that this feature is still in
  beta, and not all accounts have access to it. It is also subject to heavy
  modification in both functionality and API without backward compatibility.


.. _Release Notes_IBMQ_provider_0.9.0_Bug Fixes:

Bug Fixes
---------

- Fixes an issue that may raise a ``ValueError`` if
  :meth:`~qiskit.providers.ibmq.IBMQBackend.retrieve_job` is used to retrieve
  a job submitted via the IBM Quantum Experience Composer.

- :class:`~qiskit.providers.ibmq.managed.IBMQJobManager` has been updated so
  that if a time out happens while waiting for an old job to finish, the
  time out error doesn't prevent a new job to be submitted. Fixes
  `#737 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/737>`_


*************
Qiskit 0.20.1
*************

Terra 0.15.2
============

.. _Release Notes_0.15.2_Bug Fixes:

Bug Fixes
---------

- When accessing the ``definition`` attribute of a parameterized ``Gate``
  instance, the generated ``QuantumCircuit`` had been generated with an invalid
  ``ParameterTable``, such that reading from ``QuantumCircuit.parameters`` or
  calling ``QuantumCircuit.bind_parameters`` would incorrectly report the
  unbound parameters. This has been resolved.

- ``SXGate().inverse()`` had previously returned an 'sx_dg' gate with a correct
  ``definition`` but incorrect ``to_matrix``. This has been updated such that
  ``SXGate().inverse()`` returns an ``SXdgGate()`` and vice versa.

- ``Instruction.inverse()``, when not overridden by a subclass, would in some
  cases return a ``Gate`` instance with an incorrect ``to_matrix`` method. The
  instances of incorrect ``to_matrix`` methods have been removed.

- For ``C3XGate`` with a non-zero ``angle``, inverting the gate via
  ``C3XGate.inverse()`` had previously generated an incorrect inverse gate.
  This has been corrected.

- The ``MCXGate`` modes have been updated to return a gate of the same mode
  when calling ``.inverse()``. This resolves an issue where in some cases,
  transpiling a circuit containing the inverse of an ``MCXVChain`` gate would
  raise an error.

- Previously, when creating a multiply controlled phase gate via
  ``PhaseGate.control``, an ``MCU1Gate`` gate had been returned. This has been
  had corrected so that an ``MCPhaseGate`` is returned.

- Previously, attempting to decompose a circuit containing an
  ``MCPhaseGate`` would raise an error due to an inconsistency in the
  definition of the ``MCPhaseGate``. This has been corrected.

- ``QuantumCircuit.compose`` and ``DAGCircuit.compose`` had, in some cases,
  incorrectly translated conditional gates if the input circuit contained
  more than one ``ClassicalRegister``. This has been resolved.

- Fixed an issue when creating a :class:`qiskit.result.Counts` object from an
  empty data dictionary. Now this will create an empty
  :class:`~qiskit.result.Counts` object. The
  :meth:`~qiskit.result.Counts.most_frequent` method is also updated to raise
  a more descriptive exception when the object is empty. Fixes
  `#5017 <https://github.com/Qiskit/qiskit-terra/issues/5017>`__

- Extending circuits with differing registers updated the ``qregs`` and
  ``cregs`` properties accordingly, but not the ``qubits`` and ``clbits``
  lists. As these are no longer generated from the registers but are cached
  lists, this lead to a discrepancy of registers and bits. This has been
  fixed and the ``extend`` method explicitly updates the cached bit lists.

- Fix bugs of the concrete implementations of
  meth:`~qiskit.circuit.ControlledGate.inverse` method which do not preserve
  the ``ctrl_state`` parameter.

- A bug was fixed that caused long pulse schedules to throw a recursion error.

Aer 0.6.1
=========

No change

Ignis 0.4.0
===========

No change

Aqua 0.7.5
==========

No change

IBM Q Provider 0.8.0
====================

No change


*************
Qiskit 0.20.0
*************

Terra 0.15.1
============

.. _Release Notes_0.15.0_Prelude:

Prelude
-------


The 0.15.0 release includes several new features and bug fixes. Some
highlights for this release are:

This release includes the introduction of arbitrary
basis translation to the transpiler. This includes support for directly
targeting a broader range of device basis sets, e.g. backends
implementing RZ, RY, RZ, CZ or iSwap gates.

The :class:`~qiskit.circuit.QuantumCircuit` class now tracks global
phase. This means controlling a circuit which has global phase now
correctly adds a relative phase, and gate matrix definitions are now
exact rather than equal up to a global phase.


.. _Release Notes_0.15.0_New Features:

New Features
------------


- A new DAG class :class:`qiskit.dagcircuit.DAGDependency` for representing
  the dependency form of circuit, In this DAG, the nodes are
  operations (gates, measure, barrier, etc...) and the edges corresponds to
  non-commutation between two operations.

- Four new functions are added to :mod:`qiskit.converters` for converting back and
  forth to :class:`~qiskit.dagcircuit.DAGDependency`. These functions are:

  * :func:`~qiskit.converters.circuit_to_dagdependency` to convert
    from a :class:`~qiskit.circuit.QuantumCircuit` object to a
    :class:`~qiskit.dagcircuit.DAGDependency` object.
  * :func:`~qiskit.converters.dagdependency_to_circuit` to convert from a
    :class:`~qiskit.dagcircuit.DAGDependency` object to a
    :class:`~qiskit.circuit.QuantumCircuit` object.
  * :func:`~qiskit.converters.dag_to_dagdependency` to convert from
    a :class:`~qiskit.dagcircuit.DAGCircuit` object to a
    :class:`~qiskit.dagcircuit.DAGDependency` object.
  * :func:`~qiskit.converters.dagdependency_to_dag` to convert from
    a :class:`~qiskit.dagcircuit.DAGDependency` object to a
    :class:`~qiskit.dagcircuit.DAGCircuit` object.

  For example::

    from qiskit.converters.dagdependency_to_circuit import dagdependency_to_circuit
    from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

    circuit_in = QuantumCircuit(2)
    circuit_in.h(qr[0])
    circuit_in.h(qr[1])

    dag_dependency = circuit_to_dagdependency(circuit_in)
    circuit_out = dagdepency_to_circuit(dag_dependency)

- Two new transpiler passes have been added to :mod:`qiskit.transpiler.passes`
  The first, :class:`~qiskit.transpiler.passes.UnrollCustomDefinitions`,
  unrolls all instructions in the
  circuit according to their :attr:`~qiskit.circuit.Instruction.definition`
  property, stopping when reaching either the specified ``basis_gates``
  or a set of gates in the provided
  :class:`~qiskit.circuit.EquivalenceLibrary`. The second,
  :class:`~qiskit.transpiler.passes.BasisTranslator`, uses the set of
  translations in the provided :class:`~qiskit.circuit.EquivalenceLibrary` to
  re-write circuit instructions in a specified basis.

- A new ``translation_method`` keyword argument has been added to
  :func:`~qiskit.compiler.transpile` to allow selection of the method to be
  used for translating circuits to the available device gates. For example,
  ``transpile(circ, backend, translation_method='translator')``. Valid
  choices are:

  * ``'unroller'``: to use the :class:`~qiskit.transpiler.passes.Unroller`
    pass
  * ``'translator'``: to use the
    :class:`~qiskit.transpiler.passes.BasisTranslator` pass.
  * ``'synthesis'``: to use the
    :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass.

  The  default value is ``'translator'``.

- A new class for handling counts result data, :class:`qiskit.result.Counts`,
  has been added. This class is a subclass of ``dict`` and can be interacted
  with like any other dictionary. But, it includes helper methods and
  attributes for dealing with counts results from experiments and also
  handles post processing and formatting of binary strings at object
  initialization. A :class:`~qiskit.result.Counts`  object can be created by
  passing a dictionary  of counts with the keys being either integers,
  hexadecimal strings of the form ``'0x4a'``,  binary strings of the form
  ``'0b1101'``, a bit string formatted across register and memory slots
  (ie ``'00 10'``), or a dit string. For example::

    from qiskit.result import Counts

    counts = Counts({"0x0': 1, '0x1', 3, '0x2': 1020})

- A new method for constructing :class:`qiskit.dagcircuit.DAGCircuit` objects
  has been added, :meth:`~qiskit.dagcircuit.DAGCircuit.from_networkx`. This
  method takes in a networkx ``MultiDiGraph`` object (in the format returned
  by :meth:`~qiskit.dagcircuit.DAGCircuit.to_networkx`) and will return a
  new :class:`~qiskit.dagcircuit.DAGCircuit` object. The intent behind this
  function is to enable transpiler pass authors to leverage networkx's
  `graph algorithm library
  <https://networkx.github.io/documentation/stable/reference/algorithms/index.html>`__
  if a function is missing from the
  `retworkx API <https://retworkx.readthedocs.io/en/latest/api.html>`_.
  Although, hopefully in such casses an issue will be opened with
  `retworkx issue tracker <https://github.com/Qiskit/retworkx/issues>`__ (or
  even better a pull request submitted).

- A new kwarg for ``init_qubits`` has been added to
  :func:`~qiskit.compiler.assemble` and :func:`~qiskit.execute.execute`.
  For backends that support this feature ``init_qubits`` can be used to
  control whether the backend executing the circuits inserts any
  initialization sequences at the start of each shot. By default this is set
  to ``True`` meaning that all qubits can assumed to be in the ground state
  at the start of each shot. However, when ``init_qubits`` is  set to
  ``False`` qubits will be uninitialized at the start of each
  experiment and between shots. Note, that the backend running the circuits
  has to support this feature for this flag to have any effect.

- A new kwarg ``rep_delay`` has been added to
  :func:`qiskit.compiler.assemble`, :func:`qiskit.execute.execute`, and the
  constructor for :class:`~qiskit.qobj.PulseQobjtConfig`.qiskit
  This new kwarg is used to denotes the time between program executions. It
  must be chosen from the list of valid values set as the
  ``rep_delays`` from a backend's
  :class:`~qiskit.providers.models.PulseBackendConfiguration` object which
  can be accessed as ``backend.configuration().rep_delays``).

  The ``rep_delay`` kwarg will only work on backends which allow for dynamic
  repetition time. This will also be indicated in the
  :class:`~qiskit.providers.models.PulseBackendConfiguration` object for a
  backend as the ``dynamic_reprate_enabled`` attribute. If
  ``dynamic_reprate_enabled`` is ``False`` then the ``rep_time`` value
  specified for :func:`qiskit.compiler.assemble`,
  :func:`qiskit.execute.execute`, or the constructor for
  :class:`~qiskit.qobj.PulseQobjtConfig` will be used rather than
  ``rep_delay``. ``rep_time`` only allows users to specify the duration of a
  program, rather than the delay between programs.

- The ``qobj_schema.json`` JSON Schema file in :mod:`qiskit.schemas` has
  been updated to include the ``rep_delay`` as an optional configuration
  property for pulse qobjs.

- The ``backend_configuration_schema.json`` JSON Schema file in
  mod:`qiskit.schemas` has been updated to include ``rep_delay_range`` and
  ``default_rep_delay`` as optional properties for a pulse backend
  configuration.

- A new attribute, :attr:`~qiskit.circuit.QuantumCircuit.global_phase`,
  which is is used for tracking the global phase has been added to the
  :class:`qiskit.circuit.QuantumCircuit` class. For example::

    import math

    from qiskit import QuantumCircuit

    circ = QuantumCircuit(1, global_phase=math.pi)
    circ.u1(0)

  The global phase may also be changed or queried with
  ``circ.global_phase`` in the above example. In either case the setting is
  in radians. If the circuit is converted to an instruction or gate the
  global phase is represented by two single qubit rotations on the first
  qubit.

  This allows for other methods and functions which consume a
  :class:`~qiskit.circuit.QuantumCircuit` object to take global phase into
  account. For example. with the
  :attr:`~qiskit.circuit.QuantumCircuit.global_phase`
  attribute the :meth:`~qiskit.circuit.Gate.to_matrix` method for a gate
  can now exactly correspond to its decompositions instead of
  just up to a global phase.

  The same attribute has also been added to the
  :class:`~qiskit.dagcircuit.DAGCircuit` class so that global phase
  can be tracked when converting between
  :class:`~qiskit.circuit.QuantumCircuit` and
  :class:`~qiskit.dagcircuit.DAGCircuit`.

- Two new classes, :class:`~qiskit.circuit.AncillaRegister` and
  :class:`~qiskit.circuit.AncillaQubit` have been added to the
  :mod:`qiskit.circuit` module. These are subclasses of
  :class:`~qiskit.circuit.QuantumRegister` and :class:`~qiskit.circuit.Qubit`
  respectively and enable marking qubits being ancillas. This will allow
  these qubits to be re-used in larger circuits and algorithms.

- A new method, :meth:`~qiskit.circuit.QuantumCircuit.control`, has been
  added to the :class:`~qiskit.circuit.QuantumCircuit`. This method will
  return a controlled version of the :class:`~qiskit.circuit.QuantumCircuit`
  object, with both open and closed controls. This functionality had
  previously only been accessible via the :class:`~qiskit.circuit.Gate`
  class.

- A new method :meth:`~qiskit.circuit.QuantumCircuit.repeat` has been added
  to the :class:`~qiskit.circuit.QuantumCircuit` class. It returns a new
  circuit object containing a specified number of repetitions of the original
  circuit. For example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    repeated_qc = qc.repeat(3)
    repeated_qc.decompose().draw(output='mpl')

  The parameters are copied by reference, meaning that if you update
  the parameters in one instance of the circuit all repetitions will be
  updated.

- A new method :meth:`~qiskit.circuit.QuantumCircuit.reverse_bits` has been
  added to the :class:`~qiskit.circuit.QuantumCircuit` class. This method
  will reverse the order of bits in a circuit (both quantum and classical
  bits). This can be used to switch a circuit from little-endian to big-endian
  and vice-versa.

- A new method, :meth:`~qiskit.transpiler.Layout.combine_into_edge_map()`,
  was added to the :class:`qiskit.transpiler.Layout` class. This method
  enables converting converting two :class:`~qiskit.transpiler.Layout` objects
  into a qubit map for composing two circuits.

- A new class, :class:`~qiskit.test.mock.utils.ConfigurableFakeBackend`, has
  been added to the :mod:`qiskit.test.mock.utils` module. This new class
  enables the creation of configurable mock backends for use in testing.
  For example::

      from qiskit.test.mock.utils import ConfigurableFakeBackend

      backend = ConfigurableFakeBackend("Tashkent",
                                        n_qubits=100,
                                        version="0.0.1",
                                        basis_gates=['u1'],
                                        qubit_t1=99.,
                                        qubit_t2=146.,
                                        qubit_frequency=5.,
                                        qubit_readout_error=0.01,
                                        single_qubit_gates=['u1'])

  will create a backend object with 100 qubits and all the other parameters
  specified in the constructor.

- A new method :meth:`~qiskit.circuit.EquivalenceLibrary.draw` has been
  added to the :class:`qiskit.circuit.EquivalenceLibrary` class. This
  method can be used for drawing the contents of an equivalence library,
  which can be useful for debugging. For example:

  .. jupyter-execute::

    from numpy import pi

    from qiskit.circuit import EquivalenceLibrary
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit import QuantumRegister
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import HGate
    from qiskit.circuit.library import U2Gate
    from qiskit.circuit.library import U3Gate

    my_equiv_library = EquivalenceLibrary()

    q = QuantumRegister(1, 'q')
    def_h = QuantumCircuit(q)
    def_h.append(U2Gate(0, pi), [q[0]], [])
    my_equiv_library.add_equivalence(HGate(), def_h)

    theta = Parameter('theta')
    phi = Parameter('phi')
    lam = Parameter('lam')
    def_u2 = QuantumCircuit(q)
    def_u2.append(U3Gate(pi / 2, phi, lam), [q[0]], [])
    my_equiv_library.add_equivalence(U2Gate(phi, lam), def_u2)

    my_equiv_library.draw()

- A new Phase instruction, :class:`~qiskit.pulse.SetPhase`, has been added
  to :mod:`qiskit.pulse`. This instruction sets the phase of the
  subsequent pulses to the specified phase (in radians. For example::

    import numpy as np

    from qiskit.pulse import DriveChannel
    from qiskit.pulse import Schedule
    from qiskit.pulse import SetPhase

    sched = Schedule()
    sched += SetPhase(np.pi, DriveChannel(0))

  In this example, the phase of the pulses applied to ``DriveChannel(0)``
  after the :class:`~qiskit.pulse.SetPhase` instruction will be set to
  :math:`\pi` radians.

- A new pulse instruction :class:`~qiskit.pulse.ShiftFrequency` has been
  added to :mod:`qiskit.pulse.instructions`. This instruction enables
  shifting the frequency of a channel from its set frequency. For example::

    from qiskit.pulse import DriveChannel
    from qiskit.pulse import Schedule
    from qiskit.pulse import ShiftFrequency

    sched = Schedule()
    sched += ShiftFrequency(-340e6, DriveChannel(0))

  In this example all the pulses applied to ``DriveChannel(0)`` after the
  :class:`~qiskit.pulse.ShiftFrequency` command will have the envelope a
  frequency decremented by 340MHz.

- A new method :meth:`~qiskit.circuit.ParameterExpression.conjugate` has
  been added to the :class:`~qiskit.circuit.ParameterExpression` class.
  This enables calling ``numpy.conj()`` without raising an error. Since a
  :class:`~qiskit.circuit.ParameterExpression` object is real, it will
  return itself. This behaviour is analogous to Python floats/ints.

- A new class :class:`~qiskit.circuit.library.PhaseEstimation` has been
  added to :mod:`qiskit.circuit.library`. This circuit library class is
  the circuit used in the original formulation of the phase estimation
  algorithm in
  `arXiv:quant-ph/9511026 <https://arxiv.org/pdf/quant-ph/9511026.pdf>`__.
  Phase estimation is the task to to estimate the phase :math:`\phi` of an
  eigenvalue :math:`e^{2\pi i\phi}` of a unitary operator :math:`U`, provided
  with the corresponding eigenstate :math:`|psi\rangle`. That is

  .. math::

      U|\psi\rangle = e^{2\pi i\phi} |\psi\rangle

  This estimation (and thereby this circuit) is a central routine to several
  well-known algorithms, such as Shor's algorithm or Quantum Amplitude
  Estimation.

- The :mod:`qiskit.visualization` function
  :func:`~qiskit.visualization.plot_state_qsphere` has a new kwarg
  ``show_state_labels`` which is used to control whether each blob in the
  qsphere visualization is labeled. By default this kwarg is set to ``True``
  and shows the basis states next to each blob by default. This feature can be
  disabled, reverting to the previous behavior, by setting the
  ``show_state_labels`` kwarg to ``False``.

- The :mod:`qiskit.visualization` function
  :func:`~qiskit.visualization.plot_state_qsphere` has a new kwarg
  ``show_state_phases`` which is set to ``False`` by default. When set to
  ``True`` it displays the phase of each basis state.

- The :mod:`qiskit.visualization` function
  :func:`~qiskit.visualization.plot_state_qsphere` has a new kwarg
  ``use_degrees`` which is set to ``False`` by default. When set to ``True``
  it displays the phase of each basis state in degrees, along with the phase
  circle at the bottom right.

- A new class, :class:`~qiskit.circuit.library.QuadraticForm` to the
  :mod:`qiskit.circuit.library` module for implementing a a quadratic form on
  binary variables. The circuit library element implements the operation

  .. math::

    |x\rangle |0\rangle \mapsto |x\rangle |Q(x) \mod 2^m\rangle

  for the quadratic form :math:`Q` and :math:`m` output qubits.
  The result is in the :math:`m` output qubits is encoded in two's
  complement. If :math:`m` is not specified, the circuit will choose
  the minimal number of qubits required to represent the result
  without applying a modulo operation.
  The quadratic form is specified using a matrix for the quadratic
  terms, a vector for the linear terms and a constant offset.
  If all terms are integers, the circuit implements the quadratic form
  exactly, otherwise it is only an approximation.

  For example::

    import numpy as np

    from qiskit.circuit.library import QuadraticForm

    A = np.array([[1, 2], [-1, 0]])
    b = np.array([3, -3])
    c = -2
    m = 4
    quad_form_circuit = QuadraticForm(m, A, b, c)

- Add :meth:`qiskit.quantum_info.Statevector.expectation_value` and
  :meth:`qiskit.quantum_info.DensityMatrix.expectation_value` methods for
  computing the expectation value of an :class:`qiskit.quantum_info.Operator`.

- For the ``seed`` kwarg in the constructor for
  :class:`qiskit.circuit.library.QuantumVolume` `numpy random Generator
  objects <https://numpy.org/doc/stable/reference/random/generator.html>`__
  can now be used. Previously, only integers were a valid input. This is
  useful when integrating :class:`~qiskit.circuit.library.QuantumVolume` as
  part of a larger function with its own random number generation, e.g.
  generating a sequence of
  :class:`~qiskit.circuit.library.QuantumVolume` circuits.

- The :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.compose` has a new kwarg ``front``
  which can be used for prepending the other circuit before the origin
  circuit instead of appending. For example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    circ1 = QuantumCircuit(2)
    circ2 = QuantumCircuit(2)

    circ2.h(0)
    circ1.cx(0, 1)

    circ1.compose(circ2, front=True).draw(output='mpl')

- Two new passes, :class:`~qiskit.transpiler.passes.SabreLayout` and
  :class:`~qiskit.transpiler.passes.SabreSwap` for layout and routing have
  been added to :mod:`qiskit.transpiler.passes`. These new passes are based
  on the algorithm presented in Li et al., "Tackling the Qubit Mapping
  Problem for NISQ-Era Quantum Devices", ASPLOS 2019. They can also be
  selected when using the :func:`~qiskit.compiler.transpile` function by
  setting the ``layout_method`` kwarg to ``'sabre'`` and/or the
  ``routing_method`` to ``'sabre'`` to use
  :class:`~qiskit.transpiler.passes.SabreLayout` and
  :class:`~qiskit.transpiler.passes.SabreSwap` respectively.

- Added the method :meth:`~qiskit.pulse.Schedule.replace` to the
  :class:`qiskit.pulse.Schedule` class which allows a
  pulse instruction to be replaced with another. For example::

  .. code-block:: python

    from qiskit import pulse

    d0 = pulse.DriveChannel(0)

    sched = pulse.Schedule()

    old = pulse.Play(pulse.Constant(100, 1.0), d0)
    new = pulse.Play(pulse.Constant(100, 0.1), d0)

    sched += old

    sched = sched.replace(old, new)

    assert sched == pulse.Schedule(new)

- Added new gate classes to :mod:`qiskit.circuit.library` for the
  :math:`\sqrt{X}`, its adjoint :math:`\sqrt{X}^\dagger`, and
  controlled :math:`\sqrt{X}` gates as
  :class:`~qiskit.circuit.library.SXGate`,
  :class:`~qiskit.circuit.library.SXdgGate`, and
  :class:`~qiskit.circuit.library.CSXGate`. They can also be added to
  a :class:`~qiskit.circuit.QuantumCircuit` object using the
  :meth:`~qiskit.circuit.QuantumCircuit.sx`,
  :meth:`~qiskit.circuit.QuantumCircuit.sxdg`, and
  :meth:`~qiskit.circuit.QuantumCircuit.csx` respectively.

- Add support for :class:`~qiskit.circuit.Reset` instructions to
  :meth:`qiskit.quantum_info.Statevector.from_instruction`. Note that this
  involves RNG sampling in choosing the projection to the zero state in the
  case where the qubit is in a superposition state. The seed for sampling
  can be set using the :meth:`~qiskit.quantum_info.Statevector.seed` method.

- The methods :meth:`qiskit.circuit.ParameterExpression.subs` and
  :meth:`qiskit.circuit.QuantumCircuit.assign_parameters` now
  accept :class:`~qiskit.circuit.ParameterExpression` as the target value
  to be substituted.

  For example,

  .. code-block::

      from qiskit.circuit import QuantumCircuit, Parameter

      p = Parameter('p')
      source = QuantumCircuit(1)
      source.rz(p, 0)

      x = Parameter('x')
      source.assign_parameters({p: x*x})

  .. parsed-literal::

           ┌──────────┐
      q_0: ┤ Rz(x**2) ├
           └──────────┘

- The :meth:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.to_gate` has a new kwarg
  ``label`` which can be used to set a label for for the output
  :class:`~qiskit.circuit.Gate` object. For example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    circuit_gate = QuantumCircuit(2)
    circuit_gate.h(0)
    circuit_gate.cx(0, 1)
    custom_gate = circuit_gate.to_gate(label='My Special Bell')
    new_circ = QuantumCircuit(2)
    new_circ.append(custom_gate, [0, 1], [])
    new_circ.draw(output='mpl')

- Added the :class:`~qiskit.circuit.library.UGate`,
  :class:`~qiskit.circuit.library.CUGate`,
  :class:`~qiskit.circuit.library.PhaseGate`, and
  :class:`~qiskit.circuit.library.CPhaseGate` with the corresponding
  :class:`~qiskit.circuit.QuantumCircuit` methods
  :meth:`~qiskit.circuit.QuantumCircuit.u`,
  :meth:`~qiskit.circuit.QuantumCircuit.cu`,
  :meth:`~qiskit.circuit.QuantumCircuit.p`, and
  :meth:`~qiskit.circuit.QuantumCircuit.cp`.
  The :class:`~qiskit.circuit.library.UGate` gate is the generic single qubit
  rotation gate with 3 Euler angles and the
  :class:`~qiskit.circuit.library.CUGate` gate its controlled version.
  :class:`~qiskit.circuit.library.CUGate` has 4 parameters to account for a
  possible global phase of the U gate. The
  :class:`~qiskit.circuit.library.PhaseGate` and
  :class:`~qiskit.circuit.library.CPhaseGate` gates are the general Phase
  gate at an arbitrary angle and it's controlled version.

- A new kwarg, ``cregbundle`` has been added to the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.draw`. When set to ``True`` the
  cregs will be bundled into a single line in circuit visualizations for the
  ``text`` and ``mpl`` drawers. The default value is ``True``.
  Addresses issue `#4290 <https://github.com/Qiskit/qiskit-terra/issues/4290>`_.

  For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      circuit = QuantumCircuit(2)
      circuit.measure_all()
      circuit.draw(output='mpl', cregbundle=True)

- A new kwarg, ``initial_state`` has been added to the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.draw`. When set to ``True`` the
  initial state will now be included in circuit visualizations for all drawers.
  Addresses issue `#4293 <https://github.com/Qiskit/qiskit-terra/issues/4293>`_.

  For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      circuit = QuantumCircuit(2)
      circuit.measure_all()
      circuit.draw(output='mpl', initial_state=True)

- Labels will now be displayed when using the 'mpl' drawer. There are 2
  types of labels - gate labels and control labels. Gate labels will
  replace the gate name in the display. Control labels will display
  above or below the controls for a gate.
  Fixes issues #3766, #4580
  Addresses issues `#3766 <https://github.com/Qiskit/qiskit-terra/issues/3766>`_
  and `#4580 <https://github.com/Qiskit/qiskit-terra/issues/4580>`_.

  For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      from qiskit.circuit.library.standard_gates import YGate
      circuit = QuantumCircuit(2)
      circuit.append(YGate(label='A Y Gate').control(label='Y Control'), [0, 1])
      circuit.draw(output='mpl')


.. _Release Notes_0.15.0_Upgrade Notes:

Upgrade Notes
-------------

- Implementations of the multi-controlled X Gate (
  :class:`~qiskit.circuit.library.MCXGrayCode`,
  :class:`~qiskit.circuit.library.MCXRecursive`, and
  :class:`~qiskit.circuit.library.MCXVChain`) have had their ``name``
  properties changed to more accurately describe their
  implementation: ``mcx_gray``, ``mcx_recursive``, and
  ``mcx_vchain`` respectively. Previously, these gates shared the
  name ``mcx`` with :class:`~qiskit.circuit.library.MCXGate`, which caused
  these gates to be incorrectly transpiled and simulated.

- By default the preset passmanagers in
  :mod:`qiskit.transpiler.preset_passmanagers` are using
  :class:`~qiskit.transpiler.passes.UnrollCustomDefinitions` and
  :class:`~qiskit.transpiler.passes.BasisTranslator` to handle basis changing
  instead of the previous default :class:`~qiskit.transpiler.passes.Unroller`.
  This was done because the new passes are more flexible and allow targeting
  any basis set, however the output may differ. To use the previous default
  you can set the ``translation_method`` kwarg on
  :func:`~qiskit.compiler.transpile` to ``'unroller'``.

- The :func:`qiskit.converters.circuit_to_gate` and
  :func`qiskit.converters.circuit_to_instruction` converter functions
  had previously automatically included the generated gate or instruction
  in the active ``SessionEquivalenceLibrary``. These converters now accept
  an optional ``equivalence_library`` keyword argument to specify if and
  where the converted instances should be registered. The default behavior
  has changed to not register the converted instance.

- The default value of the ``cregbundle`` kwarg for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function has been changed
  to ``True``. This means that by default the classical bits in the
  circuit diagram will now be bundled by default, for example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    circ = QuantumCircuit(4)
    circ.x(0)
    circ.h(1)
    circ.measure_all()
    circ.draw(output='mpl')

  If you want to have your circuit drawing retain the previous behavior
  and show each classical bit in the diagram you can set the ``cregbundle``
  kwarg to ``False``. For example:

  .. jupyter-execute::

    from qiskit.circuit import QuantumCircuit

    circ = QuantumCircuit(4)
    circ.x(0)
    circ.h(1)
    circ.measure_all()
    circ.draw(output='mpl', cregbundle=False)

- :class:`~qiskit.pulse.Schedule` plotting with
  :py:meth:`qiskit.pulse.Schedule.draw` and
  :func:`qiskit.visualization.pulse_drawer` will no
  longer display the event table by default. This can be reenabled by setting
  the ``table`` kwarg to ``True``.

- The pass :class:`~qiskit.transpiler.passes.RemoveResetInZeroState` was
  previously included in the preset pass manager
  :func:`~qiskit.transpiler.preset_passmanagers.level_0_pass_manager` which
  was used with the ``optimization_level=0`` for
  :func:`~qiskit.compiler.transpile` and :func:`~qiskit.execute.execute`
  functions. However,
  :class:`~qiskit.transpiler.passes.RemoveResetInZeroState` is an
  optimization pass and should not have been included in optimization level
  0 and was removed. If you need to run :func:`~qiskit.compiler.transpile`
  with :class:`~qiskit.transpiler.passes.RemoveResetInZeroState` either use
  a custom pass manager or ``optimization_level`` 1, 2, or 3.

- The deprecated kwarg ``line_length`` for the
  :func:`qiskit.visualization.circuit_drawer` function and
  :meth:`qiskit.circuit.QuantumCircuit.draw` method has been removed. It
  had been deprecated since the 0.10.0 release. Instead you can use the
  ``fold`` kwarg to adjust the width of the circuit diagram.

- The ``'mpl'`` output mode for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`~qiskit.visualization.circuit_drawer` now requires the
  `pylatexenc <https://pylatexenc.readthedocs.io/en/latest/latexencode/>`__
  library to be installed. This was already an optional dependency for
  visualization, but was only required for the ``'latex'`` output mode
  before. It is now also required for the matplotlib drawer because it is
  needed to handle correctly sizing gates with matplotlib's
  `mathtext <https://matplotlib.org/3.2.2/tutorials/text/mathtext.html>`__
  labels for gates.

- The deprecated ``get_tokens`` methods for the :class:`qiskit.qasm.Qasm`
  and :class:`qiskit.qasm.QasmParser` has been removed. These methods have
  been deprecated since the 0.9.0 release. The
  :meth:`qiskit.qasm.Qasm.generate_tokens` and
  :meth:`qiskit.qasm.QasmParser.generate_tokens` methods should be used
  instead.

- The deprecated kwarg ``channels_to_plot`` for
  :meth:`qiskit.pulse.Schedule.draw`,
  :meth:`qiskit.pulse.Instruction.draw`,
  ``qiskit.visualization.pulse.matplotlib.ScheduleDrawer.draw`` and
  :func:`~qiskit.visualization.pulse_drawer` has been removed. The kwarg
  has been deprecated since the 0.11.0 release and was replaced by
  the ``channels`` kwarg, which functions identically and should be used
  instead.

- The deprecated ``circuit_instruction_map`` attribute of the
  :class:`qiskit.providers.models.PulseDefaults` class has been removed.
  This attribute has been deprecated since the 0.12.0 release and was
  replaced by the ``instruction_schedule_map`` attribute which can be used
  instead.

- The ``union`` method of :py:class:`~qiskit.pulse.Schedule` and
  :py:class:`~qiskit.pulse.Instruction` have been deprecated since
  the 0.12.0 release and have now been removed. Use
  :meth:`qiskit.pulse.Schedule.insert` and
  :meth:`qiskit.pulse.Instruction.meth` methods instead with the
  kwarg``time=0``.

- The deprecated ``scaling`` argument to the ``draw`` method of
  :py:class:`~qiskit.pulse.Schedule` and :py:class:`~qiskit.pulse.Instruction`
  has been replaced with ``scale`` since the 0.12.0 release and now has been
  removed. Use the ``scale`` kwarg instead.

- The deprecated ``period`` argument to :py:mod:`qiskit.pulse.library` functions
  have been replaced by ``freq`` since the 0.13.0 release and now removed. Use the
  ``freq`` kwarg instead of ``period``.

- The ``qiskit.pulse.commands`` module containing ``Commands`` classes
  was deprecated in the 0.13.0 release and has now been removed. You will
  have to upgrade your Pulse code if you were still using commands. For
  example:

  .. list-table::
    :header-rows: 2

    * - Old
      - New
    * - ``Command(args)(channel)``
      - ``Instruction(args, channel)``
    * - .. code-block:: python

          Acquire(duration)(AcquireChannel(0))
      - .. code-block:: python

          Acquire(duration, AcquireChannel(0))
    * - .. code-block:: python

          Delay(duration)(channel)
      - .. code-block:: python

          Delay(duration, channel)
    * - .. code-block:: python

          FrameChange(angle)(DriveChannel(0))
      - .. code-block:: python

          # FrameChange was also renamed
          ShiftPhase(angle, DriveChannel(0))
    * - .. code-block:: python

          Gaussian(...)(DriveChannel(0))
      - .. code-block:: python

          # Pulses need to be `Play`d
          Play(Gaussian(...), DriveChannel(0))

- All classes and function in the ``qiskit.tool.qi`` module were deprecated
  in the 0.12.0 release and have now been removed. Instead use the
  :mod:`qiskit.quantum_info` module and the new methods and classes that
  it has for working with quantum states and operators.

- The ``qiskit.quantum_info.basis_state`` and
  ``qiskit.quantum_info.projector`` functions are deprecated as of
  Qiskit Terra 0.12.0 as are now removed. Use the
  :class:`qiskit.quantum_info.QuantumState` and its derivatives
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` to work with states.

- The interactive plotting functions from :mod:`qiskit.visualization`,
  ``iplot_bloch_multivector``, ``iplot_state_city``, ``iplot_state_qsphere``,
  ``iplot_state_hinton``, ``iplot_histogram``, ``iplot_state_paulivec`` now
  are just deprecated aliases for the matplotlib based equivalents and are
  no longer interactive. The hosted static JS code that these functions
  relied on has been removed and they no longer could work. A normal
  deprecation wasn't possible because the site they depended on no longer
  exists.

- The validation components using marshmallow from :mod:`qiskit.validation`
  have been removed from terra. Since they are no longer used to build
  any objects in terra.

- The marshmallow schema classes in :mod:`qiskit.result` have been removed
  since they are no longer used by the :class:`qiskit.result.Result` class.

- The output of the :meth:`~qiskit.result.Result.to_dict` method for the
  :class:`qiskit.result.Result` class is no longer in a format for direct
  JSON serialization. Depending on the content contained in instances of
  these classes there may be types that the default JSON encoder doesn't
  know how to handle, for example complex numbers or numpy arrays. If you're
  JSON serializing the output of the ``to_dict()`` method directly you should
  ensure that your JSON encoder can handle these types.

- The option to acquire multiple qubits at once was deprecated in the 0.12.0
  release and is now removed. Specifically, the init args ``mem_slots`` and
  ``reg_slots`` have been removed from
  :class:`qiskit.pulse.instructions.Acquire`, and ``channel``, ``mem_slot``
  and ``reg_slot`` will raise an error if a list is provided as input.

- Support for the use of the ``USE_RETWORKX`` environment variable which was
  introduced in the 0.13.0 release to provide an optional fallback to the
  legacy `networkx <https://networkx.github.io/>`__ based
  :class:`qiskit.dagcircuit.DAGCircuit` implementation
  has been removed. This flag was only intended as provide a relief valve
  for any users that encountered a problem with the new implementation for
  one release during the transition to retworkx.

- The module within :mod:`qiskit.pulse` responsible for schedule->schedule transformations
  has been renamed from ``reschedule.py`` to ``transforms.py``. The previous import
  path has been deprecated. To upgrade your code::

      from qiskit.pulse.rescheduler import <X>

  should be replaced by::

      from qiskit.pulse.transforms import <X>

- In previous releases a :class:`~qiskit.transpiler.PassManager`
  did not allow ``TransformationPass`` classes to modify the
  :class:`~qiskit.transpiler.PropertySet`.  This restriction has been lifted
  so a ``TransformationPass`` class now has read and write access to both
  the :class:`~qiskit.transpiler.PropertySet` and
  :class:`~qiskit.transpiler.DAGCircuit` during
  :meth:`~qiskit.transpiler.PassManager.run`. This change was made to
  more efficiently facilitate ``TransformationPass`` classes that have an
  internal state which may be necessary for later passes in the
  :class:`~qiskit.transpiler.PassManager`. Without this change a second
  redundant ``AnalysisPass`` would have been necessary to recreate the
  internal state, which could add significant overhead.

.. _Release Notes_0.15.0_Deprecation Notes:

Deprecation Notes
-----------------

- The name of the first positional parameter for the
  :mod:`qiskit.visualization` functions
  :func:`~qiskit.visualization.plot_state_hinton`,
  :func:`~qiskit.visualization.plot_bloch_multivector`,
  :func:`~qiskit.visualization.plot_state_city`,
  :func:`~qiskit.visualization.plot_state_paulivec`, and
  :func:`~qiskit.visualization.plot_state_qsphere` has been renamed from
  ``rho`` to ``state``. Passing in the value by name to ``rho`` is deprecated
  and will be removed in a future release. Instead you should either pass
  the argument positionally or use the new parameter name ``state``.

- The ``qiskit.pulse.pulse_lib`` module has been deprecated and will be
  removed in a future release. It has been renamed to
  :py:mod:`qiskit.pulse.library` which should be used instead.

- The :class:`qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.mirror` has been deprecated and will
  be removed in a future release. The method
  :meth:`qiskit.circuit.QuantumCircuit.reverse_ops` should be used instead,
  since mirroring could be confused with swapping the output qubits of the
  circuit. The :meth:`~qiskit.circuit.QuantumCircuit.reverse_ops` method
  only reverses the order of gates that are applied instead of mirroring.

- The :meth:`~qiskit.dagcircuit.DAGCircuit.qubits` and
  :meth:`~qiskit.dagcircuit.DAGCircuit.clbits` methods of
  :class:`qiskit.dagcircuit.DAGCircuit` have been deprecated and will be
  removed in a future release. They have been replaced with properties of
  the same name, :attr:`qiskit.dagcircuit.DAGCircuit.qubits` and
  :attr:`qiskit.dagcircuit.DAGCircuit.clbits`, and are cached so
  accessing them is much faster.

- The ``get_sample_pulse`` method for
  ``qiskit.pulse.library.ParametricPulse`` derived classes (for example
  :class:`~qiskit.pulse.library.GaussianSquare`) has been deprecated and
  will be removed in a future release. It has been replaced by the
  ``get_waveform`` method (for example
  :meth:`~qiskit.pulse.library.GaussianSquare.get_waveform`) which should
  behave identically.

- The use of the optional ``condition`` argument on
  :class:`qiskit.dagcircuit.DAGNode`,
  :meth:`qiskit.dagcircuit.DAGCircuit.apply_operation_back`, and
  :meth:`qiskit.dagcircuit.DAGCircuit.apply_operation_front` has been
  deprecated and will be removed in a future release. Instead the
  ``control`` set in :class:`qiskit.circuit.Instruction` instances being
  added to a :class:`~qiskit.dagcircuit.DAGCircuit` should be used.

- The ``set_atol`` and ``set_rtol`` class methods of the
  :class:`qiskit.quantum_info.BaseOperator` and
  :class:`qiskit.quantum_info.QuantumState` classes (and
  their subclasses such as :class:`~qiskit.quantum_info.Operator`
  and :class:`qiskit.quantum_info.DensityMatrix`) are deprecated and will
  be removed in a future release. Instead the value for the attributes
  ``.atol`` and ``.rtol`` should be set on the class instead. For example::

    from qiskit.quantum_info import ScalarOp

    ScalarOp.atol = 3e-5
    op = ScalarOp(2)

- The interactive plotting functions from :mod:`qiskit.visualization`,
  ``iplot_bloch_multivector``, ``iplot_state_city``, ``iplot_state_qsphere``,
  ``iplot_state_hinton``, ``iplot_histogram``, ``iplot_state_paulivec`` have
  been deprecated and will be removed in a future release. The matplotlib
  based equivalent functions from :mod:`qiskit.visualization`,
  :func:`~qiskit.visualization.plot_bloch_multivector`,
  :func:`~qiskit.visualization.plot_state_city`,
  :func:`~qiskit.visualization.plot_state_qsphere`,
  :func:`~qiskit.visualization.plot_state_hinton`,
  :func:`~qiskit.visualization.plot_state_histogram`, and
  :func:`~qiskit.visualization.plot_state_paulivec` should be used instead.

- The properties ``acquires``, ``mem_slots``, and ``reg_slots`` of the
  :class:`qiskit.pulse.instructions.Acquire` pulse instruction have been
  deprecated and will be removed in a future release. They are just
  duplicates of :attr:`~qiskit.pulse.instructions.Acquire.channel`,
  :attr:`~qiskit.pulse.instructions.Acquire.mem_slot`,
  and :attr:`~qiskit.pulse.instructions.Acquire.reg_slot` respectively
  now that previously deprecated support for using multiple qubits in a
  single :class:`~qiskit.pulse.instructions.Acquire` instruction has been
  removed.

- The ``SamplePulse`` class from :mod:`qiskit.pulse` has been renamed to
  :py:class:`~qiskit.pulse.library.Waveform`. ``SamplePulse`` is deprecated
  and will be removed in a future release.

- The style dictionary key ``cregbundle`` has been deprecated and will be
  removed in a future release. This has been replaced by the
  kwarg ``cregbundle`` added to the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.draw`.


.. _Release Notes_0.15.0_Bug Fixes:

Bug Fixes
---------

- The :class:`qiskit.circuit.QuantumCircuit` method
  :attr:`~qiskit.circuit.QuantumCircuit.num_nonlocal_gates` previously
  included multi-qubit :class:`qiskit.circuit.Instruction` objects
  (for example, :class:`~qiskit.circuit.library.Barrier`) in its count of
  non-local gates. This has been corrected so that only non-local
  :class:`~qiskit.circuit.Gate` objects are counted.
  Fixes `#4500 <https://github.com/Qiskit/qiskit-terra/issues/4500>`__

- :class:`~qiskit.circuit.ControlledGate` instances with a set
  ``ctrl_state`` were in some cases not being evaluated as equal, even if the
  compared gates were equivalent. This has been resolved so that
  Fixes `#4573 <https://github.com/Qiskit/qiskit-terra/issues/4573>`__

- When accessing a bit from a
  :class:`qiskit.circuit.QuantumRegister` or
  :class:`qiskit.circuit.ClassicalRegister` by index when using numpy
  `integer types` <https://numpy.org/doc/stable/user/basics.types.html>`__
  would previously raise a ``CircuitError`` exception. This has been
  resolved so numpy types can be used in addition to Python's built-in
  ``int`` type.
  Fixes `#3929 <https://github.com/Qiskit/qiskit-terra/issues/3929>`__.

- A bug was fixed where only the first :class:`qiskit.pulse.configuration.Kernel`
  or :class:`qiskit.pulse.configuration.Discriminator` for an
  :class:`qiskit.pulse.Acquire` was used when there were multiple Acquires
  at the same time in a :class:`qiskit.pulse.Schedule`.

- The SI unit use for constructing :py:class:`qiskit.pulse.SetFrequency`
  objects is in Hz, but when a :class:`~qiskit.qobj.PulseQobjInstruction`
  object is created from a :py:class:`~qiskit.pulse.SetFrequency` instance
  it needs to be converted to GHz. This conversion was missing from previous
  releases and has been fixed.

- Previously it was possible to set the number of control qubits to zero in
  which case the the original, potentially non-controlled, operation would be
  returned. This could cause an ``AttributeError`` to be raised if the caller
  attempted to access an attribute which only
  :class:`~qiskit.circuit.ControlledGate` object have. This has been fixed
  by adding a getter and setter for
  :attr:`~qiskit.circuit.ControlledGate.num_ctrl_qubits` to validate
  that a valid value is being used.
  Fixes `#4576 <https://github.com/Qiskit/qiskit-terra/issues/4576>`__

- Open controls were implemented by modifying a :class:`~qiskit.circuit.Gate`
  objects :attr:`~qiskit.circuit.Gate.definition`. However, when the gate
  already exists in the basis set, this definition was not used, which
  resulted in incorrect circuits being sent to a backend after transpilation.
  This has been fixed by modifying the :class:`~qiskit.transpiler.Unroller`
  pass to use the definition if it encounters a controlled gate with open
  controls.
  Fixes `#4437 <https://github.com/Qiskit/qiskit-terra/issues/4437>`__

- The ``insert_barriers`` keyword argument in the
  :class:`~qiskit.circuit.library.ZZFeatureMap` class didn't actually insert
  barriers in between the Hadamard layers and evolution layers. This has been
  fixed so that barriers are now properly inserted.

- Fixed issue where some gates with three or more qubits would fail to compile
  in certain instances. Refer to
  `#4577 <https://github.com/Qiskit/qiskit-terra/issues/4577` for more detail.

- The matplotlib (``'mpl'``) output backend for the
  :class:`qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.draw` and the
  :func:`qiskit.visualization.circuit_drawer` function was not properly
  scaling when the kwarg ``scale`` was set. Fonts and line widths
  did not scale with the rest of the image. This has been fixed and all
  elements of the circuit diagram now scale properly. For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      circuit = QuantumCircuit(2)
      circuit.h(0)
      circuit.cx(0, 1)
      circuit.draw(output='mpl', scale=0.5)

  Fixes `#4179 <https://github.com/Qiskit/qiskit-terra/issues/4179>`_.

- Fixes issue where initializing or evolving
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes by circuits by
  circuit containing :class:`~qiskit.circuit.Barrier` instructions would
  raise an exception. Fixes
  `#4461 <https://github.com/Qiskit/qiskit-terra/issues/4461>`__

- Previously when a :class:`~qiskit.circuit.QuantumCircuit` contained a
  :class:`~qiskit.circuit.Gate` with a classical condition the transpiler
  would sometimes fail when using ``optimization_level=3`` on
  :func:`~qiskit.compiler.transpile` or
  :func:`~qiskit.execute.execute` raising an ``UnboundLocalError``. This has
  been fixed by updating the
  :class:`~qiskit.transpiler.passes.ConsolidateBlocks` pass to account for
  the classical condition.
  Fixes `#4672 <https://github.com/Qiskit/qiskit-terra/issues/4672>`_.

- In some situations long gate and register names would overflow, or leave
  excessive empty space around them when using the ``'mpl'`` output backend
  for the :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function. This has been fixed
  by using correct text widths for a proportional font. Fixes
  `#4611 <https://github.com/Qiskit/qiskit-terra/issues/4611>`__,
  `#4605 <https://github.com/Qiskit/qiskit-terra/issues/4605>`__,
  `#4545 <https://github.com/Qiskit/qiskit-terra/issues/4545>`__,
  `#4497 <https://github.com/Qiskit/qiskit-terra/issues/4497>`__,
  `#4449 <https://github.com/Qiskit/qiskit-terra/issues/4449>`__, and
  `#3641 <https://github.com/Qiskit/qiskit-terra/issues/3641>`__.

- When using the ``style` kwarg on the
  :meth:`qiskit.circuit.QuantumCircuit.draw` or
  :func:`qiskit.visualization.circuit_drawer` with the ``'mpl'`` output
  backend the dictionary key ``'showindex'`` set to ``True``, the index
  numbers at the top of the column did not line up properly. This has been
  fixed.

- When using ``cregbunde=True`` with the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function and measuring onto
  a second fold, the measure arrow would overwrite the creg count. The count
  was moved to the left to prevent this. Fixes
  `#4148 <https://github.com/Qiskit/qiskit-terra/issues/4148>`__.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function
  :class:`~qiskit.circuit.library.CSwapGate` gates and a controlled
  :class:`~qiskit.circuit.library.RZZGate` gates now display with their
  appropriate symbols instead of in a box.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function controlled gates
  created using the :meth:`~qiskit.circuit.QuantumCircuit.to_gate` method
  were not properly spaced and could overlap with other gates in the circuit
  diagram. This issue has been fixed.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function
  gates with arrays as parameters, such as
  :class:`~qiskit.extensions.HamiltonianGate`, no longer display with
  excessive space around them. Fixes
  `#4352 <https://github.com/Qiskit/qiskit-terra/issues/4352>`__.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function
  generic gates created by directly instantiating :class:`qiskit.circuit.Gate`
  method now display the proper background color for the gate. Fixes
  `#4496 <https://github.com/Qiskit/qiskit-terra/issues/4496>`__.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function
  an ``AttributeError`` that occurred when using
  :class:`~qiskit.extensions.Isometry` or :class:`~qiskit.extensions.Initialize`
  has been fixed. Fixes
  `#4439 <https://github.com/Qiskit/qiskit-terra/issues/4439>`__.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function
  some open-controlled gates did not properly display the open controls.
  This has been corrected so that open controls are properly displayed
  as open circles. Fixes
  `#4248 <https://github.com/Qiskit/qiskit-terra/issues/4248>`__.

- When using the ``'mpl'`` output backend for the
  :meth:`qiskit.circuit.QuantumCircuit.draw` method and
  :func:`qiskit.visualization.circuit_drawer` function
  setting the ``fold`` kwarg to -1 will now properly display the circuit
  without folding. Fixes
  `#4506 <https://github.com/Qiskit/qiskit-terra/issues/4506>`__.

- Parametric pulses from :mod:`qiskit.pulse.library.discrete`
  now have zero ends of parametric pulses by default. The endpoints are
  defined such that for a function :math:`f(x)` then
  :math:`f(-1) = f(duration + 1) = 0`.
  Fixes `#4317 <https://github.com/Qiskit/qiskit-terra/issues/4317>`__


.. _Release Notes_0.15.0_Other Notes:

Other Notes
-----------

- The :class:`qiskit.result.Result` class which was previously constructed
  using the marshmallow library has been refactored to not depend on
  marshmallow anymore. This new implementation should be a seamless transition
  but some specific behavior that was previously inherited from marshmallow
  may not work. Please file issues for any incompatibilities found.

Aer 0.6.1
=========

.. _Release Notes_0.6.0_Prelude:

Prelude
-------

This 0.6.0 release includes numerous performance improvements for all
simulators in the Aer provider and significant changes to the build system
when building from source. The main changes are support for SIMD
vectorization, approximation in the matrix product state method via
bond-dimension truncation, more efficient Pauli expectation value
computation, and greatly improved efficiency in Python conversion of
C++ result objects. The build system was upgraded to use the
`Conan <https://conan.io/>`__ to manage common C++ dependencies when
building from source.

.. _Release Notes_0.6.0_New Features:

New Features
------------

- Add density matrix snapshot support to "statevector" and "statevector_gpu"
  methods of the QasmSimulator.

- Allow density matrix snapshots on specific qubits, not just all qubits.
  This computes the partial trace of the state over the remaining qubits.

- Adds Pauli expectation value snapshot support to the `"density_matrix"`
  simulation method of the :class:`qiskit.providers.aer.QasmSimulator`.
  Add snapshots to circuits using the
  :class:`qiskit.providers.aer.extensions.SnapshotExpectationValue`
  extension.

- Greatly improves performance of the Pauli expectation value snapshot
  algorithm for the `"statevector"`, `"statevector_gpu`, `"density_matrix"`,
  and `"density_matrix_gpu"` simulation methods of the
  :class:`qiskit.providers.aer.QasmSimulator`.

- Enable the gate-fusion circuit optimization from the
  :class:`qiskit.providers.aer.QasmSimulator` in both the
  :class:`qiskit.providers.aer.StatevectorSimulator` and
  :class:`qiskit.providers.aer.UnitarySimulator` backends.

- Improve the performance of average snapshot data in simulator results.
  This effects probability, Pauli expectation value, and density matrix snapshots
  using the following extensions:

  * :class:`qiskit.providers.aer.extensions.SnapshotExpectationValue`
  * :class:`qiskit.providers.aer.extensions.SnapshotProbabilities`
  * :class:`qiskit.providers.aer.extensions.SnapshotDensityMatrix`

- Add move constructor and improve memory usage of the C++ matrix class
  to minimize copies of matrices when moving output of simulators into results.

- Improve performance of unitary simulator.

- Add approximation to the `"matrix_product_state"` simulation method of the
  :class:`~qiskit.providers.aer.QasmSimulator` to limit the bond-dimension of
  the MPS.

  There are two modes of approximation. Both discard the smallest
  Schmidt coefficients following the SVD algorithm.
  There are two parameters that control the degree of approximation:
  ``"matrix_product_state_max_bond_dimension"`` (int): Sets a limit
  on the number of Schmidt coefficients retained at the end of
  the svd algorithm. Coefficients beyond this limit will be discarded.
  (Default: None, i.e., no limit on the bond dimension).
  ``"matrix_product_state_truncation_threshold"`` (double):
  Discard the smallest coefficients for which the sum of
  their squares is smaller than this threshold.
  (Default: 1e-16).

- Improve the performance of measure sampling when using the
  `"matrix_product_state"` :class:`~qiskit.providers.aer.QasmSimulator`
  simulation method.

- Add support for ``Delay``, ``Phase`` and ``SetPhase`` pulse instructions
  to the :class:`qiskit.providers.aer.PulseSimulator`.

- Improve the performance of the :class:`qiskit.providers.aer.PulseSimulator`
  by caching calls to RHS function

- Introduce alternate DE solving methods, specifiable through ``backend_options``
  in the :class:`qiskit.providers.aer.PulseSimulator`.

- Improve performance of simulator result classes by using move semantics
  and removing unnecessary copies that were happening when combining results
  from separate experiments into the final result object.

- Greatly improve performance of pybind11 conversion of simulator results by
  using move semantics where possible, and by moving vector and matrix results
  to Numpy arrays without copies.

- Change the RNG engine for simulators from 32-bit Mersenne twister to
  64-bit Mersenne twister engine.

- Improves the performance of the `"statevector"` simulation method of the
  :class:`qiskit.providers.aer.QasmSimulator` and
  :class:`qiskit.providers.aer.StatevectorSimulator` by using SIMD
  intrinsics on systems that support the AVX2 instruction set. AVX2
  support is automatically detected and enabled at runtime.


.. _Release Notes_0.6.0_Upgrade Notes:

Upgrade Notes
-------------

- Changes the build system to use the
  `Conan package manager <https://conan.io/>`__.
  This tool will handle most of the dependencies needed by the C++ source
  code. Internet connection may be needed for the first build or when
  dependencies are added or updated, in order to download the required
  packages if they are not in your Conan local repository.

  When building the standalone version of qiskit-aer you must install conan
  first with:

  .. code-block:: bash

    pip install conan

- Changes how transpilation passes are handled in the C++ Controller classes
  so that each pass must be explicitly called. This allows for greater
  customization on when each pass should be called, and with what parameters.
  In particular this enables setting different parameters for the gate
  fusion optimization pass depending on the QasmController simulation method.

- Add ``gate_length_units`` kwarg to
  :meth:`qiskit.providers.aer.noise.NoiseModel.from_device`
  for specifying custom ``gate_lengths`` in the device noise model function
  to handle unit conversions for internal code.

- Add Controlled-Y ("cy") gate to the Stabilizer simulator methods supported
  gateset.

- For Aer's backend the jsonschema validation of input qobj objects from
  terra is now opt-in instead of being enabled by default. If you want
  to enable jsonschema validation of qobj set the ``validate`` kwarg on
  the :meth:`qiskit.providers.aer.QasmSimualtor.run` method for the backend
  object to ``True``.

- Adds an OpSet object to the base simulator State class to allow easier
  validation of instructions, gates, and snapshots supported by simulators.

- Refactor OpSet class. Moved OpSet to separate header file and add
  ``contains`` and ``difference`` methods based on ``std::set::contains``
  and ``std::algorithm::set_difference``. These replace the removed invalid
  and validate instructions from OpSet, but with the order reversed. It
  returns a list of other ops not in current opset rather than opset
  instructions not in the other.

- Improves how measurement sampling optimization is checked. The expensive
  part of this operation is now done once during circuit construction where
  rather than multiple times during simulation for when checking memory
  requirements, simulation method, and final execution.


.. _Release Notes_0.6.0_Bug Fixes:

Bug Fixes
---------

- Remove "extended_stabilizer" from the automatically selected simulation
  methods. This is needed as the extended stabilizer method is not exact
  and may give incorrect results for certain circuits unless the user
  knows how to optimize its configuration parameters.

  The automatic method now only selects from "stabilizer", "density_matrix",
  and "statevector" methods. If a non-Clifford circuit that is too large for
  the statevector method is executed an exception will be raised suggesting
  you could try explicitly using the "extended_stabilizer" or
  "matrix_product_state" methods instead.

- Disables gate fusion for the matrix product state simulation method as this
  was causing issues with incorrect results being returned in some cases.

- Fixes a bug causing incorrect channel evaluation in the
  :class:`qiskit.providers.aer.PulseSimulator`.

- Fixes several minor bugs for Hamiltonian parsing edge cases in the
  :class:`qiskit.providers.aer.pulse.system_models.hamiltonian_model.HamiltonianModel`
  class.

Ignis 0.4.0
===========

.. _Release Notes_0.4.0_Prelude:

Prelude
-------

The main change made in this release is a refactor of the Randomized
Benchmarking code to integrate the updated Clifford class
:class:`qiskit.quantum_info.Clifford` from Terra and to improve the
CNOT-Dihedral class.


.. _Release Notes_0.4.0_New Features:

New Features
------------

- The :func:`qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
  function was refactored to use the updated Clifford class :class:`~qiskit.quantum_info.Clifford`,
  to allow efficient Randomized Benchmarking (RB) on Clifford sequences with more than 2 qubits.
  In addition, the code of the CNOT-Dihedral class
  :class:`qiskit.ignis.verification.randomized_benchmarking.CNOTDihedral`
  was refactored to make it more efficient, by using numpy arrays, as well not using pre-generated
  pickle files storing all the 2-qubit group elements.
  The :func:`qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
  function has a new kwarg ``rand_seed`` which can be used to specify a seed for the random number
  generator used to generate the RB circuits. This can be useful for having a reproducible circuit.

- The :func:`qiskit.ignis.verification.qv_circuits` function has a new
  kwarg ``seed`` which can be used to specify a seed for the random number
  generator used to generate the Quantum Volume circuits. This can be useful
  for having a reproducible circuit.


.. _Release Notes_0.4.0_Upgrade Notes:

Upgrade Notes
-------------

- The :func:`qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
  function is now using the updated Clifford class :class:`~qiskit.quantum_info.Clifford`
  and the updated CNOT-Dihedral class
  :class:`qiskit.ignis.verification.randomized_benchmarking.CNOTDihedral` to construct its
  output instead of using pre-generated group tables for the Clifford and CNOT-Dihedral
  group elements, which were stored in pickle files.
  This may result in subtle differences from the output from the previous version.

- A new requirement `scikit-learn <https://scikit-learn.org/stable/>`__ has
  been added to the requirements list. This dependency was added in the 0.3.0
  release but wasn't properly exposed as a dependency in that release. This
  would lead to an ``ImportError`` if the
  :mod:`qiskit.ignis.measurement.discriminator.iq_discriminators` module was
  imported. This is now correctly listed as a dependency so that
  ``scikit-learn`` will be installed with qiskit-ignis.

- The :func:`qiskit.ignis.verification.qv_circuits` function is now using
  the circuit library class :class:`~qiskit.circuit.library.QuantumVolume`
  to construct its output instead of building the circuit from scratch.
  This may result in subtle differences from the output from the previous
  version.

- Tomography fitters can now also get list of `Result` objects instead of a single `Result`
  as requested in `issue #320 <https://github.com/Qiskit/qiskit-ignis/issues/320/>`_.


.. _Release Notes_0.4.0_Deprecation Notes:

Deprecation Notes
-----------------

- The kwarg ``interleaved_gates`` for the
  :func:`qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
  function has been deprecated and will be removed in a future release.
  It is superseded by ``interleaved_elem``.
  The helper functions :class:`qiskit.ignis.verification.randomized_benchmarking.BasicUtils`,
  :class:`qiskit.ignis.verification.randomized_benchmarking.CliffordUtils` and
  :class:`qiskit.ignis.verification.randomized_benchmarking.DihedralUtils` were deprecated.
  These classes are superseded by :class:`qiskit.ignis.verification.randomized_benchmarking.RBgroup`
  that handles the group operations needed for RB.
  The class :class:`qiskit.ignis.verification.randomized_benchmarking.Clifford`
  is superseded by :class:`~qiskit.quantum_info.Clifford`.

- The kwargs ``qr`` and ``cr`` for the
  :func:`qiskit.ignis.verification.qv_circuits` function have been deprecated
  and will be removed in a future release. These kwargs were documented as
  being used for specifying a :class:`qiskit.circuit.QuantumRegister` and
  :class:`qiskit.circuit.ClassicalRegister` to use in the generated Quantum
  Volume circuits instead of creating new ones. However, the parameters were
  never actually respected and a new Register would always be created
  regardless of whether they were set or not. This behavior is unchanged and
  these kwargs still do not have any effect, but are being deprecated prior
  to removal to avoid a breaking change for users who may have been setting
  either.

- Support for passing in subsets of qubits as a list in the ``qubit_lists``
  parameter for the :func:`qiskit.ignis.verification.qv_circuits` function
  has been deprecated and will removed in a future release. In the past
  this was used to specify a layout to run the circuit on a device. In
  other words if you had a 5 qubit device and wanted to run a 2 qubit
  QV circuit on qubits 1, 3, and 4 of that device. You would pass in
  ``[1, 3, 4]`` as one of the lists in ``qubit_lists``, which would
  generate a 5 qubit virtual circuit and have qv applied to qubits 1, 3,
  and 4 in that virtual circuit. However, this functionality is not necessary
  and overlaps with the concept of ``initial_layout`` in the transpiler and
  whether a circuit has been embedded with a layout set. Moving forward
  instead you should just run :func:`~qiskit.compiler.transpile` or
  :func:`~qiskit.execute.execute` with initial layout set to do this. For
  example, running the above example would become::

    from qiskit import execute
    from qiskit.ignis.verification import qv_circuits

    initial_layout = [1, 3, 4]
    qv_circs, _ = qv_circuits([list(range3)])
    execute(qv_circuits, initial_layout=initial_layout)


.. _Release Notes_0.4.0_Bug Fixes:

Bug Fixes
---------

- Fix a bug of the position of measurement pulses inserted by
  py:func:`qiskit.ignis.characterization.calibrations.pulse_schedules.drag_schedules`.
  Fixes `#465 <https://github.com/Qiskit/qiskit-ignis/issues/465>`__

Aqua 0.7.5
==========

.. _Release Notes_0.7.5_New Features:

New Features
------------

- Removed soft dependency on CPLEX in ADMMOptimizer. Now default optimizers used by ADMMOptimizer
  are MinimumEigenOptimizer for QUBO problems and SlsqpOptimizer as a continuous optimizer. You
  can still use CplexOptimizer as an optimizer for ADMMOptimizer, but it should be set explicitly.

- New Yahoo! finance provider created.

- Introduced ``QuadraticProgramConverter`` which is an abstract class for converters.
  Added ``convert``/``interpret`` methods for converters instead of ``encode``/``decode``.
  Added ``to_ising`` and ``from_ising`` to ``QuadraticProgram`` class.
  Moved all parameters from ``convert`` to constructor except ``name``.
  Created setter/getter for converter parameters.
  Added ``auto_define_penalty`` and ``interpret`` for``LinearEqualityToPenalty``.
  Now error messages of converters are more informative.

- Added an SLSQP optimizer ``qiskit.optimization.algorithms.SlsqpOptimizer`` as a wrapper
  of the corresponding SciPy optimization method. This is a classical optimizer, does not depend
  on quantum algorithms and may be used as a replacement for ``CobylaOptimizer``.

- Cobyla optimizer has been modified to accommodate a multi start feature introduced
  in the SLSQP optimizer. By default, the optimizer does not run in the multi start mode.

- The ``SummedOp`` does a mathematically more correct check for equality, where
  expressions such as ``X + X == 2*X`` and ``X + Z == Z + X`` evaluate to ``True``.


.. _Release Notes_0.7.5_Deprecation Notes:

Deprecation Notes
-----------------

- GSLS optimizer class deprecated ``__init__`` parameter ``max_iter`` in favor of ``maxiter``.
  SPSA optimizer class deprecated ``__init__`` parameter ``max_trials`` in favor of ``maxiter``.
  optimize_svm function deprecated ``max_iters`` parameter in favor of ``maxiter``.
  ADMMParameters class deprecated ``__init__`` parameter ``max_iter`` in favor of ``maxiter``.

- The ising convert classes
  :class:`qiskit.optimization.converters.QuadraticProgramToIsing` and
  :class:`qiskit.optimization.converters.IsingToQuadraticProgram` have
  been deprecated and will be removed in a future release. Instead the
  :class:`qiskit.optimization.QuadraticProgram` methods
  :meth:`~qiskit.optimization.QuadraticProgram.to_ising` and
  :meth:`~qiskit.optimization.QuadraticPrgraom.from_ising` should be used
  instead.

- The ``pprint_as_string`` method for
  :class:`qiskit.optimization.QuadraticProgram` has been deprecated and will
  be removed in a future release. Instead you should just run
  ``.pprint_as_string()`` on the output from
  :meth:`~qiskit.optimization.QuadraticProgram.to_docplex`

- The ``prettyprint`` method for
  :class:`qiskit.optimization.QuadraticProgram` has been deprecated and will
  be removed in a future release. Instead you should just run
  ``.prettyprint()`` on the output from
  :meth:`~qiskit.optimization.QuadraticProgram.to_docplex`

.. _Release Notes_0.7.5_Bug Fixes:

Bug Fixes
---------

- Changed in python version 3.8: On macOS, the spawn start method is now the
  default. The fork start method should be considered unsafe as it can
  lead to crashes in subprocesses.
  However P_BFGS doesn't support spawn, so we revert to single process.
  Refer to
  `#1109 <https://github.com/Qiskit/qiskit-aqua/issues/1109>` for more details.

- Binding parameters in the ``CircuitStateFn`` did not copy
  the value of ``is_measurement`` and always set ``is_measurement=False``.
  This has been fixed.

- Previously, SummedOp.to_matrix_op built a list MatrixOp's (with numpy
  matrices) and then summed them, returning a single MatrixOp. Some
  algorithms (for example vqe) require summing thousands of matrices, which
  exhausts memory when building the list of matrices. With this change,
  no list is constructed. Rather, each operand in the sum is converted to
  a matrix, added to an accumulator, and discarded.

- Changing backends in VQE from statevector to qasm_simulator or real device
  was causing an error due to CircuitSampler incompatible reuse. VQE was changed
  to always create a new CircuitSampler and create a new  expectation in case not
  entered by user.
  Refer to
  `#1153 <https://github.com/Qiskit/qiskit-aqua/issues/1153>` for more details.

- Exchange and Wikipedia finance providers were fixed to correctly handle Quandl data.
  Refer to
  `#775 <https://github.com/Qiskit/qiskit-aqua/issues/775>` for more details.
  Fixes a divide by 0 error on finance providers mean vector and covariance matrix
  calculations. Refer to
  `#781 <https://github.com/Qiskit/qiskit-aqua/issues/781>` for more details.

- The ``ListOp.combo_fn`` property has been lost in several transformations,
  such as converting to another operator type, traversing, reducing or
  multiplication. Now this attribute is propagated to the resulting operator.

- The evaluation of some operator expressions, such as of ``SummedOp``s
  and evaluations with the ``CircuitSampler`` did not treat coefficients
  correctly or ignored them completely. E.g. evaluating
  ``~StateFn(0 * (I + Z)) @ Plus`` did not yield 0 or the normalization
  of ``~StateFn(I) @ ((Plus + Minus) / sqrt(2))`` missed a factor
  of ``sqrt(2)``. This has been fixed.

- ``OptimizationResult`` included some public setters and class variables
  were ``Optional``. This fix makes all class variables read-only so that
  mypy and pylint can check types more effectively.
  ``MinimumEigenOptimizer.solve`` generated bitstrings in a result as ``str``.
  This fix changed the result into ``List[float]`` as the other algorithms do.
  Some public classes related to optimization algorithms were missing in
  the documentation of ``qiskit.optimization.algorithms``. This fix added
  all such classes to the docstring.
  `#1131 <https://github.com/Qiskit/qiskit-aqua/issues/1131>` for more details.

- ``OptimizationResult.__init__`` did not check whether the sizes of ``x`` and
  ``variables`` match or not (they should match). This fix added the check to
  raise an error if they do not match and fixes bugs detected by the check.
  This fix also adds missing unit tests related to ``OptimizationResult.variable_names``
  and ``OptimizationResult.variables_dict`` in ``test_converters``.
  `#1167 <https://github.com/Qiskit/qiskit-aqua/issues/1167>` for more details.

- Fix parameter binding in the ``OperatorStateFn``, which did not bind
  parameters of the underlying primitive but just the coefficients.

- ``op.eval(other)``, where ``op`` is of type ``OperatorBase``, sometimes
  silently returns a nonsensical value when the number of qubits in ``op``
  and ``other`` are not equal. This fix results in correct behavior, which
  is to throw an error rather than return a value, because the input in
  this case is invalid.

- The ``construct_circuit`` method of ``VQE`` previously returned the
  expectation value to be evaluated as type ``OperatorBase``.
  This functionality has been moved into ``construct_expectation`` and
  ``construct_circuit`` returns a list of the circuits that are evaluated
  to compute the expectation value.


IBM Q Provider 0.8.0
====================

.. _Release Notes_0.8.0_New Features:

New Features
------------

- :class:`~qiskit.providers.ibmq.IBMQBackend` now has a new
  :meth:`~qiskit.providers.ibmq.IBMQBackend.reservations` method that
  returns reservation information for the backend, with optional filtering.
  In addition, you can now use
  :meth:`provider.backends.my_reservations()<qiskit.providers.ibmq.IBMQBackendService.my_reservations>`
  to query for your own reservations.

- :meth:`qiskit.providers.ibmq.job.IBMQJob.result` raises an
  :class:`~qiskit.providers.ibmq.job.IBMQJobFailureError` exception if
  the job has failed. The exception message now contains the reason
  the job failed, if the entire job failed for a single reason.

- A new attribute ``client_version`` was added to
  :class:`~qiskit.providers.ibmq.job.IBMQJob` and
  :class:`qiskit.result.Result` object retrieved via
  :meth:`qiskit.providers.ibmq.job.IBMQJob.result`.
  ``client_version`` is a dictionary with the key being the name
  and the value being the version of the client used to submit
  the job, such as Qiskit.

- The :func:`~qiskit.providers.ibmq.least_busy` function now takes a new,
  optional parameter ``reservation_lookahead``. If specified or defaulted to,
  a backend is considered unavailable if it has reservations in the next
  ``n`` minutes, where ``n`` is the value of ``reservation_lookahead``.
  For example, if the default value of 60 is used, then any
  backends that have reservations in the next 60 minutes are considered unavailable.

- :class:`~qiskit.providers.ibmq.managed.ManagedResults` now has a new
  :meth:`~qiskit.providers.ibmq.managed.ManagedResults.combine_results` method
  that combines results from all managed jobs and returns a single
  :class:`~qiskit.result.Result` object. This ``Result`` object can
  be used, for example, in ``qiskit-ignis`` fitter methods.


.. _Release Notes_0.8.0_Upgrade Notes:

Upgrade Notes
-------------

- Timestamps in the following fields are now in local time instead of UTC:

  * Backend properties returned by
    :meth:`qiskit.providers.ibmq.IBMQBackend.properties`.
  * Backend properties returned by
    :meth:`qiskit.providers.ibmq.job.IBMQJob.properties`.
  * ``estimated_start_time`` and ``estimated_complete_time`` in
    :class:`~qiskit.providers.ibmq.job.QueueInfo`, returned by
    :meth:`qiskit.providers.ibmq.job.IBMQJob.queue_info`.
  * ``date`` in :class:`~qiskit.result.Result`, returned by
    :meth:`qiskit.providers.ibmq.job.IBMQJob.result`.

  In addition, the ``datetime`` parameter for
  :meth:`qiskit.providers.ibmq.IBMQBackend.properties` is also expected to be
  in local time unless it has UTC timezone information.

- ``websockets`` 8.0 or above is now required if Python 3.7 or above is used.
  ``websockets`` 7.0 will continue to be used for Python 3.6 or below.

- On Windows, the event loop policy is set to ``WindowsSelectorEventLoopPolicy``
  instead of using the default ``WindowsProactorEventLoopPolicy``. This fixes
  the issue that the :meth:`qiskit.providers.ibmq.job.IBMQJob.result` method
  could hang on Windows. Fixes
  `#691 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/691>`_


.. _Release Notes_0.8.0_Deprecation Notes:

Deprecation Notes
-----------------

- Use of ``Qconfig.py`` to save IBM Quantum Experience credentials is deprecated
  and will be removed in the next release. You should use ``qiskitrc``
  (the default) instead.


.. _Release Notes_0.8.0_Bug Fixes:

Bug Fixes
---------

- Fixes an issue wherein a call to :meth:`qiskit.providers.ibmq.IBMQBackend.jobs`
  can hang if the number of jobs being returned is large. Fixes
  `#674 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/674>`_

- Fixes an issue which would raise a ``ValueError`` when building
  error maps in Jupyter for backends that are offline. Fixes
  `#706 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/706>`_

- :meth:`qiskit.providers.ibmq.IBMQBackend.jobs` will now return the correct
  list of :class:`~qiskit.providers.ibmq.job.IBMQJob` objects when the
  ``status`` kwarg is set to ``'RUNNING'``.

- The package metadata has been updated to properly reflect the dependency
  on ``qiskit-terra`` >= 0.14.0. This dependency was implicitly added as
  part of the 0.7.0 release but was not reflected in the package requirements
  so it was previously possible to install ``qiskit-ibmq-provider`` with a
  version of ``qiskit-terra`` which was too old. Fixes
  `#677 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/677>`_

*************
Qiskit 0.19.6
*************

Terra 0.14.2
============

No Change

Aer 0.5.2
=========

No Change

Ignis 0.3.3
===========

.. _Release Notes_0.3.3_Upgrade Notes:

Upgrade Notes
-------------

- A new requirement `scikit-learn <https://scikit-learn.org/stable/>`__ has
  been added to the requirements list. This dependency was added in the 0.3.0
  release but wasn't properly exposed as a dependency in that release. This
  would lead to an ``ImportError`` if the
  :mod:`qiskit.ignis.measurement.discriminator.iq_discriminators` module was
  imported. This is now correctly listed as a dependency so that
  ``scikit-learn`` will be installed with qiskit-ignis.


.. _Release Notes_0.3.3_Bug Fixes:

Bug Fixes
---------

- Fixes an issue in qiskit-ignis 0.3.2 which would raise an ``ImportError``
  when :mod:`qiskit.ignis.verification.tomography.fitters.process_fitter` was
  imported without ``cvxpy`` being installed.

Aqua 0.7.3
==========

No Change

IBM Q Provider 0.7.2
====================

No Change


*************
Qiskit 0.19.5
*************

Terra 0.14.2
============

No Change

Aer 0.5.2
=========

No Change

Ignis 0.3.2
===========

Bug Fixes
---------

- The :meth:`qiskit.ignis.verification.TomographyFitter.fit` method has improved
  detection logic for the default fitter. Previously, the ``cvx`` fitter method
  was used whenever `cvxpy <https://www.cvxpy.org/>`__ was installed. However,
  it was possible to install cvxpy without an SDP solver that would work for the
  ``cvx`` fitter method. This logic has been reworked so that the ``cvx``
  fitter method is only used if ``cvxpy`` is installed and an SDP solver is present
  that can be used. Otherwise, the ``lstsq`` fitter is used.

- Fixes an edge case in
  :meth:`qiskit.ignis.mitigation.measurement.fitters.MeasurementFitter.apply`
  for input that has invalid or incorrect state labels that don't match
  the calibration circuit. Previously, this would not error and just return
  an empty result. Instead now this case is correctly caught and a
  ``QiskitError`` exception is raised when using incorrect labels.

Aqua 0.7.3
==========

.. _Release Notes_0.7.3_Upgrade Notes:

Upgrade Notes
-------------

- The `cvxpy <https://www.cvxpy.org/>`__ dependency which is required for
  the svm classifier has been removed from the requirements list and made
  an optional dependency. This is because installing cvxpy is not seamless
  in every environment and often requires a compiler be installed to run.
  To use the svm classifier now you'll need to install cvxpy by either
  running ``pip install cvxpy<1.1.0`` or to install it with aqua running
  ``pip install qiskit-aqua[cvx]``.


.. _Release Notes_0.7.3_Bug Fixes:

Bug Fixes
---------

- The ``compose`` method of the ``CircuitOp`` used ``QuantumCircuit.combine`` which has been
  changed to use ``QuantumCircuit.compose``. Using combine leads to the problem that composing
  an operator with a ``CircuitOp`` based on a named register does not chain the operators but
  stacks them. E.g. composing ``Z ^ 2`` with a circuit based on a 2-qubit named register yielded
  a 4-qubit operator instead of a 2-qubit operator.

- The ``MatrixOp.to_instruction`` method previously returned an operator and not
  an instruction. This method has been updated to return an Instruction.
  Note that this only works if the operator primitive is unitary, otherwise
  an error is raised upon the construction of the instruction.

- The ``__hash__`` method of the ``PauliOp`` class used the ``id()`` method
  which prevents set comparisons to work as expected since they rely on hash
  tables and identical objects used to not have identical hashes. Now, the
  implementation uses a hash of the string representation inline with the
  implementation in the ``Pauli`` class.

IBM Q Provider 0.7.2
====================

No Change


*************
Qiskit 0.19.4
*************

Terra 0.14.2
============

.. _Release Notes_0.14.2_Upgrade Notes:

Upgrade Notes
-------------

- The ``circuit_to_gate`` and ``circuit_to_instruction`` converters had
  previously automatically included the generated gate or instruction in the
  active ``SessionEquivalenceLibrary``. These converters now accept an
  optional ``equivalence_library`` keyword argument to specify if and where
  the converted instances should be registered. The default behavior is not
  to register the converted instance.


.. _Release Notes_0.14.2_Bug Fixes:

Bug Fixes
---------

- Implementations of the multi-controlled X Gate (``MCXGrayCode``,
  ``MCXRecursive`` and ``MCXVChain``) have had their ``name``
  properties changed to more accurately describe their
  implementation (``mcx_gray``, ``mcx_recursive``, and
  ``mcx_vchain`` respectively.) Previously, these gates shared the
  name ``mcx` with ``MCXGate``, which caused these gates to be
  incorrectly transpiled and simulated.

- ``ControlledGate`` instances with a set ``ctrl_state`` were in some cases
  not being evaluated as equal, even if the compared gates were equivalent.
  This has been resolved.

- Fixed the SI unit conversion for :py:class:`qiskit.pulse.SetFrequency`. The
  ``SetFrequency`` instruction should be in Hz on the frontend and has to be
  converted to GHz when ``SetFrequency`` is converted to ``PulseQobjInstruction``.

- Open controls were implemented by modifying a gate\'s
  definition. However, when the gate already exists in the basis,
  this definition is not used, which yields incorrect circuits sent
  to a backend. This modifies the unroller to output the definition
  if it encounters a controlled gate with open controls.

Aer 0.5.2
=========

No Change

Ignis 0.3.0
===========

No Change

Aqua 0.7.2
==========

Prelude
-------
VQE expectation computation with Aer qasm_simulator now defaults to a
computation that has the expected shot noise behavior.

Upgrade Notes
-------------
- `cvxpy <https://github.com/cvxgrp/cvxpy/>`_ is now in the requirements list
  as a dependency for qiskit-aqua. It is used for the quadratic program solver
  which is used as part of the :class:`qiskit.aqua.algorithms.QSVM`. Previously
  ``cvxopt`` was an optional dependency that needed to be installed to use
  this functionality. This is no longer required as cvxpy will be installed
  with qiskit-aqua.
- For state tomography run as part of :class:`qiskit.aqua.algorithms.HHL` with
  a QASM backend the tomography fitter function
  :meth:`qiskit.ignis.verification.StateTomographyFitter.fit` now gets called
  explicitly with the method set to ``lstsq`` to always use the least-squares
  fitting. Previously it would opportunistically try to use the ``cvx`` fitter
  if ``cvxpy`` were installed. But, the ``cvx`` fitter depends on a
  specifically configured ``cvxpy`` installation with an SDP solver installed
  as part of ``cvxpy`` which is not always present in an environment with
  ``cvxpy`` installed.
- The VQE expectation computation using qiskit-aer's
  :class:`qiskit.providers.aer.extensions.SnapshotExpectationValue` instruction
  is not enabled by default anymore. This was changed to be the default in
  0.7.0 because it is significantly faster, but it led to unexpected ideal
  results without shot noise (see
  `#1013 <https://github.com/Qiskit/qiskit-aqua/issues/1013>`_ for more
  details). The default has now changed back to match user expectations. Using
  the faster expectation computation is now opt-in by setting the new
  ``include_custom`` kwarg to ``True`` on the
  :class:`qiskit.aqua.algorithms.VQE` constructor.

New Features
------------
- A new kwarg ``include_custom`` has been added to the constructor for
  :class:`qiskit.aqua.algorithms.VQE` and it's subclasses (mainly
  :class:`qiskit.aqua.algorithms.QAOA`). When set to true and the
  ``expectation`` kwarg is set to ``None`` (the default) this will enable
  the use of VQE expectation computation with Aer's ``qasm_simulator``
  :class:`qiskit.providers.aer.extensions.SnapshotExpectationValue` instruction.
  The special Aer snapshot based computation is much faster but with the ideal
  output similar to state vector simulator.

IBM Q Provider 0.7.2
====================

No Change

*************
Qiskit 0.19.3
*************

Terra 0.14.1
============

No Change

Aer 0.5.2
=========

Bug Fixes
---------

- Fixed bug with statevector and unitary simulators running a number of (parallel)
  shots equal to the number of CPU threads instead of only running a single shot.

- Fixes the "diagonal" qobj gate instructions being applied incorrectly
  in the density matrix Qasm Simulator method.

- Fixes bug where conditional gates were not being applied correctly
  on the density matrix simulation method.

- Fix bug in CZ gate and Z gate for "density_matrix_gpu" and
  "density_matrix_thrust" QasmSimulator methods.

- Fixes issue where memory requirements of simulation were not being checked
  on the QasmSimulator when using a non-automatic simulation method.

- Fixed a memory leak that effected the GPU simulator methods

Ignis 0.3.0
===========

No Change

Aqua 0.7.1
==========

No Change

IBM Q Provider 0.7.2
====================

Bug Fixes
---------

- :meth:`qiskit.provider.ibmq.IBMQBackend.jobs` will now return the correct
  list of :class:`~qiskit.provider.ibmq.job.IBMQJob` objects when the
  ``status`` kwarg is set to ``'RUNNING'``. Fixes
  `#523 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/523>`_

- The package metadata has been updated to properly reflect the dependency
  on ``qiskit-terra`` >= 0.14.0. This dependency was implicitly added as
  part of the 0.7.0 release but was not reflected in the package requirements
  so it was previously possible to install ``qiskit-ibmq-provider`` with a
  version of ``qiskit-terra`` which was too old. Fixes
  `#677 <https://github.com/Qiskit/qiskit-ibmq-provider/issues/677>`_

*************
Qiskit 0.19.0
*************

Terra 0.14.0
============

.. _Release Notes_0.14.0_Prelude:

Prelude
-------

The 0.14.0 release includes several new features and bug fixes. The biggest
change for this release is the introduction of a quantum circuit library
in :mod:`qiskit.circuit.library`, containing some circuit families of
interest.

The circuit library gives users access to a rich set of well-studied
circuit families, instances of which can be used as benchmarks,
as building blocks in building more complex circuits, or
as a tool to explore quantum computational advantage over classical.
The contents of this library will continue to grow and mature.

The initial release of the circuit library contains:

* ``standard_gates``: these are fixed-width gates commonly used as primitive
  building blocks, consisting of 1, 2, and 3 qubit gates. For example
  the :class:`~qiskit.circuit.library.XGate`,
  :class:`~qiskit.circuit.library.RZZGate` and
  :class:`~qiskit.circuit.library.CSWAPGate`. The old location of these
  gates under ``qiskit.extensions.standard`` is deprecated.
* ``generalized_gates``: these are families that can generalize to arbitrarily
  many qubits, for example a :class:`~qiskit.circuit.library.Permutation` or
  :class:`~qiskit.circuit.library.GMS` (Global Molmer-Sorensen gate).
* ``boolean_logic``: circuits that transform basis states according to simple
  Boolean logic functions, such as :class:`~qiskit.circuit.library.ADD` or
  :class:`~qiskit.circuit.library.XOR`.
* ``arithmetic``: a set of circuits for doing classical arithmetic such as
  :class:`~qiskit.circuit.library.WeightedAdder` and
  :class:`~qiskit.circuit.library.IntegerComparator`.
* ``basis_changes``: circuits such as the quantum Fourier transform,
  :class:`~qiskit.circuit.library.QFT`, that mathematically apply basis
  changes.
* ``n_local``: patterns to easily create large circuits with rotation and
  entanglement layers, such as  :class:`~qiskit.circuit.library.TwoLocal`
  which uses single-qubit rotations and two-qubit entanglements.
* ``data_preparation``: circuits that take classical input data and encode it
  in a quantum state that is difficult to simulate, e.g.
  :class:`~qiskit.circuit.library.PauliFeatureMap` or
  :class:`~qiskit.circuit.library.ZZFeatureMap`.
* Other circuits that have proven interesting in the literature, such as
  :class:`~qiskit.circuit.library.QuantumVolume`,
  :class:`~qiskit.circuit.library.GraphState`, or
  :class:`~qiskit.circuit.library.IQP`.

To allow easier use of these circuits as building blocks, we have introduced
a :meth:`~qiskit.circuit.QuantumCircuit.compose` method of
:class:`qiskit.circuit.QuantumCircuit` for composition of circuits either
with other circuits (by welding them at the ends and optionally permuting
wires) or with other simpler gates::

  >>> lhs.compose(rhs, qubits=[3, 2], inplace=True)

.. parsed-literal::
                  ┌───┐                   ┌─────┐                ┌───┐
      lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────
                  ├───┤              ┌─┴─┐└─────┘                ├───┤
      lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────
               ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐
      lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────
               └─────────┘                                    └─────────┘└─┬─┘┌─────┐
      lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├
                  ┌─┴─┐                                          ┌─┴─┐        └─────┘
      lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────
                  └───┘                                          └───┘
      lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════
      lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════

With this, Qiskit's circuits no longer assume an implicit
initial state of :math:`|0\rangle`, and will not be drawn with this
initial state. The all-zero initial state is still assumed on a backend
when a circuit is executed.


.. _Release Notes_0.14.0_New Features:

New Features
------------

- A new method, :meth:`~qiskit.circuit.EquivalenceLibrary.has_entry`, has been
  added to the :class:`qiskit.circuit.EquivalenceLibrary` class to quickly
  check if a given gate has any known decompositions in the library.

- A new class :class:`~qiskit.circuit.library.IQP`, to construct an
  instantaneous quantum polynomial circuit, has been added to the circuit
  library module :mod:`qiskit.circuit.library`.

- A new :meth:`~qiskit.circuit.QuantumCircuit.compose` method has been added
  to :class:`qiskit.circuit.QuantumCircuit`. It allows
  composition of two quantum circuits without having to turn one into
  a gate or instruction. It also allows permutations of qubits/clbits
  at the point of composition, as well as optional inplace modification.
  It can also be used in place of
  :meth:`~qiskit.circuit.QuantumCircuit.append()`, as it allows
  composing instructions and operators onto the circuit as well.

- :class:`qiskit.circuit.library.Diagonal` circuits have been added to the
  circuit library. These circuits implement diagonal quantum operators
  (consisting of non-zero elements only on the diagonal). They are more
  efficiently simulated by the Aer simulator than dense matrices.

- Add :meth:`~qiskit.quantum_info.Clifford.from_label` method to the
  :class:`qiskit.quantum_info.Clifford` class for initializing as the
  tensor product of single-qubit I, X, Y, Z, H, or S gates.

- Schedule transformer :func:`qiskit.pulse.reschedule.compress_pulses`
  performs an optimization pass to reduce the usage of waveform
  memory in hardware by replacing multiple identical instances of
  a pulse in a pulse schedule with a single pulse.
  For example::

      from qiskit.pulse import reschedule

      schedules = []
      for _ in range(2):
          schedule = Schedule()
          drive_channel = DriveChannel(0)
          schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)
          schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)
          schedules.append(schedule)

      compressed_schedules = reschedule.compress_pulses(schedules)

- The :class:`qiskit.transpiler.Layout` has a new method
  :meth:`~qiskit.transpiler.Layout.reorder_bits` that is used to reorder a
  list of virtual qubits based on the layout object.

- Two new methods have been added to the
  :class:`qiskit.providers.models.PulseBackendConfiguration` for
  interacting with channels.

  * :meth:`~qiskit.providers.models.PulseBackendConfiguration.get_channel_qubits`
    to get a list of all qubits operated by the given channel and
  * :meth:`~qiskit.providers.models.PulseBackendConfiguration.get_qubit_channel`
    to get a list of channels operating on the given qubit.

- New :class:`qiskit.extensions.HamiltonianGate` and
  :meth:`qiskit.circuit.QuantumCircuit.hamiltonian()` methods are
  introduced, representing Hamiltonian evolution of the circuit
  wavefunction by a user-specified Hermitian Operator and evolution time.
  The evolution time can be a :class:`~qiskit.circuit.Parameter`, allowing
  the creation of parameterized UCCSD or QAOA-style circuits which compile to
  ``UnitaryGate`` objects if ``time`` parameters are provided. The Unitary of
  a ``HamiltonianGate`` with Hamiltonian Operator ``H`` and time parameter
  ``t`` is :math:`e^{-iHt}`.

- The circuit library module :mod:`qiskit.circuit.library` now provides a
  new boolean logic AND circuit, :class:`qiskit.circuit.library.AND`, and
  OR circuit, :class:`qiskit.circuit.library.OR`, which implement the
  respective operations on a variable number of provided qubits.

- New fake backends are added under :mod:`qiskit.test.mock`. These include
  mocked versions of ``ibmq_armonk``, ``ibmq_essex``, ``ibmq_london``,
  ``ibmq_valencia``, ``ibmq_cambridge``, ``ibmq_paris``, ``ibmq_rome``, and
  ``ibmq_athens``. As with other fake backends, these include snapshots of
  calibration data (i.e. ``backend.defaults()``) and error data (i.e.
  ``backend.properties()``) taken from the real system, and can be used for
  local testing, compilation and simulation.

- The ``last_update_date`` parameter for
  :class:`~qiskit.providers.models.BackendProperties` can now also be
  passed in as a ``datetime`` object. Previously only a string in
  ISO8601 format was accepted.

- Adds :meth:`qiskit.quantum_info.Statevector.from_int` and
  :meth:`qiskit.quantum_info.DensityMatrix.from_int` methods that allow
  constructing a computational basis state for specified system dimensions.

- The methods on the :class:`qiskit.circuit.QuantumCircuit` class for adding
  gates (for example :meth:`~qiskit.circuit.QuantumCircuit.h`) which were
  previously added dynamically at run time to the class definition have been
  refactored to be statically defined methods of the class. This means that
  static analyzer (such as IDEs) can now read these methods.


.. _Release Notes_0.14.0_Upgrade Notes:

Upgrade Notes
-------------

- A new package,
  `python-dateutil <https://pypi.org/project/python-dateutil/>`_, is now
  required and has been added to the requirements list. It is being used
  to parse datetime strings received from external providers in
  :class:`~qiskit.providers.models.BackendProperties` objects.

- The marshmallow schema classes in :mod:`qiskit.providers.models` have been
  removed since they are no longer used by the BackendObjects.

- The output of the ``to_dict()`` method for the classes in
  :mod:`qiskit.providers.models` is no longer in a format for direct JSON
  serialization. Depending on the content contained in instances of these
  class there may be numpy arrays and/or complex numbers in the fields of the dict.
  If you're JSON serializing the output of the to_dict methods you should
  ensure your JSON encoder can handle numpy arrays and complex numbers. This
  includes:

  * :meth:`qiskit.providers.models.BackendConfiguration.to_dict`
  * :meth:`qiskit.providers.models.BackendProperties.to_dict`
  * :meth:`qiskit.providers.models.BackendStatus.to_dict`
  * :meth:`qiskit.providers.models.QasmBackendConfiguration.to_dict`
  * :meth:`qiskit.providers.models.PulseBackendConfiguration.to_dict`
  * :meth:`qiskit.providers.models.UchannelLO.to_dict`
  * :meth:`qiskit.providers.models.GateConfig.to_dict`
  * :meth:`qiskit.providers.models.PulseDefaults.to_dict`
  * :meth:`qiskit.providers.models.Command.to_dict`
  * :meth:`qiskit.providers.models.JobStatus.to_dict`
  * :meth:`qiskit.providers.models.Nduv.to_dict`
  * :meth:`qiskit.providers.models.Gate.to_dict`


.. _Release Notes_0.14.0_Deprecation Notes:

Deprecation Notes
-----------------

- The :meth:`qiskit.dagcircuit.DAGCircuit.compose` method now takes a list
  of qubits/clbits that specify the positional order of bits to compose onto.
  The dictionary-based method of mapping using the ``edge_map`` argument is
  deprecated and will be removed in a future release.

- The ``combine_into_edge_map()`` method for the
  :class:`qiskit.transpiler.Layout` class has been deprecated and will be
  removed in a future release. Instead, the new method
  :meth:`~qiskit.transpiler.Layout.reorder_bits` should be used to reorder
  a list of virtual qubits according to the layout object.

- Passing a :class:`qiskit.pulse.ControlChannel` object in via the
  parameter ``channel`` for the
  :class:`qiskit.providers.models.PulseBackendConfiguration` method
  :meth:`~qiskit.providers.models.PulseBackendConfiguration.control` has been
  deprecated and will be removed in a future release. The
  ``ControlChannel`` objects are now generated from the backend configuration
  ``channels`` attribute which has the information of all channels and the
  qubits they operate on. Now, the method
  :meth:`~qiskit.providers.models.PulseBackendConfiguration.control`
  is expected to take the parameter ``qubits`` of the form
  ``(control_qubit, target_qubit)`` and type ``list``
  or ``tuple``, and returns a list of control channels.

- The ``AND`` and ``OR`` methods of :class:`qiskit.circuit.QuantumCircuit`
  are deprecated and will be removed in a future release. Instead you should
  use the circuit library boolean logic classes
  :class:`qiskit.circuit.library.AND` amd :class:`qiskit.circuit.library.OR`
  and then append those objects to your class. For example::

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import AND

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qc_and = AND(2)

    qc.compose(qc_and, inplace=True)

- The ``qiskit.extensions.standard`` module is deprecated and will be
  removed in a future release. The gate classes in that module have been
  moved to :mod:`qiskit.circuit.library.standard_gates`.


.. _Release Notes_0.14.0_Bug Fixes:

Bug Fixes
---------

- The :class:`qiskit.circuit.QuantumCircuit` methods
  :meth:`~qiskit.circuit.QuantumCircuit.inverse`,
  :meth:`~qiskit.circuit.QuantumCircuit.mirror` methods, as well as
  the ``QuantumCircuit.data`` setter would generate an invalid circuit when
  used on a parameterized circuit instance. This has been resolved and
  these methods should now work with a parameterized circuit. Fixes
  `#4235 <https://github.com/Qiskit/qiskit-terra/issues/4235>`_

- Previously when creating a controlled version of a standard qiskit
  gate if a ``ctrl_state`` was specified a generic ``ControlledGate``
  object would be returned whereas without it a standard qiskit
  controlled gate would be returned if it was defined. This PR
  allows standard qiskit controlled gates to understand
  ``ctrl_state``.

  Additionally, this PR fixes what might be considered a bug where
  setting the ``ctrl_state`` of an already controlled gate would
  assume the specified state applied to the full control width
  instead of the control qubits being added. For instance,::

    circ = QuantumCircuit(2)
    circ.h(0)
    circ.x(1)
    gate = circ.to_gate()
    cgate = gate.control(1)
    c3gate = cgate.control(2, ctrl_state=0)

  would apply ``ctrl_state`` to all three control qubits instead of just
  the two control qubits being added.

- Fixed a bug in :func:`~qiskit.quantum_info.random_clifford` that stopped it
  from sampling the full Clifford group. Fixes
  `#4271 <https://github.com/Qiskit/qiskit-terra/issues/4271>`_

- The :class:`qiskit.circuit.Instruction` method
  :meth:`qiskit.circuit.Instruction.is_parameterized` method had previously
  returned ``True`` for any ``Instruction`` instance which had a
  :class:`qiskit.circuit.Parameter` in any element of its ``params`` array,
  even if that ``Parameter`` had been fully bound. This has been corrected so
  that ``.is_parameterized`` will return ``False`` when the instruction is
  fully bound.

- :meth:`qiskit.circuit.ParameterExpression.subs` had not correctly detected
  some cases where substituting parameters would result in a two distinct
  :class:`~qiskit.circuit.Parameters` objects in an expression with the same
  name. This has been corrected so a ``CircuitError`` will be raised in these
  cases.

- Improve performance of :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` for low-qubit circuit
  simulations by optimizing the class ``__init__`` methods. Fixes
  `#4281 <https://github.com/Qiskit/qiskit-terra/issues/4281>`_

- The function :func:`qiskit.compiler.transpile` now correctly handles when
  the parameter ``basis_gates`` is set to ``None``. This will allow any gate
  in the output tranpiled circuit, including gates added by the transpilation
  process. Note that using this parameter may have some
  unintended consequences during optimization. Some transpiler passes
  depend on having a ``basis_gates`` set. For example,
  :class:`qiskit.transpiler.passes.Optimize1qGates` only optimizes the chains
  of u1, u2, and u3 gates and without ``basis_gates`` it is unable to unroll
  gates that otherwise could be optimized:

  .. code-block:: python

    from qiskit import *

    q = QuantumRegister(1, name='q')
    circuit = QuantumCircuit(q)
    circuit.h(q[0])
    circuit.u1(0.1, q[0])
    circuit.u2(0.1, 0.2, q[0])
    circuit.h(q[0])
    circuit.u3(0.1, 0.2, 0.3, q[0])

    result = transpile(circuit, basis_gates=None, optimization_level=3)
    result.draw()

  .. parsed-literal::
        ┌───┐┌─────────────┐┌───┐┌─────────────────┐
   q_0: ┤ H ├┤ U2(0.1,0.3) ├┤ H ├┤ U3(0.1,0.2,0.3) ├
        └───┘└─────────────┘└───┘└─────────────────┘

  Fixes `#3017 <https://github.com/Qiskit/qiskit-terra/issues/3017>`_


.. _Release Notes_0.14.0_Other Notes:

Other Notes
-----------

- The objects in :mod:`qiskit.providers.models` which were previously
  constructed using the marshmallow library have been refactored to not
  depend on marshmallow. This includes:

  * :class:`~qiskit.providers.models.BackendConfiguration`
  * :class:`~qiskit.providers.models.BackendProperties`
  * :class:`~qiskit.providers.models.BackendStatus`
  * :class:`~qiskit.providers.models.QasmBackendConfiguration`
  * :class:`~qiskit.providers.models.PulseBackendConfiguration`
  * :class:`~qiskit.providers.models.UchannelLO`
  * :class:`~qiskit.providers.models.GateConfig`
  * :class:`~qiskit.providers.models.PulseDefaults`
  * :class:`~qiskit.providers.models.Command`
  * :class:`~qiskit.providers.models.JobStatus`
  * :class:`~qiskit.providers.models.Nduv`
  * :class:`~qiskit.providers.models.Gate`

  These should be drop-in replacements without any noticeable change but
  specifics inherited from marshmallow may not work. Please file issues for
  any incompatibilities found.

Aer 0.5.1
=========

No Change


Ignis 0.3.0
===========

No Change

Aqua 0.7.0
==========

Prelude
-------

The Qiskit Aqua 0.7.0 release introduces a lot of new functionality along
with an improved integration with :class:`qiskit.circuit.QuantumCircuit`
objects. The central contributions are the Qiskit's optimization module,
a complete refactor on Operators, using circuits as native input for the
algorithms and removal of the declarative JSON API.

Optimization module
^^^^^^^^^^^^^^^^^^^
The :mod:`qiskit.optimization`` module now offers functionality for modeling
and solving quadratic programs. It provides various near-term quantum and
conventional algorithms, such as the ``MinimumEigenOptimizer``
(covering e.g. ``VQE`` or ``QAOA``) or ``CplexOptimizer``, as well as
a set of converters to translate between different
problem representations, such as ``QuadraticProgramToQubo``.
See the
`changelog <https://github.com/Qiskit/qiskit-aqua/blob/master/CHANGELOG.md>`_
for a list of the added features.

Operator flow
^^^^^^^^^^^^^
The operator logic provided in :mod:`qiskit.aqua.operators`` was completely
refactored and is now a full set of tools for constructing
physically-intuitive quantum computations. It contains state functions,
operators and measurements and internally relies on Terra's Operator
objects. Computing expectation values and evolutions was heavily simplified
and objects like the ``ExpectationFactory`` produce the suitable, most
efficient expectation algorithm based on the Operator input type.
See the `changelog <https://github.com/Qiskit/qiskit-aqua/blob/master/CHANGELOG.md>`_
for a overview of the added functionality.

Native circuits
^^^^^^^^^^^^^^^
Algorithms commonly use parameterized circuits as input, for example the
VQE, VQC or QSVM. Previously, these inputs had to be of type
``VariationalForm`` or ``FeatureMap`` which were wrapping the circuit
object. Now circuits are natively supported in these algorithms, which
means any individually constructed ``QuantumCircuit`` can be passed to
these algorithms. In combination with the release of the circuit library
which offers a wide collection of circuit families, it is now easy to
construct elaborate circuits as algorithm input.

Declarative JSON API
^^^^^^^^^^^^^^^^^^^^
The ability of running algorithms using dictionaries as parameters as well
as using the Aqua interfaces GUI has been removed.


IBM Q Provider 0.7.0
====================

.. _Release Notes_0.7.0_New Features:

New Features
------------

- A new exception, :class:`qiskit.providers.ibmq.IBMQBackendJobLimitError`,
  is now raised if a job could not be submitted because the limit on active
  jobs has been reached.

- :class:`qiskit.providers.ibmq.job.IBMQJob` and
  :class:`qiskit.providers.ibmq.managed.ManagedJobSet` each has two new methods
  ``update_name`` and ``update_tags``.
  They are used to change the name and tags of a job or a job set, respectively.

- :meth:`qiskit.providers.ibmq.IBMQFactory.save_account` and
  :meth:`qiskit.providers.ibmq.IBMQFactory.enable_account` now accept optional
  parameters ``hub``, ``group``, and ``project``, which allow specifying a default
  provider to save to disk or use, respectively.


.. _Release Notes_0.7.0_Upgrade Notes:

Upgrade Notes
-------------

- The :class:`qiskit.providers.ibmq.job.IBMQJob` methods ``creation_date`` and
  ``time_per_step`` now return date time information as a ``datetime`` object in
  local time instead of UTC. Similarly, the parameters ``start_datetime`` and
  ``end_datetime``, of
  :meth:`qiskit.providers.ibmq.IBMQBackendService.jobs` and
  :meth:`qiskit.providers.ibmq.IBMQBackend.jobs` can now be specified in local time.

- The :meth:`qiskit.providers.ibmq.job.QueueInfo.format` method now uses a custom
  ``datetime`` to string formatter, and the package
  `arrow <https://pypi.org/project/arrow/>`_ is no longer required and has been
  removed from the requirements list.


.. _Release Notes_0.7.0_Deprecation Notes:

Deprecation Notes
-----------------

- The :meth:`~qiskit.providers.ibmq.job.IBMQJob.from_dict` and
  :meth:`~qiskit.providers.ibmq.job.IBMQJob.to_dict` methods of
  :class:`qiskit.providers.ibmq.job.IBMQJob` are deprecated and will be removed in
  the next release.


.. _Release Notes_0.7.0_Bug Fixes:

Bug Fixes
---------

- Fixed an issue where ``nest_asyncio.apply()`` may raise an exception if there is
  no asyncio loop due to threading.


*************
Qiskit 0.18.3
*************

Terra 0.13.0
============

No Change

Aer 0.5.1
==========

.. _Release Notes_0.5.1_Upgrade Notes:

Upgrade Notes
-------------

- Changes how transpilation passes are handled in the C++ Controller classes
  so that each pass must be explicitly called. This allows for greater
  customization on when each pass should be called, and with what parameters.
  In particular this enables setting different parameters for the gate
  fusion optimization pass depending on the QasmController simulation method.

- Add ``gate_length_units`` kwarg to
  :meth:`qiskit.providers.aer.noise.NoiseModel.from_device`
  for specifying custom ``gate_lengths`` in the device noise model function
  to handle unit conversions for internal code.

- Add Controlled-Y ("cy") gate to the Stabilizer simulator methods supported
  gateset.

- For Aer's backend the jsonschema validation of input qobj objects from
  terra is now opt-in instead of being enabled by default. If you want
  to enable jsonschema validation of qobj set the ``validate`` kwarg on
  the :meth:`qiskit.providers.aer.QasmSimualtor.run` method for the backend
  object to ``True``.


.. _Release Notes_0.5.1_Bug Fixes:

Bug Fixes
---------

- Remove "extended_stabilizer" from the automatically selected simulation
  methods. This is needed as the extended stabilizer method is not exact
  and may give incorrect results for certain circuits unless the user
  knows how to optimize its configuration parameters.

  The automatic method now only selects from "stabilizer", "density_matrix",
  and "statevector" methods. If a non-Clifford circuit that is too large for
  the statevector method is executed an exception will be raised suggesting
  you could try explicitly using the "extended_stabilizer" or
  "matrix_product_state" methods instead.

- Fixes Controller classes so that the ReduceBarrier transpilation pass is
  applied first. This prevents barrier instructions from preventing truncation
  of unused qubits if the only instruction defined on them was a barrier.

- Disables gate fusion for the matrix product state simulation method as this
  was causing issues with incorrect results being returned in some cases.

- Fix error in gate time unit conversion for device noise model with thermal
  relaxation errors and gate errors. The error probability the depolarizing
  error was being  calculated with gate time in microseconds, while for
  thermal relaxation it was being calculated in nanoseconds. This resulted
  in no depolarizing error being applied as the incorrect units would make
  the device seem to be coherence limited.

- Fix bug in incorrect composition of QuantumErrors when the qubits of
  composed instructions differ.

- Fix issue where the "diagonal" gate is checked to be unitary with too
  high a tolerance. This was causing diagonals generated from Numpy functions
  to often fail the test.

- Fix remove-barrier circuit optimization pass to be applied before qubit
  trucation. This fixes an issue where barriers inserted by the Terra
  transpiler across otherwise inactive qubits would prevent them from being
  truncated.

Ignis 0.3.0
===========

No Change


Aqua 0.6.6
==========

No Change


IBM Q Provider 0.6.1
====================

No Change


*************
Qiskit 0.18.0
*************

.. _Release Notes_0.13.0:

Terra 0.13.0
============

.. _Release Notes_0.13.0_Prelude:

Prelude
-------

The 0.13.0 release includes many big changes. Some highlights for this
release are:

For the transpiler we have switched the graph library used to build the
:class:`qiskit.dagcircuit.DAGCircuit` class which is the underlying data
structure behind all operations to be based on
`retworkx <https://pypi.org/project/retworkx/>`_ for greatly improved
performance. Circuit transpilation speed in the 0.13.0 release should
be significanlty faster than in previous releases.

There has been a significant simplification to the style in which Pulse
instructions are built. Now, ``Command`` s are deprecated and a unified
set of :class:`~qiskit.pulse.instructions.Instruction` s are supported.

The :mod:`qiskit.quantum_info` module includes several new functions
for generating random operators (such as Cliffords and quantum channels)
and for computing the diamond norm of quantum channels; upgrades to the
:class:`~qiskit.quantum_info.Statevector` and
:class:`~qiskit.quantum_info.DensityMatrix` classes to support
computing measurement probabilities and sampling measurements; and several
new classes are based on the symplectic representation
of Pauli matrices. These new classes include Clifford operators
(:class:`~qiskit.quantum_info.Clifford`), N-qubit matrices that are
sparse in the Pauli basis (:class:`~qiskit.quantum_info.SparsePauliOp`),
lists of Pauli's (:class:`~qiskit.quantum_info.PauliTable`),
and lists of stabilizers (:class:`~qiskit.quantum_info.StabilizerTable`).

This release also has vastly improved documentation across Qiskit,
including improved documentation for the :mod:`qiskit.circuit`,
:mod:`qiskit.pulse` and :mod:`qiskit.quantum_info` modules.

Additionally, the naming of gate objects and
:class:`~qiskit.circuit.QuantumCircuit` methods have been updated to be
more consistent. This has resulted in several classes and methods being
deprecated as things move to a more consistent naming scheme.

For full details on all the changes made in this release see the detailed
release notes below.


.. _Release Notes_0.13.0_New Features:

New Features
------------

- Added a new circuit library module :mod:`qiskit.circuit.library`. This will
  be a place for constructors of commonly used circuits that can be used as
  building blocks for larger circuits or applications.

- The :class:`qiskit.providers.BaseJob` class has four new methods:

  * :meth:`~qiskit.providers.BaseJob.done`
  * :meth:`~qiskit.providers.BaseJob.running`
  * :meth:`~qiskit.providers.BaseJob.cancelled`
  * :meth:`~qiskit.providers.BaseJob.in_final_state`

  These methods are used to check wheter a job is in a given job status.

- Add ability to specify control conditioned on a qubit being in the
  ground state. The state of the control qubits is represented by an
  integer. For example::

    from qiskit import QuantumCircuit
    from qiskit.extensions.standard import XGate

    qc = QuantumCircuit(4)
    cgate = XGate().control(3, ctrl_state=6)
    qc.append(cgate, [0, 1, 2, 3])

  Creates a four qubit gate where the fourth qubit gets flipped if
  the first qubit is in the ground state and the second and third
  qubits are in the excited state. If ``ctrl_state`` is ``None``, the
  default, control is conditioned on all control qubits being
  excited.

- A new jupyter widget, ``%circuit_library_info`` has been added to
  :mod:`qiskit.tools.jupyter`. This widget is used for visualizing
  details about circuits built from the circuit library. For example

  .. jupyter-execute::

      from qiskit.circuit.library import XOR
      import qiskit.tools.jupyter
      circuit = XOR(5, seed=42)
      %circuit_library_info circuit

- A new kwarg option, ``formatted`` ,  has been added to
  :meth:`qiskit.circuit.QuantumCircuit.qasm` . When set to ``True`` the
  method will print a syntax highlighted version (using pygments) to
  stdout and return ``None`` (which differs from the normal behavior of
  returning the QASM code as a string).

- A new kwarg option, ``filename`` , has been added to
  :meth:`qiskit.circuit.QuantumCircuit.qasm`. When set to a path the method
  will write the QASM code to that file. It will then continue to output as
  normal.

- A new instruction :py:class:`~qiskit.pulse.SetFrequency` which allows users
  to change the frequency of the :class:`~qiskit.pulse.PulseChannel`. This is
  done in the following way::

      from qiskit.pulse import Schedule
      from qiskit.pulse import SetFrequency

      sched = pulse.Schedule()
      sched += SetFrequency(5.5e9, DriveChannel(0))

  In this example, the frequency of all pulses before the ``SetFrequency``
  command will be the default frequency and all pulses applied to drive
  channel zero after the ``SetFrequency`` command will be at 5.5 GHz. Users
  of ``SetFrequency`` should keep in mind any hardware limitations.

- A new method, :meth:`~qiskit.circuit.QuantumCircuit.assign_parameters`
  has been added to the :class:`qiskit.circuit.QuantumCircuit` class. This
  method accepts a parameter dictionary with both floats and Parameters
  objects in a single dictionary. In other words this new method allows you
  to bind floats, Parameters or both in a single dictionary.

  Also, by using the ``inplace`` kwarg it can be specified you can optionally
  modify the original circuit in place. By default this is set to ``False``
  and a copy of the original circuit will be returned from the method.

- A new method :meth:`~qiskit.circuit.QuantumCircuit.num_nonlocal_gates`
  has been added to the :class:`qiskit.circuit.QuantumCircuit` class.
  This method will return the number of gates in a circuit that involve 2 or
  or more qubits. These gates are more costly in terms of time and error to
  implement.

- The :class:`qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.iso` for adding an
  :class:`~qiskit.extensions.Isometry` gate to the circuit has a new alias. You
  can now call :meth:`qiskit.circuit.QuantumCircuit.isometry` in addition to
  calling ``iso``.

- A ``description`` attribute has been added to the
  :class:`~qiskit.transpiler.CouplingMap` class for storing a short
  description for different coupling maps (e.g. full, grid, line, etc.).

- A new method :meth:`~qiskit.dagcircuit.DAGCircuit.compose` has been added to
  the :class:`~qiskit.dagcircuit.DAGCircuit` class for composing two circuits
  via their DAGs.

  .. code-block:: python

      dag_left.compose(dag_right, edge_map={right_qubit0: self.left_qubit1,
                                        right_qubit1: self.left_qubit4,
                                        right_clbit0: self.left_clbit1,
                                        right_clbit1: self.left_clbit0})

  .. parsed-literal::

                  ┌───┐                    ┌─────┐┌─┐
      lqr_1_0: ───┤ H ├───     rqr_0: ──■──┤ Tdg ├┤M├
                  ├───┤               ┌─┴─┐└─┬─┬─┘└╥┘
      lqr_1_1: ───┤ X ├───     rqr_1: ┤ X ├──┤M├───╫─
               ┌──┴───┴──┐            └───┘  └╥┘   ║
      lqr_1_2: ┤ U1(0.1) ├  +  rcr_0: ════════╬════╩═  =
               └─────────┘                    ║
      lqr_2_0: ─────■─────     rcr_1: ════════╩══════
                  ┌─┴─┐
      lqr_2_1: ───┤ X ├───
                  └───┘
      lcr_0:   ═══════════

      lcr_1:   ═══════════

                  ┌───┐
      lqr_1_0: ───┤ H ├──────────────────
                  ├───┤        ┌─────┐┌─┐
      lqr_1_1: ───┤ X ├─────■──┤ Tdg ├┤M├
               ┌──┴───┴──┐  │  └─────┘└╥┘
      lqr_1_2: ┤ U1(0.1) ├──┼──────────╫─
               └─────────┘  │          ║
      lqr_2_0: ─────■───────┼──────────╫─
                  ┌─┴─┐   ┌─┴─┐  ┌─┐   ║
      lqr_2_1: ───┤ X ├───┤ X ├──┤M├───╫─
                  └───┘   └───┘  └╥┘   ║
      lcr_0:   ═══════════════════╩════╬═
                                       ║
      lcr_1:   ════════════════════════╩═

- The mock backends in ``qiskit.test.mock`` now have a functional ``run()``
  method that will return results similar to the real devices. If
  ``qiskit-aer`` is installed a simulation will be run with a noise model
  built from the device snapshot in the fake backend.  Otherwise,
  :class:`qiskit.providers.basicaer.QasmSimulatorPy` will be used to run an
  ideal simulation. Additionally, if a pulse experiment is passed to ``run``
  and qiskit-aer is installed the ``PulseSimulator`` will be used to simulate
  the pulse schedules.

- The :meth:`qiskit.result.Result` method
  :meth:`~qiskit.result.Result.get_counts` will now return a list of all the
  counts available when there are multiple circuits in a job. This works when
  ``get_counts()`` is called with no arguments.

  The main consideration for this feature was for drawing all the results
  from multiple circuits in the same histogram. For example it is now
  possible to do something like:

  .. jupyter-execute::

      from qiskit import execute
      from qiskit import QuantumCircuit
      from qiskit.providers.basicaer import BasicAer
      from qiskit.visualization import plot_histogram

      sim = BasicAer.get_backend('qasm_simulator')

      qc = QuantumCircuit(2)
      qc.h(0)
      qc.cx(0, 1)
      qc.measure_all()
      result = execute([qc, qc, qc], sim).result()

      plot_histogram(result.get_counts())

- A new kwarg, ``initial_state`` has been added to the
  :func:`qiskit.visualization.circuit_drawer` function and the
  :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.draw`. When set to ``True`` the
  initial state will be included in circuit visualizations for all backends.
  For example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit

      circuit = QuantumCircuit(2)
      circuit.measure_all()
      circuit.draw(output='mpl', initial_state=True)

- It is now possible to insert a callable into a :class:`qiskit.pulse.InstructionScheduleMap`
  which returns a new :class:`qiskit.pulse.Schedule` when it is called with parameters.
  For example:

  .. code-block::

     def test_func(x):
        sched = Schedule()
        sched += pulse_lib.constant(int(x), amp_test)(DriveChannel(0))
        return sched

     inst_map = InstructionScheduleMap()
     inst_map.add('f', (0,), test_func)
     output_sched = inst_map.get('f', (0,), 10)
     assert output_sched.duration == 10

- Two new gate classes, :class:`qiskit.extensions.iSwapGate` and
  :class:`qiskit.extensions.DCXGate`, along with their
  :class:`~qiskit.circuit.QuantumCircuit` methods
  :meth:`~qiskit.circuit.QuantumCircuit.iswap` and
  :meth:`~qiskit.circuit.QuantumCircuit.dcx` have been added to the standard
  extensions. These gates, which are locally equivalent to each other, can be
  used to enact particular XY interactions. A brief motivation for these gates
  can be found in:
  `arxiv.org/abs/quant-ph/0209035 <https://arxiv.org/abs/quant-ph/0209035>`_

- The :class:`qiskit.providers.BaseJob` class now has a new method
  :meth:`~qiskit.providers.BaseJob.wait_for_final_state` that polls for the
  job status until the job reaches a final state (such as ``DONE`` or
  ``ERROR``). This method also takes an optional ``callback`` kwarg which
  takes a Python callable that will be called during each iteration of the
  poll loop.

- The ``search_width`` and ``search_depth`` attributes of the
  :class:`qiskit.transpiler.passes.LookaheadSwap` pass are now settable when
  initializing the pass. A larger search space can often lead to more
  optimized circuits, at the cost of longer run time.

- The number of qubits in
  :class:`~qiskit.providers.models.BackendConfiguration` can now be accessed
  via the property
  :py:attr:`~qiskit.providers.models.BackendConfiguration.num_qubits`. It
  was previously only accessible via the ``n_qubits`` attribute.

- Two new methods, :meth:`~qiskit.quantum_info.OneQubitEulerDecomposer.angles`
  and :meth:`~qiskit.quantum_info.OneQubitEulerDecomposer.angles_and_phase`,
  have been added to the :class:`qiskit.quantum_info.OneQubitEulerDecomposer`
  class. These methods will return the relevant parameters without
  validation, and calling the ``OneQubitEulerDecomposer`` object will
  perform the full synthesis with validation.

- An ``RR`` decomposition basis has been added to the
  :class:`qiskit.quantum_info.OneQubitEulerDecomposer` for decomposing an
  arbitrary 2x2 unitary into a two :class:`~qiskit.extensions.RGate`
  circuit.

- Adds the ability to set ``qargs`` to objects which are subclasses
  of the abstract ``BaseOperator`` class. This is done by calling the
  object ``op(qargs)`` (where ``op`` is an operator class) and will return
  a shallow copy of the original object with a qargs property set. When
  such an object is used with the
  :meth:`~qiskit.quantum_info.Operator.compose` or
  :meth:`~qiskit.quantum_info.Operator.dot` methods the internal value for
  qargs will be used when the ``qargs`` method kwarg is not used. This
  allows for subsystem composition using binary operators, for example::

      from qiskit.quantum_info import Operator

      init = Operator.from_label('III')
      x = Operator.from_label('X')
      h = Operator.from_label('H')
      init @ x([0]) @ h([1])

- Adds :class:`qiskit.quantum_info.Clifford` operator class to the
  `quantum_info` module. This operator is an efficient symplectic
  representation an N-qubit unitary operator from the Clifford group. This
  class includes a :meth:`~qiskit.quantum_info.Clifford.to_circuit` method
  for compilation into a :class:`~qiskit.QuantumCircuit` of Clifford gates
  with a minimal number of CX gates for up to 3-qubits. It also providers
  general compilation for N > 3 qubits but this method is not optimal in
  the number of two-qubit gates.

- Adds :class:`qiskit.quantum_info.SparsePauliOp` operator class. This is an
  efficient representaiton of an N-qubit matrix that is sparse in the Pauli
  basis and uses a :class:`qiskit.quantum_info.PauliTable` and vector of
  complex coefficients for its data structure.

  This class supports much of the same functionality of the
  :class:`qiskit.quantum_info.Operator` class so
  :class:`~qiskit.quantum_info.SparsePauliOp` objects can be tensored,
  composed, scalar multiplied, added and subtracted.

  Numpy arrays or :class:`~qiskit.quantum_info.Operator` objects can be
  converted to a :class:`~qiskit.quantum_info.SparsePauliOp` using the
  `:class:`~qiskit.quantum_info.SparsePauliOp.from_operator` method.
  :class:`~qiskit.quantum_info.SparsePauliOp` can be convered to a sparse
  csr_matrix or dense Numpy array using the
  :class:`~qiskit.quantum_info.SparsePauliOp.to_matrix` method, or to an
  :class:`~qiskit.quantum_info.Operator` object using the
  :class:`~qiskit.quantum_info.SparsePauliOp.to_operator` method.

  A :class:`~qiskit.quantum_info.SparsePauliOp` can be iterated over
  in terms of its :class:`~qiskit.quantum_info.PauliTable` components and
  coefficients, its coefficients and Pauli string labels using the
  :meth:`~qiskit.quantum_info.SparsePauliOp.label_iter` method, and the
  (dense or sparse) matrix components using the
  :meth:`~qiskit.quantum_info.SparsePauliOp.matrix_iter` method.

- Add :meth:`qiskit.quantum_info.diamond_norm` function for computing the
  diamond norm (completely-bounded trace-norm) of a quantum channel. This
  can be used to compute the distance between two quantum channels using
  ``diamond_norm(chan1 - chan2)``.

- A new class :class:`qiskit.quantum_info.PauliTable` has been added. This
  is an efficient symplectic representation of a list of N-qubit Pauli
  operators. Some features of this class are:

    * :class:`~qiskit.quantum_info.PauliTable` objects may be composed, and
      tensored which will return a :class:`~qiskit.quantum_info.PauliTable`
      object with the combination of the operation (
      :meth:`~qiskit.quantum_info.PauliTable.compose`,
      :meth:`~qiskit.quantum_info.PauliTable.dot`,
      :meth:`~qiskit.quantum_info.PauliTable.expand`,
      :meth:`~qiskit.quantum_info.PauliTable.tensor`) between each element
      of  the first table, with each element of the second table.

    * Addition of two tables acts as list concatination of the terms in each
      table (``+``).

    * Pauli tables can be sorted by lexicographic (tensor product) order or
      by Pauli weights (:meth:`~qiskit.quantum_info.PauliTable.sort`).

    * Duplicate elements can be counted and deleted
      (:meth:`~qiskit.quantum_info.PauliTable.unique`).

    * The PauliTable may be iterated over in either its native symplectic
      boolean array representation, as Pauli string labels
      (:meth:`~qiskit.quantum_info.PauliTable.label_iter`), or as dense
      Numpy array or sparse CSR matrices
      (:meth:`~qiskit.quantum_info.PauliTable.matrix_iter`).

    * Checking commutation between elements of the Pauli table and another
      Pauli (:meth:`~qiskit.quantum_info.PauliTable.commutes`) or Pauli
      table (:meth:`~qiskit.quantum_info.PauliTable.commutes_with_all`)

  See the :class:`qiskit.quantum_info.PauliTable` class API documentation for
  additional details.

- Adds :class:`qiskit.quantum_info.StabilizerTable` class. This is a subclass
  of the :class:`qiskit.quantum_info.PauliTable` class which includes a
  boolean phase vector along with the Pauli table array. This represents a
  list of Stabilizer operators which are real-Pauli operators with +1 or -1
  coefficient. Because the stabilizer matrices are real the ``"Y"`` label
  matrix is defined as ``[[0, 1], [-1, 0]]``. See the API documentation for
  additional information.

- Adds :func:`qiskit.quantum_info.pauli_basis` function which returns an N-qubit
  Pauli basis as a :class:`qiskit.quantum_info.PauliTable` object. The ordering
  of this basis can either be by standard lexicographic (tensor product) order,
  or by the number of non-identity Pauli terms (weight).

- Adds :class:`qiskit.quantum_info.ScalarOp` operator class that represents
  a scalar multiple of an identity operator. This can be used to initialize
  an identity on arbitrary dimension subsystems and it will be implicitly
  converted to other ``BaseOperator`` subclasses (such as an
  :class:`qiskit.quantum_info.Operator` or
  :class:`qiskit.quantum_info.SuperOp`) when it is composed with,
  or added to, them.

  Example: Identity operator

  .. code-block::

      from qiskit.quantum_info import ScalarOp, Operator

      X = Operator.from_label('X')
      Z = Operator.from_label('Z')

      init = ScalarOp(2 ** 3)  # 3-qubit identity
      op = init @ X([0]) @ Z([1]) @ X([2])  # Op XZX

- A new method, :meth:`~qiskit.quantum_info.Operator.reshape`, has been added
  to the :class:`qiskit.quantum_innfo.Operator` class that returns a shallow
  copy of an operator subclass with reshaped subsystem input or output dimensions.
  The combined dimensions of all subsystems must be the same as the original
  operator or an exception will be raised.

- Adds :func:`qiskit.quantum_info.random_clifford` for generating a random
  :class:`qiskit.quantum_info.Clifford` operator.

- Add :func:`qiskit.quantum_info.random_quantum_channel` function
  for generating a random quantum channel with fixed
  :class:`~qiskit.quantum_info.Choi`-rank in the
  :class:`~qiskit.quantum_info.Stinespring` representation.

- Add :func:`qiskit.quantum_info.random_hermitian` for generating
  a random Hermitian :class:`~qiskit.quantum_info.Operator`.

- Add :func:`qiskit.quantum_info.random_statevector` for generating
  a random :class:`~qiskit.quantum_info.Statevector`.

- Adds :func:`qiskit.quantum_info.random_pauli_table` for generating a random
  :class:`qiskit.quantum_info.PauliTable`.

- Adds :func:`qiskit.quantum_info.random_stabilizer_table` for generating a random
  :class:`qiskit.quantum_info.StabilizerTable`.

- Add a ``num_qubits`` attribute to :class:`qiskit.quantum_info.StateVector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes. This returns the number of
  qubits for N-qubit states and returns ``None`` for non-qubit states.

- Adds :meth:`~qiskit.quantum_info.Statevector.to_dict` and
  :meth:`~qiskit.quantum_info.DensityMatrix.to_dict` methods to convert
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` objects into Bra-Ket notation
  dictionary.

  Example

  .. jupyter-execute::

    from qiskit.quantum_info import Statevector

    state = Statevector.from_label('+0')
    print(state.to_dict())

  .. jupyter-execute::

    from qiskit.quantum_info import DensityMatrix

    state = DensityMatrix.from_label('+0')
    print(state.to_dict())

- Adds :meth:`~qiskit.quantum_info.Statevector.probabilities` and
  :meth:`~qiskit.quantum_info.DensityMatrix.probabilities` to
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes which return an
  array of measurement outcome probabilities in the computational
  basis for the specified subsystems.

  Example

  .. jupyter-execute::

    from qiskit.quantum_info import Statevector

    state = Statevector.from_label('+0')
    print(state.probabilities())

  .. jupyter-execute::

    from qiskit.quantum_info import DensityMatrix

    state = DensityMatrix.from_label('+0')
    print(state.probabilities())

- Adds :meth:`~qiskit.quantum_info.Statevector.probabilities_dict` and
  :meth:`~qiskit.quantum_info.DensityMatrix.probabilities_dict` to
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes which return a
  count-style dictionary array of measurement outcome probabilities
  in the computational basis for the specified subsystems.

  .. jupyter-execute::

    from qiskit.quantum_info import Statevector

    state = Statevector.from_label('+0')
    print(state.probabilities_dict())

  .. jupyter-execute::

    from qiskit.quantum_info import DensityMatrix

    state = DensityMatrix.from_label('+0')
    print(state.probabilities_dict())

- Add :meth:`~qiskit.quantum_info.Statevector.sample_counts` and
  :meth:`~qiskit.quantum_info.Statevector.sample_memory` methods to the
  :class:`~qiskit.quantum_info.Statevector`
  and :class:`~qiskit.quantum_info.DensityMatrix` classes for sampling
  measurement outcomes on subsystems.

  Example:

    Generate a counts dictionary by sampling from a statevector

    .. jupyter-execute::

      from qiskit.quantum_info import Statevector

      psi = Statevector.from_label('+0')
      shots = 1024

      # Sample counts dictionary
      counts = psi.sample_counts(shots)
      print('Measure both:', counts)

      # Qubit-0
      counts0 = psi.sample_counts(shots, [0])
      print('Measure Qubit-0:', counts0)

      # Qubit-1
      counts1 = psi.sample_counts(shots, [1])
      print('Measure Qubit-1:', counts1)

    Return the array of measurement outcomes for each sample

    .. jupyter-execute::

      from qiskit.quantum_info import Statevector

      psi = Statevector.from_label('-1')
      shots = 10

      # Sample memory
      mem = psi.sample_memory(shots)
      print('Measure both:', mem)

      # Qubit-0
      mem0 = psi.sample_memory(shots, [0])
      print('Measure Qubit-0:', mem0)

      # Qubit-1
      mem1 = psi.sample_memory(shots, [1])
      print('Measure Qubit-1:', mem1)

- Adds a :meth:`~qiskit.quantum_info.Statevector.measure` method to the
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` quantum state classes. This
  allows sampling a single measurement outcome from the specified subsystems
  and collapsing the statevector to the post-measurement computational basis
  state. For example

  .. jupyter-execute::

    from qiskit.quantum_info import Statevector

    psi = Statevector.from_label('+1')

    # Measure both qubits
    outcome, psi_meas = psi.measure()
    print("measure([0, 1]) outcome:", outcome, "Post-measurement state:")
    print(psi_meas)

    # Measure qubit-1 only
    outcome, psi_meas = psi.measure([1])
    print("measure([1]) outcome:", outcome, "Post-measurement state:")
    print(psi_meas)

- Adds a :meth:`~qiskit.quantum_info.Statevector.reset` method to the
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` quantum state classes. This
  allows reseting some or all subsystems to the :math:`|0\rangle` state.
  For example

  .. jupyter-execute::

    from qiskit.quantum_info import Statevector

    psi = Statevector.from_label('+1')

    # Reset both qubits
    psi_reset = psi.reset()
    print("Post reset state: ")
    print(psi_reset)

    # Reset qubit-1 only
    psi_reset = psi.reset([1])
    print("Post reset([1]) state: ")
    print(psi_reset)

- A new visualization function
  :func:`qiskit.visualization.visualize_transition` for visualizing
  single qubit gate transitions has been added. It takes in a single qubit
  circuit and returns an animation of qubit state transitions on a Bloch
  sphere. To use this function you must have installed
  the dependencies for and configured globally a matplotlib animtion
  writer. You can refer to the `matplotlib documentation
  <https://matplotlib.org/api/animation_api.html#writer-classes>`_ for
  more details on this. However, in the default case simply ensuring
  that `FFmpeg <https://www.ffmpeg.org/>`_ is installed is sufficient to
  use this function.

  It supports circuits with the following gates:

  * :class:`~qiskit.extensions.HGate`
  * :class:`~qiskit.extensions.XGate`
  * :class:`~qiskit.extensions.YGate`
  * :class:`~qiskit.extensions.ZGate`
  * :class:`~qiskit.extensions.RXGate`
  * :class:`~qiskit.extensions.RYGate`
  * :class:`~qiskit.extensions.RZGate`
  * :class:`~qiskit.extensions.SGate`
  * :class:`~qiskit.extensions.SdgGate`
  * :class:`~qiskit.extensions.TGate`
  * :class:`~qiskit.extensions.TdgGate`
  * :class:`~qiskit.extensions.U1Gate`

  For example:

  .. jupyter-execute::

    from qiskit.visualization import visualize_transition
    from qiskit import *

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(70,0)
    qc.rx(90,0)
    qc.rz(120,0)

    visualize_transition(qc, fpg=20, spg=1, trace=True)

- :func:`~qiskit.execute.execute` has a new kwarg ``schedule_circuit``. By
  setting ``schedule_circuit=True`` this enables scheduling of the circuit
  into a :class:`~qiskit.pulse.Schedule`. This allows users building
  :class:`qiskit.circuit.QuantumCircuit` objects to make use of custom
  scheduler  methods, such as the ``as_late_as_possible`` and
  ``as_soon_as_possible`` methods.
  For example::

      job = execute(qc, backend, schedule_circuit=True,
                    scheduling_method="as_late_as_possible")

- A new environment variable ``QISKIT_SUPPRESS_PACKAGING_WARNINGS`` can be
  set to ``Y`` or ``y`` which will suppress the warnings about
  ``qiskit-aer`` and ``qiskit-ibmq-provider`` not being installed at import
  time. This is useful for users who are only running qiskit-terra (or just
  not qiskit-aer and/or qiskit-ibmq-provider) and the warnings are not an
  indication of a potential packaging problem. You can set the environment
  variable to ``N`` or ``n`` to ensure that warnings are always enabled
  even if the user config file is set to disable them.

- A new user config file option, ``suppress_packaging_warnings`` has been
  added. When set to ``true`` in your user config file like::

      [default]
      suppress_packaging_warnings = true

  it will suppress the warnings about  ``qiskit-aer`` and
  ``qiskit-ibmq-provider`` not being installed at import time. This is useful
  for users who are only running qiskit-terra (or just not qiskit-aer and/or
  qiskit-ibmq-provider) and the warnings are not an indication of a potential
  packaging problem. If the user config file is set to disable the warnings
  this can be overriden by setting the ``QISKIT_SUPPRESS_PACKAGING_WARNINGS``
  to ``N`` or ``n``

- :func:`qiskit.compiler.transpile()` has two new kwargs, ``layout_method``
  and ``routing_method``. These allow you to select a particular method for
  placement and routing of circuits on constrained architectures. For,
  example::

      transpile(circ, backend, layout_method='dense',
                routing_method='lookahead')

  will run :class:`~qiskit.transpiler.passes.DenseLayout` layout pass and
  :class:`~qiskit.transpiler.passes.LookaheadSwap` routing pass.

- There has been a significant simplification to the style in which Pulse
  instructions are built.

  With the previous style, ``Command`` s were called with channels to make
  an :py:class:`~qiskit.pulse.instructions.Instruction`. The usage of both
  commands and instructions was a point of confusion. This was the previous
  style::

      sched += Delay(5)(DriveChannel(0))
      sched += ShiftPhase(np.pi)(DriveChannel(0))
      sched += SamplePulse([1.0, ...])(DriveChannel(0))
      sched += Acquire(100)(AcquireChannel(0), MemorySlot(0))

  or, equivalently (though less used)::

      sched += DelayInstruction(Delay(5), DriveChannel(0))
      sched += ShiftPhaseInstruction(ShiftPhase(np.pi), DriveChannel(0))
      sched += PulseInstruction(SamplePulse([1.0, ...]), DriveChannel(0))
      sched += AcquireInstruction(Acquire(100), AcquireChannel(0),
                                  MemorySlot(0))

  Now, rather than build a command *and* an instruction, each command has
  been migrated into an instruction::

      sched += Delay(5, DriveChannel(0))
      sched += ShiftPhase(np.pi, DriveChannel(0))
      sched += Play(SamplePulse([1.0, ...]), DriveChannel(0))
      sched += SetFrequency(5.5, DriveChannel(0))  # New instruction!
      sched += Acquire(100, AcquireChannel(0), MemorySlot(0))

- There is now a :py:class:`~qiskit.pulse.instructions.Play` instruction
  which takes a description of a pulse envelope and a channel. There is a
  new :py:class:`~qiskit.pulse.pulse_lib.Pulse` class in the
  :mod:`~qiskit.pulse.pulse_lib` from which the pulse envelope description
  should subclass.

  For example::

      Play(SamplePulse([0.1]*10), DriveChannel(0))
      Play(ConstantPulse(duration=10, amp=0.1), DriveChannel(0))


.. _Release Notes_0.13.0_Upgrade Notes:

Upgrade Notes
-------------

- The :class:`qiskit.dagcircuit.DAGNode` method ``pop`` which was deprecated
  in the 0.9.0 release has been removed. If you were using this method you
  can leverage Python's ``del`` statement or ``delattr()`` function
  to perform the same task.

- A new optional visualization requirement,
  `pygments <https://pygments.org/>`_ , has been added. It is used for
  providing syntax highlighting of OpenQASM 2.0 code in Jupyter widgets and
  optionally for the :meth:`qiskit.circuit.QuantumCircuit.qasm` method. It
  must be installed (either with ``pip install pygments`` or
  ``pip install qiskit-terra[visualization]``) prior to using the
  ``%circuit_library_info`` widget in :mod:`qiskit.tools.jupyter` or
  the ``formatted`` kwarg on the :meth:`~qiskit.circuit.QuantumCircuit.qasm`
  method.

- The pulse ``buffer`` option found in :class:`qiskit.pulse.Channel` and
  :class:`qiskit.pulse.Schedule` was deprecated in Terra 0.11.0 and has now
  been removed. To add a delay on a channel or in a schedule, specify it
  explicitly in your Schedule with a Delay::

      sched = Schedule()
      sched += Delay(5)(DriveChannel(0))

- ``PulseChannelSpec``, which was deprecated in Terra 0.11.0, has now been
  removed. Use BackendConfiguration instead::

      config = backend.configuration()
      drive_chan_0 = config.drives(0)
      acq_chan_0 = config.acquires(0)

  or, simply reference the channel directly, such as ``DriveChannel(index)``.

- An import path was deprecated in Terra 0.10.0 and has now been removed: for
  ``PulseChannel``, ``DriveChannel``, ``MeasureChannel``, and
  ``ControlChannel``, use ``from qiskit.pulse.channels import X`` in place of
  ``from qiskit.pulse.channels.pulse_channels import X``.

- The pass :class:`qiskit.transpiler.passes.CSPLayout` (which was introduced
  in the 0.11.0 release) has been added to the preset pass manager for
  optimization levels 2 and 3. For level 2, there is a call limit of 1,000
  and a timeout of 10 seconds. For level 3, the call limit is 10,000 and the
  timeout is 1 minute.

  Now that the pass is included in the preset pass managers the
  `python-constraint <https://pypi.org/project/python-constraint/>`_ package
  is not longer an optional dependency and has been added to the requirements
  list.

- The ``TranspileConfig`` class which was previously used to set
  run time configuration for a :class:`qiskit.transpiler.PassManager` has
  been removed and replaced by a new class
  :class:`qiskit.transpile.PassManagerConfig`. This new class has been
  structured to include only the information needed to construct a
  :class:`~qiskit.transpiler.PassManager`. The attributes of this class are:

  * ``initial_layout``
  * ``basis_gates``
  * ``coupling_map``
  * ``backend_properties``
  * ``seed_transpiler``

- The function ``transpile_circuit`` in
  :mod:`qiskit.transpiler` has been removed. To transpile a circuit with a
  custom :class:`~qiskit.transpiler.PassManager` now you should use the
  :meth:`~qiskit.transpiler.PassManager.run` method of the
  :class:~qiskit.transpiler.PassManager` object.

- The :class:`~qiskit.circuit.QuantumCircuit` method
  :meth:`~qiskit.circuit.QuantumCircuit.draw` and
  :func:`qiskit.visualization.circuit_drawer` function will no longer include
  the initial state included in visualizations by default. If you would like to
  retain the initial state in the output visualization you need to set the
  ``initial_state`` kwarg to ``True``. For example, running:

  .. jupyter-execute::

      from qiskit import QuantumCircuit

      circuit = QuantumCircuit(2)
      circuit.measure_all()
      circuit.draw(output='text')

  This no longer includes the initial state. If you'd like to retain it you can run:

  .. jupyter-execute::

      from qiskit import QuantumCircuit

      circuit = QuantumCircuit(2)
      circuit.measure_all()
      circuit.draw(output='text', initial_state=True)


- :func:`qiskit.compiler.transpile` (and :func:`qiskit.execute.execute`,
  which uses ``transpile`` internally) will now raise an error when the
  ``pass_manager`` kwarg is set and a value is set for other kwargs that
  are already set in an instantiated :class:`~qiskit.transpiler.PassManager`
  object. Previously, these conflicting kwargs would just be silently
  ignored and the values in the ``PassManager`` instance would be used. For
  example::

      from qiskit.circuit import QuantumCircuit
      from qiskit.transpiler.pass_manager_config import PassManagerConfig
      from qiskit.transpiler import preset_passmanagers
      from qiskit.compiler import transpile

      qc = QuantumCircuit(5)

      config = PassManagerConfig(basis_gates=['u3', 'cx'])
      pm = preset_passmanagers.level_0_pass_manager(config)
      transpile(qc, optimization_level=3, pass_manager=pm)

  will now raise an error while prior to this release the value in ``pm``
  would just silently be used and the value for the ``optimization_level``
  kwarg would be ignored. The ``transpile`` kwargs this applies to are:

  * ``optimization_level``
  * ``basis_gates``
  * ``coupling_map``
  * ``seed_transpiler``
  * ``backend_properties``
  * ``initial_layout``
  * ``layout_method``
  * ``routing_method``
  * ``backend``

- The :class:`~qiskit.quantum_info.Operator`,
  :class:`~qiskit.quantum_info.Clifford`,
  :class:`~qiskit.quantum_info.SparsePauliOp`,
  :class:`~qiskit.quantum_info.PauliTable`,
  :class:`~qiskit.quantum_info.StabilizerTable`, operator classes have an added
  ``call`` method that allows them to assign a `qargs` to the operator for use
  with the :meth:`~qiskit.quantum_info.Operator.compose`,
  :meth:`~qiskit.quantum_info.Operator.dot`,
  :meth:`~qiskit.quantum_info.Statevector.evolve`,``+``, and ``-`` operations.

- The addition method of the :class:`qiskit.quantum_info.Operator`, class now accepts a
  ``qarg`` kwarg to allow adding a smaller operator to a larger one assuming identities
  on the other subsystems (same as for ``qargs`` on
  :meth:`~qiskit.quantum_info.Operator.compose` and
  :meth:`~qiskit.quantum_info.Operator.dot` methods). This allows
  subsystem addition using the call method as with composition. This support is
  added to all BaseOperator subclasses (:class:`~qiskit.quantum_info.ScalarOp`,
  :class:`~qiskit.quantum_info.Operator`,
  :class:`~qiskit.quantum_info.QuantumChannel`).

  For example:

  .. code-block::

    from qiskit.quantum_info import Operator, ScalarOp

    ZZ = Operator.from_label('ZZ')

    # Initialize empty Hamiltonian
    n_qubits = 10
    ham = ScalarOp(2 ** n_qubits, coeff=0)

    # Add 2-body nearest neighbour terms
    for j in range(n_qubits - 1):
        ham = ham + ZZ([j, j+1])

- The ``BaseOperator`` class has been updated so that addition,
  subtraction and scalar multiplication are no longer abstract methods. This
  means that they are no longer required to be implemented in subclasses if
  they are not supported. The base class will raise a ``NotImplementedError``
  when the methods are not defined.

- The :func:`qiskit.quantum_info.random_density_matrix` function will
  now return a random :class:`~qiskit.quantum_info.DensityMatrix` object. In
  previous releases it returned a numpy array.

- The :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes no longer copy the
  input array if it is already the correct dtype.

- `fastjsonschema <https://pypi.org/project/fastjsonschema/>`_ is added as a
  dependency. This is used for much faster validation of qobj dictionaries
  against the JSON schema when the ``to_dict()`` method is called on qobj
  objects with the ``validate`` keyword argument set to ``True``.

- The qobj construction classes in :mod:`qiskit.qobj` will no longer validate
  against the qobj jsonschema by default. These include the following classes:

  * :class:`qiskit.qobj.QasmQobjInstruction`
  * :class:`qiskit.qobj.QobjExperimentHeader`
  * :class:`qiskit.qobj.QasmQobjExperimentConfig`
  * :class:`qiskit.qobj.QasmQobjExperiment`
  * :class:`qiskit.qobj.QasmQobjConfig`
  * :class:`qiskit.qobj.QobjHeader`
  * :class:`qiskit.qobj.PulseQobjInstruction`
  * :class:`qiskit.qobj.PulseQobjExperimentConfig`
  * :class:`qiskit.qobj.PulseQobjExperiment`
  * :class:`qiskit.qobj.PulseQobjConfig`
  * :class:`qiskit.qobj.QobjMeasurementOption`
  * :class:`qiskit.qobj.PulseLibraryItem`
  * :class:`qiskit.qobj.QasmQobjInstruction`
  * :class:`qiskit.qobj.QasmQobjExperimentConfig`
  * :class:`qiskit.qobj.QasmQobjExperiment`
  * :class:`qiskit.qobj.QasmQobjConfig`
  * :class:`qiskit.qobj.QasmQobj`
  * :class:`qiskit.qobj.PulseQobj`

  If you were relying on this validation or would like to validate them
  against the qobj schema this can be done by setting the ``validate`` kwarg
  to ``True`` on :meth:`~qiskit.qobj.QasmQobj.to_dict` method from either of
  the top level Qobj classes :class:`~qiskit.qobj.QasmQobj` or
  :class:`~qiskit.qobj.PulseQobj`. For example:

  .. code-block:

      from qiskit import qobj

      my_qasm = qobj.QasmQobj(
          qobj_id='12345',
          header=qobj.QobjHeader(),
          config=qobj.QasmQobjConfig(shots=1024, memory_slots=2,
                                     max_credits=10),
          experiments=[
              qobj.QasmQobjExperiment(instructions=[
                  qobj.QasmQobjInstruction(name='u1', qubits=[1],
                                           params=[0.4]),
                  qobj.QasmQobjInstruction(name='u2', qubits=[1],
                                           params=[0.4, 0.2])
              ])
          ]
      )
      qasm_dict = my_qasm.to_dict(validate=True)

  which will validate the output dictionary against the Qobj jsonschema.

- The output dictionary from :meth:`qiskit.qobj.QasmQobj.to_dict` and
  :meth:`qiskit.qobj.PulseQobj.to_dict` is no longer in a format for direct
  json serialization as expected by IBMQ's API. These Qobj objects are
  the current format we use for passing experiments to providers/backends
  and while having a dictionary format that could just be passed to the IBMQ
  API directly was moderately useful for ``qiskit-ibmq-provider``, it made
  things more difficult for other providers. Especially for providers that
  wrap local simulators. Moving forward the definitions of what is passed
  between providers and the IBMQ API request format will be further decoupled
  (in a backwards compatible manner) which should ease the burden of writing
  providers and backends.

  In practice, the only functional difference between the output of these
  methods now and previous releases is that complex numbers are represented
  with the ``complex`` type and numpy arrays are not silently converted to
  list anymore. If you were previously calling ``json.dumps()`` directly on
  the output of ``to_dict()`` after this release a custom json encoder will
  be needed to handle these cases. For example::

      import json

      from qiskit.circuit import ParameterExpression
      from qiskit import qobj

      my_qasm = qobj.QasmQobj(
          qobj_id='12345',
          header=qobj.QobjHeader(),
          config=qobj.QasmQobjConfig(shots=1024, memory_slots=2,
                                     max_credits=10),
          experiments=[
              qobj.QasmQobjExperiment(instructions=[
                  qobj.QasmQobjInstruction(name='u1', qubits=[1],
                                           params=[0.4]),
                  qobj.QasmQobjInstruction(name='u2', qubits=[1],
                                           params=[0.4, 0.2])
              ])
          ]
      )
      qasm_dict = my_qasm.to_dict()

      class QobjEncoder(json.JSONEncoder):
          """A json encoder for pulse qobj"""
          def default(self, obj):
              # Convert numpy arrays:
              if hasattr(obj, 'tolist'):
                  return obj.tolist()
              # Use Qobj complex json format:
              if isinstance(obj, complex):
                  return (obj.real, obj.imag)
              if isinstance(obj, ParameterExpression):
                  return float(obj)
              return json.JSONEncoder.default(self, obj)

      json_str = json.dumps(qasm_dict, cls=QobjEncoder)

  will generate a json string in the same exact manner that
  ``json.dumps(my_qasm.to_dict())`` did in previous releases.

- ``CmdDef`` has been deprecated since Terra 0.11.0 and has been removed.
  Please continue to use :py:class:`~qiskit.pulse.InstructionScheduleMap`
  instead.

- The methods ``cmds`` and ``cmd_qubits`` in
  :py:class:`~qiskit.pulse.InstructionScheduleMap` have been deprecated
  since Terra 0.11.0 and have been removed. Please use ``instructions``
  and ``qubits_with_instruction`` instead.

- PulseDefaults have reported ``qubit_freq_est`` and ``meas_freq_est`` in
  Hz rather than GHz since Terra release 0.11.0. A warning which notified
  of this change has been removed.

- The previously deprecated (in the 0.11.0 release) support for passsing in
  :class:`qiskit.circuit.Instruction` parameters of types ``sympy.Basic``,
  ``sympy.Expr``, ``qiskit.qasm.node.node.Node`` (QASM AST node) and
  ``sympy.Matrix`` has been removed. The supported types for instruction
  parameters are:

  * ``int``
  * ``float``
  * ``complex``
  * ``str``
  * ``list``
  * ``np.ndarray``
  * :class:`qiskit.circuit.ParameterExpression`

- The following properties of
  :py:class:`~qiskit.providers.models.BackendConfiguration`:

  * ``dt``
  * ``dtm``
  * ``rep_time``

  all have units of seconds. Prior to release 0.11.0, ``dt`` and ``dtm`` had
  units of nanoseconds. Prior to release 0.12.0, ``rep_time`` had units of
  microseconds. The warnings alerting users of these changes have now been
  removed from ``BackendConfiguration``.

- A new requirement has been added to the requirements list,
  `retworkx <https://pypi.org/project/retworkx/>`_. It is an Apache 2.0
  licensed graph library that has a similar API to networkx and is being used
  to significantly speed up the :class:`qiskit.dagcircuit.DAGCircuit`
  operations as part of the transpiler. There are binaries published on PyPI
  for all the platforms supported by Qiskit Terra but if you're using a
  platform where there aren't precompiled binaries published refer to the
  `retworkx documentation
  <https://retworkx.readthedocs.io/en/stable/README.html#installing-retworkx>`_
  for instructions on pip installing from sdist.

  If you encounter any issues with the transpiler or DAGCircuit class as part
  of the transition you can switch back to the previous networkx
  implementation by setting the environment variable ``USE_RETWORKX`` to
  ``N``. This option will be removed in the 0.14.0 release.


.. _Release Notes_0.13.0_Deprecation Notes:

Deprecation Notes
-----------------

- Passing in the data to the constructor for
  :class:`qiskit.dagcircuit.DAGNode` as a dictionary arg ``data_dict``
  is deprecated and will be removed in a future release. Instead you should
  now pass the fields in as kwargs to the constructor. For example the
  previous behavior of::

    from qiskit.dagcircuit import DAGNode

    data_dict = {
        'type': 'in',
        'name': 'q_0',
    }
    node = DAGNode(data_dict)

  should now be::

    from qiskit.dagcircuit import DAGNode

    node = DAGNode(type='in', name='q_0')

- The naming of gate objects and methods have been updated to be more
  consistent. The following changes have been made:

  * The Pauli gates all have one uppercase letter only (``I``, ``X``, ``Y``,
    ``Z``)
  * The parameterized Pauli gates (i.e. rotations) prepend the uppercase
    letter ``R`` (``RX``, ``RY``, ``RZ``)
  * A controlled version prepends the uppercase letter ``C`` (``CX``,
    ``CRX``, ``CCX``)
  * Gates are named according to their action, not their alternative names
    (``CCX``, not ``Toffoli``)

  The old names have been deprecated and will be removed in a future release.
  This is a list of the changes showing the old and new class, name attribute,
  and methods. If a new column is blank then there is no change for that.

  .. list-table:: Gate Name Changes
     :header-rows: 1

     * - Old Class
       - New Class
       - Old Name Attribute
       - New Name Attribute
       - Old :class:`qiskit.circuit.QuantumCircuit` method
       - New :class:`qiskit.circuit.QuantumCircuit` method
     * - ``ToffoliGate``
       - :class:`~qiskit.extensions.CCXGate`
       - ``ccx``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.ccx` and
         :meth:`~qiskit.circuit.QuantumCircuit.toffoli`
       -
     * - ``CrxGate``
       - :class:`~qiskit.extensions.CRXGate`
       - ``crx``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.crx`
       -
     * - ``CryGate``
       - :class:`~qiskit.extensions.CRYGate`
       - ``cry``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cry`
       -
     * - ``CrzGate``
       - :class:`~qiskit.extensions.CRZGate`
       - ``crz``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.crz`
       -
     * - ``FredkinGate``
       - :class:`~qiskit.extensions.CSwapGate`
       - ``cswap``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cswap` and
         :meth:`~qiskit.circuit.QuantumCircuit.fredkin`
       -
     * - ``Cu1Gate``
       - :class:`~qiskit.extensions.CU1Gate`
       - ``cu1``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cu1`
       -
     * - ``Cu3Gate``
       - :class:`~qiskit.extensions.CU3Gate`
       - ``cu3``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cu3`
       -
     * - ``CnotGate``
       - :class:`~qiskit.extensions.CXGate`
       - ``cx``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cx` and
         :meth:`~qiskit.circuit.QuantumCircuit.cnot`
       -
     * - ``CyGate``
       - :class:`~qiskit.extensions.CYGate`
       - ``cy``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cy`
       -
     * - ``CzGate``
       - :class:`~qiskit.extensions.CZGate`
       - ``cz``
       -
       - :meth:`~qiskit.circuit.QuantumCircuit.cz`
       -
     * - ``DiagGate``
       - :class:`~qiskit.extensions.DiagonalGate`
       - ``diag``
       - ``diagonal``
       - ``diag_gate``
       - :meth:`~qiskit.circuit.QuantumCircuit.diagonal`
     * - ``IdGate``
       - :class:`~qiskit.extensions.IGate`
       - ``id``
       -
       - ``iden``
       - :meth:`~qiskit.circuit.QuantumCircuit.i` and
         :meth:`~qiskit.circuit.QuantumCircuit.id`
     * - :class:`~qiskit.extensions.Isometry`
       -
       - ``iso``
       - ``isometry``
       - :meth:`~qiskit.circuit.QuantumCircuit.iso`
       - :meth:`~qiskit.circuit.QuantumCircuit.isometry`
         and :meth:`~qiskit.circuit.QuantumCircuit.iso`
     * - ``UCG``
       - :class:`~qiskit.extensions.UCGate`
       - ``multiplexer``
       -
       - ``ucg``
       - :meth:`~qiskit.circuit.QuantumCircuit.uc`
     * - ``UCRot``
       - :class:`~qiskit.extensions.UCPauliRotGate`
       -
       -
       -
       -
     * - ``UCX``
       - :class:`~qiskit.extensions.UCRXGate`
       - ``ucrotX``
       - ``ucrx``
       - ``ucx``
       - :meth:`~qiskit.circuit.QuantumCircuit.ucrx`
     * - ``UCY``
       - :class:`~qiskit.extensions.UCRYGate`
       - ``ucroty``
       - ``ucry``
       - ``ucy``
       - :meth:`~qiskit.circuit.QuantumCircuit.ucry`
     * - ``UCZ``
       - :class:`~qiskit.extensions.UCRZGate`
       - ``ucrotz``
       - ``ucrz``
       - ``ucz``
       - :meth:`~qiskit.circuit.QuantumCircuit.ucrz`

- The kwarg ``period`` for the function
  :func:`~qiskit.pulse.pulse_lib.square`,
  :func:`~qiskit.pulse.pulse_lib.sawtooth`, and
  :func:`~qiskit.pulse.pulse_lib.triangle` in
  :mod:`qiskit.pulse.pulse_lib` is now deprecated and will be removed in a
  future release. Instead you should now use the ``freq`` kwarg to set
  the frequency.

- The ``DAGCircuit.compose_back()`` and ``DAGCircuit.extend_back()`` methods
  are deprecated and will be removed in a future release. Instead you should
  use the :meth:`qiskit.dagcircuit.DAGCircuit.compose` method, which is a more
  general and more flexible method that provides the same functionality.

- The ``callback`` kwarg of the :class:`qiskit.transpiler.PassManager` class's
  constructor has been deprecated and will be removed in a future release.
  Instead of setting it at the object level during creation it should now
  be set as a kwarg parameter on the :meth:`qiskit.transpiler.PassManager.run`
  method.

- The ``n_qubits`` and ``numberofqubits`` keywords are deprecated throughout
  Terra and replaced by ``num_qubits``. The old names will be removed in
  a future release. The objects affected by this change are listed below:

  .. list-table:: New Methods
     :header-rows: 1

     * - Class
       - Old Method
       - New Method
     * - :class:`~qiskit.circuit.QuantumCircuit`
       - ``n_qubits``
       - :meth:`~qiskit.circuit.QuantumCircuit.num_qubits`
     * - :class:`~qiskit.quantum_info.Pauli`
       - ``numberofqubits``
       - :meth:`~qiskit.quantum_info.Pauli.num_qubits`

  .. list-table:: New arguments
     :header-rows: 1

     * - Function
       - Old Argument
       - New Argument
     * - :func:`~qiskit.circuit.random.random_circuit`
       - ``n_qubits``
       - ``num_qubits``
     * - :class:`~qiskit.extensions.MSGate`
       - ``n_qubit``
       - ``num_qubits``

- The function ``qiskit.quantum_info.synthesis.euler_angles_1q`` is now
  deprecated. It has been superseded by the
  :class:`qiskit.quantum_info.OneQubitEulerDecomposer` class which provides
  the same functionality through::

      OneQubitEulerDecomposer().angles(mat)

- The ``pass_manager`` kwarg for the :func:`qiskit.compiler.transpile`
  has been deprecated and will be removed in a future release. Moving forward
  the preferred way to transpile a circuit with a custom
  :class:`~qiskit.transpiler.PassManager` object is to use the
  :meth:`~qiskit.transpiler.PassManager.run` method of the ``PassManager``
  object.

- The :func:`qiskit.quantum_info.random_state` function has been deprecated
  and will be removed in a future release. Instead you should use the
  :func:`qiskit.quantum_info.random_statevector` function.

- The ``add``, ``subtract``, and ``multiply`` methods of the
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes are deprecated and will
  be removed in a future release. Instead you shoulde use ``+``, ``-``, ``*``
  binary operators instead.

- Deprecates :meth:`qiskit.quantum_info.Statevector.to_counts`,
  :meth:`qiskit.quantum_info.DensityMatrix.to_counts`, and
  :func:`qiskit.quantum_info.counts.state_to_counts`. These functions
  are superseded by the class methods
  :meth:`qiskit.quantum_info.Statevector.probabilities_dict` and
  :meth:`qiskit.quantum_info.DensityMatrix.probabilities_dict`.

- :py:class:`~qiskit.pulse.pulse_lib.SamplePulse` and
  :py:class:`~qiskit.pulse.pulse_lib.ParametricPulse` s (e.g. ``Gaussian``)
  now subclass from :py:class:`~qiskit.pulse.pulse_lib.Pulse` and have been
  moved to the :mod:`qiskit.pulse.pulse_lib`. The previous path via
  ``pulse.commands`` is deprecated and will be removed in a future release.

- ``DelayInstruction`` has been deprecated and replaced by
  :py:class:`~qiskit.pulse.instruction.Delay`. This new instruction has been
  taken over the previous ``Command`` ``Delay``. The migration pattern is::

      Delay(<duration>)(<channel>) -> Delay(<duration>, <channel>)
      DelayInstruction(Delay(<duration>), <channel>)
          -> Delay(<duration>, <channel>)

  Until the deprecation period is over, the previous ``Delay`` syntax of
  calling a command on a channel will also be supported::

      Delay(<phase>)(<channel>)

  The new ``Delay`` instruction does not support a ``command`` attribute.

- ``FrameChange`` and ``FrameChangeInstruction`` have been deprecated and
  replaced by :py:class:`~qiskit.pulse.instructions.ShiftPhase`. The changes
  are::

      FrameChange(<phase>)(<channel>) -> ShiftPhase(<phase>, <channel>)
      FrameChangeInstruction(FrameChange(<phase>), <channel>)
          -> ShiftPhase(<phase>, <channel>)

  Until the deprecation period is over, the previous FrameChange syntax of
  calling a command on a channel will be supported::

      ShiftPhase(<phase>)(<channel>)

- The ``call`` method of :py:class:`~qiskit.pulse.pulse_lib.SamplePulse` and
  :py:class:`~qiskit.pulse.pulse_lib.ParametricPulse` s have been deprecated.
  The migration is as follows::

      Pulse(<*args>)(<channel>) -> Play(Pulse(*args), <channel>)

- ``AcquireInstruction`` has been deprecated and replaced by
  :py:class:`~qiskit.pulse.instructions.Acquire`. The changes are::

      Acquire(<duration>)(<**channels>) -> Acquire(<duration>, <**channels>)
      AcquireInstruction(Acquire(<duration>), <**channels>)
          -> Acquire(<duration>, <**channels>)

  Until the deprecation period is over, the previous Acquire syntax of
  calling the command on a channel will be supported::

      Acquire(<duration>)(<**channels>)


.. _Release Notes_0.13.0_Bug Fixes:

Bug Fixes
---------

- The :class:`~qiskit.transpiler.passes.BarrierBeforeFinalMeasurements`
  transpiler pass, included in the preset transpiler levels when targeting
  a physical device, previously inserted a barrier across only measured
  qubits. In some cases, this allowed the transpiler to insert a swap after a
  measure operation, rendering the circuit invalid for current
  devices. The pass has been updated so that the inserted barrier
  will span all qubits on the device. Fixes
  `#3937 <https://github.com/Qiskit/qiskit-terra/issues/3937>`_

- When extending a :class:`~qiskit.circuit.QuantumCircuit` instance
  (extendee) with another circuit (extension), the circuit is taken via
  reference. If a circuit is extended with itself that leads to an infinite
  loop as extendee and extension are the same. This bug has been resolved by
  copying the extension if it is the same object as the extendee.
  Fixes `#3811 <https://github.com/Qiskit/qiskit-terra/issues/3811>`_

- Fixes a case in :meth:`qiskit.result.Result.get_counts`, where the results
  for an expirement could not be referenced if the experiment was initialized
  as a Schedule without a name. Fixes
  `#2753 <https://github.com/Qiskit/qiskit-terra/issues/2753>`_

- Previously, replacing :class:`~qiskit.circuit.Parameter` objects in a
  circuit with new Parameter objects prior to decomposing a circuit would
  result in the substituted values not correctly being substituted into the
  decomposed gates. This has been resolved such that binding and
  decomposition may occur in any order.

- The matplotlib output backend for the
  :func:`qiskit.visualization.circuit_drawer` function and
  :meth:`qiskit.circuit.QuantumCircuit.draw` method drawer has been fixed
  to render :class:`~qiskit.extensions.CU1Gate` gates correctly.
  Fixes `#3684 <https://github.com/Qiskit/qiskit-terra/issues/3684>`_

- A bug in :meth:`qiskit.circuit.QuantumCircuit.from_qasm_str` and
  :meth:`qiskit.circuit.QuantumCircuit.from_qasm_file` when
  loading QASM with custom gates defined has been fixed. Now, loading
  this QASM::

      OPENQASM 2.0;
      include "qelib1.inc";
      gate rinv q {sdg q; h q; sdg q; h q; }
      qreg q[1];
      rinv q[0];

  is equivalent to the following circuit::

      rinv_q = QuantumRegister(1, name='q')
      rinv_gate = QuantumCircuit(rinv_q, name='rinv')
      rinv_gate.sdg(rinv_q)
      rinv_gate.h(rinv_q)
      rinv_gate.sdg(rinv_q)
      rinv_gate.h(rinv_q)
      rinv = rinv_gate.to_instruction()
      qr = QuantumRegister(1, name='q')
      expected = QuantumCircuit(qr, name='circuit')
      expected.append(rinv, [qr[0]])

  Fixes `#1566 <https://github.com/Qiskit/qiskit-terra/issues/1566>`_

- Allow quantum circuit Instructions to have list parameter values. This is
  used in Aer for expectation value snapshot parameters for example
  ``params = [[1.0, 'I'], [1.0, 'X']]]`` for :math:`\langle I + X\rangle`.

- Previously, for circuits containing composite gates (those created via
  :meth:`qiskit.circuit.QuantumCircuit.to_gate` or
  :meth:`qiskit.circuit.QuantumCircuit.to_instruction` or their corresponding
  converters), attempting to bind the circuit more than once would result in
  only the first bind value being applied to all circuits when transpiled.
  This has been resolved so that the values provided for subsequent binds are
  correctly respected.


.. _Release Notes_0.13.0_Other Notes:

Other Notes
-----------

- The qasm and pulse qobj classes:

  * :class:`~qiskit.qobj.QasmQobjInstruction`
  * :class:`~qiskit.qobj.QobjExperimentHeader`
  * :class:`~qiskit.qobj.QasmQobjExperimentConfig`
  * :class:`~qiskit.qobj.QasmQobjExperiment`
  * :class:`~qiskit.qobj.QasmQobjConfig`
  * :class:`~qiskit.qobj.QobjHeader`
  * :class:`~qiskit.qobj.PulseQobjInstruction`
  * :class:`~qiskit.qobj.PulseQobjExperimentConfig`
  * :class:`~qiskit.qobj.PulseQobjExperiment`
  * :class:`~qiskit.qobj.PulseQobjConfig`
  * :class:`~qiskit.qobj.QobjMeasurementOption`
  * :class:`~qiskit.qobj.PulseLibraryItem`
  * :class:`~qiskit.qobj.QasmQobjInstruction`
  * :class:`~qiskit.qobj.QasmQobjExperimentConfig`
  * :class:`~qiskit.qobj.QasmQobjExperiment`
  * :class:`~qiskit.qobj.QasmQobjConfig`
  * :class:`~qiskit.qobj.QasmQobj`
  * :class:`~qiskit.qobj.PulseQobj`

  from :mod:`qiskit.qobj` have all been reimplemented without using the
  marsmallow library. These new implementations are designed to be drop-in
  replacement (except for as noted in the upgrade release notes) but
  specifics inherited from marshmallow may not work. Please file issues for
  any incompatibilities found.

Aer 0.5.0
=========

Added
-----
 - Add support for terra diagonal gate
 - Add support for parameterized qobj

Fixed
-----
 - Added postfix for linux on Raspberry Pi
 - Handle numpy array inputs from qobj

Ignis 0.3.0
===========

Added
-----

* API documentation
* CNOT-Dihedral randomized benchmarking
* Accreditation module for output accrediation of noisy devices
* Pulse calibrations for single qubits
* Pulse Discriminator
* Entanglement verification circuits
* Gateset tomography for single-qubit gate sets
* Adds randomized benchmarking utility functions ``calculate_1q_epg``,
  ``calculate_2q_epg`` functions to calculate 1 and 2-qubit error per gate from
  error per Clifford
* Adds randomized benchmarking utility functions ``calculate_1q_epc``,
  ``calculate_2q_epc`` for calculating 1 and 2-qubit error per Clifford from error
  per gate

Changed
-------
* Support integer labels for qubits in tomography
* Support integer labels for measurement error mitigation

Deprecated
----------
* Deprecates ``twoQ_clifford_error`` function. Use ``calculate_2q_epc`` instead.
* Python 3.5 support in qiskit-ignis is deprecated. Support will be removed on
  the upstream python community's end of life date for the version, which is
  09/13/2020.

Aqua 0.6.5
==========

No Change

IBM Q Provider 0.6.0
====================

No Change

*************
Qiskit 0.17.0
*************

Terra 0.12.0
============

No Change

Aer 0.4.1
=========

No Change

Ignis 0.2.0
===========

No Change

Aqua 0.6.5
==========

No Change

IBM Q Provider 0.6.0
====================

New Features
------------

- There are three new exceptions: ``VisualizationError``, ``VisualizationValueError``,
  and ``VisualizationTypeError``. These are now used in the visualization modules when
  an exception is raised.
- You can now set the logging level and specify a log file using the environment
  variables ``QSIKIT_IBMQ_PROVIDER_LOG_LEVEL`` and ``QISKIT_IBMQ_PROVIDER_LOG_FILE``,
  respectively. Note that the name of the logger is ``qiskit.providers.ibmq``.
- :class:`qiskit.providers.ibmq.job.IBMQJob` now has a new method
  :meth:`~qiskit.providers.ibmq.job.IBMQJob.scheduling_mode` that returns the scheduling
  mode the job is in.
- IQX-related tutorials that used to be in ``qiskit-iqx-tutorials`` are now in
  ``qiskit-ibmq-provider``.

Changed
-------

- :meth:`qiskit.providers.ibmq.IBMQBackend.jobs` now accepts a new boolean parameter
  ``descending``, which can be used to indicate whether the jobs should be returned in
  descending or ascending order.
- :class:`qiskit.providers.ibmq.managed.IBMQJobManager` now looks at the job limit and waits
  for old jobs to finish before submitting new ones if the limit has been reached.
- :meth:`qiskit.providers.ibmq.IBMQBackend.status` now raises a
  :class:`qiskit.providers.ibmq.IBMQBackendApiProtocolError` exception
  if there was an issue with validating the status.

*************
Qiskit 0.16.0
*************

Terra 0.12.0
============

No Change

Aer 0.4.0
=========

No Change

Ignis 0.2.0
===========

No Change

Aqua 0.6.4
==========

No Change

IBM Q Provider 0.5.0
====================

New Features
------------

- Some of the visualization and Jupyter tools, including gate/error map and
  backend information, have been moved from ``qiskit-terra`` to ``qiskit-ibmq-provider``.
  They are now under the :mod:`qiskit.providers.ibmq.jupyter` and
  :mod:`qiskit.providers.ibmq.visualization`. In addition, you can now
  use ``%iqx_dashboard`` to get a dashboard that provides both job and
  backend information.

Changed
-------

- JSON schema validation is no longer run by default on Qobj objects passed
  to :meth:`qiskit.providers.ibmq.IBMQBackend.run`. This significantly speeds
  up the execution of the `run()` method. Qobj objects are still validated on the
  server side, and invalid Qobjs will continue to raise exceptions. To force local
  validation, set ``validate_qobj=True`` when you invoke ``run()``.

*************
Qiskit 0.15.0
*************

Terra 0.12.0
============

Prelude
-------

The 0.12.0 release includes several new features and bug fixes. The biggest
change for this release is the addition of support for parametric pulses to
OpenPulse. These are Pulse commands which take parameters rather than sample
points to describe a pulse. 0.12.0 is also the first release to include
support for Python 3.8. It also marks the beginning of the deprecation for
Python 3.5 support, which will be removed when the upstream community stops
supporting it.


.. _Release Notes_0.12.0_New Features:

New Features
------------

- The pass :class:`qiskit.transpiler.passes.CSPLayout` was extended with two
  new parameters: ``call_limit`` and ``time_limit``. These options allow
  limiting how long the pass will run. The option ``call_limit`` limits the
  number of times that the recursive function in the backtracking solver may
  be called. Similarly, ``time_limit`` limits how long (in seconds) the solver
  will be allowed to run. The defaults are ``1000`` calls and ``10`` seconds
  respectively.

- :class:`qiskit.pulse.Acquire` can now be applied to a single qubit.
  This makes pulse programming more consistent and easier to reason
  about, as now all operations apply to a single channel.
  For example::

    acquire = Acquire(duration=10)
    schedule = Schedule()
    schedule.insert(60, acquire(AcquireChannel(0), MemorySlot(0), RegisterSlot(0)))
    schedule.insert(60, acquire(AcquireChannel(1), MemorySlot(1), RegisterSlot(1)))

- A new method :meth:`qiskit.transpiler.CouplingMap.draw` was added to
  :class:`qiskit.transpiler.CouplingMap` to generate a graphviz image from
  the coupling map graph. For example:

  .. jupyter-execute::

      from qiskit.transpiler import CouplingMap

      coupling_map = CouplingMap(
          [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]])
      coupling_map.draw()

- Parametric pulses have been added to OpenPulse. These are pulse commands
  which are parameterized and understood by the backend. Arbitrary pulse
  shapes are still supported by the SamplePulse Command. The new supported
  pulse classes are:

    - :class:`qiskit.pulse.ConstantPulse`
    - :class:`qiskit.pulse.Drag`
    - :class:`qiskit.pulse.Gaussian`
    - :class:`qiskit.pulse.GaussianSquare`

  They can be used like any other Pulse command. An example::

      from qiskit.pulse import (Schedule, Gaussian, Drag, ConstantPulse,
                                GaussianSquare)

      sched = Schedule(name='parametric_demo')
      sched += Gaussian(duration=25, sigma=4, amp=0.5j)(DriveChannel(0))
      sched += Drag(duration=25, amp=0.1, sigma=5, beta=4)(DriveChannel(1))
      sched += ConstantPulse(duration=25, amp=0.3+0.1j)(DriveChannel(1))
      sched += GaussianSquare(duration=1500, amp=0.2, sigma=8,
                              width=140)(MeasureChannel(0)) << sched.duration

  The resulting schedule will be similar to a SamplePulse schedule built
  using :mod:`qiskit.pulse.pulse_lib`, however, waveform sampling will be
  performed by the backend. The method :meth:`qiskit.pulse.Schedule.draw`
  can still be used as usual. However, the command will be converted to a
  ``SamplePulse`` with the
  :meth:`qiskit.pulse.ParametricPulse.get_sample_pulse` method, so the
  pulse shown may not sample the continuous function the same way that the
  backend will.

  This feature can be used to construct Pulse programs for any backend, but
  the pulses will be converted to ``SamplePulse`` objects if the backend does
  not support parametric pulses. Backends which support them will have the
  following new attribute::

      backend.configuration().parametric_pulses: List[str]
      # e.g. ['gaussian', 'drag', 'constant']

  Note that the backend does not need to support all of the parametric
  pulses defined in Qiskit.

  When the backend supports parametric pulses, and the Pulse schedule is
  built with them, the assembled Qobj is significantly smaller. The size
  of a PulseQobj built entirely with parametric pulses is dependent only
  on the number of instructions, whereas the size of a PulseQobj built
  otherwise will grow with the duration of the instructions (since every
  sample must be specified with a value).

- Added utility functions, :func:`qiskit.scheduler.measure` and
  :func:`qiskit.scheduler.measure_all` to `qiskit.scheduler` module. These
  functions return a :class:`qiskit.pulse.Schedule` object which measures
  qubits using OpenPulse. For example::

      from qiskit.scheduler import measure, measure_all

      measure_q0_schedule = measure(qubits=[0], backend=backend)
      measure_all_schedule = measure_all(backend)
      measure_custom_schedule = measure(qubits=[0],
                                        inst_map=backend.defaults().instruction_schedule_map,
                                        meas_map=[[0]],
                                        qubit_mem_slots={0: 1})

- Pulse :class:`qiskit.pulse.Schedule` objects now have better
  representations that for simple schedules should be valid Python
  expressions.

- The :class:`qiskit.circuit.QuantumCircuit` methods
  :meth:`qiskit.circuit.QuantumCircuit.measure_active`,
  :meth:`qiskit.circuit.QuantumCircuit.measure_all`, and
  :meth:`qiskit.circuit.QuantumCircuit.remove_final_measurements` now have
  an addition kwarg ``inplace``. When ``inplace`` is set to ``False`` the
  function will return a modified **copy** of the circuit. This is different
  from the default behavior which will modify the circuit object in-place and
  return nothing.

- Several new constructor methods were added to the
  :class:`qiskit.transpiler.CouplingMap` class for building objects
  with basic qubit coupling graphs. The new constructor methods are:

    - :meth:`qiskit.transpiler.CouplingMap.from_full`
    - :meth:`qiskit.transpiler.CouplingMap.from_line`
    - :meth:`qiskit.transpiler.CouplingMap.from_ring`
    - :meth:`qiskit.transpiler.CouplingMap.from_grid`

  For example, to use the new constructors to get a coupling map of 5
  qubits connected in a linear chain you can now run:

  .. jupyter-execute::

      from qiskit.transpiler import CouplingMap

      coupling_map = CouplingMap.from_line(5)
      coupling_map.draw()

- Introduced a new pass
  :class:`qiskit.transpiler.passes.CrosstalkAdaptiveSchedule`. This
  pass aims to reduce the impact of crosstalk noise on a program. It
  uses crosstalk characterization data from the backend to schedule gates.
  When a pair of gates has high crosstalk, they get serialized using a
  barrier. Naive serialization is harmful because it incurs decoherence
  errors. Hence, this pass uses a SMT optimization approach to compute a
  schedule which minimizes the impact of crosstalk as well as decoherence
  errors.

  The pass takes as input a circuit which is already transpiled onto
  the backend i.e., the circuit is expressed in terms of physical qubits and
  swap gates have been inserted and decomposed into CNOTs if required. Using
  this circuit and crosstalk characterization data, a
  `Z3 optimization <https://github.com/Z3Prover/z3>`_ is used to construct a
  new scheduled circuit as output.

  To use the pass on a circuit circ::

    dag = circuit_to_dag(circ)
    pass_ = CrosstalkAdaptiveSchedule(backend_prop, crosstalk_prop)
    scheduled_dag = pass_.run(dag)
    scheduled_circ = dag_to_circuit(scheduled_dag)

  ``backend_prop`` is a :class:`qiskit.providers.models.BackendProperties`
  object for the target backend. ``crosstalk_prop`` is a dict which specifies
  conditional error rates. For two gates ``g1`` and ``g2``,
  ``crosstalk_prop[g1][g2]`` specifies the conditional error rate of ``g1``
  when ``g1`` and ``g2`` are executed simultaneously. A method for generating
  ``crosstalk_prop`` will be added in a future release of qiskit-ignis. Until
  then you'll either have to already know the crosstalk properties of your
  device, or manually write your own device characterization experiments.

- In the preset pass manager for optimization level 1,
  :func:`qiskit.transpiler.preset_passmanagers.level_1_pass_manager` if
  :class:`qiskit.transpiler.passes.TrivialLayout` layout pass is not a
  perfect match for a particular circuit, then
  :class:`qiskit.transpiler.passes.DenseLayout` layout pass is used
  instead.

- Added a new abstract method
  :meth:`qiskit.quantum_info.Operator.dot` to
  the abstract ``BaseOperator`` class, so it is included for all
  implementations of that abstract
  class, including :class:`qiskit.quantum_info.Operator` and
  ``QuantumChannel`` (e.g., :class:`qiskit.quantum_info.Choi`)
  objects. This method returns the right operator multiplication
  ``a.dot(b)`` :math:`= a \cdot b`. This is equivalent to
  calling the operator
  :meth:`qiskit.quantum_info.Operator.compose` method with the kwarg
  ``front`` set to ``True``.

- Added :func:`qiskit.quantum_info.average_gate_fidelity` and
  :func:`qiskit.quantum_info.gate_error` functions to the
  :mod:`qiskit.quantum_info` module for working with
  :class:`qiskit.quantum_info.Operator` and ``QuantumChannel``
  (e.g., :class:`qiskit.quantum_info.Choi`) objects.

- Added the :func:`qiskit.quantum_info.partial_trace` function to the
  :mod:`qiskit.quantum_info` that works with
  :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` quantum state classes.
  For example::

      from qiskit.quantum_info.states import Statevector
      from qiskit.quantum_info.states import DensityMatrix
      from qiskit.quantum_info.states import partial_trace

      psi = Statevector.from_label('10+')
      partial_trace(psi, [0, 1])
      rho = DensityMatrix.from_label('10+')
      partial_trace(rho, [0, 1])

- When :meth:`qiskit.circuit.QuantumCircuit.draw` or
  :func:`qiskit.visualization.circuit_drawer` is called with the
  ``with_layout`` kwarg set True (the default) the output visualization
  will now display the physical qubits as integers to clearly
  distinguish them from the virtual qubits.

  For Example:

  .. jupyter-execute::

      from qiskit import QuantumCircuit
      from qiskit import transpile
      from qiskit.test.mock import FakeVigo

      qc = QuantumCircuit(3)
      qc.h(0)
      qc.cx(0, 1)
      qc.cx(0, 2)
      transpiled_qc = transpile(qc, FakeVigo())
      transpiled_qc.draw(output='mpl')

- Added new state measure functions to the :mod:`qiskit.quantum_info`
  module: :func:`qiskit.quantum_info.entropy`,
  :func:`qiskit.quantum_info.mutual_information`,
  :func:`qiskit.quantum_info.concurrence`, and
  :func:`qiskit.quantum_info.entanglement_of_formation`. These functions work
  with the :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.DensityMatrix` classes.

- The decomposition methods for single-qubit gates in
  :class:`qiskit.quantum_info.synthesis.one_qubit_decompose.OneQubitEulerDecomposer` have
  been expanded to now also include the ``'ZXZ'`` basis, characterized by three rotations
  about the  Z,X,Z axis. This now means that a general 2x2 Operator can be
  decomposed into following bases: ``U3``, ``U1X``, ``ZYZ``, ``ZXZ``,
  ``XYX``, ``ZXZ``.


.. _Release Notes_0.12.0_Known Issues:

Known Issues
------------

- Running functions that use :func:`qiskit.tools.parallel_map` (for example
  :func:`qiskit.execute.execute`, :func:`qiskit.compiler.transpile`, and
  :meth:`qiskit.transpiler.PassManager.run`) may not work when
  called from a script running outside of a ``if __name__ == '__main__':``
  block when using Python 3.8 on MacOS. Other environments are unaffected by
  this issue. This is due to changes in how parallel processes are launched
  by Python 3.8 on MacOS. If ``RuntimeError`` or ``AttributeError`` are
  raised by scripts that are directly calling ``parallel_map()`` or when
  calling a function that uses it internally with Python 3.8 on MacOS
  embedding the script calls inside ``if __name__ == '__main__':`` should
  workaround the issue. For example::

      from qiskit import QuantumCircuit, QiskitError
      from qiskit import execute, BasicAer

      qc1 = QuantumCircuit(2, 2)
      qc1.h(0)
      qc1.cx(0, 1)
      qc1.measure([0,1], [0,1])
      # making another circuit: superpositions
      qc2 = QuantumCircuit(2, 2)
      qc2.h([0,1])
      qc2.measure([0,1], [0,1])
      execute([qc1, qc2], BasicAer.get_backend('qasm_simulator'))

  should be changed to::

      from qiskit import QuantumCircuit, QiskitError
      from qiskit import execute, BasicAer

      def main():
          qc1 = QuantumCircuit(2, 2)
          qc1.h(0)
          qc1.cx(0, 1)
          qc1.measure([0,1], [0,1])
          # making another circuit: superpositions
          qc2 = QuantumCircuit(2, 2)
          qc2.h([0,1])
          qc2.measure([0,1], [0,1])
          execute([qc1, qc2], BasicAer.get_backend('qasm_simulator'))

      if __name__ == '__main__':
          main()

  if errors are encountered with Python 3.8 on MacOS.


.. _Release Notes_0.12.0_Upgrade Notes:

Upgrade Notes
-------------

- The value of the ``rep_time`` parameter for Pulse backend's configuration
  object is now in units of seconds, not microseconds. The first time a
  ``PulseBackendConfiguration`` object is initialized it will raise a single
  warning to the user to indicate this.

- The ``rep_time`` argument for :func:`qiskit.compiler.assemble` now takes
  in a value in units of seconds, not microseconds. This was done to make
  the units with everything else in pulse. If you were passing in a value for
  ``rep_time`` ensure that you update the value to account for this change.

- The value of the ``base_gate`` property of
  :class:`qiskit.circuit.ControlledGate` objects has been changed from the
  class of the base gate to an instance of the class of the base gate.

- The ``base_gate_name`` property of :class:`qiskit.circuit.ControlledGate`
  has been removed; you can get the name of the base gate by accessing
  ``base_gate.name`` on the object. For example::

      from qiskit import QuantumCircuit
      from qiskit.extensions import HGate

      qc = QuantumCircuit(3)
      cch_gate = HGate().control(2)
      base_gate_name = cch_gate.base_gate.name

- Changed :class:`qiskit.quantum_info.Operator` magic methods so that
  ``__mul__`` (which gets executed by python's multiplication operation,
  if the left hand side of the operation has it defined) implements right
  matrix multiplication (i.e. :meth:`qiskit.quantum_info.Operator.dot`), and
  ``__rmul__`` (which gets executed by python's multiplication operation
  from the right hand side of the operation if the left does not have
  ``__mul__`` defined) implements scalar multiplication (i.e.
  :meth:`qiskit.quantum_info.Operator.multiply`). Previously both methods
  implemented scalar multiplciation.

- The second argument of the :func:`qiskit.quantum_info.process_fidelity`
  function, ``target``, is now optional. If a target unitary is not
  specified, then process fidelity of the input channel with the identity
  operator will be returned.

- :func:`qiskit.compiler.assemble` will now respect the configured
  ``max_shots`` value for a backend. If a value for the ``shots`` kwarg is
  specified that exceed the max shots set in the backend configuration the
  function will now raise a ``QiskitError`` exception. Additionally, if no
  shots argument is provided the default value is either 1024 (the previous
  behavior) or ``max_shots`` from the backend, whichever is lower.


.. _Release Notes_0.12.0_Deprecation Notes:

Deprecation Notes
-----------------

- Methods for adding gates to a :class:`qiskit.circuit.QuantumCircuit` with
  abbreviated keyword arguments (e.g. ``ctl``, ``tgt``) have had their keyword
  arguments renamed to be more descriptive (e.g. ``control_qubit``,
  ``target_qubit``). The old names have been deprecated. A table including the
  old and new calling signatures for the ``QuantumCircuit`` methods is included below.

  .. list-table:: New signatures for ``QuantumCircuit`` gate methods
     :header-rows: 1

     * - Instruction Type
       - Former Signature
       - New Signature
     * - :class:`qiskit.extensions.HGate`
       - ``qc.h(q)``
       - ``qc.h(qubit)``
     * - :class:`qiskit.extensions.CHGate`
       - ``qc.ch(ctl, tgt)``
       - ``qc.ch((control_qubit, target_qubit))``
     * - :class:`qiskit.extensions.IdGate`
       - ``qc.iden(q)``
       - ``qc.iden(qubit)``
     * - :class:`qiskit.extensions.RGate`
       - ``qc.iden(q)``
       - ``qc.iden(qubit)``
     * - :class:`qiskit.extensions.RGate`
       - ``qc.r(theta, phi, q)``
       - ``qc.r(theta, phi, qubit)``
     * - :class:`qiskit.extensions.RXGate`
       - ``qc.rx(theta, q)``
       - ``qc.rx(theta, qubit)``
     * - :class:`qiskit.extensions.CrxGate`
       - ``qc.crx(theta, ctl, tgt)``
       - ``qc.crx(theta, control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.RYGate`
       - ``qc.ry(theta, q)``
       - ``qc.ry(theta, qubit)``
     * - :class:`qiskit.extensions.CryGate`
       - ``qc.cry(theta, ctl, tgt)``
       - ``qc.cry(theta, control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.RZGate`
       - ``qc.rz(phi, q)``
       - ``qc.rz(phi, qubit)``
     * - :class:`qiskit.extensions.CrzGate`
       - ``qc.crz(theta, ctl, tgt)``
       - ``qc.crz(theta, control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.SGate`
       - ``qc.s(q)``
       - ``qc.s(qubit)``
     * - :class:`qiskit.extensions.SdgGate`
       - ``qc.sdg(q)``
       - ``qc.sdg(qubit)``
     * - :class:`qiskit.extensions.FredkinGate`
       - ``qc.cswap(ctl, tgt1, tgt2)``
       - ``qc.cswap(control_qubit, target_qubit1, target_qubit2)``
     * - :class:`qiskit.extensions.TGate`
       - ``qc.t(q)``
       - ``qc.t(qubit)``
     * - :class:`qiskit.extensions.TdgGate`
       - ``qc.tdg(q)``
       - ``qc.tdg(qubit)``
     * - :class:`qiskit.extensions.U1Gate`
       - ``qc.u1(theta, q)``
       - ``qc.u1(theta, qubit)``
     * - :class:`qiskit.extensions.Cu1Gate`
       - ``qc.cu1(theta, ctl, tgt)``
       - ``qc.cu1(theta, control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.U2Gate`
       - ``qc.u2(phi, lam, q)``
       - ``qc.u2(phi, lam, qubit)``
     * - :class:`qiskit.extensions.U3Gate`
       - ``qc.u3(theta, phi, lam, q)``
       - ``qc.u3(theta, phi, lam, qubit)``
     * - :class:`qiskit.extensions.Cu3Gate`
       - ``qc.cu3(theta, phi, lam, ctl, tgt)``
       - ``qc.cu3(theta, phi, lam, control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.XGate`
       - ``qc.x(q)``
       - ``qc.x(qubit)``
     * - :class:`qiskit.extensions.CnotGate`
       - ``qc.cx(ctl, tgt)``
       - ``qc.cx(control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.ToffoliGate`
       - ``qc.ccx(ctl1, ctl2, tgt)``
       - ``qc.ccx(control_qubit1, control_qubit2, target_qubit)``
     * - :class:`qiskit.extensions.YGate`
       - ``qc.y(q)``
       - ``qc.y(qubit)``
     * - :class:`qiskit.extensions.CyGate`
       - ``qc.cy(ctl, tgt)``
       - ``qc.cy(control_qubit, target_qubit)``
     * - :class:`qiskit.extensions.ZGate`
       - ``qc.z(q)``
       - ``qc.z(qubit)``
     * - :class:`qiskit.extensions.CzGate`
       - ``qc.cz(ctl, tgt)``
       - ``qc.cz(control_qubit, target_qubit)``

- Running :class:`qiskit.pulse.Acquire` on multiple qubits has been
  deprecated and will be removed in a future release. Additionally, the
  :class:`qiskit.pulse.AcquireInstruction` parameters ``mem_slots`` and
  ``reg_slots`` have been deprecated. Instead ``reg_slot`` and ``mem_slot``
  should be used instead.

- The attribute of the :class:`qiskit.providers.models.PulseDefaults` class
  ``circuit_instruction_map`` has been deprecated and will be removed in a
  future release. Instead you should use the new attribute
  ``instruction_schedule_map``. This was done to match the type of the
  value of the attribute, which is an ``InstructionScheduleMap``.

- The :class:`qiskit.pulse.PersistentValue` command is deprecated and will
  be removed in a future release. Similar functionality can be achieved with
  the :class:`qiskit.pulse.ConstantPulse` command (one of the new parametric
  pulses). Compare the following::

      from qiskit.pulse import Schedule, PersistentValue, ConstantPulse, \
                               DriveChannel

      # deprecated implementation
      sched_w_pv = Schedule()
      sched_w_pv += PersistentValue(value=0.5)(DriveChannel(0))
      sched_w_pv += PersistentValue(value=0)(DriveChannel(0)) << 10

      # preferred implementation
      sched_w_const = Schedule()
      sched_w_const += ConstantPulse(duration=10, amp=0.5)(DriveChannel(0))

- Python 3.5 support in qiskit-terra is deprecated. Support will be
  removed in the first release after the upstream Python community's end of
  life date for the version, which is 09/13/2020.

- The ``require_cptp`` kwarg of the
  :func:`qiskit.quantum_info.process_fidelity` function has been
  deprecated and will be removed in a future release. It is superseded by
  two separate kwargs ``require_cp`` and ``require_tp``.

- Setting the ``scale`` parameter for
  :meth:`qiskit.circuit.QuantumCircuit.draw` and
  :func:`qiskit.visualization.circuit_drawer` as the first positional
  argument is deprecated and will be removed in a future release. Instead you
  should use ``scale`` as keyword argument.

- The :mod:`qiskit.tools.qi.qi` module is deprecated and will be removed in a
  future release. The legacy functions in the module have all been superseded
  by functions and classes in the :mod:`qiskit.quantum_info` module. A table
  of the deprecated functions and their replacement are below:

  .. list-table:: ``qiskit.tools.qi.qi`` replacements
     :header-rows: 1

     * - Deprecated
       - Replacement
     * - :func:`qiskit.tools.partial_trace`
       - :func:`qiskit.quantum_info.partial_trace`
     * - :func:`qiskit.tools.choi_to_pauli`
       - :class:`qiskit.quantum_info.Choi` and :class:`quantum_info.PTM`
     * - :func:`qiskit.tools.chop`
       - ``numpy.round``
     * - ``qiskit.tools.qi.qi.outer``
       - ``numpy.outer``
     * - :func:`qiskit.tools.concurrence`
       - :func:`qiskit.quantum_info.concurrence`
     * - :func:`qiskit.tools.shannon_entropy`
       - :func:`qiskit.quantum_info.shannon_entropy`
     * - :func:`qiskit.tools.entropy`
       - :func:`qiskit.quantum_info.entropy`
     * - :func:`qiskit.tools.mutual_information`
       - :func:`qiskit.quantum_info.mutual_information`
     * - :func:`qiskit.tools.entanglement_of_formation`
       - :func:`qiskit.quantum_info.entanglement_of_formation`
     * - :func:`qiskit.tools.is_pos_def`
       - ``quantum_info.operators.predicates.is_positive_semidefinite_matrix``

- The :mod:`qiskit.quantum_info.states.states` module is deprecated and will
  be removed in a future release. The legacy functions in the module have
  all been superseded by functions and classes in the
  :mod:`qiskit.quantum_info` module.

  .. list-table:: ``qiskit.quantum_info.states.states`` replacements
     :header-rows: 1

     * - Deprecated
       - Replacement
     * - ``qiskit.quantum_info.states.states.basis_state``
       - :meth:`qiskit.quantum_info.Statevector.from_label`
     * - ``qiskit.quantum_info.states.states.projector``
       - :class:`qiskit.quantum_info.DensityMatrix`

- The ``scaling`` parameter of the ``draw()`` method for the ``Schedule`` and
  ``Pulse`` objects was deprecated and will be removed in a future release.
  Instead the new ``scale`` parameter should be used. This was done to have
  a consistent argument between pulse and circuit drawings. For example::

      #The consistency in parameters is seen below
      #For circuits
      circuit = QuantumCircuit()
      circuit.draw(scale=0.2)
      #For pulses
      pulse = SamplePulse()
      pulse.draw(scale=0.2)
      #For schedules
      schedule = Schedule()
      schedule.draw(scale=0.2)


.. _Release Notes_0.12.0_Bug Fixes:

Bug Fixes
---------

- Previously, calling :meth:`qiskit.circuit.QuantumCircuit.bind_parameters`
  prior to decomposing a circuit would result in the bound values not being
  correctly substituted into the decomposed gates. This has been resolved
  such that binding and decomposition may occur in any order. Fixes
  `issue #2482 <https://github.com/Qiskit/qiskit-terra/issues/2482>`_ and
  `issue #3509 <https://github.com/Qiskit/qiskit-terra/issues/3509>`_

- The ``Collect2qBlocks`` pass had previously not considered classical
  conditions when determining whether to include a gate within an
  existing block. In some cases, this resulted in classical
  conditions being lost when transpiling with
  ``optimization_level=3``. This has been resolved so that classically
  conditioned gates are never included in a block.
  Fixes `issue #3215 <https://github.com/Qiskit/qiskit-terra/issues/3215>`_

- All the output types for the circuit drawers in
  :meth:`qiskit.circuit.QuantumCircuit.draw` and
  :func:`qiskit.visualization.circuit_drawer` have fixed and/or improved
  support for drawing controlled custom gates. Fixes
  `issue #3546 <https://github.com/Qiskit/qiskit-terra/issues/3546>`_,
  `issue #3763 <https://github.com/Qiskit/qiskit-terra/issues/3763>`_,
  and `issue #3764 <https://github.com/Qiskit/qiskit-terra/issues/3764>`_

- Explanation and examples have been added to documentation for the
  :class:`qiskit.circuit.QuantumCircuit` methods for adding gates:
  :meth:`qiskit.circuit.QuantumCircuit.ccx`,
  :meth:`qiskit.circuit.QuantumCircuit.ch`,
  :meth:`qiskit.circuit.QuantumCircuit.crz`,
  :meth:`qiskit.circuit.QuantumCircuit.cswap`,
  :meth:`qiskit.circuit.QuantumCircuit.cu1`,
  :meth:`qiskit.circuit.QuantumCircuit.cu3`,
  :meth:`qiskit.circuit.QuantumCircuit.cx`,
  :meth:`qiskit.circuit.QuantumCircuit.cy`,
  :meth:`qiskit.circuit.QuantumCircuit.cz`,
  :meth:`qiskit.circuit.QuantumCircuit.h`,
  :meth:`qiskit.circuit.QuantumCircuit.iden`,
  :meth:`qiskit.circuit.QuantumCircuit.rx`,
  :meth:`qiskit.circuit.QuantumCircuit.ry`,
  :meth:`qiskit.circuit.QuantumCircuit.rz`,
  :meth:`qiskit.circuit.QuantumCircuit.s`,
  :meth:`qiskit.circuit.QuantumCircuit.sdg`,
  :meth:`qiskit.circuit.QuantumCircuit.swap`,
  :meth:`qiskit.circuit.QuantumCircuit.t`,
  :meth:`qiskit.circuit.QuantumCircuit.tdg`,
  :meth:`qiskit.circuit.QuantumCircuit.u1`,
  :meth:`qiskit.circuit.QuantumCircuit.u2`,
  :meth:`qiskit.circuit.QuantumCircuit.u3`,
  :meth:`qiskit.circuit.QuantumCircuit.x`,
  :meth:`qiskit.circuit.QuantumCircuit.y`,
  :meth:`qiskit.circuit.QuantumCircuit.z`. Fixes
  `issue #3400 <https://github.com/Qiskit/qiskit-terra/issues/3400>`_

- Fixes for handling of complex number parameter in circuit visualization.
  Fixes `issue #3640 <https://github.com/Qiskit/qiskit-terra/issues/3640>`_


.. _Release Notes_0.12.0_Other Notes:

Other Notes
-----------

- The transpiler passes in the :mod:`qiskit.transpiler.passes` directory have
  been organized into subdirectories to better categorize them by
  functionality. They are still all accessible under the
  ``qiskit.transpiler.passes`` namespace.

Aer 0.4.0
=========

Added
-----
 * Added ``NoiseModel.from_backend`` for building a basic device noise model for an IBMQ
   backend (\#569)
 * Added multi-GPU enabled simulation methods to the ``QasmSimulator``,
   ``StatevectorSimulator``, and ``UnitarySimulator``. The qasm simulator has gpu version
   of the density matrix and statevector methods and can be accessed using
   ``"method": "density_matrix_gpu"`` or ``"method": "statevector_gpu"`` in ``backend_options``.
   The statevector simulator gpu method can be accessed using ``"method": "statevector_gpu"``.
   The unitary simulator GPU method can be accessed using ``"method": "unitary_gpu"``. These
   backends use CUDA and require an NVidia GPU.(\#544)
 * Added ``PulseSimulator`` backend (\#542)
 * Added ``PulseSystemModel`` and ``HamiltonianModel`` classes to represent models to be used
   in ``PulseSimulator`` (\#496, \#493)
 * Added ``duffing_model_generators`` to generate ``PulseSystemModel`` objects from a list
   of parameters (\#516)
 * Migrated ODE function solver to C++ (\#442, \#350)
 * Added high level pulse simulator tests (\#379)
 * CMake BLAS_LIB_PATH flag to set path to look for BLAS lib (\#543)

Changed
-------

 * Changed the structure of the ``src`` directory to organise simulator source code.
   Simulator controller headers were moved to ``src/controllers`` and simulator method State
   headers are in ``src/simulators`` (\#544)
 * Moved the location of several functions (\#568):
   * Moved contents of ``qiskit.provider.aer.noise.errors`` into
   the ``qiskit.providers.noise`` module
   * Moved contents of ``qiskit.provider.aer.noise.utils`` into
   the ``qiskit.provider.aer.utils`` module.
 * Enabled optimization to aggregate consecutive gates in a circuit (fusion) by default (\#579).

Deprecated
----------
 * Deprecated ``utils.qobj_utils`` functions (\#568)
 * Deprecated ``qiskit.providers.aer.noise.device.basic_device_noise_model``. It is superseded
   by the ``NoiseModel.from_backend`` method (\#569)

Removed
-------
 * Removed ``NoiseModel.as_dict``, ``QuantumError.as_dict``, ``ReadoutError.as_dict``, and
   ``QuantumError.kron`` methods that were deprecated in 0.3 (\#568).

Ignis 0.2
=========

No Change

Aqua 0.6
========

No Change

IBM Q Provider 0.4.6
====================

Added
-----

- Several new methods were added to
  :class:`IBMQBackend<qiskit.providers.ibmq.ibmqbackend.IBMQBackend>`:

    * :meth:`~qiskit.providers.ibmq.job.IBMQJob.wait_for_final_state`
      blocks until the job finishes. It takes a callback function that it will invoke
      after every query to provide feedback.
    * :meth:`~qiskit.providers.ibmq.ibmqbackend.IBMQBackend.active_jobs` returns
      the jobs submitted to a backend that are currently in an unfinished status.
    * :meth:`~qiskit.providers.ibmq.ibmqbackend.IBMQBackend.job_limit` returns the
      job limit for a backend.
    * :meth:`~qiskit.providers.ibmq.ibmqbackend.IBMQBackend.remaining_jobs_count` returns the
      number of jobs that you can submit to the backend before job limit is reached.

- :class:`~qiskit.providers.ibmq.job.QueueInfo` now has a new
  :meth:`~qiskit.providers.ibmq.job.QueueInfo.format` method that returns a
  formatted string of the queue information.

- :class:`IBMQJob<qiskit.providers.ibmq.job.IBMQJob>` now has three new methods:
  :meth:`~qiskit.providers.ibmq.job.IBMQJob.done`,
  :meth:`~qiskit.providers.ibmq.job.IBMQJob.running`, and
  :meth:`~qiskit.providers.ibmq.job.IBMQJob.cancelled` that are used to indicate job status.

- :meth:`qiskit.providers.ibmq.ibmqbackend.IBMQBackend.run()` now accepts an optional `job_tags`
  parameter. If specified, the `job_tags` are assigned to the job, which can later be used
  as a filter in :meth:`qiskit.providers.ibmq.ibmqbackend.IBMQBackend.jobs()`.

- :class:`~qiskit.providers.ibmq.managed.IBMQJobManager` now has a new method
  :meth:`~qiskit.providers.ibmq.managed.IBMQJobManager.retrieve_job_set()` that allows
  you to retrieve a previously submitted job set using the job set ID.

Changed
-------

- The ``Exception`` hierarchy has been refined with more specialized classes.
  You can, however, continue to catch their parent exceptions (such
  as ``IBMQAccountError``). Also, the exception class ``IBMQApiUrlError``
  has been replaced by ``IBMQAccountCredentialsInvalidUrl`` and
  ``IBMQAccountCredentialsInvalidToken``.

Deprecated
----------

- The use of proxy urls without a protocol (e.g. ``http://``) is deprecated
  due to recent Python changes.

*************
Qiskit 0.14.0
*************

Terra 0.11.0
============

.. _Release Notes_0.11.0_Prelude:

Prelude
-------

The 0.11.0 release includes several new features and bug fixes. The biggest
change for this release is the addition of the pulse scheduler. This allows
users to define their quantum program as a ``QuantumCircuit`` and then map it
to the underlying pulse instructions that will control the quantum hardware to
implement the circuit.

.. _Release Notes_0.11.0_New Features:

New Features
------------

- Added 5 new commands to easily retrieve user-specific data from
  ``BackendProperties``: ``gate_property``, ``gate_error``, ``gate_length``,
  ``qubit_property``, ``t1``, ``t2``, ``readout_error`` and ``frequency``.
  They return the specific values of backend properties. For example::

    from qiskit.test.mock import FakeOurense
    backend = FakeOurense()
    properties = backend.properties()

    gate_property = properties.gate_property('u1')
    gate_error = properties.gate_error('u1', 0)
    gate_length = properties.gate_length('u1', 0)
    qubit_0_property = properties.qubit_property(0)
    t1_time_0 = properties.t1(0)
    t2_time_0 = properties.t2(0)
    readout_error_0 = properties.readout_error(0)
    frequency_0 = properties.frequency(0)

- Added method ``Instruction.is_parameterized()`` to check if an instruction
  object is parameterized. This method returns ``True`` if and only if
  instruction has a ``ParameterExpression`` or ``Parameter`` object for one
  of its params.

- Added a new analysis pass ``Layout2qDistance``. This pass allows to "score"
  a layout selection, once ``property_set['layout']`` is set.  The score will
  be the sum of distances for each two-qubit gate in the circuit, when they
  are not directly connected. This scoring does not consider direction in the
  coupling map. The lower the number, the better the layout selection is.

  For example, consider a linear coupling map ``[0]--[2]--[1]`` and the
  following circuit::

      qr = QuantumRegister(2, 'qr')
      circuit = QuantumCircuit(qr)
      circuit.cx(qr[0], qr[1])

  If the layout is ``{qr[0]:0, qr[1]:1}``, ``Layout2qDistance`` will set
  ``property_set['layout_score'] = 1``. If the layout
  is ``{qr[0]:0, qr[1]:2}``, then the result
  is ``property_set['layout_score'] = 0``. The lower the score, the better.

- Added ``qiskit.QuantumCircuit.cnot`` as an alias for the ``cx`` method of
  ``QuantumCircuit``. The names ``cnot`` and ``cx`` are often used
  interchangeably now the `cx` method can be called with either name.

- Added ``qiskit.QuantumCircuit.toffoli`` as an alias for the ``ccx`` method
  of ``QuantumCircuit``. The names ``toffoli`` and ``ccx`` are often used
  interchangeably now the `ccx` method can be called with either name.

- Added ``qiskit.QuantumCircuit.fredkin`` as an alias for the ``cswap``
  method of ``QuantumCircuit``. The names ``fredkin`` and ``cswap`` are
  often used interchangeably now the `cswap` method can be called with
  either name.

- The ``latex`` output mode for ``qiskit.visualization.circuit_drawer()`` and
  the ``qiskit.circuit.QuantumCircuit.draw()`` method now has a mode to
  passthrough raw latex from gate labels and parameters. The syntax
  for doing this mirrors matplotlib's
  `mathtext mode <https://matplotlib.org/tutorials/text/mathtext.html>`__
  syntax. Any portion of a label string between a pair of '$' characters will
  be treated as raw latex and passed directly into the generated output latex.
  This can be leveraged to add more advanced formatting to circuit diagrams
  generated with the latex drawer.

  Prior to this release all gate labels were run through a utf8 -> latex
  conversion to make sure that the output latex would compile the string as
  expected. This is still what happens for all portions of a label outside
  the '$' pair. Also if you want to use a dollar sign in your label make sure
  you escape it in the label string (ie ``'\$'``).

  You can mix and match this passthrough with the utf8 -> latex conversion to
  create the exact label you want, for example::

      from qiskit import circuit
      circ = circuit.QuantumCircuit(2)
      circ.h([0, 1])
      circ.append(circuit.Gate(name='α_gate', num_qubits=1, params=[0]), [0])
      circ.append(circuit.Gate(name='α_gate$_2$', num_qubits=1, params=[0]), [1])
      circ.append(circuit.Gate(name='\$α\$_gate', num_qubits=1, params=[0]), [1])
      circ.draw(output='latex')

  will now render the first custom gate's label as ``α_gate``, the second
  will be ``α_gate`` with a 2 subscript, and the last custom gate's label
  will be ``$α$_gate``.

- Add ``ControlledGate`` class for representing controlled
  gates. Controlled gate instances are created with the
  ``control(n)`` method of ``Gate`` objects where ``n`` represents
  the number of controls. The control qubits come before the
  controlled qubits in the new gate. For example::

    from qiskit import QuantumCircuit
    from qiskit.extensions import HGate
    hgate = HGate()
    circ = QuantumCircuit(4)
    circ.append(hgate.control(3), [0, 1, 2, 3])
    print(circ)

  generates::

    q_0: |0>──■──
              │
    q_1: |0>──■──
              │
    q_2: |0>──■──
            ┌─┴─┐
    q_3: |0>┤ H ├
            └───┘

- Allowed values of ``meas_level`` parameters and fields can now be a member
  from the `IntEnum` class ``qiskit.qobj.utils.MeasLevel``. This can be used
  when calling ``execute`` (or anywhere else ``meas_level`` is specified) with
  a pulse experiment. For example::

    from qiskit import QuantumCircuit, transpile, schedule, execute
    from qiskit.test.mock import FakeOpenPulse2Q
    from qiskit.qobj.utils import MeasLevel, MeasReturnType

    backend = FakeOpenPulse2Q()
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0,1)
    qc_transpiled = transpile(qc, backend)
    sched = schedule(qc_transpiled, backend)
    execute(sched, backend, meas_level=MeasLevel.CLASSIFIED)

  In this above example, ``meas_level=MeasLevel.CLASSIFIED`` and
  ``meas_level=2`` can be used interchangably now.

- A new layout selector based on constraint solving is included. `CSPLayout` models the problem
  of finding a layout as a constraint problem and uses recursive backtracking to solve it.

  .. code-block:: python

     cmap16 = CouplingMap(FakeRueschlikon().configuration().coupling_map)

     qr = QuantumRegister(5, 'q')
     circuit = QuantumCircuit(qr)
     circuit.cx(qr[0], qr[1])
     circuit.cx(qr[0], qr[2])
     circuit.cx(qr[0], qr[3])

     pm = PassManager(CSPLayout(cmap16))
     circuit_after = pm.run(circuit)
     print(pm.property_set['layout'])


  .. code-block:: python

      Layout({
      1: Qubit(QuantumRegister(5, 'q'), 1),
      2: Qubit(QuantumRegister(5, 'q'), 0),
      3: Qubit(QuantumRegister(5, 'q'), 3),
      4: Qubit(QuantumRegister(5, 'q'), 4),
      15: Qubit(QuantumRegister(5, 'q'), 2)
      })


  The parameter ``CSPLayout(...,strict_direction=True)`` is more restrictive
  but it will guarantee there is no need of running ``CXDirection`` after.

  .. code-block:: python

      pm = PassManager(CSPLayout(cmap16, strict_direction=True))
      circuit_after = pm.run(circuit)
      print(pm.property_set['layout'])

  .. code-block:: python

      Layout({
      8: Qubit(QuantumRegister(5, 'q'), 4),
      11: Qubit(QuantumRegister(5, 'q'), 3),
      5: Qubit(QuantumRegister(5, 'q'), 1),
      6: Qubit(QuantumRegister(5, 'q'), 0),
      7: Qubit(QuantumRegister(5, 'q'), 2)
      })

  If the constraint system is not solvable, the `layout` property is not set.

  .. code-block:: python

      circuit.cx(qr[0], qr[4])
      pm = PassManager(CSPLayout(cmap16))
      circuit_after = pm.run(circuit)
      print(pm.property_set['layout'])

  .. code-block:: python

      None

- PulseBackendConfiguration (accessed normally as backend.configuration())
  has been extended with useful methods to explore its data and the
  functionality that exists in PulseChannelSpec. PulseChannelSpec will be
  deprecated in the future. For example::

      backend = provider.get_backend(backend_name)
      config = backend.configuration()
      q0_drive = config.drive(0)  # or, DriveChannel(0)
      q0_meas = config.measure(0)  # MeasureChannel(0)
      q0_acquire = config.acquire(0)  # AcquireChannel(0)
      config.hamiltonian  # Returns a dictionary with hamiltonian info
      config.sample_rate()  # New method which returns 1 / dt

- ``PulseDefaults`` (accessed normally as ``backend.defaults()``) has an
  attribute, ``circuit_instruction_map`` which has the methods of CmdDef.
  The new `circuit_instruction_map` is an ``InstructionScheduleMap`` object
  with three new functions beyond what CmdDef had:

   * qubit_instructions(qubits) returns the operations defined for the qubits
   * assert_has(instruction, qubits) raises an error if the op isn't defined
   * remove(instruction, qubits) like pop, but doesn't require parameters

  There are some differences from the CmdDef:

   * ``__init__`` takes no arguments
   * ``cmds`` and ``cmd_qubits`` are deprecated and replaced with
     ``instructions`` and ``qubits_with_instruction``

  Example::

      backend = provider.get_backend(backend_name)
      inst_map = backend.defaults().circuit_instruction_map
      qubit = inst_map.qubits_with_instruction('u3')[0]
      x_gate = inst_map.get('u3', qubit, P0=np.pi, P1=0, P2=np.pi)
      pulse_schedule = x_gate(DriveChannel(qubit))

- A new kwarg parameter, ``show_framechange_channels`` to optionally disable
  displaying channels with only framechange instructions in pulse
  visualizations was added to the ``qiskit.visualization.pulse_drawer()``
  function and ``qiskit.pulse.Schedule.draw()`` method. When this new kwarg
  is set to ``False`` the output pulse schedule visualization will not
  include any channels that only include frame changes.

  For example:

  .. jupyter-execute::

      from qiskit.pulse import *
      from qiskit.pulse import library as pulse_lib

      gp0 = pulse_lib.gaussian(duration=20, amp=1.0, sigma=1.0)
      sched = Schedule()
      channel_a = DriveChannel(0)
      channel_b = DriveChannel(1)
      sched += Play(gp0, channel_a)
      sched = sched.insert(60, ShiftPhase(-1.57, channel_a))
      sched = sched.insert(30, ShiftPhase(-1.50, channel_b))
      sched = sched.insert(70, ShiftPhase(1.50, channel_b))

      sched.draw(show_framechange_channels=False)

- A new utility function ``qiskit.result.marginal_counts()`` is added
  which allows marginalization of the counts over some indices of interest.
  This is useful when more qubits are measured than needed, and one wishes
  to get the observation counts for some subset of them only.

- When ``passmanager.run(...)`` is invoked with more than one circuit, the
  transpilation of these circuits will run in parallel.

- PassManagers can now be sliced to create a new PassManager containing a
  subset of passes using the square bracket operator. This allow running or
  drawing a portion of the PassManager for easier testing and visualization.
  For example let's try to draw the first 3 passes of a PassManager pm, or
  run just the second pass on our circuit:

  .. code-block:: python

    pm[0:4].draw()
    circuit2 = pm[1].run(circuit)

  Also now, PassManagers can be created by adding two PassManagers or by
  directly adding a pass/list of passes to a PassManager.

  .. code-block:: python

    pm = pm1[0] + pm2[1:3]
    pm += [setLayout, unroller]

- A basic ``scheduler`` module has now been added to Qiskit. The `scheduler`
  schedules an input transpiled ``QuantumCircuit`` into a pulse
  ``Schedule``. The scheduler accepts as input a ``Schedule`` and either a
  pulse ``Backend``, or a ``CmdDef`` which relates circuit ``Instruction``
  objects on specific qubits to pulse Schedules and a ``meas_map`` which
  determines which measurements must occur together.

  Scheduling example::

    from qiskit import QuantumCircuit, transpile, schedule
    from qiskit.test.mock import FakeOpenPulse2Q

    backend = FakeOpenPulse2Q()
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0,1)
    qc_transpiled = transpile(qc, backend)
    schedule(qc_transpiled, backend)

  The scheduler currently supports two scheduling policies,
  `as_late_as_possible` (``alap``) and `as_soon_as_possible` (``asap``), which
  respectively schedule pulse instructions to occur as late as
  possible or as soon as possible across qubits in a circuit.
  The scheduling policy may be selected with the input argument ``method``,
  for example::

    schedule(qc_transpiled, backend, method='alap')

  It is easy to use a pulse ``Schedule`` within a ``QuantumCircuit`` by
  mapping it to a custom circuit instruction such as a gate which may be used
  in a ``QuantumCircuit``. To do this, first, define the custom gate and then
  add an entry into the ``CmdDef`` for the gate, for each qubit that the gate
  will be applied to. The gate can then be used in the ``QuantumCircuit``.
  At scheduling time the gate will be mapped to the underlying pulse schedule.
  Using this technique allows easy integration with preexisting qiskit modules
  such as Ignis.

  For example::

      from qiskit import pulse, circuit, schedule
      from qiskit.pulse import pulse_lib

      custom_cmd_def = pulse.CmdDef()

      # create custom gate
      custom_gate = circuit.Gate(name='custom_gate', num_qubits=1, params=[])

      # define schedule for custom gate
      custom_schedule = pulse.Schedule()
      custom_schedule += pulse_lib.gaussian(20, 1.0, 10)(pulse.DriveChannel)

      # add schedule to custom gate with same name
      custom_cmd_def.add('custom_gate', (0,), custom_schedule)

      # use custom gate in a circuit
      custom_qc = circuit.QuantumCircuit(1)
      custom_qc.append(custom_gate, qargs=[0])

      # schedule the custom gate
      schedule(custom_qc, cmd_def=custom_cmd_def, meas_map=[[0]])


.. _Release Notes_0.11.0_Known Issues:

Known Issues
------------

- The feature for transpiling in parallel when ``passmanager.run(...)`` is
  invoked with more than one circuit is not supported under Windows. See
  `#2988 <https://github.com/Qiskit/qiskit-terra/issues/2988>`__ for more
  details.


.. _Release Notes_0.11.0_Upgrade Notes:

Upgrade Notes
-------------

- The ``qiskit.pulse.channels.SystemTopology`` class was used as a helper
  class for ``PulseChannelSpec``. It has been removed since with the deprecation
  of ``PulseChannelSpec`` and changes to ``BackendConfiguration`` make it
  unnecessary.

- The previously deprecated representation of qubits and classical bits as
  tuple, which was deprecated in the 0.9 release, has been removed. The use
  of ``Qubit`` and ``Clbit`` objects is the new way to represent qubits and
  classical bits.

- The previously deprecated representation of the basis set as single string
  has been removed. A list of strings is the new preferred way.

- The method ``BaseModel.as_dict``, which was deprecated in the 0.9 release,
  has been removed in favor of the method ``BaseModel.to_dict``.

- In PulseDefaults (accessed normally as backend.defaults()),
  ``qubit_freq_est`` and ``meas_freq_est`` are now returned in Hz rather than
  GHz. This means the new return values are 1e9 * their previous value.

- `dill <https://pypi.org/project/dill/>`__ was added as a requirement. This
  is needed to enable running ``passmanager.run()`` in parallel for more than
  one circuit.

- The previously deprecated gate ``UBase``, which was deprecated
  in the 0.9 release, has been removed. The gate ``U3Gate``
  should be used instead.

- The previously deprecated gate ``CXBase``, which was deprecated
  in the 0.9 release, has been removed. The gate ``CnotGate``
  should be used instead.

- The instruction ``snapshot`` used to implicitly convert the ``label``
  parameter to string. That conversion has been removed and an error is raised
  if a string is not provided.

- The previously deprecated gate ``U0Gate``, which was deprecated
  in the 0.9 release, has been removed. The gate ``IdGate``
  should be used instead to insert delays.


.. _Release Notes_0.11.0_Deprecation Notes:

Deprecation Notes
-----------------

- The ``qiskit.pulse.CmdDef`` class has been deprecated. Instead you should
  use the ``qiskit.pulse.InstructionScheduleMap``. The
  ``InstructionScheduleMap`` object for a pulse enabled system can be
  accessed at ``backend.defaults().instruction_schedules``.

- ``PulseChannelSpec`` is being deprecated. Use ``BackendConfiguration``
  instead. The backend configuration is accessed normally as
  ``backend.configuration()``. The config has been extended with most of
  the functionality of PulseChannelSpec, with some modifications as follows,
  where `0` is an exemplary qubit index::

      pulse_spec.drives[0]   -> config.drive(0)
      pulse_spec.measures[0] -> config.measure(0)
      pulse_spec.acquires[0] -> config.acquire(0)
      pulse_spec.controls[0] -> config.control(0)

  Now, if there is an attempt to get a channel for a qubit which does not
  exist for the device, a ``BackendConfigurationError`` will be raised with
  a helpful explanation.

  The methods ``memoryslots`` and ``registerslots`` of the PulseChannelSpec
  have not been migrated to the backend configuration. These classical
  resources are not restrained by the physical configuration of a backend
  system. Please instantiate them directly::

      pulse_spec.memoryslots[0] -> MemorySlot(0)
      pulse_spec.registerslots[0] -> RegisterSlot(0)

  The ``qubits`` method is not migrated to backend configuration. The result
  of ``qubits`` can be built as such::

      [q for q in range(backend.configuration().n_qubits)]

- ``Qubit`` within ``pulse.channels`` has been deprecated. They should not
  be used. It is possible to obtain channel <=> qubit mappings through the
  BackendConfiguration (or backend.configuration()).

- The function ``qiskit.visualization.circuit_drawer.qx_color_scheme()`` has
  been deprecated. This function is no longer used internally and doesn't
  reflect the current IBM QX style. If you were using this function to
  generate a style dict locally you must save the output from it and use
  that dictionary directly.

- The Exception ``TranspilerAccessError`` has been deprecated. An
  alternative function ``TranspilerError`` can be used instead to provide
  the same functionality. This alternative function provides the exact same
  functionality but with greater generality.

- Buffers in Pulse are deprecated. If a nonzero buffer is supplied, a warning
  will be issued with a reminder to use a Delay instead. Other options would
  include adding samples to a pulse instruction which are (0.+0.j) or setting
  the start time of the next pulse to ``schedule.duration + buffer``.

- Passing in ``sympy.Basic``, ``sympy.Expr`` and ``sympy.Matrix`` types as
  instruction parameters are deprecated and will be removed in a future
  release. You'll need to convert the input to one of the supported types
  which are:

   * ``int``
   * ``float``
   * ``complex``
   * ``str``
   * ``np.ndarray``


.. _Release Notes_0.11.0_Bug Fixes:

Bug Fixes
---------

- The Collect2qBlocks and CommutationAnalysis passes in the transpiler had been
  unable to process circuits containing Parameterized gates, preventing
  Parameterized circuits from being transpiled at optimization_level 2 or
  above. These passes have been corrected to treat Parameterized gates as
  opaque.

- The align_measures function had an issue where Measure stimulus
  pulses weren't properly aligned with Acquire pulses, resulting in
  an error. This has been fixed.

- Uses of ``numpy.random.seed`` have been removed so that calls of qiskit
  functions do not affect results of future calls to ``numpy.random``

- Fixed race condition occurring in the job monitor when
  ``job.queue_position()`` returns ``None``. ``None`` is a valid
  return from ``job.queue_position()``.

- Backend support for ``memory=True`` now checked when that kwarg is passed.
  ``QiskitError`` results if not supported.

- When transpiling without a coupling map, there were no check in the amount
  of qubits of the circuit to transpile. Now the transpile process checks
  that the backend has enough qubits to allocate the circuit.


.. _Release Notes_0.11.0_Other Notes:

Other Notes
-----------

- The ``qiskit.result.marginal_counts()`` function replaces a similar utility
  function in qiskit-ignis
  ``qiskit.ignis.verification.tomography.marginal_counts()``, which
  will be deprecated in a future qiskit-ignis release.

- All sympy parameter output type support have been been removed (or
  deprecated as noted) from qiskit-terra. This includes sympy type
  parameters in ``QuantumCircuit`` objects, qasm ast nodes, or ``Qobj``
  objects.

Aer 0.3
=======

No Change

Ignis 0.2
=========

No Change

Aqua 0.6
========

No Change

IBM Q Provider 0.4
==================

Prelude
-------

The 0.4.0 release is the first release that makes use of all the features
of the new IBM Q API. In particular, the ``IBMQJob`` class has been revamped in
order to be able to retrieve more information from IBM Q, and a Job Manager
class has been added for allowing a higher-level and more seamless usage of
large or complex jobs. If you have not upgraded from the legacy IBM Q
Experience or QConsole yet, please ensure to revisit the release notes for
IBM Q Provider 0.3 (Qiskit 0.11) for more details on how to make the
transition. The legacy accounts will no longer be supported as of this release.


New Features
------------

Job modifications
^^^^^^^^^^^^^^^^^

The ``IBMQJob`` class has been revised, and now mimics more closely to the
contents of a remote job along with new features:

* You can now assign a name to a job, by specifying
  ``IBMQBackend.run(..., job_name='...')`` when submitting a job. This name
  can be retrieved via ``IBMQJob.name()`` and can be used for filtering.
* Jobs can now be shared with other users at different levels (global, per
  hub, group or project) via an optional ``job_share_level`` parameter when
  submitting the job.
* ``IBMQJob`` instances now have more attributes, reflecting the contents of the
  remote IBM Q jobs. This implies that new attributes introduced by the IBM Q
  API will automatically and immediately be available for use (for example,
  ``job.new_api_attribute``). The new attributes will be promoted to methods
  when they are considered stable (for example, ``job.name()``).
* ``.error_message()`` returns more information on why a job failed.
* ``.queue_position()`` accepts a ``refresh`` parameter for forcing an update.
* ``.result()`` accepts an optional ``partial`` parameter, for returning
  partial results, if any, of jobs that failed. Be aware that ``Result``
  methods, such as ``get_counts()`` will raise an exception if applied on
  experiments that failed.

Please note that the changes include some low-level modifications of the class.
If you were creating the instances manually, note that:

* the signature of the constructor has changed to account for the new features.
* the ``.submit()`` method can no longer be called directly, and jobs are
  expected to be submitted either via the synchronous ``IBMQBackend.run()`` or
  via the Job Manager.

Job Manager
^^^^^^^^^^^

A new Job Manager (``IBMQJobManager``) has been introduced, as a higher-level
mechanism for handling jobs composed of multiple circuits or pulse schedules.
The Job Manager aims to provide a transparent interface, intelligently splitting
the input into efficient units of work and taking full advantage of the
different components. It will be expanded on upcoming versions, and become the
recommended entry point for job submission.

Its ``.run()`` method receives a list of circuits or pulse schedules, and
returns a ``ManagedJobSet instance``, which can then be used to track the
statuses and results of these jobs. For example::

    from qiskit.providers.ibmq.managed import IBMQJobManager
    from qiskit.circuit.random import random_circuit
    from qiskit import IBMQ
    from qiskit.compiler import transpile

    provider = IBMQ.load_account()
    backend = provider.backends.ibmq_ourense

    circs = []
    for _ in range(1000000):
        circs.append(random_circuit(2, 2))
    transpile(circs, backend=backend)

    # Farm out the jobs.
    jm = IBMQJobManager()
    job_set = jm.run(circs, backend=backend, name='foo')

    job_set.statuses()    # Gives a list of job statuses
    job_set.report()    # Prints detailed job information
    results = job_set.results()
    counts = results.get_counts(5)   # Returns data for experiment 5


provider.backends modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``provider.backends`` member, which was previously a function that returned
a list of backends, has been promoted to a service. This implies that it can
be used both in the previous way, as a ``.backends()`` method, and also as a
``.backends`` attribute with expanded capabilities:

* it contains the existing backends from that provider as attributes, which
  can be used for autocompletion. For example::

      my_backend = provider.get_backend('ibmq_qasm_simulator')

  is equivalent to::

      my_backend = provider.backends.ibmq_qasm_simulator

* the ``provider.backends.jobs()`` and ``provider.backends.retrieve_job()``
  methods can be used for retrieving provider-wide jobs.


Other changes
^^^^^^^^^^^^^

* The ``backend.properties()`` function now accepts an optional ``datetime``
  parameter. If specified, the function returns the backend properties
  closest to, but older than, the specified datetime filter.
* Some ``warnings`` have been toned down to ``logger.warning`` messages.


*************
Qiskit 0.13.0
*************

Terra 0.10.0
============

.. _Release Notes_0.10.0_Prelude:

Prelude
-------

The 0.10.0 release includes several new features and bug fixes. The biggest
change for this release is the addition of initial support for using Qiskit
with trapped ion trap backends.


.. _Release Notes_0.10.0_New Features:

New Features
------------

- Introduced new methods in ``QuantumCircuit`` which allows the seamless adding or removing
  of measurements at the end of a circuit.

  ``measure_all()``
    Adds a ``barrier`` followed by a ``measure`` operation to all qubits in the circuit.
    Creates a ``ClassicalRegister`` of size equal to the number of qubits in the circuit,
    which store the measurements.

  ``measure_active()``
    Adds a ``barrier`` followed by a ``measure`` operation to all active qubits in the circuit.
    A qubit is active if it has at least one other operation acting upon it.
    Creates a ``ClassicalRegister`` of size equal to the number of active qubits in the circuit,
    which store the measurements.

  ``remove_final_measurements()``
    Removes all final measurements and preceeding ``barrier`` from a circuit.
    A measurement is considered "final" if it is not followed by any other operation,
    excluding barriers and other measurements.
    After the measurements are removed, if all of the classical bits in the ``ClassicalRegister``
    are idle (have no operations attached to them), then the ``ClassicalRegister`` is removed.

  Examples::

        # Using measure_all()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.measure_all()
        circuit.draw()

        # A ClassicalRegister with prefix measure was created.
        # It has 2 clbits because there are 2 qubits to measure

                     ┌───┐ ░ ┌─┐
             q_0: |0>┤ H ├─░─┤M├───
                     └───┘ ░ └╥┘┌─┐
             q_1: |0>──────░──╫─┤M├
                           ░  ║ └╥┘
        measure_0: 0 ═════════╩══╬═
                                 ║
        measure_1: 0 ════════════╩═


        # Using measure_active()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.measure_active()
        circuit.draw()

        # This ClassicalRegister only has 1 clbit because only 1 qubit is active

                     ┌───┐ ░ ┌─┐
             q_0: |0>┤ H ├─░─┤M├
                     └───┘ ░ └╥┘
             q_1: |0>──────░──╫─
                           ░  ║
        measure_0: 0 ═════════╩═


        # Using remove_final_measurements()
        # Assuming circuit_all and circuit_active are the circuits from the measure_all and
        # measure_active examples above respectively

        circuit_all.remove_final_measurements()
        circuit_all.draw()
        # The ClassicalRegister is removed because, after the measurements were removed,
        # all of its clbits were idle

                ┌───┐
        q_0: |0>┤ H ├
                └───┘
        q_1: |0>─────


        circuit_active.remove_final_measurements()
        circuit_active.draw()
        # This will result in the same circuit

                ┌───┐
        q_0: |0>┤ H ├
                └───┘
        q_1: |0>─────

- Initial support for executing experiments on ion trap backends has been
  added.

- An Rxx gate (rxx) and a global Mølmer–Sørensen gate (ms) have been added
  to the standard gate set.

- A Cnot to Rxx/Rx/Ry decomposer ``cnot_rxx_decompose`` and a single qubit
  Euler angle decomposer ``OneQubitEulerDecomposer`` have been added to the
  ``quantum_info.synthesis`` module.

- A transpiler pass ``MSBasisDecomposer`` has been added to unroll circuits
  defined over U3 and Cnot gates into a circuit defined over Rxx,Ry and Rx.
  This pass will be included in preset pass managers for backends which
  include the 'rxx' gate in their supported basis gates.

- The backends in ``qiskit.test.mock`` now contain a snapshot of real
  device calibration data. This is accessible via the ``properties()`` method
  for each backend. This can be used to test any code that depends on
  backend properties, such as noise-adaptive transpiler passes or device
  noise models for simulation. This will create a faster testing and
  development cycle without the need to go to live backends.

- Allows the Result class to return partial results. If a valid result schema
  is loaded that contains some experiments which succeeded and some which
  failed, this allows accessing the data from experiments that succeeded,
  while raising an exception for experiments that failed and displaying the
  appropriate error message for the failed results.

- An ``ax`` kwarg has been added to the following visualization functions:

   * ``qiskit.visualization.plot_histogram``
   * ``qiskit.visualization.plot_state_paulivec``
   * ``qiskit.visualization.plot_state_qsphere``
   * ``qiskit.visualization.circuit_drawer`` (``mpl`` backend only)
   * ``qiskit.QuantumCircuit.draw`` (``mpl`` backend only)

  This kwarg is used to pass in a ``matplotlib.axes.Axes`` object to the
  visualization functions. This enables integrating these visualization
  functions into a larger visualization workflow. Also, if an `ax` kwarg is
  specified then there is no return from the visualization functions.

- An ``ax_real`` and ``ax_imag`` kwarg has been added to the
  following visualization functions:

   * ``qiskit.visualization.plot_state_hinton``
   * ``qiskit.visualization.plot_state_city``

  These new kargs work the same as the newly added ``ax`` kwargs for other
  visualization functions. However because these plots use two axes (one for
  the real component, the other for the imaginary component). Having two
  kwargs also provides the flexibility to only generate a visualization for
  one of the components instead of always doing both. For example::

      from matplotlib import pyplot as plt
      from qiskit.visualization import plot_state_hinton

      ax = plt.gca()

      plot_state_hinton(psi, ax_real=ax)

  will only generate a plot of the real component.

- A given pass manager now can be edited with the new method `replace`. This method allows to
  replace a particular stage in a pass manager, which can be handy when dealing with preset
  pass managers. For example, let's edit the layout selector of the pass manager used at
  optimization level 0:

  .. code-block:: python

    from qiskit.transpiler.preset_passmanagers.level0 import level_0_pass_manager
    from qiskit.transpiler.transpile_config import TranspileConfig

    pass_manager = level_0_pass_manager(TranspileConfig(coupling_map=CouplingMap([[0,1]])))

    pass_manager.draw()

  .. code-block::

    [0] FlowLinear: SetLayout
    [1] Conditional: TrivialLayout
    [2] FlowLinear: FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout
    [3] FlowLinear: Unroller

  The layout selection is set in the stage `[1]`. Let's replace it with `DenseLayout`:

  .. code-block:: python

    from qiskit.transpiler.passes import DenseLayout

    pass_manager.replace(1, DenseLayout(coupling_map), condition=lambda property_set: not property_set['layout'])
    pass_manager.draw()

  .. code-block::

    [0] FlowLinear: SetLayout
    [1] Conditional: DenseLayout
    [2] FlowLinear: FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout
    [3] FlowLinear: Unroller

  If you want to replace it without any condition, you can use set-item shortcut:

  .. code-block:: python

    pass_manager[1] = DenseLayout(coupling_map)
    pass_manager.draw()

  .. code-block::

    [0] FlowLinear: SetLayout
    [1] FlowLinear: DenseLayout
    [2] FlowLinear: FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout
    [3] FlowLinear: Unroller

- Introduced a new pulse command ``Delay`` which may be inserted into a pulse
  ``Schedule``. This command accepts a ``duration`` and may be added to any
  ``Channel``. Other commands may not be scheduled on a channel during a delay.

  The delay can be added just like any other pulse command. For example::

    from qiskit import pulse
    from qiskit.pulse.utils import pad

    dc0 = pulse.DriveChannel(0)

    delay = pulse.Delay(1)
    test_pulse = pulse.SamplePulse([1.0])

    sched = pulse.Schedule()
    sched += test_pulse(dc0).shift(1)

    # build padded schedule by hand
    ref_sched = delay(dc0) | sched

    # pad schedule
    padded_sched = pad(sched)

    assert padded_sched == ref_sched

  One may also pass additional channels to be padded and a time to pad until,
  for example::

    from qiskit import pulse
    from qiskit.pulse.utils import pad

    dc0 = pulse.DriveChannel(0)
    dc1 = pulse.DriveChannel(1)

    delay = pulse.Delay(1)
    test_pulse = pulse.SamplePulse([1.0])

    sched = pulse.Schedule()
    sched += test_pulse(dc0).shift(1)

    # build padded schedule by hand
    ref_sched = delay(dc0) | delay(dc1) |  sched

    # pad schedule across both channels until up until the first time step
    padded_sched = pad(sched, channels=[dc0, dc1], until=1)

    assert padded_sched == ref_sched


.. _Release Notes_0.10.0_Upgrade Notes:

Upgrade Notes
-------------

- Assignments and modifications to the ``data`` attribute of
  ``qiskit.QuantumCircuit`` objects are now validated following the same
  rules used throughout the ``QuantumCircuit`` API. This was done to
  improve the performance of the circuits API since we can now assume the
  ``data`` attribute is in a known format. If you were manually modifying
  the ``data`` attribute of a circuit object before this may no longer work
  if your modifications resulted in a data structure other than the list
  of instructions with context in the format ``[(instruction, qargs, cargs)]``

- The transpiler default passmanager for optimization level 2 now uses the
  ``DenseLayout`` layout selection mechanism by default instead of
  ``NoiseAdaptiveLayout``. The ``Denselayout`` pass has also been modified
  to be made noise-aware.

- The deprecated ``DeviceSpecification`` class has been removed. Instead you should
  use the ``PulseChannelSpec``. For example, you can run something like::

      device = pulse.PulseChannelSpec.from_backend(backend)
      device.drives[0]    # for DeviceSpecification, this was device.q[0].drive
      device.memoryslots  # this was device.mem

- The deprecated module ``qiskit.pulse.ops`` has been removed. Use
  ``Schedule`` and ``Instruction`` methods directly. For example, rather
  than::

      ops.union(schedule_0, schedule_1)
      ops.union(instruction, schedule)  # etc

  Instead please use::

      schedule_0.union(schedule_1)
      instruction.union(schedule)

  This same pattern applies to other ``ops`` functions: ``insert``, ``shift``,
  ``append``, and ``flatten``.


.. _Release Notes_0.10.0_Deprecation Notes:

Deprecation Notes
-----------------

- Using the ``control`` property of ``qiskit.circuit.Instruction`` for
  classical control is now deprecated. In the future this property will be
  used for quantum control. Classically conditioned operations will instead
  be handled by the ``condition`` property of ``qiskit.circuit.Instruction``.

- Support for setting ``qiskit.circuit.Instruction`` parameters with an object
  of type ``qiskit.qasm.node.Node`` has been deprecated. ``Node`` objects that
  were previously used as parameters should be converted to a supported type
  prior to initializing a new ``Instruction`` object or calling the
  ``Instruction.params`` setter. Supported types are ``int``, ``float``,
  ``complex``, ``str``, ``qiskit.circuit.ParameterExpression``, or
  ``numpy.ndarray``.

- In the qiskit 0.9.0 release the representation of bits (both qubits and
  classical bits) changed from tuples of the form ``(register, index)`` to be
  instances of the classes ``qiskit.circuit.Qubit`` and
  ``qiskit.circuit.Clbit``. For backwards compatibility comparing
  the equality between a legacy tuple and the bit classes was supported as
  everything transitioned from tuples to being objects. This support is now
  deprecated and will be removed in the future. Everything should use the bit
  classes instead of tuples moving forward.

- When the ``mpl`` output is used for either ``qiskit.QuantumCircuit.draw()``
  or ``qiskit.visualization.circuit_drawer()`` and the ``style`` kwarg is
  used, passing in unsupported dictionary keys as part of the ``style```
  dictionary is now deprecated. Where these unknown arguments were previously
  silently ignored, in the future, unsupported keys will raise an exception.

- The ``line length`` kwarg for the ``qiskit.QuantumCircuit.draw()`` method
  and the ``qiskit.visualization.circuit_drawer()`` function with the text
  output mode is deprecated. It has been replaced by the ``fold`` kwarg which
  will behave identically for the text output mode (but also now supports
  the mpl output mode too). ``line_length`` will be removed in a future
  release so calls should be updated to use ``fold`` instead.

- The ``fold`` field in the ``style`` dict kwarg for the
  ``qiskit.QuantumCircuit.draw()`` method and the
  ``qiskit.visualization.circuit_drawer()`` function has been deprecated. It
  has been replaced by the ``fold`` kwarg on both functions. This kwarg
  behaves identically to the field in the style dict.


.. _Release Notes_0.10.0_Bug Fixes:

Bug Fixes
---------

- Instructions layering which underlies all types of circuit drawing has
  changed to address right/left justification. This sometimes results in
  output which is topologically equivalent to the rendering in prior versions
  but visually different than previously rendered. Fixes
  `issue #2802 <https://github.com/Qiskit/qiskit-terra/issues/2802>`_

- Add ``memory_slots`` to ``QobjExperimentHeader`` of pulse Qobj. This fixes
  a bug in the data format of ``meas_level=2`` results of pulse experiments.
  Measured quantum states are returned as a bit string with zero padding
  based on the number set for ``memory_slots``.

- Fixed the visualization of the rzz gate in the latex circuit drawer to match
  the cu1 gate to reflect the symmetry in the rzz gate. The fix is based on
  the cds command of the qcircuit latex package. Fixes
  `issue #1957 <https://github.com/Qiskit/qiskit-terra/issues/1957>`_


.. _Release Notes_0.10.0_Other Notes:

Other Notes
-----------

- ``matplotlib.figure.Figure`` objects returned by visualization functions
  are no longer always closed by default. Instead the returned figure objects
  are only closed if the configured matplotlib backend is an inline jupyter
  backend(either set with ``%matplotlib inline`` or
  ``%matplotlib notebook``). Output figure objects are still closed with
  these backends to avoid duplicate outputs in jupyter notebooks (which is
  why the ``Figure.close()`` were originally added).

Aer 0.3
=======

No Change

Ignis 0.2
=========

No Change

Aqua 0.6
========

No Change

IBM Q Provider 0.3
==================

No Change

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
  secant derviative ``SamplePulse`` object respectively.

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
  passed a set of kwargs on each call with the state of the pass manager after
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

- Additional decomposition methods for several types of gates. These methods
  will use different decomposition techniques to break down a gate into
  a sequence of CNOTs and single qubit gates. The following methods are
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
  5-qubit circuit with a depth of 10 using::

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
  ``warnings``. Users can suppress the warnings by putting these two lines
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

- All the circuit drawer backends now will express gate parameters in a
  circuit as common fractions of pi in the output visualization. If the value
  of a parameter can be expressed as a fraction of pi that will be used
  instead of the numeric equivalent.

- When using ``qiskit.assembler.assemble_schedules()`` if you do not provide
  the number of memory_slots to use the number will be inferred based on the
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
- A better 2-qubit error approximations have been included.
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
- Seed values can now be arbitrarily added to RB (not just in order)
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
  * QMolecule extended to include integrals, coefficients etc for separate beta

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
- Add ``op_converter`` module to unify the place in charge of converting
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

The ``IBMQProvider`` has been updated in order to default to use the new
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
  for creating parameterized circuits.
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
- Added gate-fusion optimization for ``QasmController``, which is enabled by
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
- The ``skip_transpiler`` argument has been deprecated from ``compile()`` and
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
The first is the ``basis`` kwarg in the ``circuit_drawer()`` function
is no longer accepted. If you were relying on the ``circuit_drawer()`` to
adjust the basis gates used in drawing a circuit diagram you will have to
do this priort to calling ``circuit_drawer()``. For example::

   from qiskit.tools import visualization
   visualization.circuit_drawer(circuit, basis_gates='x,U,CX')

will have to be adjusted to be::

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
  instantiated and initialized via a single (non-empty) constructor call
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
     IBMQ.save_account('MY_API_TOKEN', 'MY_API_URL')

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
