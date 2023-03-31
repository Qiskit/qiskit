#######################
Opflow Migration Guide
#######################

TL;DR
=====
The new :mod:`~qiskit.primitives`, in combination with the :mod:`~qiskit.quantum_info` module, have superseded
functionality of :mod:`~qiskit.opflow`. Thus, the latter is being deprecated.

In this migration guide, you will find instructions and code examples for how to migrate your code based on
the :mod:`~qiskit.opflow` module to the :mod:`~qiskit.primitives` and :mod:`~qiskit.quantum_info` modules.

.. note::

    The use of :mod:`~qiskit.opflow` was tightly coupled to the :class:`~qiskit.utils.QuantumInstance` class, which
    is also being deprecated. For more information on migrating the :class:`~qiskit.utils.QuantumInstance`, please
    read the `quantum instance migration guide <http://qisk.it/qi_migration>`_.

.. _attention_primitives:

..  attention::

    Most references to the :class:`qiskit.primitives.Sampler` or :class:`qiskit.primitives.Estimator` in this guide
    can be replaced with instances of the:

    - Aer primitives (:class:`qiskit_aer.primitives.Sampler`, :class:`qiskit_aer.primitives.Estimator`)
    - Runtime primitives (:class:`qiskit_ibm_runtime.Sampler`, :class:`qiskit_ibm_runtime.Estimator`)
    - Terra backend primitives (:class:`qiskit.primitives.BackendSampler`, :class:`qiskit.primitives.BackendEstimator`)

    Certain classes, such as the
    :class:`~qiskit.opflow.expectations.AerPauliExpectation`, can only be replaced by a specific primitive instance
    (in this case, :class:`qiskit_aer.primitives.Estimator`), or require a specific option configuration.
    If this is the case, it will be explicitly indicated in the corresponding section.


Background
==========

The :mod:`~qiskit.opflow` module was originally introduced as a layer between circuits and algorithms, a series of building blocks
for quantum algorithms research and development.

The recent release of the :mod:`qiskit.primitives` introduced a new paradigm for interacting with backends. Now, instead of
preparing a circuit to execute with a ``backend.run()`` type of method, the algorithms can leverage the :class:`.Sampler` and
:class:`.Estimator` primitives, send parametrized circuits and observables, and directly receive quasi-probability distributions or
expectation values (respectively). This workflow simplifies considerably the pre-processing and post-processing steps
that previously relied on this module; encouraging us to move away from :mod:`~qiskit.opflow`
and find new paths for developing algorithms based on the :mod:`~qiskit.primitives` interface and
the :mod:`~qiskit.quantum_info` module.

This guide traverses the opflow submodules and provides either a direct alternative
(i.e., using :mod:`~qiskit.quantum_info`), or an explanation of how to replace their functionality in algorithms.

The functional equivalency can be roughly summarized as follows:

.. list-table::
   :header-rows: 1

   * - Opflow Module
     - Alternative
   * - Operators (:class:`~qiskit.opflow.OperatorBase`, :ref:`operator_globals`,
       :mod:`~qiskit.opflow.primitive_ops`, :mod:`~qiskit.opflow.list_ops`)
     - ``qiskit.quantum_info`` :ref:`Operators <quantum_info_operators>`

   * - :mod:`qiskit.opflow.state_fns`
     - ``qiskit.quantum_info`` :ref:`States <quantum_info_states>`

   * - :mod:`qiskit.opflow.converters`
     - :mod:`qiskit.primitives`

   * - :mod:`qiskit.opflow.evolutions`
     - ``qiskit.synthesis`` :ref:`Evolution <evolution_synthesis>`

   * - :mod:`qiskit.opflow.expectations`
     - :class:`qiskit.primitives.Estimator`

   * - :mod:`qiskit.opflow.gradients`
     - :mod:`qiskit.algorithms.gradients`

Contents
========

This document covers the migration from these opflow submodules:

**Operators**

- `Operator Base Class`_
- `Operator Globals`_
- `Primitive and List Ops`_
- `State Functions`_

**Converters**

- `Converters`_
- `Evolutions`_
- `Expectations`_

**Gradients**

- `Gradients`_


Operator Base Class
===================
*Back to* `Contents`_

The :class:`qiskit.opflow.OperatorBase` abstract class can be replaced with :class:`qiskit.quantum_info.BaseOperator` ,
keeping in mind that :class:`qiskit.quantum_info.BaseOperator` is more generic than its opflow counterpart.

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative
   * - :class:`qiskit.opflow.OperatorBase`
     - :class:`qiskit.quantum_info.BaseOperator`

..  attention::

    Despite the similar class names, :class:`qiskit.opflow.OperatorBase` and
    :class:`qiskit.quantum_info.BaseOperator` are not completely equivalent to each other, and the transition
    should be handled with care. Namely:

    1. :class:`qiskit.opflow.OperatorBase` implements a broader algebra mixin. Some operator overloads that were
    commonly used :mod:`~qiskit.opflow` (for example ``~`` for ``.adjoint()``) are not defined for
    :class:`qiskit.quantum_info.BaseOperator`. You might want to check the specific
    :mod:`~qiskit.quantum_info` subclass instead.

    2. :class:`qiskit.opflow.OperatorBase` also implements methods such as ``.to_matrix()`` or ``.to_spmatrix()``,
    which are only found in some of the :class:`qiskit.quantum_info.BaseOperator` subclasses.

    See :class:`~qiskit.opflow.OperatorBase` and :class:`~qiskit.quantum_info.BaseOperator` API references
    for more information.


Operator Globals
================
*Back to* `Contents`_

Opflow provided shortcuts to define common single qubit states, operators, and non-parametrized gates in the
:ref:`operator_globals` module.

These were mainly used for didactic purposes or quick prototyping, and can easily be replaced by their corresponding
:mod:`~qiskit.quantum_info` class: :class:`~qiskit.quantum_info.Pauli`, :class:`~qiskit.quantum_info.Clifford` or
:class:`~qiskit.quantum_info.Statevector`.


1-Qubit Paulis
--------------
*Back to* `Contents`_

The 1-qubit paulis were commonly used for quick testing of algorithms, as they could be combined to create more complex operators
(for example, ``0.39 * (I ^ Z) + 0.5 * (X ^ X)``).
These operations implicitly created operators of type  :class:`~qiskit.opflow.primitive_ops.PauliSumOp`, and can be replaced by
directly creating a corresponding :class:`~qiskit.quantum_info.SparsePauliOp`, as shown in the examples below.


.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative
   * - :class:`~qiskit.opflow.X`, :class:`~qiskit.opflow.Y`, :class:`~qiskit.opflow.Z`, :class:`~qiskit.opflow.I`
     - :class:`~qiskit.quantum_info.Pauli`

       ..  tip::

           For direct compatibility with classes in :mod:`~qiskit.algorithms`, wrap in :class:`~qiskit.quantum_info.SparsePauliOp`.


.. _1_q_pauli:


.. dropdown:: Example 1: Defining the XX operator
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import X

        operator = X ^ X
        print(repr(operator))

    .. testoutput::

        PauliOp(Pauli('XX'), coeff=1.0)

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import Pauli, SparsePauliOp

        operator = Pauli('XX')

        # equivalent to:
        X = Pauli('X')
        operator = X ^ X
        print("As Pauli Op: ", repr(operator))

        # another alternative is:
        operator = SparsePauliOp('XX')
        print("As Sparse Pauli Op: ", repr(operator))

    .. testoutput::

        As Pauli Op:  Pauli('XX')
        As Sparse Pauli Op:  SparsePauliOp(['XX'],
                      coeffs=[1.+0.j])

.. dropdown:: Example 2: Defining a more complex operator
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import I, X, Z, PauliSumOp

        operator = 0.39 * (I ^ Z ^ I) + 0.5 * (I ^ X ^ X)

        # equivalent to:
        operator = PauliSumOp.from_list([("IZI", 0.39), ("IXX", 0.5)])

        print(repr(operator))

    .. testoutput::

        PauliSumOp(SparsePauliOp(['IZI', 'IXX'],
                      coeffs=[0.39+0.j, 0.5 +0.j]), coeff=1.0)

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp

        operator = SparsePauliOp(["IZI", "IXX"], coeffs = [0.39, 0.5])

        # equivalent to:
        operator = SparsePauliOp.from_list([("IZI", 0.39), ("IXX", 0.5)])

        # equivalent to:
        operator = SparsePauliOp.from_sparse_list([("Z", [1], 0.39), ("XX", [0,1], 0.5)], num_qubits = 3)

        print(repr(operator))

    .. testoutput::

        SparsePauliOp(['IZI', 'IXX'],
                      coeffs=[0.39+0.j, 0.5 +0.j])

Common non-parametrized gates (Clifford)
----------------------------------------
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.CX`, :class:`~qiskit.opflow.S`, :class:`~qiskit.opflow.H`, :class:`~qiskit.opflow.T`,
       :class:`~qiskit.opflow.CZ`, :class:`~qiskit.opflow.Swap`
     - Append corresponding gate to :class:`~qiskit.circuit.QuantumCircuit`. If necessary,
       :class:`qiskit.quantum_info.Operator`\s can be directly constructed from quantum circuits.
       Another alternative is to wrap the circuit in :class:`~qiskit.quantum_info.Clifford` and call
       ``Clifford.to_operator()``.

       ..  note::

            Constructing :mod:`~qiskit.quantum_info` operators from circuits is not efficient, as it is a dense operation and
            scales exponentially with the size of the circuit, use with care.

.. dropdown:: Example 1: Defining the HH operator
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import H

        operator = H ^ H
        print(operator)

    .. testoutput::

             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             └───┘

    **Alternative**

    .. testcode::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Clifford, Operator

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        print(qc)

    .. testoutput::

             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             └───┘

    If we want to turn this circuit into an operator, we can do:

    .. testcode::

        operator = Clifford(qc).to_operator()

        # or, directly
        operator = Operator(qc)

        print(operator)

    .. testoutput::

        Operator([[ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j],
                  [ 0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j],
                  [ 0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j],
                  [ 0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5+0.j]],
                 input_dims=(2, 2), output_dims=(2, 2))


1-Qubit States
--------------
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.Zero`, :class:`~qiskit.opflow.One`, :class:`~qiskit.opflow.Plus`, :class:`~qiskit.opflow.Minus`
     - :class:`~qiskit.quantum_info.Statevector` or simply :class:`~qiskit.circuit.QuantumCircuit`, depending on the use case.

       ..  note::

           For efficient simulation of stabilizer states, :mod:`~qiskit.quantum_info` includes a
           :class:`~qiskit.quantum_info.StabilizerState` class. See API reference of :class:`~qiskit.quantum_info.StabilizerState` for more info.

.. dropdown:: Example 1: Working with stabilizer states
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import Zero, One, Plus, Minus

        # Zero, One, Plus, Minus are all stabilizer states
        state1 = Zero ^ One
        state2 = Plus ^ Minus

        print("State 1: ", state1)
        print("State 2: ", state2)

    .. testoutput::

        State 1:  DictStateFn({'01': 1})
        State 2:  CircuitStateFn(
             ┌───┐┌───┐
        q_0: ┤ X ├┤ H ├
             ├───┤└───┘
        q_1: ┤ H ├─────
             └───┘
        )

    **Alternative**

    .. testcode::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import StabilizerState, Statevector

        qc_zero = QuantumCircuit(1)
        qc_one = qc_zero.copy()
        qc_one.x(0)
        state1 = Statevector(qc_zero) ^ Statevector(qc_one)
        print("State 1: ", state1)

        qc_plus = qc_zero.copy()
        qc_plus.h(0)
        qc_minus = qc_one.copy()
        qc_minus.h(0)
        state2 = StabilizerState(qc_plus) ^ StabilizerState(qc_minus)
        print("State 2: ", state2)

    .. testoutput::

        State 1:  Statevector([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    dims=(2, 2))
        State 2:  StabilizerState(StabilizerTable: ['-IX', '+XI'])

Primitive and List Ops
======================
*Back to* `Contents`_

Most of the workflows that previously relied on components from :mod:`~qiskit.opflow.primitive_ops` and
:mod:`~qiskit.opflow.list_ops` can now leverage elements from :mod:`~qiskit.quantum_info`\'s
operators instead.
Some of these classes do not require a 1-1 replacement because they were created to interface with other
opflow components.

Primitive Ops
-------------
*Back to* `Contents`_

:class:`~qiskit.opflow.primitive_ops.PrimitiveOp` is the :mod:`~qiskit.opflow.primitive_ops` module's base class.
It also acts as a factory to instantiate a corresponding sub-class depending on the computational primitive used
to initialize it.

.. tip::

    Interpreting :class:`~qiskit.opflow.primitive_ops.PrimitiveOp` as a factory class:

    .. list-table::
       :header-rows: 1

       * - Class passed to :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`
         - Subclass returned

       * - :class:`~qiskit.quantum_info.Pauli`
         - :class:`~qiskit.opflow.primitive_ops.PauliOp`

       * - :class:`~qiskit.circuit.Instruction`, :class:`~qiskit.circuit.QuantumCircuit`
         - :class:`~qiskit.opflow.primitive_ops.CircuitOp`

       * - ``list``, ``np.ndarray``, ``scipy.sparse.spmatrix``, :class:`~qiskit.quantum_info.Operator`
         - :class:`~qiskit.opflow.primitive_ops.MatrixOp`

Thus, when migrating opflow code, it is important to look for alternatives to replace the specific subclasses that
are used "under the hood" in the original code:

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`
     - As mentioned above, this class is used to generate an instance of one of the classes below, so there is
       no direct replacement.

   * - :class:`~qiskit.opflow.primitive_ops.CircuitOp`
     - :class:`~qiskit.circuit.QuantumCircuit`

   * - :class:`~qiskit.opflow.primitive_ops.MatrixOp`
     - :class:`~qiskit.quantum_info.Operator`

   * - :class:`~qiskit.opflow.primitive_ops.PauliOp`
     - :class:`~qiskit.quantum_info.Pauli`. For direct compatibility with classes in :mod:`qiskit.algorithms`,
       wrap in :class:`~qiskit.quantum_info.SparsePauliOp`.

   * - :class:`~qiskit.opflow.primitive_ops.PauliSumOp`
     - :class:`~qiskit.quantum_info.SparsePauliOp`. See example :ref:`below <example_pauli_sum_op>`.

   * - :class:`~qiskit.opflow.primitive_ops.TaperedPauliSumOp`
     - This class was used to combine a :class:`.PauliSumOp` with its identified symmetries in one object.
       This functionality is not currently used in any workflow, and has been deprecated without replacement.
       See :class:`qiskit.quantum_info.analysis.Z2Symmetries` example for updated workflow.

   * - :class:`qiskit.opflow.primitive_ops.Z2Symmetries`
     - :class:`qiskit.quantum_info.analysis.Z2Symmetries`. See example :ref:`below <example_z2_sym>`.

.. _example_pauli_sum_op:

.. dropdown:: Example 1: ``PauliSumOp``
    :animate: fade-in-slide-down


    **Opflow**

    .. testcode::

        from qiskit.opflow import PauliSumOp
        from qiskit.quantum_info import SparsePauliOp, Pauli

        qubit_op = PauliSumOp(SparsePauliOp(Pauli("XYZY"), coeffs=[2]), coeff=-3j)
        print(repr(qubit_op))

    .. testoutput::

        PauliSumOp(SparsePauliOp(['XYZY'],
                      coeffs=[2.+0.j]), coeff=(-0-3j))

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp, Pauli

        qubit_op = SparsePauliOp(Pauli("XYZY"), coeffs=[-6j])
        print(repr(qubit_op))

    .. testoutput::

        SparsePauliOp(['XYZY'],
                      coeffs=[0.-6.j])

.. _example_z2_sym:

.. dropdown:: Example 2: ``Z2Symmetries`` and ``TaperedPauliSumOp``
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import PauliSumOp, Z2Symmetries, TaperedPauliSumOp

        qubit_op = PauliSumOp.from_list(
            [
            ("II", -1.0537076071291125),
            ("IZ", 0.393983679438514),
            ("ZI", -0.39398367943851387),
            ("ZZ", -0.01123658523318205),
            ("XX", 0.1812888082114961),
            ]
        )
        z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
        print(z2_symmetries)

        tapered_op = z2_symmetries.taper(qubit_op)
        print("Tapered Op from Z2 symmetries: ", tapered_op)

        # can be represented as:
        tapered_op = TaperedPauliSumOp(qubit_op.primitive, z2_symmetries)
        print("Tapered PauliSumOp: ", tapered_op)

    .. testoutput::

        Z2 symmetries:
        Symmetries:
        ZZ
        Single-Qubit Pauli X:
        IX
        Cliffords:
        0.7071067811865475 * ZZ
        + 0.7071067811865475 * IX
        Qubit index:
        [0]
        Tapering values:
          - Possible values: [1], [-1]
        Tapered Op from Z2 symmetries:  ListOp([
          -1.0649441923622942 * I
          + 0.18128880821149604 * X,
          -1.0424710218959303 * I
          - 0.7879673588770277 * Z
          - 0.18128880821149604 * X
        ])
        Tapered PauliSumOp:  -1.0537076071291125 * II
        + 0.393983679438514 * IZ
        - 0.39398367943851387 * ZI
        - 0.01123658523318205 * ZZ
        + 0.1812888082114961 * XX


    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp
        from qiskit.quantum_info.analysis import Z2Symmetries

        qubit_op = SparsePauliOp.from_list(
            [
            ("II", -1.0537076071291125),
            ("IZ", 0.393983679438514),
            ("ZI", -0.39398367943851387),
            ("ZZ", -0.01123658523318205),
            ("XX", 0.1812888082114961),
            ]
        )
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        print(z2_symmetries)

        tapered_op = z2_symmetries.taper(qubit_op)
        print("Tapered Op from Z2 symmetries: ", tapered_op)

    .. testoutput::

        Z2 symmetries:
        Symmetries:
        ZZ
        Single-Qubit Pauli X:
        IX
        Cliffords:
        SparsePauliOp(['ZZ', 'IX'],
                      coeffs=[0.70710678+0.j, 0.70710678+0.j])
        Qubit index:
        [0]
        Tapering values:
          - Possible values: [1], [-1]
        Tapered Op from Z2 symmetries:  [SparsePauliOp(['I', 'X'],
                      coeffs=[-1.06494419+0.j,  0.18128881+0.j]), SparsePauliOp(['I', 'Z', 'X'],
                      coeffs=[-1.04247102+0.j, -0.78796736+0.j, -0.18128881+0.j])]

ListOps
--------
*Back to* `Contents`_

The :mod:`~qiskit.opflow.list_ops` module contained classes for manipulating lists of :mod:`~qiskit.opflow.primitive_ops`
or :mod:`~qiskit.opflow.state_fns`. The :mod:`~qiskit.quantum_info` alternatives for this functionality are the
:class:`~qiskit.quantum_info.PauliList` and :class:`~qiskit.quantum_info.SparsePauliOp` (for sums of :class:`~qiskit.quantum_info.Pauli`\s).

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.list_ops.ListOp`
     - No direct replacement. This is the base class for operator lists. In general, these could be replaced with
       Python ``list``\s. For :class:`~qiskit.quantum_info.Pauli` operators, there are a few alternatives, depending on the use-case.
       One alternative is :class:`~qiskit.quantum_info.PauliList`.

   * - :class:`~qiskit.opflow.list_ops.ComposedOp`
     - No direct replacement. Current workflows do not require composition of states and operators within
       one object (no lazy evaluation).

   * - :class:`~qiskit.opflow.list_ops.SummedOp`
     - No direct replacement. For :class:`~qiskit.quantum_info.Pauli` operators, use :class:`~qiskit.quantum_info.SparsePauliOp`.

   * - :class:`~qiskit.opflow.list_ops.TensoredOp`
     - No direct replacement. For :class:`~qiskit.quantum_info.Pauli` operators, use :class:`~qiskit.quantum_info.SparsePauliOp`.


State Functions
===============
*Back to* `Contents`_

The :mod:`~qiskit.opflow.state_fns` module can be generally replaced by subclasses of :mod:`~qiskit.quantum_info`\'s
:class:`qiskit.quantum_info.QuantumState`.

Similarly to :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`, :class:`~qiskit.opflow.state_fns.StateFn`
acts as a factory to create the corresponding subclass depending on the computational primitive used to initialize it.

.. tip::

    Interpreting :class:`~qiskit.opflow.state_fns.StateFn` as a factory class:

    .. list-table::
       :header-rows: 1

       * - Class passed to :class:`~qiskit.opflow.state_fns.StateFn`
         - Sub-class returned

       * - ``str``, ``dict``, :class:`~qiskit.result.Result`
         - :class:`~qiskit.opflow.state_fns.DictStateFn`

       * - ``list``, ``np.ndarray``, :class:`~qiskit.quantum_info.Statevector`
         - :class:`~qiskit.opflow.state_fns.VectorStateFn`

       * - :class:`~qiskit.circuit.QuantumCircuit`, :class:`~qiskit.circuit.Instruction`
         - :class:`~qiskit.opflow.state_fns.CircuitStateFn`

       * - :class:`~qiskit.opflow.OperatorBase`
         - :class:`~qiskit.opflow.state_fns.OperatorStateFn`

This means that references to :class:`~qiskit.opflow.state_fns.StateFn` in opflow code should be examined to
identify the subclass that is being used, to then look for an alternative.

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.state_fns.StateFn`
     - In most cases, :class:`~qiskit.quantum_info.Statevector`. However, please remember that :class:`~qiskit.opflow.state_fns.StateFn` is a factory class.

   * - :class:`~qiskit.opflow.state_fns.CircuitStateFn`
     - :class:`~qiskit.quantum_info.Statevector`

   * - :class:`~qiskit.opflow.state_fns.DictStateFn`
     - This class was used to store efficient representations of sparse measurement results. The
       :class:`~qiskit.primitives.Sampler` now returns the measurements as an instance of
       :class:`~qiskit.result.QuasiDistribution` (see example in `Converters`_).

   * - :class:`~qiskit.opflow.state_fns.VectorStateFn`
     - This class can be replaced with :class:`~qiskit.quantum_info.Statevector` or
       :class:`~qiskit.quantum_info.StabilizerState` (for Clifford-based vectors).

   * - :class:`~qiskit.opflow.state_fns.SparseVectorStateFn`
     - No direct replacement. This class was used for sparse statevector representations.

   * - :class:`~qiskit.opflow.state_fns.OperatorStateFn`
     - No direct replacement. This class was used to represent measurements against operators.

   * - :class:`~qiskit.opflow.state_fns.CVaRMeasurement`
     - Used in :class:`~qiskit.opflow.expectations.CVaRExpectation`.
       Functionality now covered by :class:`.SamplingVQE`. See example in `Expectations`_.


.. dropdown:: Example 1: Applying an operator to a state
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import StateFn, X, Y
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.z(1)
        op = X ^ Y
        state = StateFn(qc)

        comp = ~op @ state
        eval = comp.eval()

        print(state)
        print(comp)
        print(repr(eval))

    .. testoutput::

        CircuitStateFn(
             ┌───┐
        q_0: ┤ X ├
             ├───┤
        q_1: ┤ Z ├
             └───┘
        )
        CircuitStateFn(
             ┌───┐┌────────────┐
        q_0: ┤ X ├┤0           ├
             ├───┤│  Pauli(XY) │
        q_1: ┤ Z ├┤1           ├
             └───┘└────────────┘
        )
        VectorStateFn(Statevector([ 0.0e+00+0.j,  0.0e+00+0.j, -6.1e-17-1.j,  0.0e+00+0.j],
                    dims=(2, 2)), coeff=1.0, is_measurement=False)

    **Alternative**

    .. testcode::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp, Statevector

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.z(1)
        op = SparsePauliOp("XY")
        state = Statevector(qc)

        eval = state.evolve(op)

        print(state)
        print(eval)

    .. testoutput::

        Statevector([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    dims=(2, 2))
        Statevector([0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
                    dims=(2, 2))

See more applied examples in `Expectations`_  and `Converters`_.


Converters
==========

*Back to* `Contents`_

The role of the :class:`qiskit.opflow.converters` submodule was to convert the operators into other opflow operator classes
(:class:`~qiskit.opflow.converters.TwoQubitReduction`, :class:`~qiskit.opflow.converters.PauliBasisChange`...).
In the case of the :class:`~qiskit.opflow.converters.CircuitSampler`, it traversed an operator and outputted
approximations of its state functions using a quantum backend.
Notably, this functionality has been replaced by the :mod:`~qiskit.primitives`.

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.converters.CircuitSampler`
     - :class:`~qiskit.primitives.Sampler` or :class:`~qiskit.primitives.Estimator` if used with
       :class:`~qiskit.oflow.expectations`. See examples :ref:`below <example_convert_state>`.
   * - :class:`~qiskit.opflow.converters.AbelianGrouper`
     - This class allowed a sum a of Pauli operators to be grouped, a similar functionality can be achieved
       through the :meth:`~qiskit.quantum_info.SparsePauliOp.group_commuting` method of
       :class:`qiskit.quantum_info.SparsePauliOp`, although this is not a 1-1 replacement, as you can see
       in the example :ref:`below <example_commuting>`.
   * - :class:`~qiskit.opflow.converters.DictToCircuitSum`
     - No direct replacement. This class was used to convert from :class:`~qiskit.opflow.state_fns.DictStateFn`\s or
       :class:`~qiskit.opflow.state_fns.VectorStateFn`\s to equivalent :class:`~qiskit.opflow.state_fns.CircuitStateFn`\s.
   * - :class:`~qiskit.opflow.converters.PauliBasisChange`
     - No direct replacement. This class was used for changing Paulis into other bases.
   * -  :class:`~qiskit.opflow.converters.TwoQubitReduction`
     -  No direct replacement. This class implements a chemistry-specific reduction for the :class:`.ParityMapper`
        class in :mod:`qiskit_nature`.
        The general symmetry logic this mapper depends on has been refactored to other classes in :mod:`~qiskit.quantum_info`,
        so this specific :mod:`~qiskit.opflow` implementation is no longer necessary.


.. _example_convert_state:

.. dropdown:: Example 1: ``CircuitSampler`` for sampling parametrized circuits
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.opflow import ListOp, StateFn, CircuitSampler
        from qiskit_aer import AerSimulator

        x, y = Parameter("x"), Parameter("y")

        circuit1 = QuantumCircuit(1)
        circuit1.p(0.2, 0)
        circuit2 = QuantumCircuit(1)
        circuit2.p(x, 0)
        circuit3 = QuantumCircuit(1)
        circuit3.p(y, 0)

        bindings = {x: -0.4, y: 0.4}
        listop = ListOp([StateFn(circuit) for circuit in [circuit1, circuit2, circuit3]])

        sampler = CircuitSampler(AerSimulator())
        sampled = sampler.convert(listop, params=bindings).eval()

        for s in sampled:
          print(s)

    .. testoutput::

        SparseVectorStateFn(  (0, 0)	1.0)
        SparseVectorStateFn(  (0, 0)	1.0)
        SparseVectorStateFn(  (0, 0)	1.0)

    **Alternative**

    .. testcode::

        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.primitives import Sampler

        x, y = Parameter("x"), Parameter("y")

        circuit1 = QuantumCircuit(1)
        circuit1.p(0.2, 0)
        circuit1.measure_all()     # Sampler primitive requires measurement readout
        circuit2 = QuantumCircuit(1)
        circuit2.p(x, 0)
        circuit2.measure_all()
        circuit3 = QuantumCircuit(1)
        circuit3.p(y, 0)
        circuit3.measure_all()

        circuits = [circuit1, circuit2, circuit3]
        param_values = [[], [-0.4], [0.4]]

        sampler = Sampler()
        sampled = sampler.run(circuits, param_values).result().quasi_dists

        print(sampled)

    .. testoutput::

        [{0: 1.0}, {0: 1.0}, {0: 1.0}]


.. dropdown:: Example 2: ``CircuitSampler`` for computing expectation values
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit import QuantumCircuit
        from qiskit.opflow import X, Z, StateFn, CircuitStateFn, CircuitSampler
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(1)
        qc.h(0)
        state = CircuitStateFn(qc)
        hamiltonian = X + Z

        expr = StateFn(hamiltonian, is_measurement=True).compose(state)
        backend = AerSimulator(method="statevector")
        sampler = CircuitSampler(backend)
        expectation = sampler.convert(expr)
        expectation_value = expectation.eval().real

        print(expectation_value)

    .. testoutput::

        1.0000000000000002

    **Alternative**

    .. testcode::

        from qiskit import QuantumCircuit
        from qiskit.primitives import Estimator
        from qiskit.quantum_info import SparsePauliOp

        state = QuantumCircuit(1)
        state.h(0)
        hamiltonian = SparsePauliOp.from_list([('X', 1), ('Z',1)])

        estimator = Estimator()
        expectation_value = estimator.run(state, hamiltonian).result().values.real

        print(expectation_value)

    .. testoutput::

        [1.]

.. _example_commuting:

.. dropdown:: Example 3: ``AbelianGrouper`` for grouping operators
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import PauliSumOp, AbelianGrouper

        op = PauliSumOp.from_list([("XX", 2), ("YY", 1), ("IZ",2j), ("ZZ",1j)])

        grouped_sum = AbelianGrouper.group_subops(op)

        print(repr(grouped_sum))

    .. testoutput::

        SummedOp([PauliSumOp(SparsePauliOp(['XX'],
                      coeffs=[2.+0.j]), coeff=1.0), PauliSumOp(SparsePauliOp(['YY'],
                      coeffs=[1.+0.j]), coeff=1.0), PauliSumOp(SparsePauliOp(['IZ', 'ZZ'],
                      coeffs=[0.+2.j, 0.+1.j]), coeff=1.0)], coeff=1.0, abelian=False)

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp

        op = SparsePauliOp.from_list([("XX", 2), ("YY", 1), ("IZ",2j), ("ZZ",1j)])

        grouped = op.group_commuting()
        grouped_sum = op.group_commuting(qubit_wise=True)

        print(repr(grouped))
        print(repr(grouped_sum))

    .. testoutput::

        [SparsePauliOp(['IZ', 'ZZ'],
                      coeffs=[0.+2.j, 0.+1.j]), SparsePauliOp(['XX', 'YY'],
                      coeffs=[2.+0.j, 1.+0.j])]
        [SparsePauliOp(['XX'],
                      coeffs=[2.+0.j]), SparsePauliOp(['YY'],
                      coeffs=[1.+0.j]), SparsePauliOp(['IZ', 'ZZ'],
                      coeffs=[0.+2.j, 0.+1.j])]

Evolutions
==========
*Back to* `Contents`_

The :mod:`qiskit.opflow.evolutions` submodule was created to provide building blocks for Hamiltonian simulation algorithms,
including various methods for Trotterization. The original opflow workflow for Hamiltonian simulation did not allow for
delayed synthesis of the gates or efficient transpilation of the circuits, so this functionality was migrated to the
``qiskit.synthesis`` :ref:`Evolution <evolution_synthesis>` module.

.. note::

    The :class:`qiskit.opflow.evolutions.PauliTrotterEvolution` class computes evolutions for exponentiated
    sums of Paulis by converting to the Z basis, rotating with an RZ, changing back, and Trotterizing.
    When calling ``.convert()``, the class follows a recursive strategy that involves creating
    :class:`~qiskit.opflow.evolutions.EvolvedOp` placeholders for the operators,
    constructing :class:`.PauliEvolutionGate`\s out of the operator primitives, and supplying one of
    the desired synthesis methods to perform the Trotterization. The methods can be specified via
    ``string``, which is then inputted into a :class:`~qiskit.opflow.evolutions.TrotterizationFactory`,
    or by supplying a method instance of :class:`qiskit.opflow.evolutions.Trotter`,
    :class:`qiskit.opflow.evolutions.Suzuki` or :class:`qiskit.opflow.evolutions.QDrift`.

    The different Trotterization methods that extend :class:`qiskit.opflow.evolutions.TrotterizationBase` were migrated to
    :mod:`qiskit.synthesis`,
    and now extend the :class:`qiskit.synthesis.ProductFormula` base class. They no longer contain a ``.convert()`` method for
    standalone use, but are now designed to be plugged into the :class:`.PauliEvolutionGate` and called via ``.synthesize()``.
    In this context, the job of the :class:`qiskit.opflow.evolutions.PauliTrotterEvolution` class can now be handled directly by the algorithms
    (for example, :class:`~qiskit.algorithms.time_evolvers.trotterization.TrotterQRTE`\).

    In a similar manner, the :class:`qiskit.opflow.evolutions.MatrixEvolution` class performs evolution by classical matrix exponentiation,
    constructing a circuit with :class:`.UnitaryGate`\s or :class:`.HamiltonianGate`\s containing the exponentiation of the operator.
    This class is no longer necessary, as the :class:`.HamiltonianGate`\s can be directly handled by the algorithms.

Trotterizations
---------------
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.evolutions.TrotterizationFactory`
     - No direct replacement. This class was used to create instances of one of the classes listed below.

   * - :class:`~qiskit.opflow.evolutions.Trotter`
     - :class:`qiskit.synthesis.SuzukiTrotter` or :class:`qiskit.synthesis.LieTrotter`

   * - :class:`~qiskit.opflow.evolutions.Suzuki`
     - :class:`qiskit.synthesis.SuzukiTrotter`

   * - :class:`~qiskit.opflow.evolutions.QDrift`
     - :class:`qiskit.synthesis.QDrift`

Other Evolution Classes
-----------------------
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.evolutions.EvolutionFactory`
     - No direct replacement. This class was used to create instances of one of the classes listed below.

   * - :class:`~qiskit.opflow.evolutions.EvolvedOp`
     - No direct replacement. The workflow no longer requires a specific operator for evolutions.

   * - :class:`~qiskit.opflow.evolutions.MatrixEvolution`
     - :class:`.HamiltonianGate`

   * - :class:`~qiskit.opflow.evolutions.PauliTrotterEvolution`
     - :class:`.PauliEvolutionGate`


.. dropdown:: Example 1: Trotter evolution
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import Trotter, PauliTrotterEvolution, PauliSumOp

        hamiltonian = PauliSumOp.from_list([('X', 1), ('Z',1)])
        evolution = PauliTrotterEvolution(trotter_mode=Trotter(), reps=2)
        evol_result = evolution.convert(hamiltonian.exp_i())
        evolved_state = evol_result.to_circuit()

        print(evolved_state)

    .. testoutput::

           ┌─────────────────────┐
        q: ┤ exp(-it (X + Z))(1) ├
           └─────────────────────┘

    **Alternative**

    .. testcode::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import SuzukiTrotter

        hamiltonian = SparsePauliOp.from_list([('X', 1), ('Z',1)])
        evol_gate = PauliEvolutionGate(hamiltonian, time=1, synthesis=SuzukiTrotter(reps=2))
        evolved_state = QuantumCircuit(1)
        evolved_state.append(evol_gate, [0])

        print(evolved_state)

    .. testoutput::

           ┌─────────────────────┐
        q: ┤ exp(-it (X + Z))(1) ├
           └─────────────────────┘

.. dropdown:: Example 2: Evolution with time-dependent Hamiltonian
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import Trotter, PauliTrotterEvolution, PauliSumOp
        from qiskit.circuit import Parameter

        time = Parameter('t')
        hamiltonian = PauliSumOp.from_list([('X', 1), ('Y',1)])
        evolution = PauliTrotterEvolution(trotter_mode=Trotter(), reps=1)
        evol_result = evolution.convert((time * hamiltonian).exp_i())
        evolved_state = evol_result.to_circuit()

        print(evolved_state)

    .. testoutput::

           ┌─────────────────────────┐
        q: ┤ exp(-it (X + Y))(1.0*t) ├
           └─────────────────────────┘

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import LieTrotter
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        time = Parameter('t')
        hamiltonian = SparsePauliOp.from_list([('X', 1), ('Y',1)])
        evol_gate = PauliEvolutionGate(hamiltonian, time=time, synthesis=LieTrotter())
        evolved_state = QuantumCircuit(1)
        evolved_state.append(evol_gate, [0])

        print(evolved_state)

    .. testoutput::

           ┌─────────────────────┐
        q: ┤ exp(-it (X + Y))(t) ├
           └─────────────────────┘


.. dropdown:: Example 3: Matrix evolution
    :animate: fade-in-slide-down


    **Opflow**

    .. testcode::

        from qiskit.opflow import MatrixEvolution, MatrixOp

        hamiltonian = MatrixOp([[0, 1], [1, 0]])
        evolution = MatrixEvolution()
        evol_result = evolution.convert(hamiltonian.exp_i())
        evolved_state = evol_result.to_circuit()

        print(evolved_state.decompose().decompose())

    .. testoutput::

           ┌────────────────┐
        q: ┤ U3(2,-π/2,π/2) ├
           └────────────────┘

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp
        from qiskit.extensions import HamiltonianGate
        from qiskit import QuantumCircuit

        evol_gate = HamiltonianGate([[0, 1], [1, 0]], 1)
        evolved_state = QuantumCircuit(1)
        evolved_state.append(evol_gate, [0])

        print(evolved_state.decompose().decompose())

    .. testoutput::

           ┌────────────────┐
        q: ┤ U3(2,-π/2,π/2) ├
           └────────────────┘


Expectations
============
*Back to* `Contents`_

Expectations are converters which enable the computation of the expectation value of an observable with respect to some state function.
This functionality can now be found in the :class:`~qiskit.primitives.Estimator` primitive. Please remember that there
are different ``Estimator`` implementations, as noted :ref:`here <attention_primitives>`

Algorithm-Agnostic Expectations
-------------------------------
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.expectations.ExpectationFactory`
     - No direct replacement. This class was used to create instances of one of the classes listed below.

   * - :class:`~qiskit.opflow.expectations.AerPauliExpectation`
     - Use :class:`qiskit_aer.primitives.Estimator`  with ``approximation=True`` and ``shots=None`` as ``run_options``.
       See example below.

   * - :class:`~qiskit.opflow.expectations.MatrixExpectation`
     - Use :class:`qiskit.primitives.Estimator` primitive (if no shots are set, it performs an exact Statevector calculation).
       See example below.

   * - :class:`~qiskit.opflow.expectations.PauliExpectation`
     - Use any Estimator primitive (for :class:`qiskit.primitives.Estimator`, set ``shots!=None`` for a shot-based
       simulation, for :class:`qiskit_aer.primitives.Estimator` , this is the default).


.. _expect_state:

.. dropdown:: Example 1: Aer Pauli expectation
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import X, Minus, StateFn, AerPauliExpectation, CircuitSampler
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        q_instance = QuantumInstance(backend)

        sampler = CircuitSampler(q_instance, attach_results=True)
        expectation = AerPauliExpectation()

        state = Minus
        operator = 1j * X

        converted_meas = expectation.convert(StateFn(operator, is_measurement=True) @ state)
        expectation_value = sampler.convert(converted_meas).eval()

        print(expectation_value)

    .. testoutput::

        -1j

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp
        from qiskit import QuantumCircuit
        from qiskit_aer.primitives import Estimator

        estimator = Estimator(run_options={"approximation": True, "shots": None})

        op = SparsePauliOp.from_list([("X", 1j)])
        states_op = QuantumCircuit(1)
        states_op.x(0)
        states_op.h(0)

        expectation_value = estimator.run(states_op, op).result().values

        print(expectation_value)

    .. testoutput::

        [0.-1.j]


.. _matrix_state:

.. dropdown:: Example 2: Matrix expectation
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import X, H, I, MatrixExpectation, ListOp, StateFn
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method='statevector')
        q_instance = QuantumInstance(backend)
        sampler = CircuitSampler(q_instance, attach_results=True)
        expect = MatrixExpectation()

        mixed_ops = ListOp([X.to_matrix_op(), H])
        converted_meas = expect.convert(~StateFn(mixed_ops))

        plus_mean = converted_meas @ Plus
        values_plus = sampler.convert(plus_mean).eval()

        print(values_plus)

    .. testoutput::

        [(1+0j), (0.7071067811865476+0j)]

    **Alternative**

    .. testcode::

        from qiskit.primitives import Estimator
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.quantum_info import Clifford

        X = SparsePauliOp("X")

        qc = QuantumCircuit(1)
        qc.h(0)
        H = Clifford(qc).to_operator()

        plus = QuantumCircuit(1)
        plus.h(0)

        estimator = Estimator()
        values_plus = estimator.run([plus, plus], [X, H]).result().values

        print(values_plus)

    .. testoutput::

        [1.         0.70710678]


CVaRExpectation
---------------
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.expectations.CVaRExpectation`
     - Functionality migrated into new VQE algorithm: :class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQE`

..  _cvar:

.. dropdown:: Example 1: VQE with CVaR
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.opflow import CVaRExpectation, PauliSumOp

        from qiskit.algorithms import VQE
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.circuit.library import TwoLocal
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        ansatz = TwoLocal(2, 'ry', 'cz')
        op = PauliSumOp.from_list([('ZZ',1), ('IZ',1), ('II',1)])
        alpha = 0.2
        cvar_expectation = CVaRExpectation(alpha=alpha)
        opt = SLSQP(maxiter=1000)
        vqe = VQE(ansatz, expectation=cvar_expectation, optimizer=opt, quantum_instance=backend)
        result = vqe.compute_minimum_eigenvalue(op)

        print(result.eigenvalue)

    .. testoutput::

        (-1+0j)

    **Alternative**

    .. testcode::

        from qiskit.quantum_info import SparsePauliOp

        from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.circuit.library import TwoLocal
        from qiskit.primitives import Sampler

        ansatz = TwoLocal(2, 'ry', 'cz')
        op = SparsePauliOp.from_list([('ZZ',1), ('IZ',1), ('II',1)])
        opt = SLSQP(maxiter=1000)
        alpha = 0.2
        vqe = SamplingVQE(Sampler(), ansatz, opt, aggregation=alpha)
        result = vqe.compute_minimum_eigenvalue(op)

        print(result.eigenvalue)

    .. testoutput::

        -1.0


Gradients
=========
*Back to* `Contents`_

The opflow :mod:`~qiskit.opflow.gradients` framework has been replaced by the new :mod:`qiskit.algorithms.gradients`
module. The new gradients are **primitive-based subroutines** commonly used by algorithms and applications, which
can also be executed in a standalone manner. For this reason, they now reside under :mod:`qiskit.algorithms`.

The former gradient framework contained base classes, converters and derivatives. The "derivatives"
followed a factory design pattern, where different methods could be provided via string identifiers
to each of these classes. The new gradient framework contains two main families of subroutines:
**Gradients** and **QGT/QFI**. The **Gradients** can either be Sampler or Estimator based, while the current
**QGT/QFI** implementations are Estimator-based.

This leads to a change in the workflow, where instead of doing:

.. code-block:: python

    from qiskit.opflow import Gradient

    grad = Gradient(method="param_shift")

    # task based on expectation value computations + gradients

We now import explicitly the desired class, depending on the target primitive (Sampler/Estimator) and target method:

.. code-block:: python

    from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
    from qiskit.primitives import Estimator

    grad = ParamShiftEstimatorGradient(Estimator())

    # task based on expectation value computations + gradients

This works similarly for the QFI class, where instead of doing:

.. code-block:: python

    from qiskit.opflow import QFI

    qfi = QFI(method="lin_comb_full")

    # task based on expectation value computations + QFI

You now have a generic QFI implementation that can be initialized with different QGT (Quantum Gradient Tensor)
implementations:

.. code-block:: python

    from qiskit.algorithms.gradients import LinCombQGT, QFI
    from qiskit.primitives import Estimator

    qgt = LinCombQGT(Estimator())
    qfi = QFI(qgt)

    # task based on expectation value computations + QFI

.. note::

    Here is a quick guide for migrating the most common gradient settings. Please note that all new gradient
    imports follow the format:

        .. code-block:: python

            from qiskit.algorithms.gradients import MethodPrimitiveGradient, QFI

    .. dropdown:: Gradients
        :animate: fade-in-slide-down

        .. list-table::
           :header-rows: 1

           * - Opflow
             - Alternative

           * - ``Gradient(method="lin_comb")``
             - ``LinCombEstimatorGradient(estimator=estimator)`` or ``LinCombSamplerGradient(sampler=sampler)``
           * - ``Gradient(method="param_shift")``
             - ``ParamShiftEstimatorGradient(estimator=estimator)`` or ``ParamShiftSamplerGradient(sampler=sampler)``
           * - ``Gradient(method="fin_diff")``
             - ``FiniteDiffEstimatorGradient(estimator=estimator)`` or ``ParamShiftSamplerGradient(sampler=sampler)``

    .. dropdown:: QFI/QGT
        :animate: fade-in-slide-down

        .. list-table::
           :header-rows: 1

           * - Opflow
             - Alternative

           * - ``QFI(method="lin_comb_full")``
             - ``qgt=LinCombQGT(Estimator())``
               ``QFI(qgt=qgt)``


Other auxiliary classes in the legacy gradient framework have now been deprecated. Here is the complete migration
list:

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.gradients.DerivativeBase`
     - No replacement. This was the base class for the gradient, hessian and QFI base classes.
   * - :class:`.GradientBase` and :class:`~qiskit.opflow.gradients.Gradient`
     - :class:`.BaseSamplerGradient` or :class:`.BaseEstimatorGradient`, and specific subclasses per method,
       as explained above.
   * - :class:`.HessianBase` and :class:`~qiskit.opflow.gradients.Hessian`
     - No replacement. The new gradient framework does not work with hessians as independent objects.
   * - :class:`.QFIBase` and :class:`~qiskit.opflow.gradients.QFI`
     - The new :class:`~qiskit.algorithms.gradients.QFI` class extends :class:`~qiskit.algorithms.gradients.QGT`, so the
       corresponding base class is :class:`~qiskit.algorithms.gradients.BaseQGT`
   * - :class:`~qiskit.opflow.gradients.CircuitGradient`
     - No replacement. This class was used to convert between circuit and gradient
       :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`, and this functionality is no longer necessary.
   * - :class:`~qiskit.opflow.gradients.CircuitQFI`
     - No replacement. This class was used to convert between circuit and QFI
       :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`, and this functionality is no longer necessary.
   * - :class:`~qiskit.opflow.gradients.NaturalGradient`
     - No replacement. The same functionality can be achieved with the QFI module.

.. dropdown:: Example 1: Finite Differences Batched Gradient
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.opflow import Gradient, X, Z, StateFn, CircuitStateFn
        import numpy as np

        ham = 0.5 * X - 1 * Z

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        params = [a,b,c]

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        # the gradient class acted similarly opflow converters,
        # with a .convert() step and an .eval() step
        state_grad = Gradient(grad_method="param_shift").convert(operator=op, params=params)

        # the old workflow did not allow for batched evaluation of parameter values
        values_dict = [{a: np.pi / 4, b: 0, c: 0}, {a: np.pi / 4, b: np.pi / 4, c: np.pi / 4}]
        gradients = []
        for i, value_dict in enumerate(values_dict):
             gradients.append(state_grad.assign_parameters(value_dict).eval())

        print(gradients)

    .. testoutput::

        [[(0.35355339059327356+0j), (-1.182555756156289e-16+0j), (-1.6675e-16+0j)], [(0.10355339059327384+0j), (0.8535533905932734+0j), (1.103553390593273+0j)]]

    **Alternative**

    .. testcode::

        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.primitives import Estimator
        from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
        from qiskit.quantum_info import SparsePauliOp
        import numpy as np

        ham = SparsePauliOp.from_list([("X", 0.5), ("Z", -1)])

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)

        estimator = Estimator()
        gradient = ParamShiftEstimatorGradient(estimator)

        # the new workflow follows an interface close to the primitives'
        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]

        # for batched evaluations, the number of circuits must match the
        # number of parameter value sets
        gradients = gradient.run([qc] * 2, [ham] * 2, param_list).result().gradients

        print(gradients)

    .. testoutput::

        [array([ 3.53553391e-01,  0.00000000e+00, -1.80411242e-16]), array([0.10355339, 0.85355339, 1.10355339])]


.. dropdown:: Example 2: QFI
    :animate: fade-in-slide-down

    **Opflow**

    .. testcode::

        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.opflow import QFI, CircuitStateFn
        import numpy as np

        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        # convert the circuit to a QFI object
        op = CircuitStateFn(qc)
        qfi = QFI(qfi_method="lin_comb_full").convert(operator=op)

        # bind parameters and evaluate
        values_dict = {a: np.pi / 4, b: 0.1}
        qfi = qfi.bind_parameters(values_dict).eval()

        print(qfi)

    .. testoutput::

        [[ 1.00000000e+00+0.j -3.63575685e-16+0.j]
         [-3.63575685e-16+0.j  5.00000000e-01+0.j]]

    **Alternative**

    .. testcode::

        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.primitives import Estimator
        from qiskit.algorithms.gradients import LinCombQGT, QFI
        import numpy as np

        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        # initialize QFI
        estimator = Estimator()
        qgt = LinCombQGT(estimator)
        qfi = QFI(qgt)

        # evaluate
        values_list = [[np.pi / 4, 0.1]]
        qfi = qfi.run(qc, values_list).result().qfis

        print(qfi)

    .. testoutput::

        [array([[ 1.00000000e+00, -1.50274614e-16],
               [-1.50274614e-16,  5.00000000e-01]])]
