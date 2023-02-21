=======================
Opflow Migration Guide
=======================

TL;DR
-----
The new :mod:`~qiskit.primitives`, in combination with the :mod:`~qiskit.quantum_info` module, have superseded
functionality of :mod:`~qiskit.opflow`. Thus, the latter is being deprecated.

.. note::

    The use of :mod:`~qiskit.opflow` was tightly coupled to the :class:`~qiskit.utils.QuantumInstance` class, which
    is also being deprecated. For more information on migrating the :class:`~qiskit.utils.QuantumInstance`, please
    read the `quantum instance migration guide <http://qisk.it/qi_migration>`_.


Background
----------

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
(i.e. using :mod:`~qiskit.quantum_info`), or an explanation of how to replace their functionality in algorithms.

The function equivalency can be roughly summarized as follows:

.. |operator_globals| replace:: ``operator_globals``
.. _operator_globals: https://qiskit.org/documentation/apidoc/opflow.html#operator-globals/

.. list-table::
   :header-rows: 1

   * - Opflow Module
     - Alternative
   * - Operators (:class:`~qiskit.opflow.OperatorBase`, |operator_globals|_ ,
       :mod:`~qiskit.opflow.primitive_ops`,
       :mod:`~qiskit.opflow.list_ops`\)
     - :mod:`qiskit.quantum_info` *Operators*

   * - :mod:`qiskit.opflow.state_fns`
     - :mod:`qiskit.quantum_info` *States*

   * - :mod:`qiskit.opflow.converters`
     - :mod:`qiskit.primitives`

   * - :mod:`qiskit.opflow.evolutions`
     - :mod:`qiskit.quantum_info` *Synthesis*

   * - :mod:`qiskit.opflow.expectations`
     - :class:`qiskit.primitives.Estimator`

   * - :mod:`qiskit.opflow.gradients`
     - :mod:`qiskit.algorithms.gradients`

.. |qiskit_aer.primitives| replace:: ``qiskit_aer.primitives``
.. _qiskit_aer.primitives: https://qiskit.org/documentation/locale/de_DE/apidoc/aer_primitives.html

.. |qiskit_aer.primitives.Estimator| replace:: ``qiskit_aer.primitives.Estimator``
.. _qiskit_aer.primitives.Estimator: https://qiskit.org/documentation/locale/de_DE/stubs/qiskit_aer.primitives.Estimator.html

.. |qiskit_ibm_runtime| replace:: ``qiskit_ibm_runtime``
.. _qiskit_ibm_runtime: https://qiskit.org/documentation/partners/qiskit_ibm_runtime/primitives.html

..  attention::

    Most references to the :class:`qiskit.primitives.Sampler` or :class:`qiskit.primitives.Estimator` in this guide
    can be replaced with instances of the Aer primitives (|qiskit_aer.primitives|_ ), Runtime primitives
    (|qiskit_ibm_runtime|_ ) or Terra backend primitives (:class:`qiskit.primitives.BackendSampler`,
    :class:`qiskit.primitives.BackendEstimator`). Certain classes, such as the
    :class:`~qiskit.opflow.expectations.AerPauliExpectation`, are only replaced by a specific primitive instance
    (in this case, |qiskit_aer.primitives.Estimator|_ ), or require a specific option configuration.
    This will be explicitly indicated in the corresponding section.

Contents
--------

This document covers the migration from these opflow sub-modules:

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
-------------------
*Back to* `Contents`_

.. |qiskit.quantum_info.BaseOperator| replace:: ``qiskit.quantum_info.BaseOperator``
.. _qiskit.quantum_info.BaseOperator: https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/operators/base_operator.py

The :class:`qiskit.opflow.OperatorBase` abstract class can be replaced with |qiskit.quantum_info.BaseOperator|_ ,
keeping in mind that |qiskit.quantum_info.BaseOperator|_ is more generic than its opflow counterpart.

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative
   * - :class:`qiskit.opflow.OperatorBase`
     - |qiskit.quantum_info.BaseOperator|_

..  attention::

    Despite the similar class names, :class:`qiskit.opflow.OperatorBase` and
    |qiskit.quantum_info.BaseOperator|_ are not completely equivalent to each other, and the transition
    should be handled with care. Namely:

    1. :class:`qiskit.opflow.OperatorBase` implements a broader algebra mixin. Some operator overloads that were
    commonly used :mod:`~qiskit.opflow` (for example ``~`` for ``.adjoint()``) are not defined for
    |qiskit.quantum_info.BaseOperator|_. You might want to check the specific
    :mod:`~qiskit.quantum_info` subclass instead.

    2. :class:`qiskit.opflow.OperatorBase` also implements methods such as ``.to_matrix()`` or ``.to_spmatrix()``,
    which are only found in some of the |qiskit.quantum_info.BaseOperator|_ subclasses.

    See API reference for more information.


Operator Globals
----------------
*Back to* `Contents`_

Opflow provided shortcuts to define common single qubit states, operators, and non-parametrized gates in the
|operator_globals|_ module.

These were mainly used for didactic purposes or quick prototyping, and can easily be replaced by their corresponding
:mod:`~qiskit.quantum_info` class: :class:`~qiskit.quantum_info.Pauli`, :class:`~qiskit.quantum_info.Clifford` or
:class:`~qiskit.quantum_info.Statevector`.


1-Qubit Paulis
~~~~~~~~~~~~~~
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

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Defining the XX operator</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import X

    operator = X ^ X


**Alternative**

.. code-block:: python

    from qiskit.quantum_info import Pauli, SparsePauliOp

    X = Pauli('X')
    op = X ^ X

    # equivalent to:
    op = Pauli('XX')

    # equivalent to:
    op = SparsePauliOp('XX')

.. raw:: html

   </details>

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: Defining a more complex operator</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import I, X, Z, PauliSumOp

    op = 0.39 * (I ^ Z ^ I) + 0.5 * (I ^ X ^ X)

    # or ...
    op = PauliSumOp.from_list([("IZI", 0.39), ("IXX", 0.5)])


**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp

    op = SparsePauliOp(["IZI", "IXX"], coeffs = [0.39, 0.5])

    # or...
    op = SparsePauliOp.from_list([("IZI", 0.39), ("IXX", 0.5)])

    # or...
    op = SparsePauliOp.from_sparse_list([("Z", [1], 0.39), ("XX", [0,1], 0.5)], num_qubits = 3)

.. raw:: html

   </details>

Common non-parametrized gates (Clifford)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.CX`, :class:`~qiskit.opflow.S`, :class:`~qiskit.opflow.H`, :class:`~qiskit.opflow.T`,
       :class:`~qiskit.opflow.CZ`, :class:`~qiskit.opflow.Swap`
     - Append corresponding gate to :class:`~qiskit.circuit.QuantumCircuit`. :mod:`~qiskit.quantum_info`
       :class:`~qiskit.quantum_info.Operator`\s can be also directly constructed from quantum circuits.
       Another alternative is to wrap the circuit in :class:`~qiskit.quantum_info.Clifford` and call
       ``Clifford.to_operator()``.

       ..  note::

            Constructing :mod:`~qiskit.quantum_info` operators from circuits is not efficient, as it is a dense operation and
            scales exponentially with the size of the circuit, use with care.


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Defining the HH operator</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import H

    op = H ^ H

**Alternative**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, Operator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    op = Clifford(qc).to_operator()

    # or...
    qc = QuantumCircuit(1)
    qc.h(0)
    H = Clifford(qc).to_operator()
    op = H ^ H

    # or, directly
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    op = Operator(qc)

.. raw:: html

   </details>

1-Qubit States
~~~~~~~~~~~~~~
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.Zero`, :class:`~qiskit.opflow.One`, :class:`~qiskit.opflow.Plus`, :class:`~qiskit.opflow.Minus`
     - :class:`~qiskit.quantum_info.Statevector` or simply :class:`~qiskit.circuit.QuantumCircuit`, depending on the use case.

       ..  note::

           For efficient simulation of stabilizer states, :mod:`~qiskit.quantum_info` includes a
           :class:`~qiskit.quantum_info.StabilizerState` class. See API ref. for more info.

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Working with stabilizer states</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import Zero, One, Plus, Minus

    # Zero, One, Plus, Minus are all stabilizer states
    state1 = Zero ^ One
    state2 = Plus ^ Minus

**Alternative**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import StabilizerState, Statevector

    qc_zero = QuantumCircuit(1)
    qc_one = qc_zero.copy()
    qc_one.x(0)
    state1 = Statevector(qc_zero) ^ Statevector(qc_one)

    qc_plus = qc_zero.copy()
    qc_plus.h(0)
    qc_minus = qc_one.copy()
    qc_minus.h(0)

    state2 = StabilizerState(qc_plus) ^ StabilizerState(qc_minus)


.. raw:: html

   </details>


Primitive and List Ops
----------------------
*Back to* `Contents`_

Most of the workflows that previously relied in components from :mod:`~qiskit.opflow.primitive_ops` and
:mod:`~qiskit.opflow.list_ops` can now leverage elements from :mod:`~qiskit.quantum_info`\'s
operators instead.
Some of these classes do not require a 1-1 replacement because they were created to interface with other
opflow components.

Primitive Ops
~~~~~~~~~~~~~~
*Back to* `Contents`_

:class:`~qiskit.opflow.primitive_ops.PrimitiveOp` is the :mod:`~qiskit.opflow.primitive_ops` module's base class.
It also acts as a factory to instantiate a corresponding sub-class depending on the computational primitive used
to initialize it.

.. tip::

    Interpreting :class:`~qiskit.opflow.primitive_ops.PrimitiveOp` as a factory class:

    .. list-table::
       :header-rows: 1

       * - Class passed to :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`
         - Sub-class returned

       * - :class:`~qiskit.quantum_info.Pauli`
         - :class:`~qiskit.opflow.primitive_ops.PauliOp`

       * - :class:`~qiskit.circuit.Instruction`, :class:`~qiskit.circuit.QuantumCircuit`
         - :class:`~qiskit.opflow.primitive_ops.CircuitOp`

       * - ``list``, ``np.ndarray``, ``scipy.sparse.spmatrix``, :class:`~qiskit.quantum_info.Operator`
         - :class:`~qiskit.opflow.primitive_ops.MatrixOp`

Thus, when migrating opflow code, it is important to look for alternatives to replace the specific subclasses that
might have been used "under the hood" in the original code:

.. |qiskit.quantum_info.Z2Symmetries| replace:: ``qiskit.quantum_info.Z2Symmetries``
.. _qiskit.quantum_info.Z2Symmetries: https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/analysis/z2_symmetries.py

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
       wrap in :class:`~qiskit.quantum_info.SparsePauliOp`

   * - :class:`~qiskit.opflow.primitive_ops.PauliSumOp`
     - :class:`~qiskit.quantum_info.SparsePauliOp`. See example below

   * - :class:`~qiskit.opflow.primitive_ops.TaperedPauliSumOp`
     - This class was used to combine a :class:`.PauliSumOp` with its identified symmetries in one object.
       This functionality is not currently used in any workflow, and has been deprecated without replacement.
       See |qiskit.quantum_info.Z2Symmetries|_ example for updated workflow.

   * - :class:`qiskit.opflow.primitive_ops.Z2Symmetries`
     - |qiskit.quantum_info.Z2Symmetries|_ . See example below.

.. _pauli_sum_op:

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: <code>PauliSumOp</code></font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import PauliSumOp
    from qiskit.quantum_info import SparsePauliOp, Pauli

    qubit_op = PauliSumOp(SparsePauliOp(Pauli("XYZY"), coeffs=[2]), coeff=-3j)

**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp, Pauli

    qubit_op = SparsePauliOp(Pauli("XYZY")), coeff=-6j)

.. raw:: html

   </details>

.. _z2_sym:

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: <code>Z2Symmetries</code> and <code>TaperedPauliSumOp</code></font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import PuliSumOp, Z2Symmetries, TaperedPauliSumOp

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
    tapered_op = z2_symmetries.taper(qubit_op)
    # can be represented as:
    tapered_op = TaperedPauliSumOp(primitive, z2_symmetries)

**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp, Z2Symmetries

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
    tapered_op = z2_symmetries.taper(qubit_op)

.. raw:: html

   </details>

ListOps
~~~~~~~
*Back to* `Contents`_

The :mod:`~qiskit.opflow.list_ops` module contained classes for manipulating lists of :mod:`~qiskit.opflow.primitive_ops`
or :mod:`~qiskit.opflow.state_fns`. The :mod:`~qiskit.quantum_info` alternatives for this functionality are the
:class:`~qiskit.quantum_info.PauliList`, :class:`~qiskit.quantum_info.SparsePauliOp` (for sums of :class:`~qiskit.quantum_info.Pauli`\s).

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
---------------
*Back to* `Contents`_

.. |qiskit.quantum_info.QuantumState| replace:: ``qiskit.quantum_info.QuantumState``
.. _qiskit.quantum_info.QuantumState: https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/states/quantum_state.py


The :mod:`~qiskit.opflow.state_fns` module can be generally replaced by subclasses of :mod:`~qiskit.quantum_info`\'s
|qiskit.quantum_info.QuantumState|_ , with some differences to keep in mind:

1. The primitives-based workflow does not rely on constructing state functions as opflow did
2. Algorithm-specific functionality has been migrated to the respective algorithm's module


Similarly to :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`, :class:`~qiskit.opflow.state_fns.StateFn`
acts as a factory to create the corresponding sub-class depending on the computational primitive used to initialize it.

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
identify the sub-class that is being used, to then look for an alternative.

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.state_fns.StateFn`
     - In most cases, :class:`~qiskit.quantum_info.Statevector`. Remember that this is a factory class.

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



.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Applying an operator to a state</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import StateFn, X, Y

    qc = QuantumCircuit(2)
    op = X ^ Y
    state = StateFn(qc)

    comp = ~op @ state
    # returns a CircuitStateFn

    eval = comp.eval()
    # returns a VectorStateFn (Statevector)

**Alternative**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    qc = QuantumCircuit(2)
    op = SparsePauliOp("XY")
    state = Statevector(qc)

    eval = state.evolve(operator)
    # returns a Statevector

.. raw:: html

   </details>
   <br>

See more applied examples in `Expectations`_  and `Converters`_.


Converters
----------
*Back to* `Contents`_

The role of this sub-module was to convert the operators into other opflow operator classes
(:class:`~qiskit.opflow.converters.TwoQubitReduction`, :class:`~qiskit.opflow.converters.PauliBasisChange`...).
In the case of the :class:`~qiskit.opflow.converters.CircuitSampler`, it traversed an operator and outputted
approximations of its state functions using a quantum backend.
Notably, this functionality has been replaced by the :mod:`~qiskit.primitives`.

.. |ParityMapper| replace:: ``ParityMapper``
.. _ParityMapper: https://qiskit.org/documentation/nature/stubs/qiskit_nature.second_q.mappers.ParityMapper.html#qiskit_nature.second_q.mappers.ParityMapper


.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.converters.CircuitSampler`
     - :class:`~qiskit.primitives.Sampler` or :class:`~qiskit.primitives.Estimator` if used with
       :class:`~qiskit.oflow.expectations`. See examples below.
   * - :class:`~qiskit.opflow.converters.AbelianGrouper`
     - This class allowed a sum a of Pauli operators to be grouped, a similar functionality can be achieved
       through the :meth:`~qiskit.quantum_info.SparsePauliOp.group_commuting` method of
       :class:`qiskit.quantum_info.SparsePauliOp`, although this is not a 1-1 replacement, as you can see
       in the example below.
   * - :class:`~qiskit.opflow.converters.DictToCircuitSum`
     - No direct replacement. This class was used to convert from :class:`~qiskit.opflow.state_fns.DictStateFn`\s or
       :class:`~qiskit.opflow.state_fns.VectorStateFn`\s to equivalent :class:`~qiskit.opflow.state_fns.CircuitStateFn`\s.
   * - :class:`~qiskit.opflow.converters.PauliBasisChange`
     - No direct replacement. This class was used for changing Paulis into other bases.
   * -  :class:`~qiskit.opflow.converters.TwoQubitReduction`
     -  No direct replacement. This class implements a chemistry-specific reduction for the |ParityMapper|_ class in ``qiskit-nature``.
        The general symmetry logic this mapper depends on has been refactored to other classes in :mod:`~qiskit.quantum_info`,
        so this specific :mod:`~qiskit.opflow` implementation is no longer necessary.


.. _convert_state:

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: <code>CircuitSampler</code> for sampling parametrized circuits</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit_aer import Aer
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.opflow import ListOp, StateFn, CircuitSampler

    x, y = Parameter("x"), Parameter("y")

    circuit1 = QuantumCircuit(1)
    circuit1.p(0.2, 0)
    circuit2 = QuantumCircuit(1)
    circuit2.p(x, 0)
    circuit3 = QuantumCircuit(1)
    circuit3.p(y, 0)

    bindings = {x: -0.4, y: 0.4}
    listop = ListOp([StateFn(circuit) for circuit in [circuit1, circuit2, circuit3]])

    sampler = CircuitSampler(Aer.get_backend("aer_simulator"))
    sampled = sampler.convert(listop, params=bindings).eval()
    # returns list of SparseVectorStateFn

**Alternative**

.. code-block:: python

    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.primitives import Sampler

    x, y = Parameter("x"), Parameter("y")

    circuit1 = QuantumCircuit(1)
    circuit1.p(0.2, 0)
    # Don't forget to add measurements!!!!!
    circuit1.measure_all()
    circuit2 = QuantumCircuit(1)
    circuit2.p(x, 0)
    circuit2.measure_all()
    circuit3 = QuantumCircuit(1)
    circuit3.p(y, 0)
    circuit3.measure_all()

    circuits = [circuit1, circuit2, circuit3]
    param_values = [None, [-0.4], [0.4]]

    sampler = Sampler()
    sampled = sampler.run(circuits, param_values).result().quasi_dists
    # returns qiskit.result.QuasiDist

.. raw:: html

    </details>


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: <code>CircuitSampler</code> for computing expectation values</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.opflow import X, Z, StateFn, CircuitStateFn, CircuitSampler
    from qiskit.providers.aer import AerSimulator

    qc = QuantumCircuit(1)
    qc.h(0)
    state = CircuitStateFn(qc)
    hamiltonian = X + Z

    expr = StateFn(hamiltonian, is_measurement=True).compose(state)
    backend = AerSimulator()
    sampler = CircuitSampler(backend)
    expectation = sampler.convert(expr)
    expectation_value = expectation.eval().real

**Alternative**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp

    state = QuantumCircuit(1)
    state.h(0)
    hamiltonian = SparsePauliOp.from_list([('X', 1), ('Z',1)])

    estimator = Estimator()
    expectation_value = estimator.run(state, hamiltonian).result().values.real

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><a><font size="+1">Example 3: <code>AbelianGrouper</code> for grouping operators</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import PauliSumOp, AbelianGrouper

    op = PauliSumOp.from_list([("XX", 2), ("YY", 1), ("IZ",2j), ("ZZ",1j)])

    grouped_sum = AbelianGrouper.group_subops(op)
    # returns: SummedOp([PauliSumOp(SparsePauliOp(['XX'], coeffs=[2.+0.j]), coeff=1.0),
    #                   PauliSumOp(SparsePauliOp(['YY'], coeffs=[1.+0.j]), coeff=1.0),
    #                   PauliSumOp(SparsePauliOp(['IZ', 'ZZ'], coeffs=[0.+2.j, 0.+1.j]),
    #                   coeff=1.0)], coeff=1.0, abelian=False)


**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp

    op = SparsePauliOp.from_list([("XX", 2), ("YY", 1), ("IZ",2j), ("ZZ",1j)])

    grouped = op.group_commuting()
    # returns: [SparsePauliOp(["IZ", "ZZ"], coeffs=[0.+2.j, 0.+1j]),
    #           SparsePauliOp(["XX", "YY"], coeffs=[2.+0.j, 1.+0.j])]

    grouped = op.group_commuting(qubit_wise=True)
    # returns: [SparsePauliOp(['XX'], coeffs=[2.+0.j]),
    #           SparsePauliOp(['YY'], coeffs=[1.+0.j]),
    #           SparsePauliOp(['IZ', 'ZZ'], coeffs=[0.+2.j, 0.+1.j])]

.. raw:: html

    </details>

Evolutions
----------
*Back to* `Contents`_

The :mod:`qiskit.opflow.evolutions` sub-module was created to provide building blocks for Hamiltonian simulation algorithms,
including various methods for trotterization. The original opflow workflow for hamiltonian simulation did not allow for
delayed synthesis of the gates or efficient transpilation of the circuits, so this functionality was migrated to the
:mod:`qiskit.synthesis` evolution module.

.. note::

    The :class:`qiskit.opflow.evolutions.PauliTrotterEvolution` class computes evolutions for exponentiated sums of Paulis by changing them each to the
    Z basis, rotating with an RZ, changing back, and trotterizing following the desired scheme. Within its ``.convert`` method,
    the class follows a recursive strategy that involves creating :class:`qiskit.opflow.evolutions.EvolvedOp` placeholders for the operators,
    constructing :class:`.PauliEvolutionGate`\s out of the operator primitives and supplying one of the desired synthesis methods to
    perform the trotterization (either via a ``string``\, which is then inputted into a :class:`qiskit.opflow.evolutions.TrotterizationFactory`,
    or by supplying a method instance of :class:`qiskit.opflow.evolutions.Trotter`, :class:`qiskit.opflow.evolutions.Suzuki` or :class:`qiskit.opflow.evolutions.QDrift`).

    The different trotterization methods that extend :class:`qiskit.opflow.evolutions.TrotterizationBase` were migrated to
    :mod:`qiskit.synthesis`,
    and now extend the :class:`qiskit.synthesis.ProductFormula` base class. They no longer contain a ``.convert()`` method for
    standalone use, but now are designed to be plugged into the :class:`.PauliEvolutionGate` and called via ``.synthesize()``.
    In this context, the job of the :class:`qiskit.opflow.evolutions.PauliTrotterEvolution` class can now be handled directly by the algorithms
    (for example, :class:`~qiskit.algorithms.time_evolvers.trotterization.TrotterQRTE`\).

    In a similar manner, the :class:`qiskit.opflow.evolutions.MatrixEvolution` class performs evolution by classical matrix exponentiation,
    constructing a circuit with :class:`.UnitaryGate`\s or :class:`.HamiltonianGate`\s containing the exponentiation of the operator.
    This class is no longer necessary, as the :class:`.HamiltonianGate`\s can be directly handled by the algorithms.

Trotterizations
~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~~~~~~
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



.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Trotter evolution</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import Trotter, PauliTrotterEvolution, PauliSumOp

    hamiltonian = PauliSumOp.from_list([('X', 1), ('Z',1)])
    evolution = PauliTrotterEvolution(trotter_mode=Trotter(), reps=2)
    evol_result = evolution.convert(hamiltonian.exp_i())
    evolved_state = evol_result.to_circuit()

**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit import QuantumCircuit

    hamiltonian = SparsePauliOp.from_list([('X', 1), ('Z',1)])
    evol_gate = PauliEvolutionGate(hamiltonian, time=1, synthesis=SuzukiTrotter(reps=2))
    evolved_state = QuantumCircuit(1)
    evolved_state.append(evol_gate, [0])

.. raw:: html

    </details>


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: Evolution with time-dependent Hamiltonian</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import Trotter, PauliTrotterEvolution, PauliSumOp
    from qiskit.circuit import Parameter

    time = Parameter('t')
    hamiltonian = PauliSumOp.from_list([('X', 1), ('Y',1)])
    evolution = PauliTrotterEvolution(trotter_mode=Trotter(), reps=1)
    evol_result = evolution.convert((time * hamiltonian).exp_i())
    evolved_state = evol_result.to_circuit()

**Alternative**

.. code-block:: python

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

.. raw:: html

    </details>



.. raw:: html

    <details>
    <summary><a><font size="+1">Example 3: Matrix evolution</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import MatrixEvolution, MatrixOp

    hamiltonian = MatrixOp([[0, 1], [1, 0]])
    evolution = MatrixEvolution()
    evol_result = evolution.convert(hamiltonian.exp_i())
    evolved_state = evol_result.to_circuit()

**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp
    from qiskit.extensions import HamiltonianGate
    from qiskit import QuantumCircuit

    evol_gate = HamiltonianGate([[0, 1], [1, 0]], 1)
    evolved_state = QuantumCircuit(1)
    evolved_state.append(evol_gate, [0])

.. raw:: html

    </details>

Expectations
------------
*Back to* `Contents`_

Expectations are converters which enable the computation of the expectation value of an observable with respect to some state function.
This functionality can now be found in the estimator primitive.

Algorithm-Agnostic Expectations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.expectations.ExpectationFactory`
     - No direct replacement. This class was used to create instances of one of the classes listed below.

   * - :class:`~qiskit.opflow.expectations.AerPauliExpectation`
     - Use ``Estimator`` primitive from |qiskit_aer.primitives|_ with ``approximation=True`` and ``shots=None`` as ``run_options``.
       See example below.

   * - :class:`~qiskit.opflow.expectations.MatrixExpectation`
     - Use :class:`~qiskit.primitives.Estimator` primitive from :mod:`qiskit` (if no shots are set, it performs an exact Statevector calculation).
       See example below.

   * - :class:`~qiskit.opflow.expectations.PauliExpectation`
     - Use any Estimator primitive (for :class:`qiskit.primitives.Estimator`, set ``shots!=None`` for a shot-based
       simulation, for |qiskit_aer.primitives.Estimator|_ , this is the default).


.. _expect_state:


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: Aer Pauli expectation</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import Z, CX, H, I, Zero, StateFn, AerPauliExpectation, CircuitSampler
    from qiskit.utils import QuantumInstance
    from qiskit_aer import Aer

    backend = Aer.get_backend("aer_simulator")
    q_instance = QuantumInstance(backend)

    sampler = CircuitSampler(q_instance, attach_results=True)
    expect = AerPauliExpectation()

    op = Z ^ Z
    wvf = CX @ (H ^ I) @ Zero

    converted_meas = expect.convert(~StateFn(op) @ wvf)
    expect_values = sampler.convert(converted_meas).eval()

**Alternative**

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp
    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import Estimator as AerEstimator

    estimator = AerEstimator(run_options={"approximation": True, "shots": None})

    op = SparsePauliOp.from_list([("ZZ", 1)])
    wvf = QuantumCircuit(2)
    wvf.h(1)
    wvf.cx(0,1)

    expect_values = estimator.run(wvf,op).result().values

.. raw:: html

    </details>

.. _matrix_state:


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 2: Matrix expectation</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit_aer import Aer
    from qiskit.opflow import X, H, I, MatrixExpectation, ListOp, StateFn
    from qiskit.utils import QuantumInstance

    backend = Aer.get_backend("statevector_simulator")
    q_instance = QuantumInstance(backend)
    sampler = CircuitSampler(q_instance, attach_results=True)
    expect = MatrixExpectation()

    mixed_ops = ListOp([X.to_matrix_op(), H])
    converted_meas = expect.convert(~StateFn(mixed_ops))

    plus_mean = converted_meas @ Plus
    values_plus = sampler.convert(plus_mean).eval()

**Alternative**

.. code-block:: python

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

.. raw:: html

    </details>

CVaRExpectation
~~~~~~~~~~~~~~~
*Back to* `Contents`_

.. list-table::
   :header-rows: 1

   * - Opflow
     - Alternative

   * - :class:`~qiskit.opflow.expectations.CVaRExpectation`
     - Functionality migrated into new VQE algorithm: :class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQE`

..  _cvar:


.. raw:: html

    <details>
    <summary><a><font size="+1">Example 1: VQE with CVaR</font></a></summary>
    <br>

**Opflow**

.. code-block:: python

    from qiskit.opflow import CVaRExpectation, PauliSumOp

    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import SLSQP
    from qiskit.circuit.library import TwoLocal
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    ansatz = TwoLocal(2, 'ry', 'cz')
    op = PauliSumOp.from_list([('ZZ',1), ('IZ',1), ('II',1)])
    alpha = 0.2
    cvar_expectation = CVaRExpectation(alpha=alpha)
    opt = SLSQP(maxiter=1000)
    vqe = VQE(ansatz, expectation=cvar_expectation, optimizer=opt, quantum_instance=backend)
    result = vqe.compute_minimum_eigenvalue(op)

**Alternative**

.. code-block:: python

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

.. raw:: html

    </details>

Gradients
---------
*Back to* `Contents`_

Replaced by the new :mod:`qiskit.algorithms.gradients` module. You can see further details in the
`algorithms migration guide <http://qisk.it/algo_migration>`_.

