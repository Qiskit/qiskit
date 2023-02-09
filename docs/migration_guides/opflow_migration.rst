=======================
Opflow Migration Guide
=======================

*Jump to* `TL;DR`_.

Background
----------

The :mod:`~qiskit.opflow` module was originally introduced as a layer between circuits and algorithms, a series of building blocks
for quantum algorithms research and development.

The recent introduction of the :mod:`~qiskit.primitives` provided a new interface for interacting with backends that disrupted
the "opflow way" of doing things. Now, instead of
preparing a circuit to execute with a ``backend.run()`` type of method, the algorithms can leverage the :class:`~Sampler` and
:class:`~Estimator` primitives, send parametrized circuits and observables, and directly receive quasi-probability distributions or
expectation values (respectively). This workflow simplifies considerably the pre-processing and post-processing steps
that previously relied on opflow. For example, the :class:`~Estimator` primitive returns expectation values from a series of
circuit-observable pairs, superseding most of the functionality of the :mod:`~qiskit.opflow.expectations` submodule. Without the need for
building opflow expectations, most of the components in ``operators`` also became redundant, as they commonly wrapped
elements from :mod:`~qiskit.quantum_info`.

Higher-level opflow sub-modules, such as the :mod:`~qiskit.opflow.gradients` sub-module, were refactored to take full advantage
of the primitives interface. They can now be accessed as part of the :mod:`~qiskit.algorithms` module,
together with other primitive-based subroutines. Similarly, the :mod:`~qiskit.opflow.evolutions` sub-module got refactored, and now
can be easily integrated into a primitives-based workflow (as seen in the new :mod:`~qiskit.algorithms.time_evolvers` algorithms).

All of these reasons have encouraged us to move away from opflow, and find new paths of developing algorithms based on
the :mod:`~qiskit.primitives` interface and the :mod:`~qiskit.quantum_info` module, which is a powerful tool for representing
and manipulating quantum operators.

This guide traverses the opflow submodules and provides either a direct alternative
(i.e. using :mod:`~qiskit.quantum_info`), or an explanation of how to replace their functionality in algorithms.

TL;DR
-----
The new :mod:`~qiskit.primitives` have superseded most of the :mod:`~qiskit.opflow` functionality. Thus, the latter is being deprecated.

Index
-----
This guide covers the migration from these opflow sub-modules:

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

The :class:`~opflow.OperatorBase` abstract class can generally be replaced with :class:`~quantum_info.BaseOperator`, keeping in
mind that :class:`~quantum_info.BaseOperator` is more generic than its opflow counterpart. In particular, you should consider that:

1. :class:`~opflow.OperatorBase` implements a broader algebra mixin. Some operator overloads are not available in
:class:`~quantum_info.BaseOperator`.

2. :class:`~opflow.OperatorBase` also implements methods such as ``.to_matrix()`` or ``.to_spmatrix()``, which are only found
in some of the :class:`~quantum_info.BaseOperator` subclasses.

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative
   * - :class:`~opflow.OperatorBase`
     - :class:`~quantum_info.BaseOperator`.
       For more information, check out the :class:`~quantum_info.BaseOperator` source code.

Operator Globals
----------------
Opflow provided shortcuts to define common single qubit states, operators, and non-parametrized gates in the
:mod:`~qiskit.opflow.operator_globals` module.

These were mainly used for didactic purposes or quick prototyping, and can easily be replaced by their corresponding
:mod:`~qiskit.quantum_info` class: :class:`~qiskit.quantum_info.Pauli`, :class:`~qiskit.quantum_info.Clifford` or :class:`~qiskit.quantum_info.Statevector`.


1-Qubit Paulis
~~~~~~~~~~~~~~
The 1-qubit paulis were commonly used for quick testing of algorithms, as they could be combined to create more complex operators
(for example, ``0.39 * (I ^ Z) + 0.5 * (X ^ X)``).
These operations implicitly created operators of type  :class:`~qiskit.opflow.PauliSumOp`, and can be replaced by
directly creating a corresponding :class:`~qiskit.quantum_info.SparsePauliOp`, as shown in the example below.


.. list-table::
   :header-rows: 1

   * - opflow
     - alternative
   * - :class:`~qiskit.opflow.X`, :class:`~qiskit.opflow.Y`, :class:`~qiskit.opflow.Z`, :class:`~qiskit.opflow.I`
     - :class:`~qiskit.quantum_info.Pauli`.
       For direct compatibility with classes in :mod:`~qiskit.algorithms`, wrap in :class:`~qiskit.quantum_info.SparsePauliOp`.
   * -
        .. code-block:: python

            from qiskit.opflow import X
            operator = X ^ X

     -
        .. code-block:: python

            from qiskit.quantum_info import Pauli, SparsePauliOp
            X = Pauli('X')
            op = X ^ X

            # equivalent to:
            op = SparsePauliOp('XX')

Common non-parametrized gates (Clifford)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.CX`, :class:`~qiskit.opflow.S`, :class:`~qiskit.opflow.H`, :class:`~qiskit.opflow.T`, :class:`~qiskit.opflow.CZ`, :class:`~qiskit.opflow.Swap`
     - Append corresponding gate to :class:`~qiskit.QuantumCircuit` + :class:`~qiskit.quantum_info.Clifford` + ``.to_operator()``.
       This alternative is not too efficient, as ``.to_operator()`` is dense and scales exponentially with the size of the circuit.

   * -
        .. code-block:: python

            from qiskit.opflow import H
            op = H ^ H

     -
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Clifford
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.h(1)
            op = Clifford(qc).to_operator()

            # or...
            qc = QuantumCircuit(1)
            qc.h(0)
            H = Clifford(qc).to_operator()
            op = H ^ H

1-Qubit States
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.Zero`, :class:`~qiskit.opflow.One`, :class:`~qiskit.opflow.Plus`, :class:`~qiskit.opflow.Minus`
     - :class:`~qiskit.quantum_info.StabilizerState` or :class:`~qiskit.quantum_info.Statevector` or :class:`~qiskit.QuantumCircuit`, depending on the use case.
       In principle, :class:`~qiskit.quantum_info.StabilizerState` is the most efficient replacement for :class:`~qiskit.opflow` states, but the functionality is not identical. See API ref. for more info.

   * -

        .. code-block:: python

            from qiskit.opflow import Zero, One, Plus, Minus

            state1 = Zero ^ One
            state2 = Plus ^ Minus

     -

        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import StabilizerState

            qc_zero = QuantumCircuit(1)
            qc_one = copy(qc_zero)
            qc_one.x(0)
            state1 = StabilizerState(qc_zero) ^ StabilizerState(qc_one)

            qc_plus = copy(qc_zero)
            qc_plus.h(0)
            qc_minus = copy(qc_one)
            qc_minus.h(0)
            state2 = StabilizerState(qc_plus) ^ StabilizerState(qc_minus)

Primitive and List Ops
----------------------
Most of the workflows that previously relied in components from :mod:`~qiskit.opflow.primitive_ops` and
:mod:`~qiskit.opflow.list_ops` can now leverage elements from ``quantum_info``\'s :mod:`~qiskit.quantum_info.operators` instead.
Some of these classes do not require a 1-1 replacement because they were created to interface with other
opflow components.

Primitive Ops
~~~~~~~~~~~~~~
:class:`~qiskit.opflow.primitive_ops.PrimitiveOp` is the :mod:`~qiskit.opflow.primitive_ops` module's base class.
It also acts as a factory to instantiate a corresponding sub-class depending on the computational primitive used
to initialize it:

.. list-table::
   :header-rows: 1

   * - class passed to constructor
     - sub-class returned

   * - :class:`~qiskit.quantum_info.Pauli`
     - :class:`~qiskit.opflow.primitive_ops.PauliOp`

   * - :class:`~qiskit.circuit.Instruction`, :class:`~qiskit.circuit.QuantumCircuit`
     - :class:`~qiskit.opflow.primitive_ops.CircuitOp`

   * - ``list``, ``np.ndarray``, ``scipy.sparse.spmatrix``, :class:`~qiskit.quantum_info.Operator`
     - :class:`~qiskit.opflow.primitive_ops.MatrixOp`

Thus, when migrating opflow code, it is important to look for alternatives to replace the specific subclasses that
might have been used "under the hood" in the original code:

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`
     - No direct replacement. In most use-cases (representing generic operators),
       the alternative is :class:`~qiskit.quantum_info.Operator`.

   * - :class:`~qiskit.opflow.primitive_ops.CircuitOp`
     - No direct replacement. :class:`~qiskit.QuantumCircuit` could be used as an alternative in some workflows.

   * - :class:`~qiskit.opflow.primitive_ops.MatrixOp`
     - :class:`~qiskit.quantum_info.Operator`

   * - :class:`~qiskit.opflow.primitive_ops.PauliOp`
     - :class:`~qiskit.quantum_info.Pauli`. For direct compatibility with classes in :mod:`~qiskit.algorithms`,
       wrap in :class:`~qiskit.quantum_info.SparsePauliOp`

   * - :class:`~qiskit.opflow.primitive_ops.PauliSumOp`
     - :class:`~qiskit.quantum_info.SparsePauliOp`. See example below.

   * - :class:`~qiskit.opflow.primitive_ops.TaperedPauliSumOp`
     - This class was used to combine a :class:`~PauliSumOp` with its identified symmetries in one object.
       This functionality is not currently used in any workflow, and has been deprecated without replacement.
       See ``Z2Symmetries`` example for updated workflow.

   * - :class:`~qiskit.opflow.primitive_ops.Z2Symmetries`
     - :class:`~qiskit.quantum_info.Z2Symmetries`. See example below.


Primitive Ops Examples
~~~~~~~~~~~~~~~~~~~~~~

``PauliSumOp`` Example
#######################

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -
        .. code-block:: python

            from qiskit.opflow import PauliSumOp
            from qiskit.quantum_info import SparsePauliOp, Pauli

            qubit_op = PauliSumOp(SparsePauliOp(Pauli("XYZY"), coeffs=[2]), coeff=-3j)

     -
        .. code-block:: python

            from qiskit.quantum_info import SparsePauliOp, Pauli

            qubit_op = SparsePauliOp(Pauli("XYZY")), coeff=-6j)

``Z2Symmetries`` and ``TaperedPauliSumOp`` Example
##################################################
.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -
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
     -
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

ListOps
~~~~~~~

The :mod:`~qiskit.opflow.list_ops` module contained classes for manipulating lists of :mod:`~qiskit.opflow.primitive_ops`
or :mod:`~qiskit.opflow.state_fns`. The :mod:`~qiskit.quantum_info` alternatives for this functionality are the
:class:`~qiskit.quantum_info.PauliList`, :class:`~qiskit.quantum_info.SparsePauliOp` (for sums of ``Pauli``\s),
:class:`~qiskit.quantum_info.PauliTable` (symplectic representation of lists of Pauli operators) and
:class:`~qiskit.quantum_info.StabilizerTable` (symplectic representation of lists of state functions).

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.list_ops.ListOp`
     - No direct replacement. This is the base class for operator lists. For ``Pauli`` operators, the
       alternative is :class:`~qiskit.quantum_info.PauliList`. For lists of state representations, an
       option is :class:`~qiskit.quantum_info.StabilizerTable`

   * - :class:`~qiskit.opflow.list_ops.ComposedOp`
     - No direct replacement. Current workflows do not require composition of states and operators within
       one object.

   * - :class:`~qiskit.opflow.list_ops.SummedOp`
     - No direct replacement. For ``Pauli`` operators, use :class:`~qiskit.quantum_info.SparsePauliOp`.

   * - :class:`~qiskit.opflow.list_ops.TensoredOp`
     - No direct replacement. For ``Pauli`` operators, use :class:`~qiskit.quantum_info.SparsePauliOp`.

ListOp Examples
~~~~~~~~~~~~~~~

``ListOp`` Example:
####################
.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -
        .. code-block:: python

            from qiskit.opflow import Zero, One, ListOp

            op1 = # list op with operators
            op2 = ~ListOp([One, Zero]) @ ListOp([One, Zero])
     -
        .. code-block:: python

            from qiskit.quantum_info import StabilizerTable


State Functions
---------------

The :mod:`~qiskit.opflow.state_fns` module can be generally replaced by :class:`~qiskit.quantum_info.QuantumState`,
with some differences to keep in mind:

1. The primitives-based workflow does not rely on constructing state functions as opflow did
2. Algorithm-specific functionality has been migrated to the respective algorithm's module

Similarly to :class:`~qiskit.opflow.primitive_ops.PrimitiveOp`, :class:`~qiskit.opflow.state_fns.StateFn`
acts as a factory to create the corresponding sub-class depending on the computational primitive used to initialize it:

.. list-table::
   :header-rows: 1

   * - class passed to constructor
     - sub-class returned

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

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.state_fns.StateFn`
     - No direct replacement

   * - :class:`~qiskit.opflow.state_fns.CircuitStateFn`
     - No direct replacement

   * - :class:`~qiskit.opflow.state_fns.DictStateFn`
     - No direct replacement

   * - :class:`~qiskit.opflow.state_fns.VectorStateFn`
     - There's the :class:`~qiskit.quantum_info.Statevector` class and the :class:`~qiskit.quantum_info.StabilizerState` (Clifford based vector).

   * - :class:`~qiskit.opflow.state_fns.SparseVectorStateFn`
     - No direct replacement. See :class:`~qiskit.opflow.state_fns.VectorStateFn`

   * - :class:`~qiskit.opflow.state_fns.OperatorStateFn`
     - No direct replacement

   * - :class:`~qiskit.opflow.state_fns.CVaRMeasurement`
     - Used in :class:`~qiskit.opflow.expectations.CVaRExpectation`. Functionality now covered by :class:`~SamplingEstimator`. See example in expectations.

StateFn Examples
~~~~~~~~~~~~~~~~~

``StateFn`` Example:
####################

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -
        .. code-block:: python

            from qiskit.opflow import PuliSumOp
            from qiskit.quantum_info import SparsePauliOp, Pauli

            qubit_op = PauliSumOp(SparsePauliOp(Pauli("XYZY"), coeffs=[2]), coeff=-3j)

     -
        .. code-block:: python

            from qiskit.quantum_info import SparsePauliOp, Pauli

            qubit_op = SparsePauliOp(Pauli("XYZY")), coeff=-6j)

Converters
----------

The role of this sub-module was to convert the operators into other opflow operator classes
(:class:`~qiskit.opflow.converters.TwoQubitReduction`, :class:`~qiskit.opflow.converters.PauliBasisChange`...).
In the case of the :class:`~qiskit.opflow.converters.CircuitSampler`, it traversed an operator and outputted
approximations of its state functions using a quantum backend.
Notably, this functionality has been replaced by the :mod:`~qiskit.primitives`.

Circuit Sampler
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.converters.CircuitSampler`
     - :class:`~primitives.Estimator`

   * -
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

     -
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.primitives import Estimator
            from qiskit.quantum_info import SparsePauliOp

            state = QuantumCircuit(1)
            state.h(0)
            hamiltonian = SparsePauliOp.from_list([('X', 1), ('Z',1)])

            estimator = Estimator()
            expectation_value = estimator.run(state, hamiltonian).result().values


Two Qubit Reduction
~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -  :class:`~qiskit.opflow.converters.TwoQubitReduction`
     -  No direct replacement. This class implements a chemistry-specific reduction for the ``ParityMapper`` class in ``qiskit-nature``.
        The general symmetry logic this mapper depends on has been refactored to other classes in :mod:`~qiskit.quantum_info`,
        so this specific :mod:`~qiskit.opflow` implementation is no longer necessary.

Other Converters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.converters.AbelianGrouper`
     - No direct replacement. This class allowed a sum a of Pauli operators to be grouped. These type of groupings are now left to the primitives to handle.
   * - :class:`~qiskit.opflow.converters.DictToCircuitSum`
     - No direct replacement
   * - :class:`~qiskit.opflow.converters.PauliBasisChange`
     - No direct replacement

Evolutions
----------

The :mod:`~qiskit.opflow.evolutions` sub-module was created to provide building blocks for hamiltonian simulation algorithms,
including various methods for trotterization. The original opflow workflow for hamiltonian simulation did not allow for
delayed synthesis of the gates or efficient transpilation of the circuits, so this functionality was migrated to the
:mod:`~qiskit.synthesis.evolution` module.

The :class:`~qiskit.opflow.evolutions.PauliTrotterEvolution` class computes evolutions for exponentiated sums of Paulis by changing them each to the
Z basis, rotating with an RZ, changing back, and trotterizing following the desired scheme. Within its ``.convert`` method,
the class follows a recursive strategy that involves creating :class:`~qiskit.opflow.evolutions.EvolvedOp` placeholders for the operators,
constructing :class:`~PauliEvolutionGate`\s out of the operator primitives and supplying one of the desired synthesis methods to
perform the trotterization (either via a ``string``\, which is then inputted into a :class:`~qiskit.opflow.evolutions.TrotterizationFactory`,
or by supplying a method instance of :class:`~qiskit.opflow.evolutions.Trotter`, :class:`~qiskit.opflow.evolutions.Suzuki` or :class:`~qiskit.opflow.evolutions.QDrift`).

The different trotterization methods that extend :class:`~qiskit.opflow.evolutions.TrotterizationBase` were migrated to :mod:`~qiskit.synthesis`,
and now extend the :class:`~qiskit.synthesis.evolution.ProductFormula` base class. They no longer contain a ``.convert()`` method for
standalone use, but now are designed to be plugged into the :class:`~qiskit.synthesis.PauliEvolutionGate` and called via ``.synthesize()``.
In this context, the job of the :class:`~qiskit.opflow.evolutions.PauliTrotterEvolution` class can now be handled directly by the algorithms
(for example, :class:`~qiskit.algorithms.time_evolvers.TrotterQRTE`\), as shown in the following example:

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -
        .. code-block:: python

            from qiskit.opflow import Trotter, PauliTrotterEvolution, PauliSumOp

            hamiltonian = PauliSumOp.from_list([('X', 1), ('Z',1)])
            evolution = PauliTrotterEvolution(trotter_mode=Trotter(), reps=1)
            evol_result = evolution.convert(hamiltonian.exp_i())
            evolved_state = evol_result.to_circuit()
     -
        .. code-block:: python

            from qiskit.quantum_info import SparsePauliOp
            from qiskit.synthesis import SuzukiTrotter
            from qiskit.circuit.library import PauliEvolutionGate
            from qiskit import QuantumCircuit

            hamiltonian = SparsePauliOp.from_list([('X', 1), ('Z',1)])
            evol_gate = PauliEvolutionGate(hamiltonian, 1, synthesis=SuzukiTrotter())
            evolved_state = QuantumCircuit(1)
            evolved_state.append(evol_gate, [0])

In a similar manner, the :class:`~qiskit.opflow.evolutions.MatrixEvolution` class performs evolution by classical matrix exponentiation,
constructing a circuit with :class:`~UnitaryGate`\s or :class:`~HamiltonianGate`\s containing the exponentiation of the operator.
This class is no longer necessary, as the :class:`~HamiltonianGate`\s can be directly handled by the algorithms.

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * -
        .. code-block:: python

            from qiskit.opflow import MatrixEvolution, MatrixOp

            hamiltonian = MatrixOp([[0, 1], [1, 0]])
            evolution = MatrixEvolution()
            evol_result = evolution.convert(hamiltonian.exp_i())
            evolved_state = evol_result.to_circuit()
     -
        .. code-block:: python

            from qiskit.quantum_info import SparsePauliOp
            from qiskit.extensions import HamiltonianGate
            from qiskit import QuantumCircuit

            evol_gate = HamiltonianGate([[0, 1], [1, 0]], 1)
            evolved_state = QuantumCircuit(1)
            evolved_state.append(evol_gate, [0])

To summarize:

.. list-table::
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.evolutions.TrotterizationFactory`
     -

   * - :class:`~qiskit.opflow.evolutions.Trotter`
     - :class:`~synthesis.SuzukiTrotter` or :class:`~synthesis.LieTrotter`

   * - :class:`~qiskit.opflow.evolutions.Suzuki`
     - :class:~synthesis.SuzukiTrotter`

   * - :class:`~qiskit.opflow.evolutions.QDrift`
     - :class:`~synthesis.QDrift`


.. list-table:: Migration of ``qiskit.opflow.evolutions.evolutions``
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.evolutions.EvolutionFactory`
     -

   * - :class:`~qiskit.opflow.evolutions.EvolvedOp`
     - :class:`~synthesis.SuzukiTrotter`

   * - :class:`~qiskit.opflow.evolutions.MatrixEvolution`
     - :class:`~HamiltonianGate`

   * - :class:`~qiskit.opflow.evolutions.PauliTrotterEvolution`
     - :class:`~PauliEvolutionGate`

Expectations
------------
Expectations are converters which enable the computation of the expectation value of an observable with respect to some state function.
This functionality can now be found in the estimator primitive.

Algorithm-Agnostic Expectations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.expectations``
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.expectations.ExpectationFactory`
     - No direct replacement

   * - :class:`~qiskit.opflow.expectations.AerPauliExpectation`
     - Use :class:`~Estimator` primitive from ``qiskit_aer`` with ``approximation=True`` and ``shots=None``

   * - :class:`~qiskit.opflow.expectations.MatrixExpectation`
     - Use :class:`~Estimator` primitive from ``qiskit`` instead (uses Statevector).

   * - :class:`~qiskit.opflow.expectations.PauliExpectation`
     - Use any :class:`~Estimator` primitive.


TODO: ADD EXAMPLES!

CVarExpectation
~~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.expectations.CVaRExpectation``
   :header-rows: 1

   * - opflow
     - alternative

   * - :class:`~qiskit.opflow.expectations.CVaRExpectation`
     - Functionality absorbed into corresponding VQE algorithm: :class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQE`

   * -

        .. code-block:: python

            from qiskit.opflow import CVaRExpectation, PauliSumOp

            from qiskit.algorithms import VQE
            from qiskit.algorithms.optimizers import SLSQP
            from qiskit.circuit.library import TwoLocal
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            ansatz = TwoLocal(2, 'ry', 'cz')
            op = PauliSumOp.from_list([('ZZ',1), ('IZ',1), ('II',1)])
            alpha=0.2
            cvar_expectation = CVaRExpectation(alpha=alpha)
            opt = SLSQP(maxiter=1000)
            vqe = VQE(ansatz, expectation=cvar_expectation, optimizer=opt, quantum_instance=backend)
            result = vqe.compute_minimum_eigenvalue(op)

     -

        .. code-block:: python

            from qiskit.quantum_info import SparsePauliOp

            from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
            from qiskit.algorithms.optimizers import SLSQP
            from qiskit.circuit.library import TwoLocal
            from qiskit.primitives import Sampler
            ansatz = TwoLocal(2, 'ry', 'cz')
            op = SparsePauliOp.from_list([('ZZ',1), ('IZ',1), ('II',1)])
            opt = SLSQP(maxiter=1000)
            alpha=0.2
            vqe = SamplingVQE(Sampler(), ansatz, optm, aggregation=alpha)
            result = vqe.compute_minimum_eigenvalue(op)

Gradients
---------
Replaced by the new :mod:`~qiskit.algorithms.gradients` module. You can see further details in the
algorithms migration guide.

