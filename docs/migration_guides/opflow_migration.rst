=======================
Opflow Migration Guide
=======================

*Jump to* `TL;DR`_.

Background
----------

The :mod:`~qiskit.opflow` module was originally introduced as a layer between circuits and algorithms, a series of building blocks
for quantum algorithms research and development. The core design of opflow was based on the assumption that the
point of access to backends (real devices or simulators) was a ``backend.run()``
type of method: a method that takes in a circuit and returns its measurement results.
Under this assumption, all the tasks related to operator handling and building expectation value
computations were left to the user to manage. Opflow helped bridge that gap, it allowed to wrap circuits and
observables into operator classes that could be algebraically manipulated, so that the final result's expectation
values could be easily computed following different methods.

This basic opflow functionality is covered by  its core submodules: the ``operators`` submodule
(including :mod:`~qiskit.opflow.operator_globals`, :mod:`~qiskit.opflow.list_ops`, :mod:`~qiskit.opflow.primitive_ops`, and :mod:`~qiskit.opflow.state_fns`), 
the :mod:`~qiskit.opflow.converters` submodule, and the :mod:`~qiskit.opflow.expectations` submodule.
Following this reference framework of ``operators``, :mod:`~qiskit.opflow.converters` and :mod:`~qiskit.opflow.expectations`, opflow includes more
algorithm-specific functionality, which can be found in the :mod:`~qiskit.opflow.evolutions` submodule (specific for hamiltonian
simulation algorithms), as well as the :mod:`~qiskit.opflow.gradients` submodule (applied in multiple machine learning and optimization
use-cases). Some classes from the core modules mentioned above are also algorithm or application-specific,
for example the :obj:`~CVarMeasurement` or the :obj:`~Z2Symmetries`.

..  With the introduction of the primitives we have a new mechanism that allows.... efficient... error mitigation...

The recent introduction of the :mod:`~qiskit.primitives` provided a new interface for interacting with backends. Now, instead of
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

.. list-table:: Migration of ``qiskit.opflow.operator_base``
   :header-rows: 1

   * - opflow
     - alternative
     - notes
   * - :class:`~opflow.OperatorBase`
     - :class:`~quantum_info.BaseOperator`
     - For more information, check the :class:`~quantum_info.BaseOperator` source code.

Operator Globals
----------------
Opflow provided shortcuts to define common single qubit states, operators, and common non-parametrized gates in the
:mod:`~qiskit.opflow.operator_globals` module. These were mainly used for didactic purposes and can easily be replaced by their corresponding
:mod:`~qiskit.quantum_info` class: :class:`~qiskit.quantum_info.Pauli`, :class:`~qiskit.quantum_info.Clifford` or :class:`~qiskit.quantum_info.Statevector`.

1-Qubit Paulis
~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.operator_globals`` (1/3)
   :header-rows: 1

   * - opflow
     - alternative
     - notes
   * - :class:`~qiskit.opflow.X`, :class:`~qiskit.opflow.Y`, :class:`~qiskit.opflow.Z`, :class:`~qiskit.opflow.I`
     - :class:`~qiskit.quantum_info.Pauli`
     - For direct compatibility with classes in :mod:`~qiskit.algorithms`, wrap in :class:`~qiskit.quantum_info.SparsePauliOp`.
   * -

        .. code-block:: python

            from qiskit.opflow import X
            operator = X ^ X

     -

        .. code-block:: python

            from qiskit.quantum_info import Pauli
            X = Pauli('X')
            op = X ^ X

     -

        .. code-block:: python

            from qiskit.quantum_info import Pauli, SparsePauliOp
            op = Pauli('X') ^ Pauli('X') 
            
            # equivalent to:
            op = SparsePauliOp('XX')

Common non-parametrized gates (Clifford)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: Migration of ``qiskit.opflow.operator_globals`` (2/3)
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.CX`, :class:`~qiskit.opflow.S`, :class:`~qiskit.opflow.H`, :class:`~qiskit.opflow.T`, :class:`~qiskit.opflow.CZ`, :class:`~qiskit.opflow.Swap`
     - Append corresponding gate to :class:`~qiskit.QuantumCircuit` + :class:`~qiskit.quantum_info.Clifford` + ``.to_operator()``
     -

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

     -

1-Qubit States
~~~~~~~~~~~~~~
.. list-table:: Migration of ``qiskit.opflow.operator_globals`` (3/3)
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.Zero`, :class:`~qiskit.opflow.One`, :class:`~qiskit.opflow.Plus`, :class:`~qiskit.opflow.Minus`
     - :class:`~qiskit.quantum_info.Statevector` or :class:`~qiskit.QuantumCircuit` directly
     -

   * -

        .. code-block:: python

            from qiskit.opflow import Zero, One, Plus, Minus

            state1 = Zero ^ One
            state2 = Plus ^ Minus

     -

        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector

            qc_zero = QuantumCircuit(1)
            qc_one = copy(qc_zero)
            qc_one.x(0)
            state1 = Statevector(qc_zero) ^ Statevector(qc_one)

            qc_plus = copy(qc_zero)
            qc_plus.h(0)
            qc_minus = copy(qc_one)
            qc_minus.h(0)
            state2 = Statevector(qc_plus) ^ Statevector(qc_minus)
     -



Primitive and List Ops
----------------------
Most of the workflows that previously relied in components from :mod:`~qiskit.opflow.primitive_ops` and :mod:`~qiskit.opflow.list_ops` can now
leverage elements from :mod:`~qiskit.quantum_info.operators` instead. Some of these classes do not require a 1-1 replacement because
they were created to interface with other opflow components.

PrimitiveOps
~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.primitive_ops``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.PrimitiveOp`
     - No replacement needed
     - Can directly use :class:`~qiskit.quantum_info.Operator``
   * - :class:`~qiskit.opflow.CircuitOp`
     - No replacement needed
     - Can directly use :class:`~qiskit.QuantumCircuit`
   * - :class:`~qiskit.opflow.MatrixOp`
     - :class:`~qiskit.quantum_info.Operator``
     -
   * - :class:`~qiskit.opflow.PauliOp`
     - :class:`~qiskit.quantum_info.Pauli`
     - For direct compatibility with classes in :mod:`~qiskit.algorithms`, wrap in :class:`~qiskit.quantum_info.SparsePauliOp`
   * - :class:`~qiskit.opflow.PauliSumOp`
     - :class:`~qiskit.quantum_info.SparsePauliOp`
     - See example below
   * - :class:`~qiskit.opflow.TaperedPauliSumOp`
     - This class was used to combine a :class:`~PauliSumOp` with its identified symmetries in one object. It has been deprecated without replacement
     - See ``Z2Symmetries`` example for updated workflow
   * - :class:`~qiskit.opflow.Z2Symmetries`
     - :class:`~qiskit.quantum_info.Z2Symmetries`
     - See example below


PrimitiveOps Examples
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * -  ``PauliSumOp`` **Example:**

        .. code-block:: python

            from qiskit.opflow import PuliSumOp
            from qiskit.quantum_info import SparsePauliOp, Pauli

            qubit_op = PauliSumOp(SparsePauliOp(Pauli("XYZY"), coeffs=[2]), coeff=-3j)

     -

        .. code-block:: python

            from qiskit.quantum_info import SparsePauliOp, Pauli

            qubit_op = SparsePauliOp(Pauli("XYZY")), coeff=-6j)

     -
   * -  ``Z2Symmetries`` **and** ``TaperedPauliSumOp`` **Example:**

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
     -


ListOps
~~~~~~~
.. list-table:: Migration of ``qiskit.opflow.list_ops``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.ListOp`
     - No replacement needed. This class was used internally within opflow.
     -

   * - :class:`~qiskit.opflow.ComposedOp`
     - No replacement needed. This class was used internally within opflow.
     -

   * - :class:`~qiskit.opflow.SummedOp`
     - No replacement needed. This class was used internally within opflow.
     -

   * - :class:`~qiskit.opflow.TensoredOp`
     - No replacement needed. This class was used internally within opflow.
     -

State Functions
---------------

This module can be generally replaced by :class:`~qiskit.quantum_info.QuantumState`, with some differences to keep in mind:

1. The primitives-based workflow does not rely on constructing state functions as opflow did
2. The equivalence is, once again, not 1-1.
3. Algorithm-specific functionality has been migrated to the respective algorithm's module

TODO: ADD EXAMPLE!

.. list-table:: Migration of ``qiskit.opflow.state_fns``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.StateFn`
     - No replacement needed. This class was used internally within opflow.
     -

   * - :class:`~qiskit.opflow.CircuitStateFn`
     - No replacement needed. This class was used internally within opflow.
     -

   * - :class:`~qiskit.opflow.DictStateFn`
     - No replacement needed. This class was used internally within opflow.
     -

   * - :class:`~qiskit.opflow.VectorStateFn`
     - This class was used internally within opflow, but there exists a :mod:`~qiskit.quantum_info` replacement. There's the :class:`~qiskit.quantum_info.Statevector` class and the :class:`~qiskit.quantum_info.StabilizerState` (Clifford based vector).
     -

   * - :class:`~qiskit.opflow.SparseVectorStateFn`
     - No replacement needed. This class was used internally within opflow.
     - See :class:`~qiskit.opflow.VectorStateFn`

   * - :class:`~qiskit.opflow.OperatorStateFn`
     - No replacement needed. This class was used internally within opflow.
     -
   * - :class:`~qiskit.opflow.CVaRMeasurement`
     - Used in :class:`~qiskit.opflow.CVaRExpectation`. Functionality now covered by :class:`~SamplingEstimator`. See example in expectations.
     -

Converters
----------

The role of this sub-module was to convert the operators into other opflow operator classes (:class:`~qiskit.opflow.TwoQubitReduction`, :class:`~qiskit.opflow.PauliBasisChange`...).
In the case of the :class:`~qiskit.opflow.CircuitSampler`, it traversed an operator and outputted approximations of its state functions using a quantum backend.
Notably, this functionality has been replaced by the :mod:`~qiskit.primitives`.

Circuit Sampler
~~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.CircuitSampler``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.CircuitSampler`
     - :class:`~primitives.Estimator`
     -

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

     -

Two Qubit Reduction
~~~~~~~~~~~~~~~~~~~~
.. list-table:: Migration of ``qiskit.opflow.TwoQubitReduction``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * -  :class:`~qiskit.opflow.TwoQubitReduction`
     -  This class used to implement a chemistry-specific reduction. It has been directly integrated in to the parity mapper class in ``qiskit-nature`` and has no replacement in ``qiskit``.
     -

Other Converters
~~~~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.converters``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.AbelianGrouper`
     - No replacement needed. This class was used internally within opflow.
     -
   * - :class:`~qiskit.opflow.DictToCircuitSum`
     - No replacement needed. This class was used internally within opflow.
     -
   * - :class:`~qiskit.opflow.PauliBasisChange`
     - No replacement needed. This class was used internally within opflow.
     -

Evolutions
----------

The :mod:`~qiskit.opflow.evolutions` sub-module was created to provide building blocks for hamiltonian simulation algorithms,
including various methods for trotterization. The original opflow workflow for hamiltonian simulation did not allow for
delayed synthesis of the gates or efficient transpilation of the circuits, so this functionality was migrated to the
:mod:`~qiskit.synthesis.evolution` module.

The :class:`~qiskit.opflow.PauliTrotterEvolution` class computes evolutions for exponentiated sums of Paulis by changing them each to the
Z basis, rotating with an RZ, changing back, and trotterizing following the desired scheme. Within its ``.convert`` method,
the class follows a recursive strategy that involves creating :class:`~qiskit.opflow.EvolvedOp` placeholders for the operators,
constructing :class:`~PauliEvolutionGate`\s out of the operator primitives and supplying one of the desired synthesis methods to
perform the trotterization (either via a ``string``\, which is then inputted into a :class:`~qiskit.opflow.TrotterizationFactory`,
or by supplying a method instance of :class:`~qiskit.opflow.Trotter`, :class:`~qiskit.opflow.Suzuki` or :class:`~qiskit.opflow.QDrift`).

The different trotterization methods that extend :class:`~qiskit.opflow.TrotterizationBase` were migrated to :mod:`~qiskit.synthesis`,
and now extend the :class:`~qiskit.synthesis.evolution.ProductFormula` base class. They no longer contain a ``.convert()`` method for
standalone use, but now are designed to be plugged into the :class:`~qiskit.synthesis.PauliEvolutionGate` and called via ``.synthesize()``.
In this context, the job of the :class:`~qiskit.opflow.PauliTrotterEvolution` class can now be handled directly by the algorithms
(for example, :class:`~qiskit.algorithms.time_evolvers.TrotterQRTE`\), as shown in the following example:

.. list-table:: Migration of ``qiskit.opflow.evolutions (1/2)``
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

In a similar manner, the :class:`~qiskit.opflow.MatrixEvolution` class performs evolution by classical matrix exponentiation,
constructing a circuit with :class:`~UnitaryGate`\s or :class:`~HamiltonianGate`\s containing the exponentiation of the operator.
This class is no longer necessary, as the :class:`~HamiltonianGate`\s can be directly handled by the algorithms.

.. list-table:: Migration of ``qiskit.opflow.evolutions (2/2)``
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

.. list-table:: Migration of ``qiskit.opflow.evolutions.trotterizations``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.TrotterizationFactory`
     - This class is no longer necessary.
     -
   * - :class:`~qiskit.opflow.Trotter`
     - :class:`~synthesis.SuzukiTrotter` or :class:`~synthesis.LieTrotter`
     -
   * - :class:`~qiskit.opflow.Suzuki`
     - `:class:`~synthesis.SuzukiTrotter`
     -
   * - :class:`~qiskit.opflow.QDrift`
     - :class:`~synthesis.QDrift`
     -

.. list-table:: Migration of ``qiskit.opflow.evolutions.evolutions``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.EvolutionFactory`
     - This class is no longer necessary.
     -
   * - :class:`~qiskit.opflow.EvolvedOp`
     - :class:`~synthesis.SuzukiTrotter`
     - This class is no longer necessary
   * - :class:`~qiskit.opflow.MatrixEvolution`
     - :class:`~HamiltonianGate`
     -
   * - :class:`~qiskit.opflow.PauliTrotterEvolution`
     - :class:`~PauliEvolutionGate`
     -

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
     - notes
   * - :class:`~qiskit.opflow.ExpectationFactory`
     - No replacement needed.
     -
   * - :class:`~qiskit.opflow.AerPauliExpectation`
     - Use :class:`~Estimator` primitive from ``qiskit_aer`` instead.
     -
   * - :class:`~qiskit.opflow.MatrixExpectation`
     - Use :class:`~Estimator` primitive from ``qiskit`` instead (uses Statevector).
     -
   * - :class:`~qiskit.opflow.PauliExpectation`
     - Use any :class:`~Estimator` primitive.
     -

TODO: ADD EXAMPLE!

CVarExpectation
~~~~~~~~~~~~~~~

.. list-table:: Migration of ``qiskit.opflow.expectations.CVaRExpectation``
   :header-rows: 1

   * - opflow
     - alternative
     - notes

   * - :class:`~qiskit.opflow.expectations.CVaRExpectation`
     - Functionality absorbed into corresponding VQE algorithm: :class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQE`
     -
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
     -

**Gradients**
-------------
Replaced by new gradients module (link) (link to new tutorial).

