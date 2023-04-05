##########################
Algorithms Migration Guide
##########################

TL;DR
=====

The :mod:`qiskit.algorithms` module has been fully refactored to use the :mod:`~qiskit.primitives`, for circuit execution, instead of the :class:`~qiskit.utils.QuantumInstance`, which is now deprecated.

There have been **3 types of refactoring**:

1. Algorithms refactored in a new location to support :mod:`~qiskit.primitives`. These algorithms have the same
   class names as the :class:`~qiskit.utils.QuantumInstance`\-based ones but are in a new sub-package.

    .. attention::

       **Careful with import paths!!** The legacy algorithms are still importable directly from
       :mod:`qiskit.algorithms`. Until the legacy imports are removed, this convenience import is not available
       for the refactored algorithms. Thus, to import the refactored algorithms you must always
       **specify the full import path** (e.g., ``from qiskit.algorithms.eigensolvers import VQD``)

    - `Minimum Eigensolvers`_
    - `Eigensolvers`_
    - `Time Evolvers`_

2. Algorithms refactored in-place (same namespace) to support both :class:`~qiskit.utils.QuantumInstance` and
   :mod:`~qiskit.primitives`. In the future, the use of :class:`~qiskit.utils.QuantumInstance` will be removed.

    - `Amplitude Amplifiers`_
    - `Amplitude Estimators`_
    - `Phase Estimators`_


3. Algorithms that were deprecated and are now removed entirely from :mod:`qiskit.algorithms`. These are algorithms that do not currently serve
   as building blocks for applications. Their main value is educational, and as such, will be kept as tutorials
   in the qiskit textbook. You can consult the tutorials in the following links:

    - `Linear Solvers (HHL) <https://learn.qiskit.org/course/ch-applications/solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation>`_ ,
    - `Factorizers (Shor) <https://learn.qiskit.org/course/ch-algorithms/shors-algorithm>`_


The remainder of this migration guide will focus on the algorithms with migration alternatives within
:mod:`qiskit.algorithms`, that is, those under refactoring types 1 and 2.

Background
==========

*Back to* `TL;DR`_

The :mod:`qiskit.algorithms` module was originally built on top of the :mod:`qiskit.opflow` library and the
:class:`~qiskit.utils.QuantumInstance` utility. The development of the :mod:`~qiskit.primitives`
introduced a higher-level execution paradigm, with the ``Estimator`` for computation of
expectation values for observables, and ``Sampler`` for executing circuits and returning probability
distributions. These tools allowed to refactor the :mod:`qiskit.algorithms` module, and deprecate both
:mod:`qiskit.opflow` and :class:`~qiskit.utils.QuantumInstance`.

.. attention::

    The transition away from :mod:`qiskit.opflow` affects the classes that algorithms take as part of the problem
    setup. As a rule of thumb, most :mod:`qiskit.opflow` dependencies have a direct :mod:`qiskit.quantum_info`
    replacement. One common example is the class :mod:`qiskit.opflow.PauliSumOp`, used to define Hamiltonians
    (for example, to plug into VQE), that can be replaced by :mod:`qiskit.quantum_info.SparsePauliOp`.
    For information on how to migrate other :mod:`~qiskit.opflow` objects, you can refer to the
    `Opflow migration guide <https://qisk.it/opflow_migration>`_.

For further background and detailed migration steps, see the:

* `Opflow migration guide <https://qisk.it/opflow_migration>`_
* `Quantum Instance migration guide <https://qisk.it/qi_migration>`_


How to choose a primitive configuration for your algorithm
==========================================================

*Back to* `TL;DR`_

The classes in :mod:`qiskit.algorithms` state the base class primitive type (``Sampler``/``Estimator``)
they require for their initialization. Once the primitive type is known, you can choose between
four different primitive implementations, depending on how you want to configure your execution:

    a. Using **local** statevector simulators for quick prototyping: **Reference Primitives** in :mod:`qiskit.primitives`
    b. Using **local** Aer simulators for finer algorithm tuning: **Aer Primitives** in :mod:`qiskit_aer.primitives`
    c. Accessing backends using the **Qiskit Runtime Service**: **Runtime Primitives** in :mod:`qiskit_ibm_runtime`
    d. Accessing backends using a **non-Runtime-enabled provider**: **Backend Primitives** in :mod:`qiskit.primitives`


For more detailed information and examples, particularly on the use of the **Backend Primitives**, please refer to
the `Quantum Instance migration guide <https://qisk.it/qi_migration>`_.

In this guide, we will cover 3 different common configurations for algorithms that determine
**which primitive import** you should be selecting:

1. Running an algorithm with a statevector simulator (i.e., using :mod:`qiskit.opflow`\'s legacy
   :class:`.MatrixExpectation`), when you want the ideal outcome without shot noise:

        - Reference Primitives with default configuration (see `QAOA`_ example):

        .. code-block:: python

            from qiskit.primitives import Sampler, Estimator

        - Aer Primitives **with statevector simulator** (see `QAOA`_ example):

        .. code-block:: python

            from qiskit_aer.primitives import Sampler, Estimator

            sampler = Sampler(backend_options={"method": "statevector"})
            estimator = Estimator(backend_options={"method": "statevector"})

2. Running an algorithm using a simulator/device with shot noise
   (i.e., using :mod:`qiskit.opflow`\'s legacy :class:`.PauliExpectation`):

        - Reference Primitives **with shots** (see `VQE`_ examples):

        .. code-block:: python

            from qiskit.primitives import Sampler, Estimator

            sampler = Sampler(options={"shots": 100})
            estimator = Estimator(options={"shots": 100})

            # or...
            sampler = Sampler()
            job = sampler.run(circuits, shots=100)

            estimator = Estimator()
            job = estimator.run(circuits, observables, shots=100)

        - Aer Primitives with default configuration (see `VQE`_ examples):

        .. code-block:: python

            from qiskit_aer.primitives import Sampler, Estimator

        - Runtime Primitives with default configuration (see `VQD`_ example):

        .. code-block:: python

            from qiskit_ibm_runtime import Sampler, Estimator


3. Running an algorithm on an Aer simulator using a custom instruction (i.e., using :mod:`qiskit.opflow`\'s legacy
:class:`.AerPauliExpectation`):

        - Aer Primitives with ``shots=None``, ``approximation=True`` (see `TrotterQRTE`_ example):

        .. code-block:: python

            from qiskit_aer.primitives import Sampler, Estimator

            sampler = Sampler(run_options={"approximation": True, "shots": None})
            estimator = Estimator(run_options={"approximation": True, "shots": None})


Minimum Eigensolvers
====================
*Back to* `TL;DR`_

The minimum eigensolver algorithms belong to the first type of refactoring listed above
(Algorithms refactored in a new location to support :mod:`~qiskit.primitives`).
Instead of a :class:`~qiskit.utils.QuantumInstance`, :mod:`qiskit.algorithms.minimum_eigensolvers` are now initialized
using an instance of the :mod:`~qiskit.primitives.Sampler` or :mod:`~qiskit.primitives.Estimator` primitive, depending
on the algorithm. The legacy classes can still be found in :mod:`qiskit.algorithms.minimum_eigen_solvers`.

.. attention::

    For the :mod:`qiskit.algorithms.minimum_eigensolvers` classes, depending on the import path,
    you will access either the primitive-based or the quantum-instance-based
    implementation. You have to be extra-careful, because the class name does not change.

    * Old import (Quantum Instance based): ``from qiskit.algorithms import VQE, QAOA, NumPyMinimumEigensolver``
    * New import (Primitives based): ``from qiskit.algorithms.minimum_eigensolvers import VQE, SamplingVQE, QAOA, NumPyMinimumEigensolver``

VQE
---

The legacy :class:`qiskit.algorithms.minimum_eigen_solvers.VQE` class has now been split according to the use-case:

- For general-purpose Hamiltonians, you can use the Estimator-based :class:`qiskit.algorithms.minimum_eigensolvers.VQE`
  class.
- If you have a diagonal Hamiltonian, and would like the algorithm to return a sampling of the state, you can use
  the new Sampler-based :class:`qiskit.algorithms.minimum_eigensolvers.SamplingVQE` algorithm. This could formerly
  be realized using the legacy :class:`~qiskit.algorithms.minimum_eigen_solvers.VQE` with
  :class:`~qiskit.opflow.expectations.CVaRExpectation`.

.. note::

    In addition to taking in an :mod:`~qiskit.primitives.Estimator` instance instead of a :class:`~qiskit.utils.QuantumInstance`,
    the new :class:`~qiskit.algorithms.minimum_eigensolvers.VQE` signature has undergone the following changes:

    1. The ``expectation`` and ``include_custom`` parameters have been removed, as this functionality is now
       defined at the ``Estimator`` level.
    2. The ``gradient`` parameter now takes in an instance of a primitive-based gradient class from
       :mod:`qiskit.algorithms.gradients` instead of the legacy :mod:`qiskit.opflow.gradients.Gradient` class.
    3. The ``max_evals_grouped`` parameter has been removed, as it can be set directly on the optimizer class.
    4. The ``estimator``, ``ansatz`` and ``optimizer`` are the only parameters that can be defined positionally
       (and in this order), all others have become keyword-only arguments.

.. note::

    The new :class:`~qiskit.algorithms.minimum_eigensolvers.VQEResult` class does not include the state anymore, as
    this output was only useful in the case of diagonal operators. However, if it is available as part of the new
    :class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQE`'s :class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQEResult`.


.. dropdown:: VQE Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms import VQE
        from qiskit.algorithms.optimizers import SPSA
        from qiskit.circuit.library import TwoLocal
        from qiskit.opflow import PauliSumOp
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator

        ansatz = TwoLocal(2, 'ry', 'cz')
        opt = SPSA(maxiter=50)

        # shot-based simulation
        backend = AerSimulator()
        qi = QuantumInstance(backend=backend, shots=2048, seed_simulator=42)
        vqe = VQE(ansatz, optimizer=opt, quantum_instance=qi)

        hamiltonian = PauliSumOp.from_list([("XX", 1), ("XY", 1)])
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        print(result.eigenvalue)

    .. testoutput::

        (-0.9775390625+0j)

    **[Updated] Using Primitives:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms.minimum_eigensolvers import VQE # new import!!!
        from qiskit.algorithms.optimizers import SPSA
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import Estimator
        from qiskit_aer.primitives import Estimator as AerEstimator

        ansatz = TwoLocal(2, 'ry', 'cz')
        opt = SPSA(maxiter=50)

        # shot-based simulation
        estimator = Estimator(options={"shots": 2048})
        vqe = VQE(estimator, ansatz, opt)

        # another option
        aer_estimator = AerEstimator(run_options={"shots": 2048, "seed": 42})
        vqe = VQE(aer_estimator, ansatz, opt)

        hamiltonian = SparsePauliOp.from_list([("XX", 1), ("XY", 1)])
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        print(result.eigenvalue)

    .. testoutput::

        -0.986328125

.. dropdown:: VQE applying CVaR (SamplingVQE) Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms import VQE
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.circuit.library import TwoLocal
        from qiskit.opflow import PauliSumOp, CVaRExpectation
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator

        ansatz = TwoLocal(2, 'ry', 'cz')
        opt = SLSQP(maxiter=50)

        # shot-based simulation
        backend = AerSimulator()
        qi = QuantumInstance(backend=backend, shots=2048)
        expectation = CVaRExpectation(alpha=0.2)
        vqe = VQE(ansatz, optimizer=opt, expectation=expectation, quantum_instance=qi)

        # diagonal Hamiltonian
        hamiltonian = PauliSumOp.from_list([("ZZ",1), ("IZ", -0.5), ("II", 0.12)])
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        print(result.eigenvalue.real)

    .. testoutput::

        -1.38

    **[Updated] Using Primitives:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms.minimum_eigensolvers import SamplingVQE # new import!!!
        from qiskit.algorithms.optimizers import SPSA
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import Sampler
        from qiskit_aer.primitives import Sampler as AerSampler

        ansatz = TwoLocal(2, 'ry', 'cz')
        opt = SPSA(maxiter=50)

        # shot-based simulation
        sampler = Sampler(options={"shots": 2048})
        vqe = SamplingVQE(sampler, ansatz, opt, aggregation=0.2)

        # another option
        aer_sampler = AerSampler(run_options={"shots": 2048, "seed": 42})
        vqe = SamplingVQE(aer_sampler, ansatz, opt, aggregation=0.2)

        # diagonal Hamiltonian
        hamiltonian = SparsePauliOp.from_list([("ZZ",1), ("IZ", -0.5), ("II", 0.12)])
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        print(result.eigenvalue.real)

    .. testoutput::

        -1.38

For complete code examples, see the following updated tutorials:

- `VQE Introduction <https://qiskit.org/documentation/tutorials/algorithms/01_algorithms_introduction.html>`_
- `VQE, Callback, Gradients, Initial Point <https://qiskit.org/documentation/tutorials/algorithms/02_vqe_advanced_options.html>`_
- `VQE with Aer Primitives <https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html>`_

QAOA
----

The legacy :class:`qiskit.algorithms.minimum_eigen_solvers.QAOA` class used to extend
:class:`qiskit.algorithms.minimum_eigen_solvers.VQE`, but now, :class:`qiskit.algorithms.minimum_eigensolvers.QAOA`
extends :class:`qiskit.algorithms.minimum_eigensolvers.SamplingVQE`.
For this reason, **the new QAOA only supports diagonal operators**.

.. note::

    In addition to taking in an :mod:`~qiskit.primitives.Sampler` instance instead of a :class:`~qiskit.utils.QuantumInstance`,
    the new :class:`~qiskit.algorithms.minimum_eigensolvers.QAOA` signature has undergone the following changes:

    1. The ``expectation`` and ``include_custom`` parameters have been removed. In return, the ``aggregation``
       parameter has been added (it used to be defined through a custom ``expectation``).
    2. The ``gradient`` parameter now takes in an instance of a primitive-based gradient class from
       :mod:`qiskit.algorithms.gradients` instead of the legacy :mod:`qiskit.opflow.gradients.Gradient` class.
    3. The ``max_evals_grouped`` parameter has been removed, as it can be set directly on the optimizer class.
    4. The ``sampler`` and ``optimizer`` are the only parameters that can be defined positionally
       (and in this order), all others have become keyword-only arguments.

.. note::

    If you want to run QAOA on a non-diagonal operator, you can use the :class:`.QAOAAnsatz` with
    :class:`qiskit.algorithms.minimum_eigensolvers.VQE`, but bear in mind there will be no state result.
    If your application requires the final probability distribution, you can instantiate a ``Sampler``
    and run it with the optimal circuit after :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`.

.. dropdown:: QAOA Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms import QAOA
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.opflow import PauliSumOp
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator

        # exact statevector simulation
        backend = AerSimulator()
        qi = QuantumInstance(backend=backend, shots=None,
                seed_simulator = 42, seed_transpiler = 42,
                backend_options={"method": "statevector"})

        optimizer = COBYLA()
        qaoa = QAOA(optimizer=optimizer, reps=2, quantum_instance=qi)

        # diagonal operator
        qubit_op = PauliSumOp.from_list([("ZIII", 1),("IZII", 1), ("IIIZ", 1), ("IIZI", 1)])
        result = qaoa.compute_minimum_eigenvalue(qubit_op)

        print(result.eigenvalue.real)

    .. testoutput::

        -4.0

    **[Updated] Using Primitives:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms.minimum_eigensolvers import QAOA
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import Sampler
        from qiskit_aer.primitives import Sampler as AerSampler

        # exact statevector simulation
        sampler = Sampler()

        # another option
        sampler = AerSampler(backend_options={"method": "statevector"},
                             run_options={"shots": None, "seed": 42})

        optimizer = COBYLA()
        qaoa = QAOA(sampler, optimizer, reps=2)

        # diagonal operator
        qubit_op = SparsePauliOp.from_list([("ZIII", 1),("IZII", 1), ("IIIZ", 1), ("IIZI", 1)])
        result = qaoa.compute_minimum_eigenvalue(qubit_op)

        print(result.eigenvalue)

    .. testoutput::

        -3.999999832366272

For complete code examples, see the following updated tutorials:

- `QAOA <https://qiskit.org/documentation/tutorials/algorithms/05_qaoa.html>`_

NumPyMinimumEigensolver
-----------------------

Because this is a classical solver, the workflow has not changed between the old and new implementation.
The import has however changed from :class:`qiskit.algorithms.minimum_eigen_solvers.NumPyMinimumEigensolver`
to :class:`qiskit.algorithms.minimum_eigensolvers.NumPyMinimumEigensolver` to conform to the new interfaces
and result classes.

.. dropdown:: NumPyMinimumEigensolver Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms import NumPyMinimumEigensolver
        from qiskit.opflow import PauliSumOp

        solver = NumPyMinimumEigensolver()

        hamiltonian = PauliSumOp.from_list([("XX", 1), ("XY", 1)])
        result = solver.compute_minimum_eigenvalue(hamiltonian)

        print(result.eigenvalue)

    .. testoutput::

        -1.4142135623730958

    **[Updated] Using Primitives:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
        from qiskit.quantum_info import SparsePauliOp

        solver = NumPyMinimumEigensolver()

        hamiltonian = SparsePauliOp.from_list([("XX", 1), ("XY", 1)])
        result = solver.compute_minimum_eigenvalue(hamiltonian)

        print(result.eigenvalue)

    .. testoutput::

        -1.414213562373095

For complete code examples, see the following updated tutorials:

- `VQE, Callback, Gradients, Initial Point <https://qiskit.org/documentation/tutorials/algorithms/02_vqe_advanced_options.html>`_

Eigensolvers
============
*Back to* `TL;DR`_

The eigensolver algorithms also belong to the first type of refactoring
(Algorithms refactored in a new location to support :mod:`~qiskit.primitives`). Instead of a
:class:`~qiskit.utils.QuantumInstance`, :mod:`qiskit.algorithms.eigensolvers` are now initialized
using an instance of the :class:`~qiskit.primitives.Sampler` or :class:`~qiskit.primitives.Estimator` primitive, or
**a primitive-based subroutine**, depending on the algorithm. The legacy classes can still be found
in :mod:`qiskit.algorithms.eigen_solvers`.

.. attention::

    For the :mod:`qiskit.algorithms.eigensolvers` classes, depending on the import path,
    you will access either the primitive-based or the quantum-instance-based
    implementation. You have to be extra-careful, because the class name does not change.

    * Old import path (Quantum Instance): ``from qiskit.algorithms import VQD, NumPyEigensolver``
    * New import path (Primitives): ``from qiskit.algorithms.eigensolvers import VQD, NumPyEigensolver``

VQD
---

The new :class:`qiskit.algorithms.eigensolvers.VQD` class is initialized with an instance of the
:class:`~qiskit.primitives.Estimator` primitive instead of a :class:`~qiskit.utils.QuantumInstance`.
In addition to this, it takes an instance of a state fidelity class from mod:`qiskit.algorithms.state_fidelities`,
such as the :class:`~qiskit.primitives.Sampler`-based :class:`~qiskit.algorithms.state_fidelities.ComputeUncompute`.

.. note::

    In addition to taking in an :mod:`~qiskit.primitives.Estimator` instance instead of a :class:`~qiskit.utils.QuantumInstance`,
    the new :class:`~qiskit.algorithms.eigensolvers.VQD` signature has undergone the following changes:

    1. The ``expectation`` and ``include_custom`` parameters have been removed, as this functionality is now
       defined at the ``Estimator`` level.
    2. The custom ``fidelity`` parameter has been added, and the custom ``gradient`` parameter has
       been removed, as current classes in :mod:`qiskit.algorithms.gradients` cannot deal with state fidelity
       gradients.
    3. The ``max_evals_grouped`` parameter has been removed, as it can be set directly on the optimizer class.
    4. The ``estimator``, ``fidelity``, ``ansatz`` and ``optimizer`` are the only parameters that can be defined positionally
       (and in this order), all others have become keyword-only arguments.

.. note::

    Similarly to VQE, the new :class:`~qiskit.algorithms.eigensolvers.VQDResult` class does not include
    the state anymore. If your application requires the final probability distribution, you can instantiate
    a ``Sampler`` and run it with the optimal circuit for the desired excited state
    after running :class:`~qiskit.algorithms.eigensolvers.VQD`.


.. dropdown:: VQD Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit import IBMQ
        from qiskit.algorithms import VQD
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.circuit.library import TwoLocal
        from qiskit.opflow import PauliSumOp
        from qiskit.utils import QuantumInstance

        ansatz = TwoLocal(3, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)
        optimizer = SLSQP(maxiter=10)
        hamiltonian = PauliSumOp.from_list([("XXZ", 1), ("XYI", 1)])

        # example executing in cloud simulator
        provider = IBMQ.load_account()
        backend = provider.get_backend("ibmq_qasm_simulator")
        qi = QuantumInstance(backend=backend)

        vqd = VQD(ansatz, k=3, optimizer=optimizer, quantum_instance=qi)
        result = vqd.compute_eigenvalues(operator=hamiltonian)

        print(result.eigenvalues)

    .. testoutput::
        :options: +SKIP

        [ 0.01765114+0.0e+00j -0.58507654+0.0e+00j -0.15003642-2.8e-17j]

    **[Updated] Using Primitives:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit_ibm_runtime import Sampler, Estimator, QiskitRuntimeService, Session
        from qiskit.algorithms.eigensolvers import VQD
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.algorithms.state_fidelities import ComputeUncompute
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp

        ansatz = TwoLocal(3, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)
        optimizer = SLSQP(maxiter=10)
        hamiltonian = SparsePauliOp.from_list([("XXZ", 1), ("XYI", 1)])

        # example executing in cloud simulator
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend("ibmq_qasm_simulator")

        with Session(service=service, backend=backend) as session:
            estimator = Estimator()
            sampler = Sampler()
            fidelity = ComputeUncompute(sampler)
            vqd = VQD(estimator, fidelity, ansatz, optimizer, k=3)
            result = vqd.compute_eigenvalues(operator=hamiltonian)

        print(result.eigenvalues)

    .. testoutput::
        :options: +SKIP

        [ 0.01765114+0.0e+00j -0.58507654+0.0e+00j -0.15003642-2.8e-17j]

.. raw:: html

    <br>

For complete code examples, see the following updated tutorial:

- `VQD <https://qiskit.org/documentation/tutorials/algorithms/04_vqd.html>`_

NumPyEigensolver
----------------
Similarly to its minimum eigensolver counterpart, because this is a classical solver, the workflow has not changed
between the old and new implementation.
The import has however changed from :class:`qiskit.algorithms.eigen_solvers.NumPyEigensolver`
to :class:`qiskit.algorithms.eigensolvers.MinimumEigensolver` to conform to the new interfaces and result classes.

.. dropdown:: NumPyEigensolver Example
    :animate: fade-in-slide-down

    **[Legacy]:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms import NumPyEigensolver
        from qiskit.opflow import PauliSumOp

        solver = NumPyEigensolver(k=2)

        hamiltonian = PauliSumOp.from_list([("XX", 1), ("XY", 1)])
        result = solver.compute_eigenvalues(hamiltonian)

        print(result.eigenvalues)

    .. testoutput::

        [-1.41421356 -1.41421356]

    **[Updated]:**

    .. testsetup::

        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = 42

    .. testcode::

        from qiskit.algorithms.eigensolvers import NumPyEigensolver
        from qiskit.quantum_info import SparsePauliOp

        solver = NumPyEigensolver(k=2)

        hamiltonian = SparsePauliOp.from_list([("XX", 1), ("XY", 1)])
        result = solver.compute_eigenvalues(hamiltonian)

        print(result.eigenvalues)

    .. testoutput::

        [-1.41421356 -1.41421356]

Time Evolvers
=============
*Back to* `TL;DR`_

The time evolvers are the last group of algorithms to undergo the first type of refactoring
(Algorithms refactored in a new location to support :mod:`~qiskit.primitives`).
Instead of a :class:`~qiskit.utils.QuantumInstance`, :mod:`qiskit.algorithms.time_evolvers` are now initialized
using an instance of the :class:`~qiskit.primitives.Estimator` primitive. The legacy classes can still be found
in :mod:`qiskit.algorithms.evolvers`.

On top of the migration, the module has been substantially expanded to include **Variational Quantum Time Evolution**
(:class:`~qiskit.algorithms.time_evolvers.VarQTE`\) solvers.

TrotterQRTE
-----------
.. attention::

    For the :class:`qiskit.algorithms.time_evolvers.TrotterQRTE` class, depending on the import path,
    you will access either the primitive-based or the quantum-instance-based
    implementation. You have to be extra-careful, because the class name does not change.

    * Old import path (Quantum Instance): ``from qiskit.algorithms import TrotterQRTE``
    * New import path (Primitives): ``from qiskit.algorithms.time_evolvers import TrotterQRTE``

.. note::

    In addition to taking in an :mod:`~qiskit.primitives.Estimator` instance instead of a :class:`~qiskit.utils.QuantumInstance`,
    the new :class:`~qiskit.algorithms.eigensolvers.VQD` signature has undergone the following changes:

    1. The ``expectation`` parameter has been removed, as this functionality is now
       defined at the ``Estimator`` level.
    2. The ``num_timesteps`` parameters has been added, to allow to define the number of steps the full evolution
       time is divided into.

.. dropdown:: TrotterQRTE Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. testcode::

        from qiskit.algorithms import EvolutionProblem, TrotterQRTE
        from qiskit.circuit import QuantumCircuit
        from qiskit.opflow import PauliSumOp, AerPauliExpectation
        from qiskit.utils import QuantumInstance
        from qiskit_aer import AerSimulator

        operator = PauliSumOp.from_list([("X", 1),("Z", 1)])
        initial_state = QuantumCircuit(1) # zero
        time = 1
        evolution_problem = EvolutionProblem(operator, 1, initial_state)

        # Aer simulator using custom instruction
        backend = AerSimulator()
        quantum_instance = QuantumInstance(backend=backend)
        expectation = AerPauliExpectation()

        # LieTrotter with 1 rep
        trotter_qrte = TrotterQRTE(expectation=expectation, quantum_instance=quantum_instance)
        evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state

        print(evolved_state)

    .. testoutput::

        CircuitStateFn(
           ┌─────────────────────┐
        q: ┤ exp(-it (X + Z))(1) ├
           └─────────────────────┘
        )

    **[Updated] Using Primitives:**

    .. testcode::

        from qiskit.algorithms.time_evolvers import TimeEvolutionProblem, TrotterQRTE  # note new import!!!
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer.primitives import Estimator as AerEstimator

        operator = SparsePauliOp.from_list([("X", 1),("Z", 1)])
        initial_state = QuantumCircuit(1) # zero
        time = 1
        evolution_problem = TimeEvolutionProblem(operator, 1, initial_state)

        # Aer simulator using custom instruction
        estimator = AerEstimator(run_options={"approximation": True, "shots": None})

        # LieTrotter with 1 rep
        trotter_qrte = TrotterQRTE(estimator=estimator)
        evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state

        print(evolved_state.decompose())

    .. testoutput::

           ┌───────────┐┌───────────┐
        q: ┤ exp(it X) ├┤ exp(it Z) ├
           └───────────┘└───────────┘

Amplitude Amplifiers
====================
*Back to* `TL;DR`_

The amplitude amplifier algorithms belong to the second type of refactoring (Algorithms refactored in-place).
Instead of a :class:`~qiskit.utils.QuantumInstance`, :mod:`qiskit.algorithms.amplitude_amplifiers` are now initialized
using an instance of any "Sampler" primitive e.g. :mod:`~qiskit.primitives.Sampler`.

.. note::
   The full :mod:`qiskit.algorithms.amplitude_amplifiers` module has been refactored in place. No need to
   change import paths.

.. dropdown:: Grover Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. code-block:: python

        from qiskit.algorithms import Grover
        from qiskit.utils import QuantumInstance

        qi = QuantumInstance(backend=backend)
        grover = Grover(quantum_instance=qi)


    **[Updated] Using Primitives:**

    .. code-block:: python

        from qiskit.algorithms import Grover
        from qiskit.primitives import Sampler

        grover = Grover(sampler=Sampler())

For complete code examples, see the following updated tutorials:

- `Amplitude Amplification and Grover <https://qiskit.org/documentation/tutorials/algorithms/06_grover.html>`_
- `Grover Examples <https://qiskit.org/documentation/tutorials/algorithms/07_grover_examples.html>`_

Amplitude Estimators
====================
*Back to* `TL;DR`_

Similarly to the amplitude amplifiers, the amplitude estimators also belong to the second type of refactoring
(Algorithms refactored in-place).
Instead of a :class:`~qiskit.utils.QuantumInstance`, :mod:`qiskit.algorithms.amplitude_estimators` are now initialized
using an instance of any "Sampler" primitive e.g. :mod:`~qiskit.primitives.Sampler`.

.. note::
   The full :mod:`qiskit.algorithms.amplitude_estimators` module has been refactored in place. No need to
   change import paths.

.. dropdown:: IAE Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. code-block:: python

        from qiskit.algorithms import IterativeAmplitudeEstimation
        from qiskit.utils import QuantumInstance

        qi = QuantumInstance(backend=backend)
        iae = IterativeAmplitudeEstimation(
            epsilon_target=0.01,  # target accuracy
            alpha=0.05,  # width of the confidence interval
            quantum_instance=qi
        )

    **[Updated] Using Primitives:**

    .. code-block:: python

        from qiskit.algorithms import IterativeAmplitudeEstimation
        from qiskit.primitives import Sampler

        iae = IterativeAmplitudeEstimation(
            epsilon_target=0.01,  # target accuracy
            alpha=0.05,  # width of the confidence interval
            sampler=Sampler()
        )

For complete code examples, see the following updated tutorials:

- `Amplitude Estimation <https://qiskit.org/documentation/finance/tutorials/00_amplitude_estimation.html>`_

Phase Estimators
================
*Back to* `TL;DR`_

Finally, the phase estimators are the last group of algorithms to undergo the first type of refactoring
(Algorithms refactored in-place).
Instead of a :class:`~qiskit.utils.QuantumInstance`, :mod:`qiskit.algorithms.phase_estimators` are now initialized
using an instance of any "Sampler" primitive e.g. :mod:`~qiskit.primitives.Sampler`.

.. note::
   The full :mod:`qiskit.algorithms.phase_estimators` module has been refactored in place. No need to
   change import paths.

.. dropdown:: IPE Example
    :animate: fade-in-slide-down

    **[Legacy] Using Quantum Instance:**

    .. code-block:: python

        from qiskit.algorithms import IterativePhaseEstimation
        from qiskit.utils import QuantumInstance

        qi = QuantumInstance(backend=backend)
        ipe = IterativePhaseEstimation(
            num_iterations=num_iter,
            quantum_instance=qi
        )

    **[Updated] Using Primitives:**

    .. code-block:: python

        from qiskit.algorithms import IterativePhaseEstimation
        from qiskit.primitives import Sampler

        ipe = IterativePhaseEstimation(
            num_iterations=num_iter,
            sampler=Sampler()
        )

For complete code examples, see the following updated tutorials:

- `Iterative Phase Estimation <https://qiskit.org/documentation/tutorials/algorithms/09_IQPE.html>`_

