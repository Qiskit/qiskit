==========================
Algorithms Migration Guide
==========================

*Jump to* `TL;DR`_.

Background
----------
Before: algorithms based on QuantumInstance
Now: algorithms based on Primitives

How to choose backends now:

1. Select algorithmic abstraction: sampler/estimator
2. Select backend --> where to import sampler/estimator from. How to define settings for each primitive.

Link to primitives tutorial for further info.

TL;DR
-----

Algorithms have been refactored to use the primitives instead of QI.

3 types of refactoring

1. New algos in new place (old algos in old place will be deprecated). Careful with import paths!!
   These must imported directly from the new folders, since names conflict with existing ones that
   are still importable from qiskit.algorithms

    - `Minimum Eigensolvers`_
    - `Eigensolvers`_
    - `Time Evolvers`_

2. Algos refactored in-place to support both QI and primitives (use of QI will be deprecated).

    - `Amplitude Amplifiers`_
    - `Amplitude Estimators`_
    - `Phase Estimators`_

3. Algos deprecated entirely in this repo --> Having an encapsulated algorithm class is not very useful.
 These are kept as educational material for the textbook. No code examples in the guide.

    - `Linear Solvers`_ (HHL) --> Add Link
    - `Factorizers`_ (Shor) --> Add Link


Minimum Eigensolvers
--------------------

``VQE`` (:class:`~qiskit.algorithms.minimum_eigensolvers.VQE`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(!) IMPORT CHANGE: ``from qiskit.algorithms.minimum_eigensolvers import VQE`` (new) instead of ``from qiskit.algorithms import VQE`` (old)

- 3 different common configs/uses (which apply to others like VQD)

1. StateVector backend (ie. using opflow Matrix Expectation) -> Terra Estimator

2. Qasm simulator/device (i.e. using PauliExpectation) -> AerEstimator, Run Estimator, or Terra Estimator with shots

3. Aer simulator using custom instruction (ie. using AerPauliExpectation - include_custom for VQE using expectation factory) -> AerEstimator shots=None, approximation=True

* Note: New result does not have state in it any more


``VQE`` + ``CVARExpectation`` -> (:class:`~qiskit.algorithms.minimum_eigensolvers.SamplingVQE`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(!) NEW IMPORT: ``from qiskit.algorithms.minimum_eigensolvers import SamplingVQE`` (new) instead of ``from qiskit.algorithms import VQE`` (old)

``QAOA`` -> (:class:`~qiskit.algorithms.minimum_eigensolvers.QAOA`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(!) NEW IMPORT: ``from qiskit.algorithms.minimum_eigensolvers import QAOA`` (new) instead of ``from qiskit.algorithms import QAOA`` (old)

This used to be based off VQE but new is based off SamplingVQE.
As such the Sampler selection now determines exact behavior or not.
The new one only supports diagonal operator.
If you want the old QAOA you can use QAOAAnsatz with VQE but bear in mind there is no state result -
if one needed counts one could use the optimal cct with a Sampler.

``NumPyMinimumEigensolver`` -> (:class:`~qiskit.algorithms.minimum_eigensolvers.NumPyMinimumEigensolver`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(!) NEW IMPORT

Eigensolvers
------------

``VQD``-> (:class:`~qiskit.algorithms.eigensolvers.VQD`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(!) NEW IMPORT


``NumPyEigensolver`` -> (:class:`~qiskit.algorithms.eigensolvers.NumPyEigensolver`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(!) NEW IMPORT

Time Evolvers
-------------

``TrotterQRTE``-> (:class:`~qiskit.algorithms.time_evolvers.TrotterQRTE`\)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(!) NEW IMPORT

(this is the only evolvers algo that was shipped before the primitives change)
See the rest of the time evolvers here.

Amplitude Amplifiers
---------------------
Inplace Algos: The QI or Sampler is a common theme in these. But maybe we show at least one algo in each category

Grover
~~~~~~
Grover + QI -> Grover + Sampler

Amplitude Estimators
--------------------
Inplace Algos: The QI or Sampler is a common theme in these. But maybe we show at least one algo in each category

(x)AE
~~~~~
(x)AE + QI -> (x)AE + Sampler

Since AE variants are similar in the old to new way maybe we only need to show one exmaple and state this fact

Phase Estimators
----------------
Inplace Algos: The QI or Sampler is a common theme in these. But maybe we show at least one algo in each category

PhaseEstimation
~~~~~~~~~~~~~~~~
PhaseEstimation + QI -> PhaseEstimation + Sampler

Similar to AE


