# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================
Primitives (:mod:`qiskit.primitives`)
=====================================

.. currentmodule:: qiskit.primitives

The primitives are computational building blocks to be used in larger applications whose input 
units, called primitive unified blocs (PUBs), require quantum resources to efficiently produce
outputs for.

Currently there are two types of primitives whose abstractions, in their latest versions, are
defined by :class:`~.BaseSamplerV2` and :class:`~.BaseEstimatorV2`. Samplers are responsible for
accepting quantum circuits (or sweeps of values over parameterized circuits) and sampling from their
classical output registers. Estimators accept combinations of circuits and observables (or sweeps
thereof) to estimate expectation values of the observables.

Qiskit implements a reference implementation for each of these abstractions, 
:class:`~.StatevectorSampler` and :class:`~.StatevectorEstimator`.

.. automodule:: qiskit.primitives.base.base_estimator
.. automodule:: qiskit.primitives.base.base_sampler

Migration from Primitives V1 to V2
==================================

The formal distinction between the Primitives V1 and V2 APIs are the base classes from which 
primitives implementations inherit, which are all listed at the bottom of the page. At a conceptual 
level, however, here are some notable differences keep in mind when migrating from V1 to V2:

1. The V2 primitives favour vectorized inputs, where single circuits can be grouped with 
   vector-valued (or more generally, array-valued) specifications. Each group is called a 
   primitive unified bloc (pub), and each pub gets its own result. For example, in the estimator, 
   you can compare the following differences:

   .. code-block:: python

      # Favoured V2 pattern. There is only one pub here, but there could be more.
      job = estimator_v2.run([(circuit, [obs1, obs2, obs3, obs4])])
      evs = job.result()[0].data.evs

      # V1 equivalent, where the same circuit must be provided four times.
      job = estimator_v1.run([circuit] * 4, [obs1, obs2, obs3, obs4])
      evs = job.result().values

   Not shown in the above example, for brevity, is that the circuit can be parametric, with arrays
   of parameter value sets broadcasted against the array of observables. The sampler is similar,
   but with no observables:

   .. code-block:: python

      # Favoured V2 pattern. There is only one pub here, but there could be more.
      job = sampler_v2.run([(circuit, [vals1, vals2, vals3])])
      samples = job.result()[0].data

      # V1 equivalent, where the same circuit must be provided three times.
      sampler_v1.run([circuit] * 3, [vals1, vals2, vals3])
      quasi_dists = job.result().quasi_dists

2. The V2 sampler returns samples of classical outcomes, preserving the shot order in which they 
   were measured. This is in contrast to the V1 sampler that outputs quasi-probability distributions
   which are instead an *estimate of the distribution* over classical outcomes. Moreover, the V2
   sampler result objects organize data in terms of their input circuits' classical register 
   names, which provides natural compatibility with dynamic circuits.

   .. code-block:: python

      circuit = QuantumCircuit(QuantumRegister(2, "qreg"), ClassicalRegister(2, "alpha"))
      circuit.h(0)
      circuit.cx(0, 1)
      circuit.measure([0, 1], [0, 1])

      # V1 sampler usage
      result = sampler_v1.run([circuit]).result()
      quasi_dists = bitstrings.get_counts()

      # V2 sampler usage
      result = sampler_v2.run([circuit]).result()
      # these are all the bitstrings from the alpha register
      bitstrings = result[0].data.alpha
      # we can use it to generate a Counts mapping, which is the most similar thing to a quasi dist
      counts = bitstrings.get_counts()           

3. The V2 primitives have brought the concept of sampling overhead, inherent to all quantum systems 
   via their inherent probabilistic nature, out of the options and into the API itself. For the
   sampler, this means that the ``shots`` argument is now part of the :meth:`~.BaseSamplerV2.run`
   signature, and moreover that each pub is able to specify its own value for ``shots``, which takes
   precedence over any value given to the method. The sampler has an analogous ``precision``
   argument that specifies the error bars that the primitive implementation should target for
   expectation values estimates.

   This concept is not present in the API of the V1 primitives, though all implementations of the 
   V1 primitives have related settings somewhere in their options.

   .. code-block:: python

      # Sample two circuits at 128 shots each.
      sampler_v2.run([circuit1, circuit2], shots=128)

      # Sample two circuits at different amounts of shots. The "None"s are necessary as placeholders
      # for the lack of parameter values in this example.
      sampler_v2.run([(circuit1, None, 123), (circuit2, None, 456)])

      # Estimate expectation values for two pubs, both with 0.05 precision.
      estimator_v2.run([(circuit1, obs_array1), (circuit2, obs_array_2)], precision=0.05)



Primitives API
==============

Primitives V2
-------------

.. autosummary::
   :toctree: ../stubs/

   BaseEstimatorV2
   StatevectorEstimator

Sampler V2
----------

.. autosummary::
   :toctree: ../stubs/

   BaseSamplerV2
   StatevectorSampler

Containers V2
-------------

.. autosummary::
   :toctree: ../stubs/

   BitArray
   DataBin
   PrimitiveResult
   PubResult

Estimator V1
------------

.. autosummary::
   :toctree: ../stubs/

   BaseEstimator
   BaseEstimatorV1
   Estimator
   BackendEstimator
   EstimatorResult


Sampler V1
----------

.. autosummary::
   :toctree: ../stubs/

   BaseSampler
   BaseSamplerV1
   Sampler
   BackendSampler
   SamplerResult

"""

from .backend_estimator import BackendEstimator
from .backend_sampler import BackendSampler
from .base import (
    BaseEstimator,
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSampler,
    BaseSamplerV1,
    BaseSamplerV2,
)
from .base.estimator_result import EstimatorResult
from .base.sampler_result import SamplerResult
from .containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    PubResult,
    EstimatorPubLike,
    SamplerPubLike,
    BindingsArrayLike,
    ObservableLike,
    ObservablesArrayLike,
)
from .estimator import Estimator
from .sampler import Sampler
from .statevector_estimator import StatevectorEstimator
from .statevector_sampler import StatevectorSampler
