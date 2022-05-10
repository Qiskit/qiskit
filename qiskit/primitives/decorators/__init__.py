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
Decorators for primitives
=========================

Allow Decorators
================

.. currentmodule:: qiskit.primitives.decorators.allow

Allow decorators enhance Primitives (Sampler and Estimator) and provide a more flexible interface.
For example, the default Primitives requires that ``circuit_indices`` cannot be optional or must
be given a list of integers. Otherwise, it returns a type error as follows:

.. jupyter-execute::
   :hide-code:

    from qiskit.primitives import (
        Estimator,
        Sampler,
        allow_broadcasting,
        allow_objects,
        allow_optional,
    )
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import RealAmplitudes

    ansatz = RealAmplitudes(num_qubits=2, reps=2).decompose()
    parameter = list(range(6))
    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.measure_all()
    observable = 0.1 * SparsePauliOp("XX") + 0.4 * SparsePauliOp("ZZ")


When an object is passed instead of an integer,

.. jupyter-execute::
   :raises:

    with Estimator([ansatz], [observable]) as est:
        result = est([ansatz], [observable], [parameter])
    print(result)


When nothing is passed,

.. jupyter-execute::
   :raises:

    with Sampler([circuit]) as sampler:
       result = sampler()
    print(result)

Allow Object
------------

:func:`~allow_objects` allows objects as inputs of indices instead of integer.
For example, the following code works:

.. jupyter-execute::

    @allow_objects
    class CustomEstimator(Estimator):
        ...

    with CustomEstimator([ansatz], [observable]) as e:
        result = e(
            circuit_indices=[ansatz],
            observable_indices=[observable],
            parameter_values=[parameter],
        )
    print(result)

Note that since qiskit's circuit objects and observable objects are not immutable,
if the object is changed after the construction of the primitives, it may behave unexpectedly.


Allow Optional
---------------

:func:`~allow_optional` allows None as inputs for circuit (and observable) indices
and prameter values.  If indices are not given, indices are the list of all circuits (and
observables) i.e. ``[0, 1, ..., len(circuits) - 1]`` (and ``[0, 1, ..., len(observables) - 1]``).
If parameter values are not given, the parameter values are circuit indices length empty lists.

.. jupyter-execute::

    @allow_optional
    class CustomEstimator(Estimator):
      ...

    with CustomEstimator([ansatz], [observable]) as e:
        result = e(parameter_values=[parameter])
    print(result)

Allow Broadcasting
------------------

:func:`~allow_broadcasting` allows broadcasting for the inputs of call in primitives.
Broadcasting means that if the number of circuits (and observables) is one
and indices are not given, it generates as many indices as the number of parameters, i.e.
``[0] * len(parameter_values)``.

.. jupyter-execute::

    @allow_broadcasting()
    class CustomSampler(Sampler):
        ...

    with CustomSampler([circuit]) as e:
        result = e(parameter_values=[[]] * 3)


Allow Decorators
=================

.. autosummary::
   :toctree: ../stubs/

   allow_optional
   allow_broadcasting
   allow_objects

Callback protocol for type hint
=================================

.. autosummary::
   :toctree: ../stubs/

   PrimitiveDecorator
"""

from .allow import allow_broadcasting, allow_objects, allow_optional
