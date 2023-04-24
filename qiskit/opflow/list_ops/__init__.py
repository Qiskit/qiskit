# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
List Operators (:mod:`qiskit.opflow.list_ops`)
==============================================

.. currentmodule:: qiskit.opflow.list_ops

.. deprecated:: 0.24.0

    The :mod:`qiskit.opflow` module is deprecated and will be removed no earlier
    than 3 months after the release date. For code migration guidelines,
    visit https://qisk.it/opflow_migration.

List Operators are classes for storing and manipulating lists of Operators, State functions,
or Measurements, and include some rule or ``combo_fn`` defining how the Operator functions of the
list constituents should be combined to form to cumulative Operator function of the
:class:`ListOp`. For example, a :class:`SummedOp` has an addition-based ``combo_fn``, so once
the Operators in its list are evaluated against some bitstring to produce a list of results,
we know to add up those results to produce the final result of the :class:`SummedOp`'s evaluation.
In theory, this ``combo_fn`` can be any function over classical complex values, but for convenience
we've chosen for them to be defined over NumPy arrays and values. This way, large numbers of
evaluations, such as after calling :meth:`~ListOp.to_matrix` on the list constituents,
can be efficiently combined. While the combination function is defined over classical values,
it should be understood as the operation by which each Operators' underlying function is
combined to form the underlying Operator function of the :class:`ListOp`. In this way, the
:mod:`.list_ops` are the basis for constructing large and sophisticated Operators,
State Functions, and Measurements.


The base :class:`ListOp` class is particularly interesting, as its ``combo_fn`` is "the identity
list Operation". Meaning, if we understand the ``combo_fn`` as a function from a list of complex
values to some output, one such function is returning the list as\-is. This is powerful for
constructing compact hierarchical Operators which return many measurements in multiple
dimensional lists. For example, if we want to estimate the gradient of some Observable
measurement with respect to some parameters in the State function, we can construct separate
evaluation Operators for each parameter's gradient which we must keep track of ourselves in a
list, or we can construct a single :class:`ListOp` containing the evaluation Operators for each
parameter, so the :meth:`~ListOp.eval` function returns the full gradient vector. Another excellent
example of this power is constructing a Quantum kernel matrix:

.. code-block::

    data_sfn_list_op = ListOp(data_circuit_state_fns)
    qkernel_op_circuits = ~data_sfn_list_op @ data_sfn_list_op
    qkernel_sampled = CircuitSampler(backend).convert(qkernel_op_circuits)
    qkernel_sampled.eval()

This will return the two dimensional Quantum kernel matrix, where each element is the inner product
of some pair of the data State functions, or in other terms, a measurement of one data
:class:`~.state_fns.CircuitStateFn` by another.

You'll encounter the :class:`ListOp` subclasses (:class:`SummedOp`, :class:`ComposedOp`,
or :class:`TensoredOp`) more often as lazy results of Operator construction operations than as
something you need to explicitly construct. Any time we don't know how to efficiently add,
compose, or tensor two :mod:`.primitive_ops` or :mod:`.state_fns` together, they're returned in
a :class:`SummedOp`, :class:`ComposedOp`, or :class:`TensoredOp`, respectively, so we can still work
with their combined function and perhaps convert them into an efficiently combine-able format later.

Note:
    Combination functions do not always behave predictably, and you must understand the
    conversions you're making when you working with :mod:`.list_ops`. Most notably - sampling a sum
    of two circuits on Quantum hardware does not incorporate interference between the
    wavefunctions! In this case, we're sending our State functions through a depolarizing channel
    before adding them, rather than adding them directly before the measurement.

List Operators
--------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   ListOp
   ComposedOp
   SummedOp
   TensoredOp

"""

from .list_op import ListOp
from .summed_op import SummedOp
from .composed_op import ComposedOp
from .tensored_op import TensoredOp

__all__ = ["ListOp", "SummedOp", "TensoredOp", "ComposedOp"]
