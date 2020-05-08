# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Operators (:mod:`qiskit.aqua.operators`)
========================================

.. currentmodule:: qiskit.aqua.operators

Operators and State functions are the building blocks of Quantum Algorithms.

A library for Quantum Algorithms & Applications is more than a collection of
algorithms wrapped in Python functions. It needs to provide tools to make writing
algorithms simple and easy. This is the layer of modules between the circuits and algorithms,
providing the language and computational primitives for QA&A research.

In Aqua, we call this layer the Operator Flow. It works by unifying computation with theory
through the common language of functions and operators, in a way which preserves physical
intuition and programming freedom. In the Operator Flow, we construct functions over binary
variables, manipulate those functions with operators, and evaluate properties of these functions
with measurements.

The Operator Flow is meant to serve as a lingua franca between the theory and implementation
of Quantum Algorithms & Applications. Meaning, the ultimate goal is that when theorists speak
their theory in the Operator Flow, they are speaking valid implementation, and when the engineers
speak their implementation in the Operator Flow, they are speaking valid physical formalism. To
be successful, it must be fast and physically formal enough for theorists to find it easier and
more natural than hacking Matlab or NumPy, and the engineers must find it straightforward enough
that they can learn it as a typical software library, and learn the physics naturally and
effortlessly as they learn the code. There can never be a point where we say "below this level
this is all hacked out, don't come down here, stay in the interface layer above." It all must
be clear and learnable.

Before getting into the details of the code, it's important to note that three mathematical
concepts unpin the Operator Flow. We derive most of the inspiration for the code structure from
`John Watrous's formalism <https://cs.uwaterloo.ca/~watrous/TQI/>`__ (but do not follow it exactly),
so it may be worthwhile to review Chapters I and II, which are free online, if you feel the
concepts are not clicking.

1. An n-qubit State function is a complex function over n binary variables, which we will
often refer to as *n-qubit binary strings*. For example, the traditional quantum "zero state" is
a 1-qubit state function, with a definition of f(0) = 1 and f(1) = 0.

2. An n-qubit Operator is a linear function taking n-qubit state functions to n-qubit state
functions. For example, the Pauli X Operator is defined by f(Zero) = One and f(One) = Zero.
Equivalently, an Operator can be defined as a complex function over two n-qubit binary strings,
and it is sometimes convenient to picture things this way. By this definition, our Pauli X can
be defined by its typical matrix elements, f(0, 0) = 0, f(1, 0) = 1, f(0, 1) = 1,
f(1, 1) = 0.

3. An n-qubit Measurement is a functional taking n-qubit State functions to complex values.
For example, a Pauli Z Measurement can be defined by f(Zero) = 0 and f(One) = 1.

Below, you'll find a base class for all Operators, some convenience immutable global variables
which simplify Operator construction, and two groups of submodules: Operators and Converters.

Operator Base Class
===================

The OperatorBase serves as the base class for all Operators, State functions and measurements, and
enforces the presence and consistency of methods to manipulate these objects conveniently.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OperatorBase

Operator Globals
================
The :mod:`operator_globals` is a set of immutable Operator instances that are convenient building
blocks to reach for while working with the Operator flow.

One qubit Pauli operators:
   :attr:`X`, :attr:`Y`, :attr:`Z`, :attr:`I`

Clifford+T, and some other common non-parameterized gates:
   :attr:`CX`, :attr:`S`, :attr:`H`, :attr:`T`, :attr:`Swap`, :attr:`CZ`

One qubit states:
   :attr:`Zero`, :attr:`One`, :attr:`Plus`, :attr:`Minus`

Submodules
==========

Operators
+++++++++

The Operators submodules include the PrimitiveOp, ListOp, and StateFn class groups which
represent the primary Operator modules used in Aqua. The :mod:`.legacy` submodule includes older
Operator classes which are currently being migrated out of usage in Aqua, but are still used in
some places.

.. autosummary::
   :toctree:

   primitive_ops
   list_ops
   state_fns
   legacy

Converters
++++++++++

The Converter submodules include objects which manipulate Operators, usually recursing over an
Operator structure and changing certain Operators' representation. For example, the
:class:`~.expectations.PauliExpectation` traverses an Operator structure, and replaces all of the
:class:`~.state_fns.OperatorStateFn` measurements containing non-diagonal Pauli terms into
diagonalizing circuits following by :class:`~.state_fns.OperatorStateFn` measurement containing
only diagonal Paulis.

.. autosummary::
   :toctree:

   converters
   evolutions
   expectations

"""

from .legacy import (evolution_instruction,
                     suzuki_expansion_slice_pauli_list,
                     pauli_measurement,
                     measure_pauli_z, covariance, row_echelon_F2,
                     kernel_F2, commutator, check_commutativity)
from .legacy import (LegacyBaseOperator, WeightedPauliOperator, Z2Symmetries,
                     TPBGroupedWeightedPauliOperator, MatrixOperator,
                     PauliGraph, op_converter)

# New Operators
from .operator_base import OperatorBase
from .primitive_ops import PrimitiveOp, PauliOp, MatrixOp, CircuitOp
from .state_fns import (StateFn, DictStateFn, VectorStateFn,
                        CircuitStateFn, OperatorStateFn)
from .list_ops import ListOp, SummedOp, ComposedOp, TensoredOp
from .converters import (ConverterBase, CircuitSampler, PauliBasisChange,
                         DictToCircuitSum, AbelianGrouper)
from .expectations import (ExpectationBase, ExpectationFactory, PauliExpectation,
                           MatrixExpectation, AerPauliExpectation)
from .evolutions import (EvolutionBase, EvolutionFactory, EvolvedOp, PauliTrotterEvolution,
                         MatrixEvolution, TrotterizationBase, TrotterizationFactory, Trotter,
                         Suzuki, QDrift)

# Convenience immutable instances
from .operator_globals import (EVAL_SIG_DIGITS,
                               X, Y, Z, I,
                               CX, S, H, T, Swap, CZ,
                               Zero, One, Plus, Minus)

__all__ = [
    # Common
    'evolution_instruction', 'suzuki_expansion_slice_pauli_list',
    'pauli_measurement', 'measure_pauli_z',
    'covariance', 'row_echelon_F2', 'kernel_F2', 'commutator', 'check_commutativity',
    # Legacy
    'PauliGraph', 'LegacyBaseOperator', 'WeightedPauliOperator',
    'Z2Symmetries', 'TPBGroupedWeightedPauliOperator',
    'MatrixOperator',
    # Operators
    'OperatorBase',
    'PrimitiveOp', 'PauliOp', 'MatrixOp', 'CircuitOp',
    'StateFn', 'DictStateFn', 'VectorStateFn', 'CircuitStateFn', 'OperatorStateFn',
    'ListOp', 'SummedOp', 'ComposedOp', 'TensoredOp',
    # Converters
    'ConverterBase', 'CircuitSampler', 'AbelianGrouper', 'DictToCircuitSum', 'PauliBasisChange',
    'ExpectationBase', 'ExpectationFactory', 'PauliExpectation', 'MatrixExpectation',
    'AerPauliExpectation',
    'EvolutionBase', 'EvolvedOp', 'EvolutionFactory', 'PauliTrotterEvolution', 'MatrixEvolution',
    'TrotterizationBase', 'TrotterizationFactory', 'Trotter', 'Suzuki', 'QDrift',
    # Convenience immutable instances
    'X', 'Y', 'Z', 'I', 'CX', 'S', 'H', 'T', 'Swap', 'CZ', 'Zero', 'One', 'Plus', 'Minus'
]
