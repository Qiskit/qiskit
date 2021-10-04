# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
===================================================
Pulse Transforms (:mod:`qiskit.pulse.transforms`)
===================================================

The pulse transforms provide transformation routines to reallocate and optimize
pulse programs for backends.

Alignments
==========

The alignment transforms define alignment policies of instructions in ``ScheduleBlock``.
These transformations are called to create ``Schedule``s from ``ScheduleBlock``s.

.. autosummary::
   :toctree: ../stubs/

   AlignEquispaced
   AlignFunc
   AlignLeft
   AlignRight
   AlignSequential


Canonicalization
================

The canonicalization transforms convert schedules to a form amenable for execution on
Openpulse backends.

.. autosummary::
   :toctree: ../stubs/

   add_implicit_acquires
   align_measures
   block_to_schedule
   compress_pulses
   flatten
   inline_subroutines
   pad
   remove_directives
   remove_trivial_barriers


DAG
===

The DAG transforms create DAG representation of input program. This can be used for
optimization of instructions and equality checks.

.. autosummary::
   :toctree: ../stubs/

   block_to_dag


Composite transform
===================

A sequence of transformations to generate a target code.

.. autosummary::
   :toctree: ../stubs/

   target_qobj_transform

"""

from qiskit.pulse.transforms.alignments import (
    AlignEquispaced,
    AlignFunc,
    AlignLeft,
    AlignRight,
    AlignSequential,
    align_equispaced,
    align_func,
    align_left,
    align_right,
    align_sequential,
)

from qiskit.pulse.transforms.base_transforms import target_qobj_transform

from qiskit.pulse.transforms.canonicalization import (
    add_implicit_acquires,
    align_measures,
    block_to_schedule,
    compress_pulses,
    flatten,
    inline_subroutines,
    pad,
    remove_directives,
    remove_trivial_barriers,
)

from qiskit.pulse.transforms.dag import block_to_dag
