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
r"""
=================================================
Pulse Transforms (:mod:`qiskit.pulse.transforms`)
=================================================

The pulse transforms provide transformation routines to reallocate and optimize
pulse programs for backends.

.. _pulse_alignments:

Alignments
==========

The alignment transforms define alignment policies of instructions in :obj:`.ScheduleBlock`.
These transformations are called to create :obj:`.Schedule`\ s from :obj:`.ScheduleBlock`\ s.

.. autosummary::
   :toctree: ../stubs/

   AlignEquispaced
   AlignFunc
   AlignLeft
   AlignRight
   AlignSequential

These are all subtypes of the abstract base class :class:`AlignmentKind`.

.. autoclass:: AlignmentKind


.. _pulse_canonical_transform:

Canonicalization
================

The canonicalization transforms convert schedules to a form amenable for execution on
OpenPulse backends.

.. autofunction:: add_implicit_acquires
.. autofunction:: align_measures
.. autofunction:: block_to_schedule
.. autofunction:: compress_pulses
.. autofunction:: flatten
.. autofunction:: inline_subroutines
.. autofunction:: pad
.. autofunction:: remove_directives
.. autofunction:: remove_trivial_barriers


.. _pulse_dag:

DAG
===

The DAG transforms create DAG representation of input program. This can be used for
optimization of instructions and equality checks.

.. autofunction:: block_to_dag


.. _pulse_transform_chain:

Composite transform
===================

A sequence of transformations to generate a target code.

.. autofunction:: target_qobj_transform

"""

from .alignments import (
    AlignEquispaced,
    AlignFunc,
    AlignLeft,
    AlignRight,
    AlignSequential,
    AlignmentKind,
)

from .base_transforms import target_qobj_transform

from .canonicalization import (
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

from .dag import block_to_dag
