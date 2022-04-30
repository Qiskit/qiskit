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
"""A collection of functions to convert ScheduleBlock to DAG representation."""

from typing import TYPE_CHECKING
import retworkx as rx

from qiskit.pulse.utils import deprecated_functionality

if TYPE_CHECKING:
    from qiskit.pulse.schedule import ScheduleBlock, BlockComponent


@deprecated_functionality
def block_to_dag(block: "ScheduleBlock") -> rx.PyDAG:
    """Convert schedule block instruction into DAG.

    ``ScheduleBlock`` can be represented as a DAG as needed.
    For example, equality of two programs are efficiently checked on DAG representation.

    .. code-block:: python

        with pulse.build() as sched1:
            with pulse.align_left():
                pulse.play(my_gaussian0, pulse.DriveChannel(0))
                pulse.shift_phase(1.57, pulse.DriveChannel(2))
                pulse.play(my_gaussian1, pulse.DriveChannel(1))

        with pulse.build() as sched2:
            with pulse.align_left():
                pulse.shift_phase(1.57, pulse.DriveChannel(2))
                pulse.play(my_gaussian1, pulse.DriveChannel(1))
                pulse.play(my_gaussian0, pulse.DriveChannel(0))

    Here the ``sched1 `` and ``sched2`` are different implementations of the same program,
    but it is difficult to confirm on the list representation.

    Another example is instruction optimization.

    .. code-block:: python

        with pulse.build() as sched:
            with pulse.align_left():
                pulse.shift_phase(1.57, pulse.DriveChannel(1))
                pulse.play(my_gaussian0, pulse.DriveChannel(0))
                pulse.shift_phase(-1.57, pulse.DriveChannel(1))

    In above program two ``shift_phase`` instructions can be cancelled out because
    they are consecutive on the same drive channel.
    This can be easily found on the DAG representation.

    Args:
        block: A schedule block to be converted.

    Returns:
        Instructions in DAG representation.
    """
    return block._blocks._dag
