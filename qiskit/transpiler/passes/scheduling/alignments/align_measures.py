# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Deprecated. Measurement alignment."""

import warnings

from qiskit.transpiler.passes.scheduling.alignments.reschedule import ConstrainedReschedule


class AlignMeasures:
    """Deprecated. Measurement alignment."""

    def __new__(cls, alignment=1) -> ConstrainedReschedule:
        """Create new pass.

        Args:
            alignment: Integer number representing the minimum time resolution to
                trigger measure instruction in units of ``dt``. This value depends on
                the control electronics of your quantum processor.

        Returns:
            ConstrainedReschedule instance that is a drop-in-replacement of this class.
        """
        warnings.warn(
            f"{cls.__name__} has been deprecated as of Qiskit 20.0. "
            f"Use ConstrainedReschedule pass instead.",
            FutureWarning,
        )
        return ConstrainedReschedule(acquire_alignment=alignment)
