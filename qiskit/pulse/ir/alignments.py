# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
=========
Pulse IR Alignments
=========
"""


class Alignment:
    """Base abstract class for IR alignments"""

    pass


class ParallelAlignment(Alignment):
    """Base abstract class for IR alignments which align instructions in parallel"""

    pass


class SequentialAlignment(Alignment):
    """Base abstract class for IR alignments which align instructions sequentially"""

    pass


class AlignLeft(ParallelAlignment):
    """Left Alignment"""

    pass


class AlignRight(ParallelAlignment):
    """Right Alignment"""

    pass


class AlignSequential(SequentialAlignment):
    """Sequential Alignment"""

    pass
