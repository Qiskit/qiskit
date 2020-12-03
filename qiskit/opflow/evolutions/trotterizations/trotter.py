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

""" Trotter Class """

from .suzuki import Suzuki


class Trotter(Suzuki):
    r"""
    Simple Trotter expansion, composing the evolution circuits of each Operator in the sum
    together ``reps`` times and dividing the evolution time of each by ``reps``.
    """
    def __init__(self,
                 reps: int = 1) -> None:
        r"""
        Args:
            reps: The number of times to repeat the Trotterization circuit.
        """
        super().__init__(order=1, reps=reps)
