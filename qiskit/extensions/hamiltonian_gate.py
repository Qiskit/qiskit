# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Gate described by the time evolution of a Hermitian Hamiltonian operator."""

import warnings
from .evolution import HamiltonianGate as NewHamiltonianGate


class HamiltonianGate(NewHamiltonianGate):
    """Gate described by the time evolution of a Hermitian Hamiltonian operator."""

    def __init__(self, data, time, label=None):
        """
        Args:
            data (matrix or Operator): a hermitian operator.
            time (float): time evolution parameter.
            label (str): unitary name for backend [Default: None].

        Raises:
            ValueError: if input data is not an N-qubit unitary operator.
        """
        warnings.warn(
            "The HamiltonianGate imported from qiskit.extensions is pending deprecation and will be "
            "and will be deprecated no sooner than 3 months after the Qiskit Terra 0.24 release. "
            "Instead, import from qiskit.circuit.library.evolution as a direct replacement.",
            stacklevel=2,
            category=PendingDeprecationWarning,
        )
        super().__init__(data=data, time=time, label=label)
