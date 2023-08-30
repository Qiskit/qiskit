# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Multi controlled single-qubit unitary up to diagonal.
"""

# pylint: disable=unused-import
from qiskit.circuit.library.generalized_gates.mcg_up_to_diagonal import MCGupDiag
from qiskit.utils.deprecation import _deprecate_extension

_deprecate_extension("MCGupDiag")
