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

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
Decomposes a diagonal matrix into elementary gates using the method described in Theorem 7 in
"Synthesis of Quantum Logic Circuits" by Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf).
"""

# pylint: disable=unused-import
from qiskit.circuit.library.generalized_gates.diagonal import DiagonalGate
from qiskit.utils.deprecation import _deprecate_extension

_deprecate_extension("DiagonalGate")
