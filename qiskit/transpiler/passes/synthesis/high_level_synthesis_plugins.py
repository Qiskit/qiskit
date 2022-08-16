# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Default plugins for synthesizing high-level-objects in Qiskit.
"""
from qiskit.quantum_info import decompose_clifford
from qiskit.transpiler.synthesis import cnot_synth


class DefaultSynthesisClifford:
    def run(self, clifford, **options):
        """Run synthesis for the given Clifford."""

        print(f"    -> Running DefaultSynthesisClifford")
        decomposition = decompose_clifford(clifford)
        return decomposition


class DefaultSynthesisLinearFunction:
    def run(self, linear_function, **options):
        """Run synthesis for the given LinearFunction."""

        print(f"    -> Running DefaultSynthesisLinearFunction")
        decomposition = cnot_synth(linear_function.linear)
        return decomposition
