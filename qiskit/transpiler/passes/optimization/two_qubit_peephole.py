# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

from __future__ import annotations

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit._accelerate.two_qubit_peephole import two_qubit_unitary_peephole_optimize


class TwoQubitPeepholeOptimization(TransformationPass):
    """Unified two qubit unitary peephole optimization

    This transpiler pass is designed to perform two qubit unitary peephole
    optimization. This pass finds all the 2 qubit blocks in the circuit,
    computes the unitary of that block, and then synthesizes that unitary.
    If the synthesized two qubit unitary is "better" than the original
    subcircuit that subcircuit is used to replace the original. The heuristic
    used to determine if it's better first looks at the two qubit gate count
    in the circuit, and prefers the synthesis with fewer two qubit gates.

    In case the target is overcomplete the pass will try all the
    decomposers supported for all the gates supported on a given qubit.
    The decomposition that has the best expected performance will be selected
    and used to replace the block.

    This pass is multithreaded, and will perform the analysis in parallel
    and use all the cores available on your local system. You can refer to
    the `configuration guide <https://docs.quantum.ibm.com/guides/configure-qiskit-local>`__
    for details on how to control the threading behavior for Qiskit more broadly
    which will also control this pass.

    Unlike :class:`.UnitarySynthesis` pass this does not use the :ref`unitary-synth-plugin`.
    This is a tradeoff for performance and it forgoes the pluggability exposed
    via that interface. Internally it currently only uses the :class:`.TwoQubitBasisDecomposer`
    and :class:`.TwoQubitControlledUDecomposer` for synthesizing the two qubit unitaries.
    You should not use this pass if you need to use the pluggable interface and the ability
    to use different synthesis algorithms, instead you should use a combination of
    :class:`.ConsolidateBlocks` and :class:`.UnitarySynthesis` to use the plugin mechanism
    """

    def __init__(
        self,
        target: Target,
        approximation_degree: float | None = 1.0,
    ):
        """Initialize the pass

        Args:
            target: The target to run the pass for
                approximation_degree: heuristic dial used for circuit approximation (1.0=no
                approximation, 0.0=maximal approximation). Approximation can decrease the number
                of gates used in the synthesized unitaries smaller at the cost of straying from the
                original unitary. If ``None``, approximation is done based on gate fidelities
                specified in the ``target``.
        """

        super().__init__()
        self._target = target
        self._approximation_degree = approximation_degree

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        return two_qubit_unitary_peephole_optimize(dag, self._target, self._approximation_degree)
