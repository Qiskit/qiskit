# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simplify multi-controlled gates by Boolean algebraic reduction of control patterns."""

from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils.control_flow import trivial_recurse
from qiskit._accelerate.control_pattern_simplification import (
    run_control_pattern_simplification,
)

# The largest number of control states of a group.
_MAX_PATTERNS = 1024


class ControlPatternSimplification(TransformationPass):
    r"""Simplify groups of controlled gates that share the same base gate and target
    by applying Boolean algebra to their control-state patterns.

    The pass collects gates that apply the same standard gate :math:`U` to the same target
    qubits under different conditions on the control qubits, and replaces them by a
    cheaper sequence of gates. Such groups arise, for example, in circuits for
    continuous-time quantum walks, where each edge of the graph becomes a multi-controlled
    rotation [1, 2].

    * The group is a multiplexor whose blocks are :math:`U` or the identity [3], and control
      states that only differ in some control qubits do not need these controls. For
      example, the control states ``11`` and ``01`` reduce to a single control qubit, as
      when merging edges of a quantum walk [1]. The control states are covered by disjoint
      sets of this kind, an exact cover [4].

    * If most control states are present, :math:`U` is applied without controls, and
      :math:`U^{-1}` is applied for the missing control states.

    * Two control states that differ in several control qubits, like ``10`` and ``01``,
      are replaced by a single gate with one control qubit less, conjugated by CX gates
      that compute their parity [2, 5].

    * Rotation gates with the same control state are merged by adding up their angles,
      and two identical self-inverse gates, such as two CX gates, cancel.

    A group of gates is only replaced if this reduces its estimated cost, which counts
    two CX gates for every control qubit of a controlled gate. A group is analyzed over all
    the control qubits of its gates, and groups covering more than 1024 control states are
    skipped, which bounds the time spent on a group.

    Example:

    .. plot::
        :include-source:
        :nofigs:

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import RXGate
        from qiskit.transpiler.passes import ControlPatternSimplification

        qc = QuantumCircuit(3)
        qc.append(RXGate(0.5).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RXGate(0.5).control(2, ctrl_state="01"), [0, 1, 2])

        # A single CRX gate controlled on qubit 0.
        optimized_qc = ControlPatternSimplification()(qc)

    References:
        1. Mostafa Atallah, Alvin Gonzales, Daniel Dilley, Igor Gaidai, Zain H. Saleem,
           and Rebekah Herrman,
           *A matching decomposition algorithm for simulating quantum walk Hamiltonians* (2026).
           `arXiv:2601.11418 [quant-ph] <https://arxiv.org/abs/2601.11418>`_
        2. Alvin Gonzales, Rebekah Herrman, Colin Campbell, Igor Gaidai, Ji Liu,
           Teague Tomesh, and Zain H. Saleem,
           *Efficient Sparse State Preparation via Quantum Walks*,
           npj Quantum Information (2025).
           `arXiv:2405.20273 [quant-ph] <https://arxiv.org/abs/2405.20273>`_
        3. Vivek V. Shende, Stephen S. Bullock, and Igor L. Markov,
           *Synthesis of Quantum Logic Circuits*,
           IEEE Transactions on Computer-Aided Design 25.6 (2006): 1000-1010.
           `arXiv:quant-ph/0406176 [quant-ph] <https://arxiv.org/abs/quant-ph/0406176>`_
        4. Anupam Chattopadhyay, Chander Chandak, and Kaushik Chakraborty,
           *Complexity Analysis of Reversible Logic Synthesis* (2014).
           `arXiv:1402.0491 [cs.ET] <https://arxiv.org/abs/1402.0491>`_
        5. Jonathan Welch, Daniel Greenbaum, Sarah Mostame, and Alán Aspuru-Guzik,
           *Efficient Quantum Circuits for Diagonal Unitaries Without Ancillas*,
           New Journal of Physics 16 (2014): 033040.
           `arXiv:1306.3991 [quant-ph] <https://arxiv.org/abs/1306.3991>`_
    """

    def __init__(self, approximation_degree: float = 1.0):
        """
        Args:
            approximation_degree: the threshold used in the average gate fidelity
                computation to decide whether two rotation gates can be considered equal,
                and whether a rotation gate can be removed.
        """
        super().__init__()
        self.approximation_degree = approximation_degree

    @trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ControlPatternSimplification pass on ``dag``.

        Args:
            dag: the DAG to be optimized.

        Returns:
            The optimized DAG.
        """
        new_dag = run_control_pattern_simplification(dag, self.approximation_degree, _MAX_PATTERNS)
        return dag if new_dag is None else new_dag
