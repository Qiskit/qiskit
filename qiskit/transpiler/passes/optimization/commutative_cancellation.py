# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel the redundant (self-adjoint) gates through commutation relations."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_library import StandardGateCommutations

from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit._accelerate import commutation_cancellation
from qiskit._accelerate.commutation_checker import CommutationChecker

from qiskit.transpiler.passes.utils.control_flow import trivial_recurse

_CUTOFF_PRECISION = 1e-5


class CommutativeCancellation(TransformationPass):
    r"""Cancel redundant gates by exploiting commutation relations.

    This pass removes gates that amount to the identity by using commutation rules to
    move matching gates next to each other, then either cancelling or merging them.
    Two kinds of simplification happen:

    * **Self-inverse gates** (``h, y, cx, cy, cz``): if an even number of copies
      of the *same* self-inverse gate on the *same* qubit(s) commute together, they
      cancel completely; an odd number leaves a single copy behind.
    * **Same-axis rotations**: consecutive Z-rotations (``z, p, u1, rz, s, sdg, t, tdg``)
      or X-rotations (``x, rx, sx, sxdg``) on a qubit are summed into a single gate.
      A total angle that is a multiple of :math:`2\pi` removes all of them entirely
      (up to global phase), so inverse pairs like ``t`` + ``tdg`` cancel out naturally.

      For Z-rotations, the pass needs to know which gate family to use for the result.
      It checks in order: (1) a ``rz``, ``p``, or ``u1`` already in the circuit,
      (2) the first of those found in ``basis_gates``/``target``. If neither is found,
      Z-rotation merging is skipped entirely.

      For X-rotations, merging always happens. The result is written as ``x`` (if the
      total is a multiple of :math:`\pi` and ``x`` is available), ``sx`` (if a multiple
      of :math:`\pi/2` and ``sx`` is available), or ``rx(total_angle)`` otherwise.

    Y-rotations are not merged: ``ry`` does not commute with ``cx``, so runs of it are
    left for other optimization passes. Gates with symbolic (:class:`~.Parameter`)
    angles are never merged.

    For example, the two ``cx`` gates below commute past the ``z`` gate (which acts
    only on the control qubit) and cancel each other, leaving just the ``z``::

                  ┌───┐              ┌───┐
        q_0: ──■──┤ Z ├──■──   ->    ┤ Z ├
             ┌─┴─┐└───┘┌─┴─┐         └───┘
        q_1: ┤ X ├─────┤ X ├   ->    ──────
             └───┘     └───┘

    .. note::

        The gate sets eligible for cancellation are fixed (listed above) and apply
        unconditionally to every circuit. ``basis_gates``/``target`` serve a single,
        narrower purpose: choosing which output gate to use when writing back a merged
        same-axis rotation, and only when the circuit itself contains no suitable gate
        already.

    This pass is multithreaded and will potentially launch a thread pool with threads
    equal to the number of CPUs by default. Tune the number of threads with the
    ``RAYON_NUM_THREADS`` environment variable, e.g. ``RAYON_NUM_THREADS=4``.

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.transpiler.passes import CommutativeCancellation

            qc = QuantumCircuit(2)
            qc.cx(0, 1)
            qc.z(0)
            qc.cx(0, 1)  # commutes past `z` and cancels the first `cx`

            optimized = CommutativeCancellation()(qc)
            optimized.count_ops()  # {'z': 1}

    See also :class:`.CommutativeInverseCancellation`, which cancels commuting inverse
    pairs of arbitrary gates, not just this pass's fixed self-inverse/rotation sets.
    """

    def __init__(self, basis_gates=None, target=None):
        """
        Args:
            basis_gates (list[str]): Specifies which gate to use when writing back a
                merged same-axis rotation result. The pass looks for ``rz``, ``p``, or
                ``u1`` (for Z-rotations) and ``x`` or ``sx`` (for X-rotations) in this
                list. This list is only consulted when the circuit itself does not
                already contain one of those gates — the circuit always takes
                precedence. If neither the circuit nor this list contains a suitable
                Z-rotation gate, Z-rotation merging is skipped entirely. Has no effect
                on which gates are eligible for cancellation; that set is fixed.
            target (Target): The :class:`~.Target` representing the target backend.
                Its operation names are extracted and used exactly like ``basis_gates``
                above — as a source of gate names for choosing the merged-rotation
                output gate. When both ``basis_gates`` and ``target`` are provided,
                ``target`` takes precedence and ``basis_gates`` is ignored entirely.
        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()
        self.target = target
        if target is not None:
            self.basis = set(target.operation_names)

        self._var_z_map = {"rz": RZGate, "p": PhaseGate, "u1": U1Gate}

        self._z_rotations = {"p", "z", "u1", "rz", "t", "s", "tdg", "sdg"}
        self._x_rotations = {"x", "rx", "sx", "sxdg"}
        self._gates = {"cx", "cy", "cz", "h", "y"}  # Now the gates supported are hard-coded

        # build a commutation checker restricted to the gates we cancel -- the others we
        # do not have to investigate, which allows to save time
        self._commutation_checker = CommutationChecker(
            StandardGateCommutations, gates=self._gates | self._z_rotations | self._x_rotations
        )

    @trivial_recurse
    def run(self, dag):
        """Run the CommutativeCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        commutation_cancellation.cancel_commutations(
            dag, self._commutation_checker, sorted(self.basis)
        )
        return dag
