"""The Omelyan-Trotter product formula."""

from __future__ import annotations

import warnings
import typing
from collections.abc import Callable
from itertools import chain
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import qiskit.quantum_info

from qiskit.synthesis.evolution.product_formula import ProductFormula, reorder_paulis

if typing.TYPE_CHECKING:
    from qiskit.circuit.quantumcircuit import ParameterValueType
    from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate


class OmelyanTrotter(ProductFormula):
    r"""The (higher order) Omelyan-Trotter product formula.

    The schemes based on the method by Omelyan et al. [1] improve the efficiency
    of product formulas within a given order, by allowing more freedom to
    choose the scheme parameters found in their exponentials.
    For example, an Omelyan improved second-order scheme with two cycles
    takes the form

    .. math::

      e^{A + B} \approx e^{a_1 A} e^{b_1 B} e^{a_2 A} e^{b_1 B} e^{a_1 A},

    with symmetric scheme parameters:

    .. math::

      a_1 = 0.1931833275037703,\quad
      a_2 = 0.6136333449924593,\quad
      b_1 = 0.5.

    The scheme parameters can be derived and optimized to improve the error
    accumulated during the evolution. Many of these schemes are already known
    and the values of their parameters can be found in the literature [1–9]
    (see Ref. [10] for a trove of schemes at orders n = 4, 6).
    For convenience, some of the most efficient schemes known are implemented
    as derivatives of this class.
    The schemes are of even order, resulting in symmetric parameters.
    Furthermore, the notation used to obtain the product formula is
    the so-called ramp notation with parameters c_k (see Ref. [5]).
    The notation enables implementations with an arbitrary number of 
    operators.
    Due to the properties of such a notation the sum of all parameters c_k
    should be equal to 1/2, otherwise the scheme becomes invalid.

    The implementation is optimized for circuit efficiency. Gates corresponding
    to identical operators with different factors are merged within a step.
    Additionally, the implementation allows merging of single-qubit gates
    where possible, as well as consolidation of repeated steps.

    References:
         [1]: I. Omelyan, I. Mryglod and R. Folk,
              "Optimized Forest–Ruth- and Suzuki-like Algorithms for Integration
              of Motion in Many-body Systems", Computer Physics Communications,
              vol. 146, no. 2, pp. 188-202, 2002.
              DOI: `10.1016/s0010-4655(02)00451-4 <https://doi.org/10.1016/s0010-4655(02)00451-4>`_
         [2]: L. Verlet, "Computer "Experiments" on Classical Fluids. I.
              Thermodynamical Properties of Lennard–Jones Molecules", Phys. Rev.,
              vol. 159, pp. 98-103, 1 1967.
              DOI: `10.1103/PhysRev.159.98 <https://doi.org/10.1103/PhysRev.159.98>`_
         [3]: E. Forest and R. D. Ruth, "Fourth-order Symplectic Integration",
              Physica D: Nonlinear Phenomena, vol. 43, no. 1, pp. 105-117, 1990.
              DOI: `10.1016//0167-2789(90)90019-L <https://doi.org/10.1016/0167-2789(90)90019-L>`_
         [4]: N. Hatano and M. Suzuki, "Finding Exponential Product Formulas of
              Higher Orders" in Quantum Annealing and Other Optimization Methods,
              Springer Berlin Heidelberg, 2005, pp. 37-68.
              DOI: `10.1007/11526216_2 <https://doi.org/10.1007/11526216_2>`_
         [5]: J. Ostmeyer, "Optimised Trotter decompositions for Classical and
              Quantum Computing", J. Phys. A, vol. 56, 285303, 2023.
              `arXiv:quant-ph/2211.02691 <https://arxiv.org/abs/2211.02691>`_
         [6]: H. Yoshida, "Construction of higher order symplectic integrators",
              Physics Letters A, vol. 150, no. 5, pp. 262-268, 1990.
              DOI: `10.1016/0375-9601(90)90092-3 <https://doi.org/10.1016/0375-9601(90)90092-3>`_
         [7]: S. Blanes and P. Moan, "Practical Symplectic Partitioned Runge–Kutta
              and Runge–Kutta–Nyström Methods", Journal of Computational
              and Applied Mathematics, vol. 142, no. 2, pp. 313-330, 2002.
              DOI: `10.1016/S0377-0427(01)00492-7 <https://doi.org/10.1016/S0377-0427(01)00492-7>`_
         [8]: M. Maležič and J. Ostmeyer, "Efficient Trotter–Suzuki Schemes
              for Long-Time Quantum Dynamics", 2026.
              `arXiv:quant-ph/2601.18756 <https://arxiv.org/abs/2601.18756>`_
         [9]: M. E. S. Morales, P. C. S. Costa, D. K. Burgarth, Y. R. Sanders
              and D. W. Berry, "Greatly improved higher-order product formulae
              for quantum simulation", 2022.
              `arXiv:quant-ph/2210.15817 <https://arxiv.org/abs/2210.15817>`_
        [10]: M. Maležič, "MarkoMalezic/efficient-trotterizations", version v1.0,
              Github: `MarkoMalezic/efficient-trotterizations <https://github.com/MarkoMalezic/efficient-trotterizations>`_
              Zenodo DOI: `10.5281/18347430 <https://doi.org/10.5281/zenodo.18347430>`_
    """

    def __init__(
        self,
        order: int,
        cycles: int,
        c_vec: list[ParameterValueType],
        reps: int = 1,
        merge_single: bool = False,
        merge_steps: bool = False,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: (
            Callable[[QuantumCircuit, qiskit.quantum_info.Pauli | SparsePauliOp, float], None]
            | None
        ) = None,
        wrap: bool = False,
        preserve_order: bool = True,
        *,
        atomic_evolution_sparse_observable: bool = False,
    ) -> None:
        r"""
        Args:
            order: The order of the product formula.
            cycles: The number of cycles of the product formula.
            c_vec: The parameters of a symmetric scheme in the ramp notation (len(c) = cycles)
            reps: The number of time steps.
            merge_single: The boolean tracking whether to merge single qubit gates if possible
            merge_steps: The boolean tracking whether to merge points where two steps meet
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be ``"chain"``,
                where next neighbor connections are used, or ``"fountain"``, where all qubits are
                connected to one. This only takes effect when ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single
                :class:`~.quantum_info.Pauli`, or :class:`.SparsePauliOp` of only commuting terms,
                to a circuit. The function takes in three arguments: the circuit to append the
                evolution to, the Pauli operator to evolve, and the evolution time. By default, a
                single Pauli evolution is decomposed into a chain of ``CX`` gates and a single
                ``RZ`` gate.
            wrap: Whether to wrap the atomic evolutions into custom gate objects. This only takes
                effect when ``atomic_evolution is None``.
            preserve_order: If ``False``, allows reordering the terms of the operator to
                potentially yield a shallower evolution circuit. Not relevant
                when synthesizing operator with a single term.
            atomic_evolution_sparse_observable: If a custom ``atomic_evolution`` is passed,
                which does not yet support :class:`.SparseObservable`\ s as input, set this
                argument to ``False`` to automatically apply a conversion to :class:`.SparsePauliOp`.
                This argument is supported until Qiskit 2.2, at which point all atomic evolutions
                are required to support :class:`.SparseObservable`\ s as input.

        Raises:
            ValueError: If order is not even (except 1), or if len(c_vec) != cycles, or if c_vec is complex
            UserWarning: If the scheme parameters in ``c_vec`` do not sum to 1/2.
        """
        if order % 2 == 1:
            raise ValueError(
                "Omelyan product formulae are symmetric and therefore only defined "
                f"for when the order is even, not {order}."
            )
        if len(c_vec) != cycles:
            raise ValueError(
                "The number of cycles does not match the length of the parameter list c."
            )
        if abs(sum(c_vec) - 0.5) > 1e-8:
            warnings.warn(
                "The parameters in c_vec do not sum up to 1/2. "
                f"The difference is {abs(sum(c_vec) - 0.5):.2e}. "
                "The scheme may be invalid!", UserWarning)

        self.cycles = cycles
        # Check if the parameters in c_vec are real
        self.c_vec = [real_or_fail(ck) for ck in c_vec]
        self.merge_single = merge_single
        self.merge_steps = merge_steps

        super().__init__(
            order,
            reps,
            insert_barriers,
            cx_structure,
            atomic_evolution,
            wrap,
            preserve_order=preserve_order,
            atomic_evolution_sparse_observable=atomic_evolution_sparse_observable,
        )

    def expand(
        self, evolution: PauliEvolutionGate
    ) -> list[tuple[str, list[int], ParameterValueType]]:
        """Expand the Hamiltonian into an Omelyan-Trotter sequence of ramps consisting of sparse gates.

        For example, the Hamiltonian ``H = IX + ZZ`` for an evolution time ``t`` and
        1 repetition for an order 2 formula would get decomposed into a list of 3-tuples
        containing ``(pauli, indices, rz_rotation_angle)``, that is:

        .. code-block:: text

            ("X", [0], t), ("ZZ", [0, 1], 2t), ("X", [0], t)

        Note that the rotation angle contains a factor of 2, such that the evolution
        of a Pauli :math:`P` over time :math:`t`, which is :math:`e^{itP}`, is represented
        by ``(P, indices, 2 * t)``.

        For ``N`` repetitions, this sequence would be repeated ``N`` times and the coefficients
        divided by ``N``.

        Args:
            evolution: The evolution gate to expand.

        Returns:
            The Pauli network implementing the Trotter expansion.
        """
        operators = evolution.operator
        time = evolution.time

        def to_sparse_list(operator):
            sparse_list = (
                operator.to_sparse_list()
                if isinstance(operator, SparsePauliOp)
                else operator.to_sparse_list()
            )
            paulis = [
                (pauli, indices, real_or_fail(coeff) * time * 2 / self.reps)
                for pauli, indices, coeff in sparse_list
            ]
            if not self.preserve_order:
                return reorder_paulis(paulis)

            return paulis
        
        # Merge same single qubit gates within a step by adding their angles
        def merge_single_qubit(step_paulis):
            merged = []
            for label, indices, angle in step_paulis:
                if len(indices) == 1:
                    merged_single = False
                    for index, element in enumerate(reversed(merged)):
                        if label == element[0] and indices == element[1]:
                            merged[-1-index] = (label, indices, angle + element[2])
                            merged_single = True
                            break
                        elif indices[0] in element[1]:
                            break
                    if not merged_single:
                      merged.append((label, indices, angle))
                else:
                    merged.append((label, indices, angle))
            return merged

        # Merge steps within a full product formula by adding up the angles at the point where they meet
        def merge_steps(full_paulis, step_length):
            merge_points = range(step_length, (self.reps-1) * step_length + 1, step_length)
            for i in reversed(merge_points):
                full_paulis[i-1] = (full_paulis[i-1][0], full_paulis[i-1][1], 2*full_paulis[i-1][2])
                del full_paulis[i]
            return full_paulis

        # construct the evolution circuit
        if isinstance(operators, list):  # already sorted into commuting bits
            non_commuting = [to_sparse_list(operator) for operator in operators]
        else:
            # Assume no commutativity here. If we were to group commuting Paulis,
            # here would be the location to do so.
            non_commuting = [[op] for op in to_sparse_list(operators)]

        # Apply the ramp approach to build one step
        one_step = self._ramp(self.c_vec, non_commuting)
        # Flatten the array
        one_step = list(chain.from_iterable(one_step))
        # Optionally merge single qubit gates
        if self.merge_single:
            one_step = merge_single_qubit(one_step)
        # Extend the single step into the full product formula
        full = self.reps * one_step
        # Optionally merge the gates where two steps meet
        if self.merge_steps:
            full = merge_steps(full, len(one_step))

        return full
    
    @staticmethod
    def _ramp(cs, grouped_paulis):
        q = len(cs)

        def add_paulis(formula, paulis, factor):
            for label, qubits, coeff in paulis:
                formula.append([(label, qubits, coeff * factor)])

        # First ramp up
        formula = []
        for paulis in grouped_paulis[:-1]:
            add_paulis(formula, paulis, cs[0])

        # Ramps in the middle (up and down)
        for k in range(q-1):
            # Point where the ramps meet above
            add_paulis(formula, grouped_paulis[-1], cs[k] + cs[q-k-1])
            # Ramp down
            for paulis in reversed(grouped_paulis[1:-1]):
                add_paulis(formula, paulis, cs[q-k-1])
            # Point where the ramps meet below
            add_paulis(formula, grouped_paulis[0], cs[q-k-1] + cs[k+1])
            # Ramp up
            for paulis in grouped_paulis[1:-1]:
                add_paulis(formula, paulis, cs[k+1])
        # Final point where the ramps meet above
        add_paulis(formula, grouped_paulis[-1], cs[q-1] + cs[0])
        # Final ramp down
        for paulis in reversed(grouped_paulis[:-1]):
            add_paulis(formula, paulis, cs[0])
        
        return formula

def real_or_fail(value, tol=100):
    """Return real if close, otherwise fail. Unbound parameters are left unchanged.

    Based on NumPy's ``real_if_close``, i.e. ``tol`` is in terms of machine precision for float.
    """
    if isinstance(value, ParameterExpression):
        return value

    abstol = tol * np.finfo(float).eps
    if abs(np.imag(value)) < abstol:
        return np.real(value)

    raise ValueError(f"Encountered complex value {value}, but expected real.")
