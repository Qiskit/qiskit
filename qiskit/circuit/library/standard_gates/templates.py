# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Standard gates."""
import numpy as np  # required for rzx templates

from qiskit.qasm import pi
from qiskit.circuit import Parameter, QuantumCircuit  # , TemplateLibrary
from qiskit.quantum_info.synthesis.ion_decompose import cnot_rxx_decompose


class TemplateLibrary:
    def __init__(self, *, base=None):
        """Create a new equivalence library.

        Args:
            base (Optional[EquivalenceLibrary]):  Base equivalence library to
                will be referenced if an entry is not found in this library.
        """
        self._base = base

        self._map = {}

    # Clifford templates

    """
    Clifford template 2_1:
    .. parsed-literal::
            q_0: ─■──■─
                  │  │
            q_1: ─■──■─
    """

    def clifford_2_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.cz(0, 1)
        return qc

    def_Clifford2_1 = clifford_2_1()

    """
    Clifford template 2_2:
    .. parsed-literal::
            q_0: ──■────■──
                 ┌─┴─┐┌─┴─┐
            q_1: ┤ X ├┤ X ├
                 └───┘└───┘
    """

    def clifford_2_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        return qc

    def_Clifford2_2 = clifford_2_2()

    """
    Clifford template 2_3:
    .. parsed-literal::
                 ┌───┐┌───┐
            q_0: ┤ H ├┤ H ├
                 └───┘└───┘
    """

    def clifford_2_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        return qc

    def_Clifford2_3 = clifford_2_3()

    """
    Clifford template 2_4:
    .. parsed-literal::
            q_0: ─X──X─
                  │  │
            q_1: ─X──X─
    """

    def clifford_2_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.swap(1, 0)
        return qc

    def_Clifford2_4 = clifford_2_4()

    """
    Clifford template 3_1:
    .. parsed-literal::
                 ┌───┐┌───┐┌───┐
            q_0: ┤ S ├┤ S ├┤ Z ├
                 └───┘└───┘└───┘
    """

    def clifford_3_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.s(0)
        qc.z(0)
        return qc

    def_Clifford3_1 = clifford_3_1()

    """
    Clifford template 4_1:
    .. parsed-literal::
                      ┌───┐
            q_0: ──■──┤ X ├──■───X─
                 ┌─┴─┐└─┬─┘┌─┴─┐ │
            q_1: ┤ X ├──■──┤ X ├─X─
                 └───┘     └───┘
    """

    def clifford_4_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.swap(0, 1)
        return qc

    def_Clifford4_1 = clifford_4_1()

    """
    Clifford template 4_2:
    .. parsed-literal::
            q_0: ───────■────────■─
                 ┌───┐┌─┴─┐┌───┐ │
            q_1: ┤ H ├┤ X ├┤ H ├─■─
                 └───┘└───┘└───┘
    """

    def clifford_4_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(1)
        qc.cz(0, 1)
        return qc

    def_Clifford4_2 = clifford_4_2()

    """
    Clifford template 4_3:
    .. parsed-literal::
                 ┌───┐     ┌─────┐
            q_0: ┤ S ├──■──┤ SDG ├──■──
                 └───┘┌─┴─┐└─────┘┌─┴─┐
            q_1: ─────┤ X ├───────┤ X ├
                      └───┘       └───┘
    """

    def clifford_4_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.s(0)
        qc.cx(0, 1)
        qc.sdg(0)
        qc.cx(0, 1)
        return qc

    def_Clifford4_3 = clifford_4_3()

    """
    Clifford template 4_4:
    .. parsed-literal::
                 ┌───┐   ┌─────┐
            q_0: ┤ S ├─■─┤ SDG ├─■─
                 └───┘ │ └─────┘ │
            q_1: ──────■─────────■─
    """

    def clifford_4_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.s(0)
        qc.cz(0, 1)
        qc.sdg(0)
        qc.cz(0, 1)
        return qc

    def_Clifford4_4 = clifford_4_4()

    """
    Clifford template 5_1:
    .. parsed-literal::
            q_0: ──■─────────■─────────■──
                 ┌─┴─┐     ┌─┴─┐       │
            q_1: ┤ X ├──■──┤ X ├──■────┼──
                 └───┘┌─┴─┐└───┘┌─┴─┐┌─┴─┐
            q_2: ─────┤ X ├─────┤ X ├┤ X ├
                      └───┘     └───┘└───┘
    """

    def clifford_5_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        return qc

    def_Clifford5_1 = clifford_5_1()

    """
    Clifford template 6_2:
    .. parsed-literal::
                 ┌───┐
            q_0: ┤ S ├──■───────────■───■─
                 ├───┤┌─┴─┐┌─────┐┌─┴─┐ │
            q_1: ┤ S ├┤ X ├┤ SDG ├┤ X ├─■─
                 └───┘└───┘└─────┘└───┘
    """

    def clifford_6_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.s(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.sdg(1)
        qc.cx(0, 1)
        qc.cz(0, 1)
        return qc

    def_Clifford6_2 = clifford_6_2()

    """
    Clifford template 6_3:
    .. parsed-literal::
                       ┌───┐     ┌───┐
            q_0: ─X──■─┤ H ├──■──┤ X ├─────
                │    │ └───┘┌─┴─┐└─┬─┘┌───┐
            q_1: ─X──■──────┤ X ├──■──┤ H ├
                            └───┘     └───┘
    """

    def clifford_6_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.cz(0, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.h(1)
        return qc

    def_Clifford6_3 = clifford_6_3()

    """
    Clifford template 6_4:
    .. parsed-literal::
                 ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
            q_0: ┤ S ├┤ H ├┤ S ├┤ H ├┤ S ├┤ H ├
                 └───┘└───┘└───┘└───┘└───┘└───┘
    """

    def clifford_6_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        return qc

    def_Clifford6_4 = clifford_6_4()

    """
    Clifford template 6_5:
    .. parsed-literal::
                          ┌───┐
            q_0: ─■───■───┤ S ├───■───────
                  │ ┌─┴─┐┌┴───┴┐┌─┴─┐┌───┐
            q_1: ─■─┤ X ├┤ SDG ├┤ X ├┤ S ├
                    └───┘└─────┘└───┘└───┘
    """

    def clifford_6_5():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.cx(0, 1)
        qc.s(0)
        qc.sdg(1)
        qc.cx(0, 1)
        qc.s(1)
        return qc

    def_Clifford6_5 = clifford_6_5()

    """
    Clifford template 8_1:
    .. parsed-literal::
                           ┌───┐ ┌───┐ ┌───┐┌─────┐
            q_0: ──■───────┤ X ├─┤ S ├─┤ X ├┤ SDG ├
                 ┌─┴─┐┌───┐└─┬─┘┌┴───┴┐└─┬─┘└┬───┬┘
            q_1: ┤ X ├┤ H ├──■──┤ SDG ├──■───┤ H ├─
                 └───┘└───┘     └─────┘      └───┘
    """

    def clifford_8_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        qc.cx(1, 0)
        qc.s(0)
        qc.sdg(1)
        qc.cx(1, 0)
        qc.sdg(0)
        qc.h(1)
        return qc

    def_Clifford8_1 = clifford_8_1()

    """
    Clifford template 8_2:
    .. parsed-literal::
                                 ┌───┐
            q_0: ──■─────────■───┤ S ├───■────────────
                 ┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐┌───┐┌───┐
            q_1: ┤ X ├┤ H ├┤ X ├┤ SDG ├┤ X ├┤ S ├┤ H ├
                 └───┘└───┘└───┘└─────┘└───┘└───┘└───┘
    """

    def clifford_8_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        qc.cx(0, 1)
        qc.s(0)
        qc.sdg(1)
        qc.cx(0, 1)
        qc.s(1)
        qc.h(1)
        return qc

    def_Clifford8_2 = clifford_8_2()

    """
    Clifford template 8_3:
    .. parsed-literal::
            q_0: ─────────────────■───────────────────────■──
                 ┌───┐┌───┐┌───┐┌─┴─┐┌─────┐┌───┐┌─────┐┌─┴─┐
            q_1: ┤ S ├┤ H ├┤ S ├┤ X ├┤ SDG ├┤ H ├┤ SDG ├┤ X ├
                 └───┘└───┘└───┘└───┘└─────┘└───┘└─────┘└───┘
    """

    def clifford_8_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.s(1)
        qc.h(1)
        qc.s(1)
        qc.cx(0, 1)
        qc.sdg(1)
        qc.h(1)
        qc.sdg(1)
        qc.cx(0, 1)
        return qc

    def_Clifford8_3 = clifford_8_3()

    """
    Template 2a_1:
    .. parsed-literal::
                 ┌───┐┌───┐
            q_0: ┤ X ├┤ X ├
                 └───┘└───┘
    """

    def template_nct_2a_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)
        return qc

    def_Template_nct_2a_1 = template_nct_2a_1()

    """
    Template 2a_2:
    .. parsed-literal::
        q_0: ──■────■──
             ┌─┴─┐┌─┴─┐
        q_1: ┤ X ├┤ X ├
             └───┘└───┘
    """

    def template_nct_2a_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        return qc

    def_Template_nct_2a_2 = template_nct_2a_2()

    """
    Template 2a_3:
    .. parsed-literal::
        q_0: ──■────■──
               │    │
        q_1: ──■────■──
             ┌─┴─┐┌─┴─┐
        q_2: ┤ X ├┤ X ├
             └───┘└───┘
    """

    def template_nct_2a_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 1, 2)
        return qc

    def_Template_nct_2a_3 = template_nct_2a_3()

    """
    Template 4a_1:
    .. parsed-literal::
        q_0: ───────■─────────■──
                    │         │
        q_1: ──■────┼────■────┼──
               │    │    │    │
        q_2: ──■────■────■────■──
               │  ┌─┴─┐  │  ┌─┴─┐
        q_3: ──┼──┤ X ├──┼──┤ X ├
             ┌─┴─┐└───┘┌─┴─┐└───┘
        q_4: ┤ X ├─────┤ X ├─────
             └───┘     └───┘
    """

    def template_nct_4a_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(5)
        qc.ccx(1, 2, 4)
        qc.ccx(0, 2, 3)
        qc.ccx(1, 2, 4)
        qc.ccx(0, 2, 3)
        return qc

    def_Template_nct_4a_1 = template_nct_4a_1()

    """
    Template 4a_2:
    .. parsed-literal::
        q_0: ──■─────────■───────
               │         │
        q_1: ──■────■────■────■──
               │  ┌─┴─┐  │  ┌─┴─┐
        q_2: ──┼──┤ X ├──┼──┤ X ├
             ┌─┴─┐└───┘┌─┴─┐└───┘
        q_3: ┤ X ├─────┤ X ├─────
             └───┘     └───┘
    """

    def template_nct_4a_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(4)
        qc.ccx(0, 1, 3)
        qc.cx(1, 2)
        qc.ccx(0, 1, 3)
        qc.cx(1, 2)
        return qc

    def_Template_nct_4a_2 = template_nct_4a_2()

    """
    Template 4a_3:
    .. parsed-literal::
        q_0: ──■────■────■────■──
               │  ┌─┴─┐  │  ┌─┴─┐
        q_1: ──┼──┤ X ├──┼──┤ X ├
             ┌─┴─┐└───┘┌─┴─┐└───┘
        q_2: ┤ X ├─────┤ X ├─────
             └───┘     └───┘
    """

    def template_nct_4a_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 1)
        return qc

    def_Template_nct_4a_3 = template_nct_4a_3()

    """
    Template 4b_1:
    .. parsed-literal::
        q_0: ───────■─────────■──
                    │         │
        q_1: ──■────┼────■────┼──
               │    │    │    │
        q_2: ──■────■────■────■──
             ┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐
        q_3: ┤ X ├┤ X ├┤ X ├┤ X ├
             └───┘└───┘└───┘└───┘
    """

    def template_nct_4b_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(4)
        qc.ccx(1, 2, 3)
        qc.ccx(0, 2, 3)
        qc.ccx(1, 2, 3)
        qc.ccx(0, 2, 3)
        return qc

    def_Template_nct_4b_1 = template_nct_4b_1()

    """
    Template 4b_2:
    .. parsed-literal::
        q_0: ──■─────────■───────
               │         │
        q_1: ──■────■────■────■──
             ┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐
        q_2: ┤ X ├┤ X ├┤ X ├┤ X ├
             └───┘└───┘└───┘└───┘
    """

    def template_nct_4b_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.cx(1, 2)
        return qc

    def_Template_nct_4b_2 = template_nct_4b_2()

    """
    Template 5a_1:
    .. parsed-literal::
        q_0: ──■────■────■────■────■──
               │  ┌─┴─┐  │  ┌─┴─┐  │
        q_1: ──■──┤ X ├──■──┤ X ├──┼──
             ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        q_2: ┤ X ├─────┤ X ├─────┤ X ├
             └───┘     └───┘     └───┘
    """

    def template_nct_5a_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.cx(0, 2)
        return qc

    def_Template_nct_5a_1 = template_nct_5a_1()

    """
    Template 5a_2:
    .. parsed-literal::
        q_0: ──■─────────■─────────■──
               │  ┌───┐  │  ┌───┐  │
        q_1: ──■──┤ X ├──■──┤ X ├──┼──
             ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        q_2: ┤ X ├─────┤ X ├─────┤ X ├
             └───┘     └───┘     └───┘
    """

    def template_nct_5a_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.x(1)
        qc.ccx(0, 1, 2)
        qc.x(1)
        qc.cx(0, 2)
        return qc

    def_Template_nct_5a_2 = template_nct_5a_2()

    """
    Template 5a_3:
    .. parsed-literal::
        q_0: ───────■─────────■────■──
                  ┌─┴─┐     ┌─┴─┐  │
        q_1: ──■──┤ X ├──■──┤ X ├──┼──
             ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        q_2: ┤ X ├─────┤ X ├─────┤ X ├
             └───┘     └───┘     └───┘
    """

    def template_nct_5a_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(0, 2)
        return qc

    def_Template_nct_5a_3 = template_nct_5a_3()

    """
    Template 5a_4:
    .. parsed-literal::
                  ┌───┐     ┌───┐
        q_0: ──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└───┘┌─┴─┐├───┤
        q_1: ┤ X ├─────┤ X ├┤ X ├
             └───┘     └───┘└───┘
    """

    def template_nct_5a_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.x(0)
        qc.cx(0, 1)
        qc.x(0)
        qc.x(1)
        return qc

    def_Template_nct_5a_4 = template_nct_5a_4()

    """
    Template 6a_1:
    .. parsed-literal::
                  ┌───┐     ┌───┐     ┌───┐
        q_0: ──■──┤ X ├──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘
        q_1: ┤ X ├──■──┤ X ├──■──┤ X ├──■──
             └───┘     └───┘     └───┘
    """

    def template_nct_6a_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        return qc

    def_Template_nct_6a_1 = template_nct_6a_1()

    """
    Template 6a_2:
    .. parsed-literal::
        q_0: ──■────■────■────■────■────■──
               │  ┌─┴─┐  │  ┌─┴─┐  │  ┌─┴─┐
        q_1: ──■──┤ X ├──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘
        q_2: ┤ X ├──■──┤ X ├──■──┤ X ├──■──
             └───┘     └───┘     └───┘
    """

    def template_nct_6a_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_6a_2 = template_nct_6a_2()

    """
    Template 6a_3:
    .. parsed-literal::
        q_0: ───────■─────────■────■────■──
                  ┌─┴─┐     ┌─┴─┐  │  ┌─┴─┐
        q_1: ──■──┤ X ├──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘
        q_2: ┤ X ├──■──┤ X ├──■──┤ X ├──■──
             └───┘     └───┘     └───┘
    """

    from qiskit.circuit.quantumcircuit import QuantumCircuit

    def template_nct_6a_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_6a_3 = template_nct_6a_3()

    """
    Template 6a_4:
    .. parsed-literal::
        q_0: ───────■──────────────■───────
                  ┌─┴─┐     ┌───┐  │  ┌───┐
        q_1: ──■──┤ X ├──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘
        q_2: ┤ X ├──■──┤ X ├──■──┤ X ├──■──
             └───┘     └───┘     └───┘
    """

    def template_nct_6a_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_6a_3 = template_nct_6a_4()

    """
    Template 6b_1:
    .. parsed-literal::
        q_0: ──■─────────■────■─────────■──
               │       ┌─┴─┐  │       ┌─┴─┐
        q_1: ──■────■──┤ X ├──■────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘
    """

    def template_nct_6b_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_6b_1 = template_nct_6b_1()

    """
    Template 6b_2:
    .. parsed-literal::
        q_0: ───────■────■─────────■────■──
                    │  ┌─┴─┐       │  ┌─┴─┐
        q_1: ──■────■──┤ X ├──■────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘
    """

    def template_nct_6b_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_6b_2 = template_nct_6b_2()

    """
    Template 6c_1:
    .. parsed-literal::
        q_0: ──■─────────■─────────■────■──
               │  ┌───┐  │  ┌───┐  │  ┌─┴─┐
        q_1: ──■──┤ X ├──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘
        q_2: ┤ X ├──■──┤ X ├──■──┤ X ├──■──
             └───┘     └───┘     └───┘
    """

    def template_nct_6c_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_6c_1 = template_nct_6c_1()

    """
    Template 7a_1:
    .. parsed-literal::
             ┌───┐                    ┌───┐
        q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
             └─┬─┘┌─┴─┐       │  ┌─┴─┐└─┬─┘  │
        q_1: ──■──┤ X ├──■────■──┤ X ├──■────■──
                  └───┘┌─┴─┐┌─┴─┐└───┘     ┌─┴─┐
        q_2: ──────────┤ X ├┤ X ├──────────┤ X ├
                       └───┘└───┘          └───┘
    """

    def template_nct_7a_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.ccx(0, 1, 2)
        return qc

    def_Template_nct_7a_1 = template_nct_7a_1()

    """
    Template 7b_1:
    .. parsed-literal::
             ┌───┐                    ┌───┐
        q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
             └───┘┌─┴─┐       │  ┌─┴─┐└───┘  │
        q_1: ─────┤ X ├──■────■──┤ X ├───────■──
                  └───┘┌─┴─┐┌─┴─┐└───┘     ┌─┴─┐
        q_2: ──────────┤ X ├┤ X ├──────────┤ X ├
                       └───┘└───┘          └───┘
    """

    def template_nct_7b_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.x(0)
        qc.ccx(0, 1, 2)
        return qc

    def_Template_nct_7b_1 = template_nct_7b_1()

    """
    Template 7c_1:
    .. parsed-literal::
             ┌───┐                    ┌───┐
        q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
             └───┘┌─┴─┐       │  ┌─┴─┐└───┘  │
        q_1: ─────┤ X ├──■────■──┤ X ├───────■──
                  └─┬─┘┌─┴─┐┌─┴─┐└─┬─┘     ┌─┴─┐
        q_2: ───────■──┤ X ├┤ X ├──■───────┤ X ├
                       └───┘└───┘          └───┘
    """

    def template_nct_7c_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.x(0)
        qc.ccx(0, 1, 2)
        return qc

    def_Template_nct_7c_1 = template_nct_7c_1()

    """
    Template 7d_1:
    .. parsed-literal::
             ┌───┐                    ┌───┐
        q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
             └─┬─┘┌─┴─┐       │  ┌─┴─┐└─┬─┘  │
        q_1: ──■──┤ X ├──■────■──┤ X ├──■────■──
                  └─┬─┘┌─┴─┐┌─┴─┐└─┬─┘     ┌─┴─┐
        q_2: ───────■──┤ X ├┤ X ├──■───────┤ X ├
                       └───┘└───┘          └───┘
    """

    def template_nct_7d_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 0)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(1, 0)
        qc.ccx(0, 1, 2)
        return qc

    def_Template_nct_7d_1 = template_nct_7d_1()

    """
    Template 7e_1:
    .. parsed-literal::
             ┌───┐                    ┌───┐
        q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
             └───┘┌─┴─┐       │  ┌─┴─┐└───┘  │
        q_1: ─────┤ X ├───────┼──┤ X ├───────┼──
                  └─┬─┘┌───┐┌─┴─┐└─┬─┘     ┌─┴─┐
        q_2: ───────■──┤ X ├┤ X ├──■───────┤ X ├
                       └───┘└───┘          └───┘
    """

    def template_nct_7e_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.x(0)
        qc.cx(0, 2)
        return qc

    def_Template_nct_7e_1 = template_nct_7e_1()

    """
    Template 9a_1:
    .. parsed-literal::
             ┌───┐     ┌───┐          ┌───┐
        q_0: ┤ X ├──■──┤ X ├──■────■──┤ X ├──■──
             └─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐
        q_1: ──■──┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├
                  └─┬─┘  │  ├───┤└─┬─┘┌───┐└─┬─┘
        q_2: ───────■────■──┤ X ├──■──┤ X ├──■──
                            └───┘     └───┘
    """

    def template_nct_9a_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 0)
        qc.ccx(0, 2, 1)
        qc.ccx(1, 2, 0)
        qc.x(2)
        qc.cx(0, 1)
        qc.ccx(0, 2, 1)
        qc.cx(1, 0)
        qc.x(2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9a_1 = template_nct_9a_1()

    """
    Template 9c_1:
    .. parsed-literal::
             ┌───┐     ┌───┐┌───┐     ┌───┐          ┌───┐
        q_0: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
        q_1: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.x(0)
        qc.cx(1, 0)
        qc.x(1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.x(1)
        qc.cx(1, 0)
        return qc

    def_Template_nct_9c_1 = template_nct_9c_1()

    """
    Template 9c_2:
    .. parsed-literal::
        q_0: ───────■────■──────────────■────■─────────■──
             ┌───┐  │  ┌─┴─┐┌───┐     ┌─┴─┐  │       ┌─┴─┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.x(2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.x(2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9c_2 = template_nct_9c_2()

    """
    Template 9c_3:
    .. parsed-literal::
        q_0: ───────■────────────────────────■────────────
             ┌───┐  │  ┌───┐┌───┐     ┌───┐  │       ┌───┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.x(1)
        qc.cx(2, 1)
        qc.x(2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.x(2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9c_3 = template_nct_9c_3()

    """
    Template 9c_4:
    .. parsed-literal::
        q_0: ──■────■─────────■──────────────■────────────
             ┌─┴─┐  │  ┌───┐┌─┴─┐     ┌───┐  │       ┌───┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.x(1)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.x(2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9c_4 = template_nct_9c_4()

    """
    Template 9c_5:
    .. parsed-literal::
        q_0: ────────────■─────────■──────────────■───────
             ┌───┐     ┌─┴─┐┌───┐  │  ┌───┐       │  ┌───┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├──┼──┤ X ├──■────┼──┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_5():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9c_5 = template_nct_9c_5()

    """
    Template 9c_6:
    .. parsed-literal::
        q_0: ───────■────■─────────■─────────■────■───────
             ┌───┐  │  ┌─┴─┐┌───┐  │  ┌───┐  │    │  ┌───┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├──┼──┤ X ├──■────┼──┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_6():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9c_6 = template_nct_9c_6()

    """
    Template 9c_7:
    .. parsed-literal::
        q_0: ──■────■────■────■────■─────────■────■───────
             ┌─┴─┐  │  ┌─┴─┐┌─┴─┐  │  ┌───┐  │    │  ┌───┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├──┼──┤ X ├──■────┼──┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_7():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9c_7 = template_nct_9c_7()

    """
    Template 9c_8:
    .. parsed-literal::
        q_0: ──■─────────■────■─────────■──────────────■──
             ┌─┴─┐     ┌─┴─┐┌─┴─┐     ┌─┴─┐          ┌─┴─┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_8():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.x(2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9c_8 = template_nct_9c_8()

    """
    Template 9c_9:
    .. parsed-literal::
        q_0: ──■────■────■────■─────────■────■─────────■──
             ┌─┴─┐  │  ┌─┴─┐┌─┴─┐     ┌─┴─┐  │       ┌─┴─┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_9():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.x(2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9c_9 = template_nct_9c_9()

    """
    Template 9c_10:
    .. parsed-literal::
        q_0: ──■─────────■────■────■────■─────────■────■──
             ┌─┴─┐     ┌─┴─┐┌─┴─┐  │  ┌─┴─┐       │  ┌─┴─┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├──┼──┤ X ├──■────┼──┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_10():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9c_10 = template_nct_9c_10()

    """
    Template 9c_11:
    .. parsed-literal::
        q_0: ───────■────■─────────■────■────■────■────■──
             ┌───┐  │  ┌─┴─┐┌───┐  │  ┌─┴─┐  │    │  ┌─┴─┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├──┼──┤ X ├──■────┼──┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    from qiskit.circuit.quantumcircuit import QuantumCircuit

    def template_nct_9c_11():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9c_11 = template_nct_9c_11()

    """
    Template 9c_12:
    .. parsed-literal::
        q_0: ──■────■────■────■────■────■────■────■────■──
             ┌─┴─┐  │  ┌─┴─┐┌─┴─┐  │  ┌─┴─┐  │    │  ┌─┴─┐
        q_1: ┤ X ├──■──┤ X ├┤ X ├──┼──┤ X ├──■────┼──┤ X ├
             └─┬─┘┌─┴─┐└───┘└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
                  └───┘          └───┘     └───┘└───┘
    """

    def template_nct_9c_12():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 1)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9c_12 = template_nct_9c_12()

    """
    Template 9d_1:
    .. parsed-literal::
                       ┌───┐          ┌───┐          ┌───┐
        q_0: ──■───────┤ X ├───────■──┤ X ├───────■──┤ X ├
             ┌─┴─┐┌───┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
        q_1: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_1():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.x(1)
        qc.cx(1, 0)
        qc.x(1)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.x(1)
        qc.cx(0, 1)
        qc.cx(1, 0)
        return qc

    def_Template_nct_9d_1 = template_nct_9d_1()

    """
    Template 9d_2:
    .. parsed-literal::
        q_0: ──■────■────■──────────────■──────────────■──
               │    │  ┌─┴─┐          ┌─┴─┐          ┌─┴─┐
        q_1: ──■────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9d_1 = template_nct_9d_1()

    """
    Template 9d_2:
    .. parsed-literal::
        q_0: ──■────■────■──────────────■──────────────■──
               │    │  ┌─┴─┐          ┌─┴─┐          ┌─┴─┐
        q_1: ──■────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_2():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9d_2 = template_nct_9d_2()

    """
    Template 9d_3:
    .. parsed-literal::
        q_0: ──■────■───────────────────■─────────────────
               │    │  ┌───┐          ┌─┴─┐          ┌───┐
        q_1: ──■────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_3():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.x(2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.cx(1, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9d_3 = template_nct_9d_3()

    """
    Template 9d_4:
    .. parsed-literal::
        q_0: ───────■─────────■──────────────■────────────
                    │  ┌───┐  │       ┌───┐  │       ┌───┐
        q_1: ──■────┼──┤ X ├──┼────■──┤ X ├──┼────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_4():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9d_4 = template_nct_9d_4()

    """
    Template 9d_5:
    .. parsed-literal::
        q_0: ──■────■─────────■─────────■────■────────────
               │    │  ┌───┐  │       ┌─┴─┐  │       ┌───┐
        q_1: ──■────┼──┤ X ├──┼────■──┤ X ├──┼────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_5():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9d_5 = template_nct_9d_5()

    """
    Template 9d_6:
    .. parsed-literal::
        q_0: ──■────■──────────────■────■─────────■───────
               │    │  ┌───┐       │  ┌─┴─┐       │  ┌───┐
        q_1: ──■────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_6():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.x(2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.ccx(0, 1, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9d_6 = template_nct_9d_6()

    """
    Template 9d_7:
    .. parsed-literal::
        q_0: ──■────■─────────■────■────■────■────■───────
               │    │  ┌───┐  │    │  ┌─┴─┐  │    │  ┌───┐
        q_1: ──■────┼──┤ X ├──┼────■──┤ X ├──┼────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_7():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 1, 2)
        qc.cx(2, 1)
        return qc

    def_Template_nct_9d_7 = template_nct_9d_7()

    """
    Template 9d_8:
    .. parsed-literal::
        q_0: ──■────■────■────■─────────■────■─────────■──
               │    │  ┌─┴─┐  │       ┌─┴─┐  │       ┌─┴─┐
        q_1: ──■────┼──┤ X ├──┼────■──┤ X ├──┼────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_8():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9d_8 = template_nct_9d_8()

    """
    Template 9d_9:
    .. parsed-literal::
        q_0: ──■────■────■─────────■────■─────────■────■──
               │    │  ┌─┴─┐       │  ┌─┴─┐       │  ┌─┴─┐
        q_1: ──■────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_9():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.x(2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9d_9 = template_nct_9d_9()

    """
    Template 9d_10:
    .. parsed-literal::
        q_0: ──■────■────■────■────■────■────■────■────■──
               │    │  ┌─┴─┐  │    │  ┌─┴─┐  │    │  ┌─┴─┐
        q_1: ──■────┼──┤ X ├──┼────■──┤ X ├──┼────■──┤ X ├
             ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
        q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
             └───┘└───┘     └───┘└───┘     └───┘└───┘
    """

    def template_nct_9d_10():
        """
        Returns:
            QuantumCircuit: template as a quantum circuit.
        """
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(0, 2)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        qc.cx(0, 2)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 2, 1)
        return qc

    def_Template_nct_9d_10 = template_nct_9d_10()

    """
    RZX based template for CX - RYGate - CX
    .. parsed-literal::
                                                           ┌──────────┐
    q_0: ──■─────────────■─────────────────────────────────┤0         ├───────────
         ┌─┴─┐┌───────┐┌─┴─┐┌────────┐┌──────────┐┌───────┐│  RZX(-ϴ) │┌─────────┐
    q_1: ┤ X ├┤ RY(ϴ) ├┤ X ├┤ RY(-ϴ) ├┤ RZ(-π/2) ├┤ RX(ϴ) ├┤1         ├┤ RZ(π/2) ├
         └───┘└───────┘└───┘└────────┘└──────────┘└───────┘└──────────┘└─────────┘
    """

    def rzx_cy(theta: float = None):
        """Template for CX - RYGate - CX."""
        if theta is None:
            theta = Parameter("ϴ")

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.ry(theta, 1)
        circ.cx(0, 1)
        circ.ry(-1 * theta, 1)
        circ.rz(-np.pi / 2, 1)
        circ.rx(theta, 1)
        circ.rzx(-1 * theta, 0, 1)
        circ.rz(np.pi / 2, 1)

        return circ

    def_rzx_cy = rzx_cy()

    """
    RZX based template for CX - RXGate - CX
    .. parsed-literal::
         ┌───┐         ┌───┐┌─────────┐┌─────────┐┌─────────┐┌──────────┐»
    q_0: ┤ X ├─────────┤ X ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├┤0         ├»
         └─┬─┘┌───────┐└─┬─┘└─────────┘└─────────┘└─────────┘│  RZX(-ϴ) │»
    q_1: ──■──┤ RX(ϴ) ├──■───────────────────────────────────┤1         ├»
              └───────┘                                      └──────────┘»
    «     ┌─────────┐┌─────────┐┌─────────┐
    «q_0: ┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├
    «     └─────────┘└─────────┘└─────────┘
    «q_1: ─────────────────────────────────
    «
    """

    def rzx_xz(theta: float = None):
        """Template for CX - RXGate - CX."""
        if theta is None:
            theta = Parameter("ϴ")

        qc = QuantumCircuit(2)
        qc.cx(1, 0)
        qc.rx(theta, 1)
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(np.pi / 2, 0)
        qc.rzx(-1 * theta, 0, 1)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def_rzx_xz = rzx_xz()

    """
    RZX based template for CX - RYGate - CX
    .. parsed-literal::
              ┌────────┐     ┌─────────┐┌─────────┐┌──────────┐
    q_0: ──■──┤ RY(-ϴ) ├──■──┤ RX(π/2) ├┤0        ├┤ RX(-π/2) ├
         ┌─┴─┐└────────┘┌─┴─┐└─────────┘│  RZX(ϴ) │└──────────┘
    q_1: ┤ X ├──────────┤ X ├───────────┤1        ├────────────
         └───┘          └───┘           └─────────┘
    """

    def rzx_yz(theta: float = None):
        """Template for CX - RYGate - CX."""
        if theta is None:
            theta = Parameter("ϴ")

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.ry(-1 * theta, 0)
        circ.cx(0, 1)
        circ.rx(np.pi / 2, 0)
        circ.rzx(theta, 0, 1)
        circ.rx(-np.pi / 2, 0)

        return circ

    def_rzx_yz = rzx_yz()

    """
    RZX based template for CX - phase - CX
    .. parsed-literal::
                                                                                »
    q_0: ──■────────────────────────────────────────────■───────────────────────»
         ┌─┴─┐┌───────┐┌────┐┌───────┐┌────┐┌────────┐┌─┴─┐┌────────┐┌─────────┐»
    q_1: ┤ X ├┤ RZ(ϴ) ├┤ √X ├┤ RZ(π) ├┤ √X ├┤ RZ(3π) ├┤ X ├┤ RZ(-ϴ) ├┤ RZ(π/2) ├»
         └───┘└───────┘└────┘└───────┘└────┘└────────┘└───┘└────────┘└─────────┘»
    «                                    ┌──────────┐                      »
    «q_0: ───────────────────────────────┤0         ├──────────────────────»
    «     ┌─────────┐┌─────────┐┌───────┐│  RZX(-ϴ) │┌─────────┐┌─────────┐»
    «q_1: ┤ RX(π/2) ├┤ RZ(π/2) ├┤ RX(ϴ) ├┤1         ├┤ RZ(π/2) ├┤ RX(π/2) ├»
    «     └─────────┘└─────────┘└───────┘└──────────┘└─────────┘└─────────┘»
    «
    «q_0: ───────────
    «     ┌─────────┐
    «q_1: ┤ RZ(π/2) ├
    «     └─────────┘
    """

    def rzx_zz1(theta: float = None):
        """Template for CX - RZGate - CX."""
        if theta is None:
            theta = Parameter("ϴ")

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rz(theta, 1)
        qc.sx(1)
        qc.rz(np.pi, 1)
        qc.sx(1)
        qc.rz(3 * np.pi, 1)
        qc.cx(0, 1)
        qc.rz(-1 * theta, 1)

        # Hadamard
        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 2, 1)

        qc.rx(theta, 1)
        qc.rzx(-1 * theta, 0, 1)
        # Hadamard
        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 2, 1)

        return qc

    def_rzx_zz1 = rzx_zz1()

    """
    RZX based template for CX - PhaseGate - CX
    .. parsed-literal::
                                                                            »
    q_0: ──■────────────■─────────────────────────────────────────────────────»
         ┌─┴─┐┌──────┐┌─┴─┐┌───────┐┌─────────┐┌─────────┐┌─────────┐┌───────┐»
    q_1: ┤ X ├┤ P(ϴ) ├┤ X ├┤ P(-ϴ) ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├┤ RX(ϴ) ├»
         └───┘└──────┘└───┘└───────┘└─────────┘└─────────┘└─────────┘└───────┘»
    «     ┌──────────┐
    «q_0: ┤0         ├─────────────────────────────────
    «     │  RZX(-ϴ) │┌─────────┐┌─────────┐┌─────────┐
    «q_1: ┤1         ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├
    «     └──────────┘└─────────┘└─────────┘└─────────┘
    """

    def rzx_zz2(theta: float = None):
        """Template for CX - RZGate - CX."""
        if theta is None:
            theta = Parameter("ϴ")

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.p(theta, 1)
        qc.cx(0, 1)
        qc.p(-1 * theta, 1)
        # Hadamard
        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 2, 1)

        qc.rx(theta, 1)
        qc.rzx(-1 * theta, 0, 1)
        # Hadamard
        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 2, 1)

        return qc

    def_rzx_zz2 = rzx_zz2()

    """
    RZX based template for CX - RZGate - CX
    .. parsed-literal::
                                                                                »
    q_0: ──■─────────────■──────────────────────────────────────────────────────»
         ┌─┴─┐┌───────┐┌─┴─┐┌────────┐┌─────────┐┌─────────┐┌─────────┐┌───────┐»
    q_1: ┤ X ├┤ RZ(ϴ) ├┤ X ├┤ RZ(-ϴ) ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├┤ RX(ϴ) ├»
         └───┘└───────┘└───┘└────────┘└─────────┘└─────────┘└─────────┘└───────┘»
    «     ┌──────────┐
    «q_0: ┤0         ├─────────────────────────────────
    «     │  RZX(-ϴ) │┌─────────┐┌─────────┐┌─────────┐
    «q_1: ┤1         ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├
    «     └──────────┘└─────────┘└─────────┘└─────────┘
    """

    import numpy as np
    from qiskit.circuit import Parameter, QuantumCircuit

    def rzx_zz3(theta: float = None):
        """Template for CX - RZGate - CX."""
        if theta is None:
            theta = Parameter("ϴ")

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rz(theta, 1)
        qc.cx(0, 1)
        qc.rz(-1 * theta, 1)
        # Hadamard
        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 2, 1)

        qc.rx(theta, 1)
        qc.rzx(-1 * theta, 0, 1)
        # Hadamard
        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 2, 1)

        return qc

    def_rzx_zz3 = rzx_zz3()
