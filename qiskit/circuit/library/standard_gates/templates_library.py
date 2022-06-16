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
from qiskit.qasm import pi
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.synthesis.ion_decompose import cnot_rxx_decompose

from . import (
    clifford_2_1,
    clifford_2_2,
    clifford_2_3,
    clifford_2_4,
    clifford_3_1,
    clifford_4_1,
    clifford_4_2,
    clifford_4_3,
    clifford_4_4,
    clifford_5_1,
    clifford_6_2,
    clifford_6_3,
    clifford_6_4,
    clifford_6_5,
    clifford_8_1,
    clifford_8_2,
    clifford_8_3,


)

_sel = StandardTemplateLibrary = TemplateLibrary()

#Clifford templates

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

_sel.add_template(clifford_2_1(), def_Clifford2_1)



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

_sel.add_template(clifford_2_2(), def_Clifford2_2)



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

_sel.add_template(clifford_2_3(), def_Clifford2_3)



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

_sel.add_template(clifford_2_4(), def_Clifford2_4)



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

_sel.add_template(clifford_3_1(), def_Clifford3_1)



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

_sel.add_template(clifford_4_1(), def_Clifford4_1)



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

_sel.add_template(clifford_4_2(), def_Clifford4_2)



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

_sel.add_template(clifford_4_3(), def_Clifford4_3)



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

_sel.add_template(clifford_4_4(), def_Clifford4_4)


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

_sel.add_template(clifford_5_1(), def_Clifford5_1)



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

_sel.add_template(clifford_6_2(), def_Clifford6_2)



"""
Clifford template 6_3:
.. parsed-literal::

                   ┌───┐     ┌───┐
        q_0: ─X──■─┤ H ├──■──┤ X ├─────
              │  │ └───┘┌─┴─┐└─┬─┘┌───┐
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

_sel.add_template(clifford_6_3(), def_Clifford6_3)



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

_sel.add_template(clifford_6_4(), def_Clifford6_4)


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

_sel.add_template(clifford_6_5(), def_Clifford6_5)



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

_sel.add_template(clifford_8_1(), def_Clifford8_1)



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

_sel.add_template(clifford_8_2(), def_Clifford8_2)

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

_sel.add_template(clifford_8_3(), def_Clifford8_3)



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

_sel.add_template(template_nct_2a_1(), def_Template_nct_2a_1)



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

_sel.add_template(template_nct_2a_2(), def_Template_nct_2a_2)



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

_sel.add_template(template_nct_2a_3(), def_Template_nct_2a_3)



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

_sel.add_template(template_nct_4a_1(), def_Template_nct_4a_1)



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

_sel.add_template(template_nct_4a_2(), def_Template_nct_4a_2)



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

_sel.add_template(template_nct_4a_3(), def_Template_nct_4a_3)



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

_sel.add_template(template_nct_4b_1(), def_Template_nct_4b_1)



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

_sel.add_template(template_nct_4b_2(), def_Template_nct_4b_2)



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

_sel.add_template(template_nct_5a_1(), def_Template_nct_5a_1)



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

_sel.add_template(template_nct_5a_2(), def_Template_nct_5a_2)

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

_sel.add_template(template_nct_5a_3(), def_Template_nct_5a_3)



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

_sel.add_template(template_nct_5a_4(), def_Template_nct_5a_4)



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

_sel.add_template(template_nct_6a_1(), def_Template_nct_6a_1)



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

_sel.add_template(template_nct_6a_2(), def_Template_nct_6a_2)



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

_sel.add_template(template_nct_6a_3(), def_Template_nct_6a_3)

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

_sel.add_template(template_nct_6a_4(), def_Template_nct_6a_4)



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

_sel.add_template(template_nct_6b_1(), def_Template_nct_6b_1)



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

_sel.add_template(template_nct_6b_2(), def_Template_nct_6b_2)



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

_sel.add_template(template_nct_6c_1(), def_Template_nct_6c_1)



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

_sel.add_template(template_nct_7a_1(), def_Template_nct_7a_1)