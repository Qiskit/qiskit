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

"""Test swap network."""

from qiskit.circuit.library import CXGate, CZGate, SwapNetwork, iSwapGate
from qiskit.test.base import QiskitTestCase


class TestSwapNetwork(QiskitTestCase):
    """Test swap network."""

    def test_swap_network(self):
        # pylint: disable=trailing-whitespace
        """Test swap network."""
        swap_network = SwapNetwork(5, register_name="q0")
        diagram = swap_network.draw()
        self.assertEqual(
            str(diagram).strip(),
            """
q0_0: ─X─────X─────X─
       │     │     │ 
q0_1: ─X──X──X──X──X─
          │     │    
q0_2: ─X──X──X──X──X─
       │     │     │ 
q0_3: ─X──X──X──X──X─
          │     │    
q0_4: ────X─────X────
""".strip(),
        )

        swap_network = SwapNetwork(5, offset=True, register_name="q0")
        diagram = swap_network.draw()
        self.assertEqual(
            str(diagram).strip(),
            """
q0_0: ────X─────X────
          │     │    
q0_1: ─X──X──X──X──X─
       │     │     │ 
q0_2: ─X──X──X──X──X─
          │     │    
q0_3: ─X──X──X──X──X─
       │     │     │ 
q0_4: ─X─────X─────X─
""".strip(),
        )

        swap_network = SwapNetwork(
            4,
            operation=lambda i, j: CXGate(),
            register_name="q0",
        )
        diagram = swap_network.draw()
        self.assertEqual(
            str(diagram).strip(),
            """
q0_0: ──■───X───────────■───X─────────
      ┌─┴─┐ │         ┌─┴─┐ │         
q0_1: ┤ X ├─X───■───X─┤ X ├─X───■───X─
      └───┘   ┌─┴─┐ │ └───┘   ┌─┴─┐ │ 
q0_2: ──■───X─┤ X ├─X───■───X─┤ X ├─X─
      ┌─┴─┐ │ └───┘   ┌─┴─┐ │ └───┘   
q0_3: ┤ X ├─X─────────┤ X ├─X─────────
      └───┘           └───┘           
""".strip(),
        )

        swap_network = SwapNetwork(
            4,
            operation=lambda i, j: CZGate(),
            swap_gate=iSwapGate(),
            offset=True,
            register_name="q0",
        )
        diagram = swap_network.draw()
        self.assertEqual(
            str(diagram).strip(),
            """
                      ┌────────┐                ┌────────┐
q0_0: ──────────────■─┤0       ├──────────────■─┤0       ├
         ┌────────┐ │ │  Iswap │   ┌────────┐ │ │  Iswap │
q0_1: ─■─┤0       ├─■─┤1       ├─■─┤0       ├─■─┤1       ├
       │ │  Iswap │   ├────────┤ │ │  Iswap │   ├────────┤
q0_2: ─■─┤1       ├─■─┤0       ├─■─┤1       ├─■─┤0       ├
         └────────┘ │ │  Iswap │   └────────┘ │ │  Iswap │
q0_3: ──────────────■─┤1       ├──────────────■─┤1       ├
                      └────────┘                └────────┘
""".strip(),
        )
