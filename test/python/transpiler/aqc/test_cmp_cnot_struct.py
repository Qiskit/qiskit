# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# TODO: I guess we don't need these tests as they compare new vs old. If the new implementation is
#  correct then there's no reason to keep the tests.
"""
Compares old vs new implementation of CNOT layout and connectivity functions.
"""
# import os
# import sys
import traceback
import unittest
import numpy as np

from qiskit.test import QiskitTestCase

# TODO: remove print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

# if os.getcwd() not in sys.path:
#     sys.path.append(os.getcwd())
from qiskit.transpiler.synthesis.aqc.cnot_structures import (
    get_connectivity_types,
    get_network_layouts,
    get_connectivity,
    make_cnot_network,
)

# from compilers.aqc_rc1.legacy.CnotStructures import \
#     full_conn, line, star, sequ, spin, cart
# Avoid excessive deprecation warnings in Qiskit on Linux system.
# import warnings
#
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestCompareCNOTStructures(QiskitTestCase):
    def test_cmp_connectivity(self):
        """Tests the new connectivity graph generators against the old ones."""
        print("")
        for n in range(1, 51):
            print(".", end="", flush=True)
            if n > 1:
                self.assertEqual(get_connectivity(num_qubits=n, connectivity="full"), full_conn(n))
                self.assertEqual(get_connectivity(num_qubits=n, connectivity="line"), line(n))
                self.assertEqual(get_connectivity(num_qubits=n, connectivity="star"), star(n))
            else:
                self.assertEqual(len(get_connectivity(num_qubits=n, connectivity="full")), 1)
                self.assertEqual(len(get_connectivity(num_qubits=n, connectivity="line")), 1)
                self.assertEqual(len(get_connectivity(num_qubits=n, connectivity="star")), 1)
        print("")

    def test_cmp_qubit_network(self):
        """Tests generators of CNOT network."""
        # Note, when we set L=0, the default lower bound is selected, which is
        # huge for n >= 10. For this reason we avoid L=0 in case of large n.
        print("")
        for conn in get_connectivity_types():
            print("=" * 80)
            print("connectivity layout:", conn)

            # Test all networks except Cartan one.
            print("-" * 40)
            for layout in get_network_layouts():
                if layout == "cart":
                    continue
                print(layout)
                for n in range(2, 51):
                    print(".", end="", flush=True)
                    links = get_connectivity(num_qubits=n, connectivity=conn)
                    for L in range(0 if n < 10 else 1, 256):
                        qn = make_cnot_network(
                            num_qubits=n,
                            network_layout=layout,
                            connectivity_type=conn,
                            depth=L,
                            verbose=1,
                        )
                        if layout == "sequ":
                            self.assertTrue(np.all(qn == sequ(n=n, links=links, L=L)))
                        elif layout == "spin":
                            self.assertTrue(np.all(qn == spin(n=n, _links=links, L=L)))
                        else:
                            self.fail("unknown CNOT-network layout type")
                print("")

        # Test Cartan network.
        print("")
        print("-" * 40)
        for n in range(3, 15):
            print("'cart' network, nqubits: {:d}".format(n))
            qn = make_cnot_network(
                num_qubits=n, network_layout="cart", connectivity_type="full", depth=0, verbose=1
            )
            self.assertTrue(np.all(qn == cart(n)))


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
