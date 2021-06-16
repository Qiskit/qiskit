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

"""
Plots CNOT-structures.
"""
print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import argparse
import os
import sys
import traceback

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from pprint import pprint
from typing import Any
import numpy as np
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit
from qiskit.transpiler.synthesis.aqc.cnot_structures import (
    get_network_layouts,
    get_connectivity_types,
)

# Avoid excessive deprecation warnings in Qiskit on Linux system.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _Main(cmd_args: Any):
    """
    Entry point of this test.
    """
    nqubits = cmd_args.nqubits
    depth = cmd_args.depth
    if depth <= 0:
        depth = 3 * nqubits
    for layout in get_network_layouts():
        for conn in get_connectivity_types():
            # Skip some infeasible or undesired structures.
            if layout == "cyclic_line" and conn != "line":
                continue
            if layout == "cyclic_spin" and conn != "full":
                continue
            if layout == "cart":
                print("WARNING: skipping Cartan layout")
                continue
            print("\n\nLayout: {:s}, connectivity: {:s}".format(layout, conn))

            circuit = ParametricCircuit(
                num_qubits=nqubits, layout=layout, connectivity=conn, depth=depth
            )
            print(circuit.to_qiskit().draw())


def _CommandLineArguments() -> Any:
    """
    Parses and returns the command-line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--nqubits", type=int, required=True, help="number of qubits")
    parser.add_argument(
        "-d", "--depth", type=int, default=0, help="circuit depth, default is 0 = auto"
    )
    cmd_args = parser.parse_args()
    print("\nInput command-line arguments:")
    pprint(cmd_args)
    print("-" * 80)
    return cmd_args


if __name__ == "__main__":
    print("\nI M P O R T A N T: run this script from the project root folder.")
    np.set_printoptions(precision=6, linewidth=256)
    try:
        _Main(_CommandLineArguments())
        print("\nfinished normally")
    except Exception as glo_ex:
        print("message length:", len(str(glo_ex)))
        traceback.print_exc()
    finally:
        print("\n\n\n")
