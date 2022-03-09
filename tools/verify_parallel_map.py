#!/usr/bin/env python
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

# pylint: disable=wrong-import-position

"""Test script to verify parallel dispatch via parallel_map() works as expected."""


import math
import os


ORIG_ENV_VAR = os.getenv("QISKIT_PARALLEL", None)
if ORIG_ENV_VAR is not None:
    print("Removing QISKIT_PARALLEL env var to verify defaults")
    del os.environ["QISKIT_PARALLEL"]


from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.test.mock import FakeRueschlikon


def run_test():
    """Run tests."""
    backend = FakeRueschlikon()
    qr = QuantumRegister(16)
    cr = ClassicalRegister(16)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    for k in range(1, 15):
        qc.cx(qr[0], qr[k])
    qc.measure(qr, cr)
    qlist = [qc for k in range(15)]
    for opt_level in [0, 1, 2, 3]:
        tqc = transpile(
            qlist, backend=backend, optimization_level=opt_level, seed_transpiler=424242
        )
        result = backend.run(tqc, seed_simulator=4242424242, shots=1000).result()
        counts = result.get_counts()
        for count in counts:
            assert math.isclose(count["0000000000000000"], 500, rel_tol=0.1)
            assert math.isclose(count["0111111111111111"], 500, rel_tol=0.1)


if __name__ == "__main__":
    run_test()
    if ORIG_ENV_VAR is not None:
        print(f"Restoring QISKIT_PARALLEL env var to {ORIG_ENV_VAR}")
        os.environ["QISKIT_PARALLEL"] = ORIG_ENV_VAR
