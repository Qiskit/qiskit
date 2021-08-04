# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Testing inverse_cancellation
"""

from qiskit import QuantumCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import CXCancellation, Cancellation
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from numpy import pi

class TestCancellation(QiskitTestCase): 

    def test_inverse_cancellation(self):
        qc = QuantumCircuit(2,2)
        qc.u(pi/2, 0, pi, 0)
        qc.u(pi/2, 0, pi, 0)
        pass_ = Cancellation(["u"])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        cir_after = new_circ.count_ops()
        self.assertNotIn("u", cir_after)

    
    
