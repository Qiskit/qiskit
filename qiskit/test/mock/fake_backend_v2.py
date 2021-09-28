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

# pylint: disable=no-name-in-module,import-error

"""Mock BackendV2 object without run implemented for testing backwards compat"""


import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.measure import Measure
from qiskit.circuit.library.standard_gates import CXGate, UGate, ECRGate, RXGate
from qiskit.providers.backend import BackendV2
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties


class FakeBackendV2(BackendV2):
    """A mock backend that doesn't implement run() to test compatibility with Terra internals."""

    def __init__(self):
        super().__init__(None)
        self._target = Target()
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lambda")
        rx_props = {
            (0,): InstructionProperties(length=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(length=4.52e-8, error=0.00032115),
        }
        self._target.add_instruction(RXGate(theta), [(0,), (1,)], properties=rx_props)
        rx_30_props = {
            (0,): InstructionProperties(length=1.23e-8, error=0.00018115),
            (1,): InstructionProperties(length=1.52e-8, error=0.00012115),
        }
        self._target.add_instruction(
            RXGate(np.pi / 6), [(0,), (1,)], name="rx_30", properties=rx_30_props
        )
        u_props = {
            (0,): InstructionProperties(length=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(length=4.52e-8, error=0.00032115),
        }
        self._target.add_instruction(UGate(theta, phi, lam), [(0,), (1,)], properties=u_props)
        cx_props = {
            (0, 1): InstructionProperties(length=5.23e-7, error=0.00098115),
            (1, 0): InstructionProperties(length=4.52e-7, error=0.00132115),
        }
        self._target.add_instruction(CXGate(), [(0, 1), (1, 0)], properties=cx_props)
        measure_props = {
            (0,): InstructionProperties(length=6e-6, error=5e-6),
            (1,): InstructionProperties(length=1e-6, error=9e-6),
        }
        self._target.add_instruction(Measure(), [(0,), (1,)], properties=measure_props)
        ecr_props = {
            (1, 0): InstructionProperties(length=4.52e-9, error=0.0000132115),
        }
        self._target.add_instruction(ECRGate(), [(1, 0)], properties=ecr_props)

    @property
    def target(self):
        return self._target

    @property
    def conditional(self):
        return False

    @property
    def max_shots(self):
        return None

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, run_input, **options):
        raise NotImplementedError
