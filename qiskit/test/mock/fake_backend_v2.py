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

import datetime

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.measure import Measure
from qiskit.circuit.library.standard_gates import CXGate, UGate, ECRGate, RXGate
from qiskit.providers.backend import BackendV2, QubitProperties
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties


class FakeBackendV2(BackendV2):
    """A mock backend that doesn't implement run() to test compatibility with Terra internals."""

    def __init__(self):
        super().__init__(
            None,
            name="FakeV2",
            description="A fake BackendV2 example",
            online_date=datetime.datetime.utcnow(),
            backend_version="0.0.1",
        )
        self._target = Target()
        self._theta = Parameter("theta")
        self._phi = Parameter("phi")
        self._lam = Parameter("lambda")
        rx_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(duration=4.52e-8, error=0.00032115),
        }
        self._target.add_instruction(RXGate(self._theta), rx_props)
        rx_30_props = {
            (0,): InstructionProperties(duration=1.23e-8, error=0.00018115),
            (1,): InstructionProperties(duration=1.52e-8, error=0.00012115),
        }
        self._target.add_instruction(RXGate(np.pi / 6), rx_30_props, name="rx_30")
        u_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(duration=4.52e-8, error=0.00032115),
        }
        self._target.add_instruction(UGate(self._theta, self._phi, self._lam), u_props)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (1, 0): InstructionProperties(duration=4.52e-7, error=0.00132115),
        }
        self._target.add_instruction(CXGate(), cx_props)
        measure_props = {
            (0,): InstructionProperties(duration=6e-6, error=5e-6),
            (1,): InstructionProperties(duration=1e-6, error=9e-6),
        }
        self._target.add_instruction(Measure(), measure_props)
        ecr_props = {
            (1, 0): InstructionProperties(duration=4.52e-9, error=0.0000132115),
        }
        self._target.add_instruction(ECRGate(), ecr_props)
        self.options.set_validator("shots", (1, 4096))
        self._qubit_properties = {
            0: QubitProperties(t1=63.48783e-6, t2=112.23246e-6, frequency=5.17538e9),
            1: QubitProperties(t1=73.09352e-6, t2=126.83382e-6, frequency=5.26722e9),
        }

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, run_input, **options):
        raise NotImplementedError

    def qubit_properties(self, qubit):
        if isinstance(qubit, int):
            return self._qubit_properties[qubit]
        return [self._qubit_properties[i] for i in qubit]
