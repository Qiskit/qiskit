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

"""Configurable backend for BackendV2."""
import datetime
from typing import Optional, List, Tuple, Union, Dict

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.backend import BackendV2, QubitProperties
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties


class ConfigurableFakeBackendV2(BackendV2):
    """Factory for generating mock backendv2 instances"""

    def __init__(
        self,
        name: str,
        description: str,
        n_qubits: int,
        gate_configuration: Dict[Gate, List[Tuple[int]]],
        measurable_qubits: List[int] = None,
        parameterized_gates: Optional[Dict[Gate, str]] = None,
        qubit_coordinates: Optional[List[List[int]]] = None,
        qubit_t1: Optional[Union[float, List[float]]] = None,
        qubit_t2: Optional[Union[float, List[float]]] = None,
        qubit_frequency: Optional[Union[float, List[float]]] = None,
        qubit_readout_error: Optional[Union[float, List[float]]] = None,
        single_qubit_gates: Optional[List[str]] = None,
        dt: Optional[float] = None,
        std: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Creates backend based on provided configuration.

        Args:
            name: Name of the backend.
            description: An optional description of the backend
            n_qubits: Number of qubits
            gate_configuration: Basis gates of the backend. Pass a dictionary where the key is the reference to the Gate object (ie XGate) whose value is a list of tuples containing qubits where the gate may be placed.
            parameterized_gates: Dictionary with keys being Gate classes that take parameters and value contains string parameter name
            measurable_qubits: Optional specify which qubits can be measured, default is all.
            qubit_coordinates: Optional specification of grid for displaying gate maps.
            qubit_t1: Longitudinal coherence times.
            qubit_t2: Transverse coherence times.
            qubit_frequency: Frequency of qubits.
            qubit_readout_error: Readout error of qubits.
            single_qubit_gates: List of single qubit gates for backend properties.
            dt: Discretization of the input time sequences.
            std: Standard deviation of the generated distributions.
            seed: Random seed.
        """
        # harcoded values taken from FakeMumbaiV2 and ConfigurableBackend

        np.random.seed(seed)

        if std is None:
            std = 0.01

        if measurable_qubits is None:
            measurable_qubits = list(range(n_qubits))

        if not isinstance(qubit_t1, list):
            qubit_t1 = np.random.normal(
                loc=qubit_t1 or 113.0, scale=std, size=n_qubits
            ).tolist()

        if not isinstance(qubit_t2, list):
            qubit_t2 = np.random.normal(
                loc=qubit_t1 or 150.2, scale=std, size=n_qubits
            ).tolist()

        if not isinstance(qubit_frequency, list):
            qubit_frequency = np.random.normal(
                loc=qubit_frequency or 4.8, scale=std, size=n_qubits
            ).tolist()

        if not isinstance(qubit_readout_error, list):
            qubit_readout_error = np.random.normal(
                loc=qubit_readout_error or 0.04, scale=std, size=n_qubits
            ).tolist()

        if dt is None:
            dt = 0.2222222222222222e-9

        if not qubit_coordinates is None:
            self.qubit_coordinates = qubit_coordinates

        self.backend_name = name
        self.description = description
        self.gate_configuration = gate_configuration
        self.measurable_qubits = measurable_qubits
        self.qubit_t1 = qubit_t1
        self.qubit_t2 = qubit_t2
        self.qubit_frequency = qubit_frequency
        self.qubit_readout_error = qubit_readout_error
        self.single_qubit_gates = single_qubit_gates
        self.std = std
        # self.n_qubits = n_qubits

        super().__init__(
            name=self.backend_name,
            description=self.description,
            online_date=datetime.datetime.utcnow(),
            backend_version="0.0.0",
        )

        self._target = Target(dt=dt)

        # Add gates to target
        # TODO: dynamic instruction properties
        for gate, qubit_tuple_list in self.gate_configuration.items():
            temp_gate_props = {
                qubit_tuple: InstructionProperties(duration=0.0, error=0)
                for qubit_tuple in qubit_tuple_list
            }
            if gate in parameterized_gates.keys():
                self._target.add_instruction(
                    gate(Parameter(parameterized_gates[gate])), temp_gate_props
                )
            else:
                self._target.add_instruction(gate(), temp_gate_props)

        # Add reset to target
        reset_props = {
            (i,): InstructionProperties(duration=3676.4444444444443)
            for i in range(n_qubits)
        }
        self._target.add_instruction(Reset(), reset_props)

        # Add Measurement properties to target
        meas_props = {
            (i,): InstructionProperties(
                duration=3.552e-06, error=self.qubit_readout_error[i]
            )
            for i in self.measurable_qubits
        }
        self._target.add_instruction(Measure(), meas_props)

        # Save qubit properties as member variable
        self._qubit_properties = {
            i: QubitProperties(
                t1=self.qubit_t1[i],
                t2=self.qubit_t2[i],
                frequency=self.qubit_frequency[i],
            )
            for i in range(n_qubits)
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
