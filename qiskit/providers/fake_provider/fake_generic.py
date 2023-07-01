# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import statistics
from typing import Optional, List, Tuple

from qiskit.transpiler import CouplingMap
from qiskit.providers.basicaer import BasicAer
from qiskit.transpiler import Target, InstructionProperties, QubitProperties
from qiskit.providers.backend import BackendV2
from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import XGate, RZGate, SXGate, CXGate, ECRGate, IGate
from qiskit.circuit import Measure, Parameter, Delay, Reset
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    BreakLoopOp,
    ContinueLoopOp,
)


class FakeGeneric(BackendV2):
    """
    Generate a generic fake backend, this backend will have properties and configuration according to the settings passed in the argument.

    Arguments:
        num_qubits:
                    Pass in the integer which is the number of qubits of the backend.
                    Example: num_qubits = 19

        coupling_map:
                        Pass in the coupling Map of the backend as a list of tuples.
                        Example: [(1, 2), (2, 3), (3, 4), (4, 5)].

                        If None passed then the coupling map will be generated.
                        This map will be in accordance with the argument coupling_map_type.

        coupling_map_type:
                            Pass in the type of coupling map to be generated. If coupling map is passed then this option will be overriden.
                            Valid types of coupling map: 'grid', 'heavy_hex'

                            Heavy Hex Lattice Reference:
                                                    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.011022


        basis_gates:
                        Pass in the basis gates of the backend as list of strings.
                        Example: ['cx', 'id', 'rz', 'sx', 'x']  --> This is the default basis gates of the backend.

        dynamic:
                    Enable/Disable dynamic circuits on this backend.
                    True: Enable
                    False: Disable (Default)

        bidirectional_cp_mp:
                             Enable/Disable bi-directional coupling map.
                             True: Enable
                             False: Disable (Default)
        replace_cx_with_ecr:
                    True: (Default) Replace every occurance of 'cx' with 'ecr'
                    False: Do not replace 'cx' with 'ecr'

        enable_reset:
                    True: (Default) this enables the reset on the backend
                    False: This disables the reset on the backend

        dt:
            The system time resolution of input signals in seconds.
            Default is 0.2222ns



    Returns:
            None

    Raises:
            QiskitError: If argument basis_gates has a gate which is not a valid basis gate.


    """

    def __init__(
        self,
        num_qubits: int,
        coupling_map: Optional[List[Tuple[str, str]]] = None,
        coupling_map_type: Optional[str] = "grid",
        basis_gates: List[str] = ["cx", "id", "rz", "sx", "x"],
        dynamic: bool = False,
        bidirectional_cp_mp: bool = False,
        replace_cx_with_ecr: bool = True,
        enable_reset: bool = True,
        dt: float = 0.222e-9,
    ):

        super().__init__(
            provider=None,
            name="fake_generic",
            description=f"This {num_qubits} qubit fake generic device, with generic settings has been generated right now!",
            backend_version="",
        )

        self.basis_gates = basis_gates
        self.__rng = np.random.default_rng(seed=123456789123456)
        self.__coupling_map_type = coupling_map_type
        if replace_cx_with_ecr:
            self.basis_gates = list(map(lambda gate: gate.replace("cx", "ecr"), basis_gates))

        if "delay" not in basis_gates:
            self.basis_gates.append("delay")
        if "measure" not in basis_gates:
            self.basis_gates.append("measure")

        if not coupling_map:
            if self.__coupling_map_type == "heavy_hex":
                distance = self._get_cmap_args(num_qubits=num_qubits)
                coupling_map = CouplingMap().from_heavy_hex(
                    distance=distance, bidirectional=bidirectional_cp_mp
                )

            elif self.__coupling_map_type == "grid":
                num_rows, num_columns = self._get_cmap_args(num_qubits=num_qubits)
                coupling_map = CouplingMap().from_grid(
                    num_rows=num_rows, num_columns=num_columns, bidirectional=bidirectional_cp_mp
                )
        else:
            coupling_map = CouplingMap(coupling_map)

        num_qubits = coupling_map.size()

        self._target = Target(
            description="Fake Generic Backend",
            num_qubits=num_qubits,
            dt=dt,
            qubit_properties=[
                QubitProperties(
                    t1=self.__rng.uniform(100e-6, 200e-6),
                    t2=self.__rng.uniform(100e-6, 200e-6),
                    frequency=self.__rng.uniform(5e9, 5.5e9),
                )
                for _ in range(num_qubits)
            ],
        )

        instruction_dict = self._get_instruction_dict(num_qubits, coupling_map)
        for gate in self.basis_gates:
            try:
                self._target.add_instruction(*instruction_dict[gate])
            except:
                raise QiskitError(f"{gate} is not a valid basis gate")

        if dynamic:
            self._target.add_instruction(IfElseOp, name="if_else")
            self._target.add_instruction(WhileLoopOp, name="while_loop")
            self._target.add_instruction(ForLoopOp, name="for_loop")
            self._target.add_instruction(SwitchCaseOp, name="switch_case")
            self._target.add_instruction(BreakLoopOp, name="break")
            self._target.add_instruction(ContinueLoopOp, name="continue")

        if enable_reset:
            self._target.add_instruction(
                Reset(), {(qubit_idx,): None for qubit_idx in range(num_qubits)}
            )

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    def _get_cmap_args(self, num_qubits):
        if self.__coupling_map_type == "heavy_hex":
            for d in range(3, 20, 2):
                # The description of the formula: 5*d**2 - 2*d -1 is explained in
                # https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.011022 Page 011022-4
                n = (5 * (d**2) - (2 * d) - 1) / 2
                if n >= num_qubits:
                    return int(d)

        elif self.__coupling_map_type == "grid":
            factors = [x for x in range(2, num_qubits + 1) if num_qubits % x == 0]
            first_factor = statistics.median_high(factors)
            second_factor = int(num_qubits / first_factor)
            return (first_factor, second_factor)

    def _get_instruction_dict(self, num_qubits, coupling_map):
        instruction_dict = {
            "ecr": (
                ECRGate(),
                {
                    edge: InstructionProperties(
                        error=self.__rng.uniform(1e-5, 5e-3),
                        duration=self.__rng.uniform(1e-8, 9e-7),
                    )
                    for edge in coupling_map
                },
            ),
            "cx": (
                CXGate(),
                {
                    edge: InstructionProperties(
                        error=self.__rng.uniform(1e-5, 5e-3),
                        duration=self.__rng.uniform(1e-8, 9e-7),
                    )
                    for edge in coupling_map
                },
            ),
            "id": (
                IGate(),
                {
                    (qubit_idx,): InstructionProperties(error=0.0, duration=0.0)
                    for qubit_idx in range(num_qubits)
                },
            ),
            "rz": (
                RZGate(Parameter("theta")),
                {
                    (qubit_idx,): InstructionProperties(error=0.0, duration=0.0)
                    for qubit_idx in range(num_qubits)
                },
            ),
            "sx": (
                SXGate(),
                {
                    (qubit_idx,): InstructionProperties(
                        error=self.__rng.uniform(1e-6, 1e-4),
                        duration=self.__rng.uniform(1e-8, 9e-7),
                    )
                    for qubit_idx in range(num_qubits)
                },
            ),
            "x": (
                XGate(),
                {
                    (qubit_idx,): InstructionProperties(
                        error=self.__rng.uniform(1e-6, 1e-4),
                        duration=self.__rng.uniform(1e-8, 9e-7),
                    )
                    for qubit_idx in range(num_qubits)
                },
            ),
            "measure": (
                Measure(),
                {
                    (qubit_idx,): InstructionProperties(
                        error=self.__rng.uniform(1e-3, 1e-1),
                        duration=self.__rng.uniform(1e-8, 9e-7),
                    )
                    for qubit_idx in range(num_qubits)
                },
            ),
            "delay": (
                Delay(Parameter("Time")),
                {(qubit_idx,): None for qubit_idx in range(num_qubits)},
            ),
        }
        return instruction_dict

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuit, **kwargs):
        noise_model = None
        return BasicAer.get_backend("qasm_simulator").run(circuit, **kwargs)
