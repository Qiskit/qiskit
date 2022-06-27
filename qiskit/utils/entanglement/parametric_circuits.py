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

"""Parametric Circuit Module"""

from typing import Any
import numpy as np


class Ansatz:

    """
    This class creates methods to create the parameteric circuits
    mentioned in https://doi.org/10.1007/s42484-021-00038-w with random
    parameters as initialization routine for testing the experessibility
    and entangling capacity of a given parametric circuit.

    """

    def __init__(self, repitition, feature_dim, circuit_id) -> None:

        """
        Args:
            repitition: The no of repitition of the layers of parametric circuit
            feature_dim: The no. of qubits required for a the parametric circuit
            circuit_id: The id of the circuit in the order mentioned in
                        https://doi.org/10.1007/s42484-021-00038-w to get the
                        particular circuit.
        """
        self.repitition = repitition
        self.feature_dim = feature_dim
        self.circuit_id = circuit_id

    def get_circ_1(self) -> Any:
        """
        Returns:
            circ: Circuit 1 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import QuantumCircuit

        num_par = 2 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_par)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[i + 1], i)
                arg_count += 2
            circ.barrier()
        return circ

    def get_circ_2(self) -> Any:
        """
        Returns:
            circ: Circuit 2 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import QuantumCircuit

        num_par = 2 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_par)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0

        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[i + 1], i)
                arg_count += 2
            for i in reversed(range(circ.num_qubits - 1)):
                circ.cx(i + 1, i)
            circ.barrier()
        return circ

    def get_circ_3(self) -> Any:
        """
        Returns:
            circ: Circuit 3 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import QuantumCircuit

        num_par = (2 * self.feature_dim * self.repitition) + (
            self.feature_dim - 1
        ) * self.repitition
        paravec = np.random.randn(num_par)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0

        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[i + 1], i)
                arg_count += 2
            for i in reversed(range(circ.num_qubits - 1)):
                circ.crz(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()
        return circ

    def get_circ_4(self) -> Any:
        """
        Returns:
            circ: Circuit 4 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import QuantumCircuit

        num_par = (2 * self.feature_dim * self.repitition) + (
            self.feature_dim - 1
        ) * self.repitition
        paravec = np.random.randn(num_par)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0

        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[i + 1], i)
                arg_count += 2
            for i in reversed(range(circ.num_qubits - 1)):
                circ.crx(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()
        return circ

    def get_circ_5(self) -> Any:
        """
        Returns:
            ansatz: Circuit 5 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit.circuit.library import NLocal
        from qiskit import QuantumCircuit

        paravec = np.random.randn(2)
        paravec_2 = np.random.randn(self.feature_dim * (self.feature_dim - 1) // 2)
        blocks = []
        for block in reversed(range(self.feature_dim)):
            block_circ = QuantumCircuit(self.feature_dim)
            p_pointer = 0
            for i in reversed(range(self.feature_dim)):
                if i != block:
                    block_circ.crz(paravec_2[p_pointer], block, i)
                    p_pointer += 1
            blocks.append(block_circ)
        rot_layer = QuantumCircuit(1)
        rot_layer.rx(paravec[0], 0)
        rot_layer.rz(paravec[1], 0)
        ansatz = NLocal(
            self.feature_dim,
            rotation_blocks=rot_layer,
            entanglement_blocks=blocks,
            reps=self.repitition,
        )
        return ansatz

    def get_circ_6(self) -> Any:
        """
        Returns:
            ansatz: Circuit 6 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit.circuit.library import NLocal
        from qiskit import QuantumCircuit

        paravec = np.random.randn(2)
        paravec_2 = np.random.randn(self.feature_dim * (self.feature_dim - 1) // 2)
        blocks = []
        for block in reversed(range(self.feature_dim)):
            block_circ = QuantumCircuit(self.feature_dim)
            p_pointer = 0
            for i in reversed(range(self.feature_dim)):
                if i != block:
                    block_circ.crx(paravec_2[p_pointer], block, i)
                    p_pointer += 1
            blocks.append(block_circ)
        rot_layer = QuantumCircuit(1)
        rot_layer.rx(paravec[0], 0)
        rot_layer.rz(paravec[1], 0)
        ansatz = NLocal(
            self.feature_dim,
            rotation_blocks=rot_layer,
            entanglement_blocks=blocks,
            reps=self.repitition,
        )
        return ansatz

    def get_circ_7(self) -> Any:
        """
        Returns:
            circ: Circuit 7 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_par = (4 * self.feature_dim + self.feature_dim) * self.repitition
        paravec = np.random.randn(num_par)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[1 + i], i)
                arg_count += 2

            for i in range(0, circ.num_qubits - 1, 2):
                circ.crz(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(1, circ.num_qubits - 1, 2):
                circ.crz(paravec[arg_count], i + 1, i)
                arg_count += 1

        return circ

    def get_circ_8(self) -> Any:
        """
        Returns:
            circ: Circuit 8 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_par = (4 * self.feature_dim + self.feature_dim) * self.repitition
        paravec = np.random.randn(num_par)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(0, circ.num_qubits - 1, 2):
                circ.crx(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(1, circ.num_qubits - 1, 2):
                circ.crx(paravec[arg_count], i + 1, i)
                arg_count += 1

        return circ

    def get_circ_9(self) -> Any:
        """
        Returns:
            circ: Circuit 9 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = self.feature_dim * self.repitition
        paravec = np.random.randn(num_pars)
        arg_count = 0
        circ = QuantumCircuit(self.feature_dim)
        for _ in range(self.repitition):
            for i in range(self.feature_dim):
                circ.h(i)
            for i in reversed(range(self.feature_dim - 1)):
                circ.cz(i + 1, i)
            for i in range(self.feature_dim):
                circ.rx(paravec[arg_count], i)
                arg_count += 1

        return circ

    def get_circ_10(self) -> Any:
        """
        Returns:
            circ: Circuit 10 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = self.feature_dim * self.repitition + self.feature_dim
        paravec = np.random.randn(num_pars)
        arg_count = 0
        circ = QuantumCircuit(self.feature_dim)
        for i in range(self.feature_dim):
            circ.ry(paravec[i], i)
        arg_count += self.feature_dim

        for _ in range(self.repitition):
            for i in range(self.feature_dim):
                circ.h(i)
            for i in range(self.feature_dim - 1):
                circ.cz(i, i + 1)
            for i in range(self.feature_dim):
                circ.rx(paravec[arg_count], i)
                arg_count += 1

        return circ

    def get_circ_11(self) -> Any:
        """
        Returns:
            circ: Circuit 11 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = (
            (2 * self.feature_dim + int(self.feature_dim / 2 - 1) * 4) * self.repitition
            if self.feature_dim % 2 == 0
            else (4 * self.feature_dim - 2) * self.repitition
        )
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(0, circ.num_qubits - 1, 2):
                circ.cx(i + 1, i)
            circ.barrier()

            for i in range(1, circ.num_qubits - 1, 2):
                circ.ry(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                circ.ry(paravec[arg_count + 2], i + 1)
                circ.rz(paravec[arg_count + 3], i + 1)
                arg_count += 4

            for i in range(1, circ.num_qubits - 1, 2):
                circ.cx(i + 1, i)

        return circ

    def get_circ_12(self) -> Any:
        """
        Returns:
            circ: Circuit 12 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = (
            (2 * self.feature_dim + int(self.feature_dim / 2 - 1) * 4) * self.repitition
            if self.feature_dim % 2 == 0
            else (4 * self.feature_dim - 2) * self.repitition
        )
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(0, circ.num_qubits - 1, 2):
                circ.cz(i + 1, i)
            circ.barrier()

            for i in range(1, circ.num_qubits - 1, 2):
                circ.ry(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                circ.ry(paravec[arg_count + 2], i + 1)
                circ.rz(paravec[arg_count + 3], i + 1)
                arg_count += 4

            for i in range(1, circ.num_qubits - 1, 2):
                circ.cz(i + 1, i)
            circ.barrier()

        return circ

    def get_circ_13(self) -> Any:
        """
        Returns:
            circ: Circuit 13 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import QuantumCircuit

        num_pars = 4 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)

        arg_count = 0

        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.ry(paravec[i], i)
                arg_count += 1

            circ.crz(paravec[arg_count], circ.num_qubits - 1, 0)
            arg_count += 1
            for i in range(circ.num_qubits - 1):
                circ.crz(paravec[arg_count], i, i + 1)
                arg_count += 1

            for i in range(circ.num_qubits):
                circ.ry(paravec[arg_count], i)
                arg_count += 1

            circ.crz(paravec[arg_count], circ.num_qubits - 1, circ.num_qubits - 2)
            arg_count += 1
            circ.crz(paravec[arg_count], 0, circ.num_qubits - 1)
            arg_count += 1

            for i in range(circ.num_qubits - 2):
                circ.crz(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()

        return circ

    def get_circ_14(self) -> Any:
        """
        Returns:
            circ: Circuit 14 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = 4 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)

        arg_count = 0

        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.ry(paravec[i], i)
                arg_count += 1

            circ.crx(paravec[arg_count], circ.num_qubits - 1, 0)
            arg_count += 1
            for i in range(circ.num_qubits - 1):
                circ.crx(paravec[arg_count], i, i + 1)
                arg_count += 1

            for i in range(circ.num_qubits):
                circ.ry(paravec[arg_count], i)
                arg_count += 1

            circ.crx(paravec[arg_count], circ.num_qubits - 1, circ.num_qubits - 2)
            arg_count += 1
            circ.crx(paravec[arg_count], 0, circ.num_qubits - 1)
            arg_count += 1

            for i in range(circ.num_qubits - 2):
                circ.crx(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()

        return circ

    def get_circ_15(self) -> Any:
        """
        Returns:
            circ: Circuit 15 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = 2 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)

        arg_count = 0

        for _ in range(self.repitition):
            for i in range(circ.num_qubits):
                circ.ry(paravec[i], i)
                arg_count += 1

            circ.cx(circ.num_qubits - 1, 0)

            for i in range(circ.num_qubits - 1):
                circ.cx(i, i + 1)

            for i in range(circ.num_qubits):
                circ.ry(paravec[arg_count], i)
                arg_count += 1

            circ.cx(circ.num_qubits - 1, circ.num_qubits - 2)

            circ.cx(0, circ.num_qubits - 1)

            for i in range(circ.num_qubits - 2):
                circ.cx(i + 1, i)

            circ.barrier()

        return circ

    def get_circ_16(self) -> Any:
        """
        Returns:
            circ: Circuit 16 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = (3 * self.feature_dim - 1) * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(0, circ.num_qubits - 1, 2):
                circ.crz(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()

            for i in range(1, circ.num_qubits - 1, 2):
                circ.crz(paravec[arg_count], i + 1, i)
                arg_count += 1

            circ.barrier()

        return circ

    def get_circ_17(self) -> Any:
        """
        Returns:
            circ: Circuit 17 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = (3 * self.feature_dim - 1) * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)
        arg_count = 0
        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.rz(paravec[arg_count + 1], i)
                arg_count += 2

            for i in range(0, circ.num_qubits - 1, 2):
                circ.crx(paravec[arg_count], i + 1, i)
                arg_count += 1
            circ.barrier()

            for i in range(1, circ.num_qubits - 1, 2):
                circ.crx(paravec[arg_count], i + 1, i)
                arg_count += 1

            circ.barrier()

        return circ

    def get_circ_18(self) -> Any:
        """
        Returns:
            circ: Circuit 18 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = 3 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)

        arg_count = 0

        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[i], i)
                arg_count += 2

            circ.crz(paravec[arg_count], circ.num_qubits - 1, 0)
            arg_count += 1

            for i in range(circ.num_qubits - 1):
                circ.crz(paravec[arg_count], i, i + 1)
                arg_count += 1
            circ.barrier()

        return circ

    def get_circ_19(self) -> Any:
        """
        Returns:
                circ: Circuit 9 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit

        from qiskit import QuantumCircuit

        num_pars = 3 * self.feature_dim * self.repitition
        paravec = np.random.randn(num_pars)
        circ = QuantumCircuit(self.feature_dim)

        arg_count = 0

        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.rz(paravec[i], i)
                arg_count += 2

            circ.crx(paravec[arg_count], circ.num_qubits - 1, 0)
            arg_count += 1

            for i in range(circ.num_qubits - 1):
                circ.crx(paravec[arg_count], i, i + 1)
                arg_count += 1
            circ.barrier()

        return circ

    def get_ansatz(self) -> Any:
        """
        Returns:
                ansatzes: Circuits mentioned in  https://doi.org/10.1007/s42484-021-00038-w by
                        declaring the particular circuit
                        identity in the class variables e.g. circuit_id = 1 will return
                        the 1st circuit mentioned in
                        https://doi.org/10.1007/s42484-021-00038-w
        """

        ansatzes = {
            1: self.get_circ_1(),
            2: self.get_circ_2(),
            3: self.get_circ_3(),
            4: self.get_circ_4(),
            5: self.get_circ_5(),
            6: self.get_circ_6(),
            7: self.get_circ_7(),
            8: self.get_circ_8(),
            9: self.get_circ_9(),
            10: self.get_circ_10(),
            11: self.get_circ_11(),
            12: self.get_circ_12(),
            13: self.get_circ_13(),
            14: self.get_circ_14(),
            15: self.get_circ_15(),
            16: self.get_circ_16(),
            17: self.get_circ_17(),
            18: self.get_circ_18(),
            19: self.get_circ_19(),
        }

        return ansatzes[self.circuit_id]
