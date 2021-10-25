from typing import List, Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class AnsatzGenerator():
    """
    Generate MRF Ansatz
    """

    def __init__(self):
        return

    def get_ansatz0(self,
                    clique_structure: List[List],
                    n: int
                    ) -> [QuantumCircuit, List[Parameter]]:
        """

        Args:
            clique_structure: Clique structure given as list
            n: Number of nodes/qubits

        Returns: MRF Ansatz quantum circuit, Ansatz parameters

        """
        qr = QuantumRegister(n)
        ansatz = QuantumCircuit(qr)
        params = []
        params_index = 0
        for clique in clique_structure:
            for i in clique:
                for j in clique:
                    if j == i:
                        continue
                    else:
                        param0 = Parameter(str(params_index))
                        param1 = Parameter(str(params_index+1))
                        param2 = Parameter(str(params_index+2))
                        param3 = Parameter(str(params_index+3))
                        param4 = Parameter(str(params_index + 4))
                        params_index += 5
                        params.extend([param0, param1, param2, param3, param4])

                        ansatz.ry(param0, qr[i])
                        ansatz.ry(param1, qr[j])
                        ansatz.cry(param4, qr[i], qr[j])
                        ansatz.ry(param2, qr[i])
                        ansatz.ry(param3, qr[j])

        return ansatz, params

    def get_ansatz1(self,
                    clique_structure: List[List],
                    n: int
                    ) -> [QuantumCircuit, List[Parameter]]:
        """

        Args:
            clique_structure: Clique structure given as list
            n: Number of nodes/qubits

        Returns: MRF Ansatz quantum circuit, Ansatz parameters

        """
        qr = QuantumRegister(n)
        ansatz = QuantumCircuit(qr)
        params = []
        params_index = 0
        for clique in clique_structure:
            for i in clique:
                for j in clique:
                    if j <= i:
                        continue
                    else:
                        param0 = Parameter(str(params_index))
                        param1 = Parameter(str(params_index+1))
                        param2 = Parameter(str(params_index+2))
                        param3 = Parameter(str(params_index+3))
                        params_index += 4
                        params.extend([param0, param1, param2, param3])
                        ansatz.ry(param0, qr[i])
                        ansatz.ry(param1, qr[j])
                        ansatz.cx(qr[i], qr[j])
                        ansatz.ry(param2, qr[i])
                        ansatz.ry(param3, qr[j])
                        ansatz.cx(qr[j], qr[i])
        for j in range(n):
            param = Parameter(str(params_index))
            params_index += 1
            ansatz.ry(param, qr[j])
            params.append(param)

        return ansatz, params
