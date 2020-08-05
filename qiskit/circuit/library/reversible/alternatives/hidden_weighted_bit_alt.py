from qiskit.circuit import QuantumCircuit
class HiddenWeightedBit4(QuantumCircuit):
    def __init__(self) -> None:
        """Create Hidden weighted bit circuit for 4 qubits.
        """

        super().__init__(4, name="Hidden Weighted Bit 4")
        self.cx(0, 1)
        self.cx(1, 3)
        self.cx(3, 2)
        self.cx(0, 3)
        self.ccx(0, 3, 1)
        self.cx(2, 1)
        self.ccx(1, 2, 0)
        self.cx(0, 1)
        self.cx(2, 3)
        self.cx(0, 2)
        self.cx(3, 0)
        self.ccx(0, 3, 1)
        self.cx(1, 3)