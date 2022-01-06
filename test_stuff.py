"""Test Stuff."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import MCXGate, MCXRecursive, MCXVChain, MCXGrayCode

# ["v-chain", "basic", "noancilla", "recursion", "advanced"]
def test_mcx(num_control_qubits, mcx_mode):
    """test"""
    print(f"==> MCX with {num_control_qubits} control qubits and mode {mcx_mode}")

    print("--> Creating gate...")

    if mcx_mode == "mcx":
        mcx = MCXGate(num_control_qubits)
    elif mcx_mode == "recursion":
        mcx = MCXRecursive(num_control_qubits)
    elif mcx_mode == "v-chain":
        mcx = MCXVChain(num_control_qubits)
    elif mcx_mode == "noancilla":
        mcx = MCXGrayCode(num_control_qubits)

    elif mcx_mode == "mcxc":
        mcy = MCXGate(num_control_qubits)
        mcx = mcy.control(2)
    elif mcx_mode == "recursionc":
        mcy = MCXRecursive(num_control_qubits)
        mcx = mcy.control(2)
    elif mcx_mode == "v-chainc":
        mcy = MCXVChain(num_control_qubits)
        mcx = mcy.control(2)
    elif mcx_mode == "noancillac":
        mcy = MCXGrayCode(num_control_qubits)
        mcx = mcy.control(2)


    #qc = QuantumCircuit(num_control_qubits + num_ancilla_qubits + 1)
    print("--> Printing gate...")
    print(mcx)
    print("--> Synthesizing...")
    mcxdef = mcx.definition
    print(mcxdef)
    #print("--> Appending to circuit")
    #print(f"{range(num_control_qubits + num_ancilla_qubits + 1)}")
    #qc.append(mcx, range(num_control_qubits + num_ancilla_qubits + 1))
    #print("--> Final Circuit...")
    #print(qc)
    print("")





if __name__ == "__main__":

    for q in range(1, 8):

        test_mcx(q, "mcx")
        test_mcx(q, "noancilla")
        test_mcx(q, "v-chain")
        test_mcx(q, "recursion")
        test_mcx(q, "mcxc")
        test_mcx(q, "noancillac")
        test_mcx(q, "v-chainc")
        test_mcx(q, "recursionc")
