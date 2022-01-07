"""Test Stuff."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import MCXGate, MCXRecursive, MCXVChain, MCXGrayCode
from qiskit.synthesis.mcx_synthesis import MCXSynthesis,MCXSynthesisRecursive, \
    MCXSynthesisVChain, MCXSynthesisGrayCode, mcx_mode_to_synthesis
from qiskit.compiler.transpiler import transpile

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

    elif mcx_mode == "mcxi":
        mcy = MCXGate(num_control_qubits)
        mcx = mcy.inverse()
    elif mcx_mode == "recursioni":
        mcy = MCXRecursive(num_control_qubits)
        mcx = mcy.inverse()
    elif mcx_mode == "v-chaini":
        mcy = MCXVChain(num_control_qubits)
        mcx = mcy.inverse()
    elif mcx_mode == "noancillai":
        mcy = MCXGrayCode(num_control_qubits)
        mcx = mcy.inverse()


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


def test_anc(num_control_qubits):
    print(f"noancilla: {MCXGate.get_num_ancilla_qubits(num_control_qubits, 'noancilla')}")
    print(f"recursion: {MCXGate.get_num_ancilla_qubits(num_control_qubits, 'recursion')}")
    print(f"v-chain:   {MCXGate.get_num_ancilla_qubits(num_control_qubits, 'v-chain')}")


def test_ancillas():
    for q in range(1, 8):
        test_anc(q)


def test_circuits():
    for q in range(5, 6):
        test_mcx(q, "mcx")
        test_mcx(q, "noancilla")
        test_mcx(q, "v-chain")
        test_mcx(q, "recursion")
        test_mcx(q, "mcxc")
        test_mcx(q, "noancillac")
        test_mcx(q, "v-chainc")
        test_mcx(q, "recursionc")
        test_mcx(q, "mcxi")
        test_mcx(q, "noancillai")
        test_mcx(q, "v-chaini")
        test_mcx(q, "recursioni")


def test_mcx_new(num_control_qubits, mcx_mode):
    """test"""
    print(f"==> MCX_NEW with {num_control_qubits} control qubits and mode {mcx_mode}")

    print("--> Creating gate...")

    if mcx_mode == "mcx":
        synthesis = MCXSynthesisGrayCode("mcx")
    elif mcx_mode == "recursion":
        synthesis = MCXSynthesisRecursive("mcx_recursive")
    elif mcx_mode == "v-chain":
        synthesis = MCXSynthesisVChain("v-chain", dirty_ancillas=False)
    elif mcx_mode == "noancilla":
        synthesis = MCXSynthesisGrayCode("noancilla")

    mcx = MCXGate(num_control_qubits, synthesis=synthesis)

    print("------> Original gate")
    print("--> Printing gate...")
    print(mcx)
    print("--> Synthesizing...")
    print(mcx.definition)
    print("")

    print("------> Creating 2-controlled")
    mcy = mcx.control(2)
    print("--> Printing gate...")
    print(mcy)
    print("--> Synthesizing...")
    print(mcy.definition)
    print("")

    print("------> Creating inverse")
    mci = mcx.inverse()
    print("--> Printing gate...")
    print(mci)
    print("--> Synthesizing...")
    print(mci.definition)
    print("")



def test_circuits_new():
    for q in range(1, 8):
        test_mcx_new(q, "mcx")
        test_mcx_new(q, "noancilla")
        test_mcx_new(q, "v-chain")
        test_mcx_new(q, "recursion")

def test_qc():
    qc = QuantumCircuit(10)
    qc.mcx([0, 1, 2, 3, 4], 9, [], "noancilla")
    qc.mcx([0, 1, 2, 3, 4], 9, [5], "recursion")
    qc.mcx([0, 1, 2, 3, 4], 9, [5, 6, 7], "v-chain")
    qc.mcx([0, 1, 2, 3, 4], 9, [])
    print(qc)
    #qc2 = transpile(qc, optimization_level=3)
    #print(qc2)


def test_synthesis_map():
    synthesis = mcx_mode_to_synthesis("noancilla")

def test_previous_map():
    num_ctrl_qubits = 6

    available_implementations = {
        "noancilla": MCXGrayCode(num_ctrl_qubits),
        "recursion": MCXRecursive(num_ctrl_qubits),
        "v-chain": MCXVChain(num_ctrl_qubits, False),
        "v-chain-dirty": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
        # outdated, previous names
        "advanced": MCXRecursive(num_ctrl_qubits),
        "basic": MCXVChain(num_ctrl_qubits, dirty_ancillas=False),
        "basic-dirty-ancilla": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
    }

    gate = available_implementations[mode]
    print(gate)

if __name__ == "__main__":
    #test_circuits();
    #test_ancillas()
    #test_circuits_new()
    #test_qc()
    #test_synthesis_map()
    test_previous_map()