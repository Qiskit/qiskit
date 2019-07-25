from qiskit.exceptions import QiskitError
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction

class ConversionError(QiskitError):
    pass

def instruction_to_gate(instruction):
    """Attempt to convert instruction to gate.

    Args:
        instruction (Instruction): instruction to convert

    Raises:
        ConversionError: Conversion fails if element of definition can't be converted.
    """
    gateSpecList = []
    for instrSpec in instruction.definition:
        instr = instrSpec[0]
        if isinstance(instr, Gate):
            thisgate = instr
        elif isinstance(instr, Instruction):
            thisgate = instruction_to_gate(instr)
        else:
            raise ConversionError('One or more instructions in this instruction '
                                  'cannot be converted to a gate')
        gateSpecList.append((thisgate, instrSpec[1], instrSpec[2]))
    gate = Gate(instruction.name, instruction.num_qubits, instruction.params)
    gate.definition = gateSpecList
    return gate
        
