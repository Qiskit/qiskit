from ._classicalregister import ClassicalRegister
from ._quantumregister import QuantumRegister
from ._quantumcircuit import QuantumCircuit
from ._gate import Gate
from ._compositegate import CompositeGate
from ._instruction import Instruction
from ._instructionset import InstructionSet
from ._qiskiterror import QISKitError
import qiskit.extensions.standard
from ._jobprocessor import JobProcessor, QuantumJob
from ._quantumprogram import QuantumProgram
from ._result import Result
#from .devices.mydevice import test
#import qiskit.backends
__version__ = '0.3.5'

