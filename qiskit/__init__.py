from IBMQuantumExperience import RegisterSizeError

from ._qiskiterror import QISKitError
from ._classicalregister import ClassicalRegister
from ._quantumregister import QuantumRegister
from ._quantumcircuit import QuantumCircuit
from ._gate import Gate
from ._compositegate import CompositeGate
from ._instruction import Instruction
from ._instructionset import InstructionSet
import qiskit.extensions.standard
from ._jobprocessor import JobProcessor
from ._quantumjob import QuantumJob
from ._quantumprogram import QuantumProgram
from ._result import Result
from ._util import _check_ibmqe_version

__version__ = '0.4.0'

_check_ibmqe_version()
