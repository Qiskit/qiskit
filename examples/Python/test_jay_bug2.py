import sys
sys.path.append("..")
from qiskit.qasm import Qasm

badqasm = """
OPENQASM 2.0;
qreg Q[5];
"""

ast = Qasm(data=badqasm).parse()
