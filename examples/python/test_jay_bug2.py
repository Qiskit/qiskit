import sys
import os

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit.qasm import Qasm

badqasm = """
OPENQASM 2.0;
qreg Q[5];
"""

ast = Qasm(data=badqasm).parse()
