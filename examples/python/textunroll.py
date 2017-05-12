"""
Unroll to a text printer.

Author: Andrew Cross
        Jesus Perez <jesusper@us.ibm.com>
"""

import sys
import os
import traceback

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit.qasm import Qasm
from qiskit.qasm._qasmexception import QasmException
import qiskit.unroll as unroll


if len(sys.argv) < 2:
    print("textunroll.py <file> [basis]\n")
    print("  file   - input QASM file")
    print("  [basis] - optional basis, \"gate,gate,...\" format\n")
    print("Default basis is [].")
    sys.exit(1)

try:
    ast = Qasm(filename=sys.argv[1]).parse()
except QasmException as e:
    print(e.msg)
except Exception as e:
    print(sys.exc_info()[0], 'Exception parsing qasm file')
    traceback.print_exc()

if ast is None:
    sys.exit(1)

if len(sys.argv) > 2:
    basis = sys.argv[2].split(",")
else:
    basis = []

unroller = unroll.Unroller(ast, unroll.PrinterBackend(basis))
unroller.execute()
