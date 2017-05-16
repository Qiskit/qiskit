"""Run some basic tests on the parser."""

import sys
import os

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from qiskit.qasm import Qasm
from qiskit.qasm import QasmException


file_except = {
    "examples/qasm/syntax_errors/e1.qasm":
    "Missing ';' at end of statement; received U",
    "./qasm_examples/syntax_errors/e2.qasm":
    "Missing ']' in indexed ID; received ;",
    "./qasm_examples/syntax_errors/e3.qasm":
    "Missing ';' at end of statement; received U",
    "./qasm_examples/syntax_errors/e4.qasm":
    "Invalid magic string. Expected '2.0;'.  Is the semicolon missing?",
    "./qasm_examples/syntax_errors/e5.qasm":
    "Expecting an integer index; received 0.3",
    "./qasm_examples/syntax_errors/e6.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e7.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e8.qasm":
    "Expected an ID, received '\"b\"'",
    "./qasm_examples/syntax_errors/e9.qasm":
    "Expected an ID, received '\"b\"'",
    "./qasm_examples/syntax_errors/e10.qasm":
    "Missing ';' in qreg or creg declaraton. Instead received 'qreg'",
    "./qasm_examples/syntax_errors/e11.qasm":
    "Expecting indexed id (ID[int]) in QREG declaration; received ;",
    "./qasm_examples/syntax_errors/e12.qasm":
    "Expecting indexed id (ID[int]) in CREG declaration; received ;",
    "./qasm_examples/syntax_errors/e13.qasm":
    "Expected an ID, received ','",
    "./qasm_examples/syntax_errors/e14.qasm":
    "Expected an ID, received '3'",
    "./qasm_examples/syntax_errors/e15.qasm":
    "Expected an ID, received '{'",
    "./qasm_examples/syntax_errors/e16.qasm":
    "Expected an ID, received '('",
    "./qasm_examples/syntax_errors/e17.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e18.qasm":
    "Expected an ID, received '['",
    "./qasm_examples/syntax_errors/e19.qasm":
    "Expected an ID, received 'gate'",
    "./qasm_examples/syntax_errors/e20.qasm":
    "Invalid gate invocation inside gate definition.",
    "./qasm_examples/syntax_errors/e21.qasm":
    "Expected an ID, received 'gate'",
    "./qasm_examples/syntax_errors/e22.qasm":
    "Expected an ID, received 'gate'",
    "./qasm_examples/syntax_errors/e23.qasm":
    "Unmatched () for gate invocation inside gate invocation.",
    "./qasm_examples/syntax_errors/e24.qasm":
    "Expected an ID, received ')'",
    "./qasm_examples/syntax_errors/e25.qasm":
    "Expected an ID, received ';'",
    "./qasm_examples/syntax_errors/e26.qasm":
    "Invalid CX inside gate definition. Expected an ID or ',', received '}'",
    "./qasm_examples/syntax_errors/e27.qasm":
    "Invalid CX inside gate definition. Expected an ID or ',', received '}'",
    "./qasm_examples/syntax_errors/e28.qasm":
    "Invalid CX inside gate definition. Expected an ID or ';', received '}'",
    "./qasm_examples/syntax_errors/e29.qasm":
    "Invalid CX inside gate definition. Expected an ID or ';', received '}'",
    "./qasm_examples/syntax_errors/e30.qasm":
    "Invalid CX inside gate definition. Expected an ID or ',', received '.'",
    "./qasm_examples/syntax_errors/e31.qasm":
    "Invalid barrier inside gate definition.",
    "./qasm_examples/syntax_errors/e32.qasm":
    "Expected an ID, received ')'",
    "./qasm_examples/syntax_errors/e33.qasm":
    "Expected an ID, received '{'",
    "./qasm_examples/syntax_errors/e34.qasm":
    "Invalid U inside gate definition. Missing bit id or ';'",
    "./qasm_examples/syntax_errors/e35.qasm":
    "Invalid U inside gate definition. Missing bit id or ';'",
    "./qasm_examples/syntax_errors/e36.qasm":
    "Missing ')' in U invocation in gate definition.",
    "./qasm_examples/syntax_errors/e37.qasm":
    "Missing ')' in U invocation in gate definition.",
    "./qasm_examples/syntax_errors/e38.qasm":
    "Expected an ID, received ','",
    "./qasm_examples/syntax_errors/e39.qasm":
    "Illegal external function call:  san",
    "./qasm_examples/syntax_errors/e40.qasm":
    "Expected an ID, received ';'",
    "./qasm_examples/syntax_errors/e41.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e42.qasm":
    "Expected an ID, received '2'",
    "./qasm_examples/syntax_errors/e43.qasm":
    "Invalid gate invocation inside gate definition.",
    "./qasm_examples/syntax_errors/e44.qasm":
    "",  # fails
    "./qasm_examples/syntax_errors/e45.qasm":
    "Invalid bit list inside gate definition or missing ';'",
    "./qasm_examples/syntax_errors/e46.qasm":
    "",  # fails
    "./qasm_examples/syntax_errors/e47.qasm":
    "Unmatched () for gate invocation inside gate invocation.",
    "./qasm_examples/syntax_errors/e48.qasm":
    "Unmatched () for gate invocation inside gate invocation.",
    "./qasm_examples/syntax_errors/e49.qasm":
    "Invalid gate invocation inside gate definition.",
    "./qasm_examples/syntax_errors/e50.qasm":
    "Illegal measure statement.=",
    "./qasm_examples/syntax_errors/e51.qasm":
    "Expected an ID, received '2'",
    "./qasm_examples/syntax_errors/e52.qasm":
    "Expected an ID, received '3'",
    "./qasm_examples/syntax_errors/e53.qasm":
    "Missing ';' at end of statement; received opaque",
    "./qasm_examples/syntax_errors/e54.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e55.qasm":
    "Poorly formed OPAQUE statement.",
    "./qasm_examples/syntax_errors/e56.qasm":
    "Missing ';' at end of statement; received )",
    "./qasm_examples/syntax_errors/e57.qasm":
    "Expected an ID, received ';'",
    "./qasm_examples/syntax_errors/e58.qasm":
    "Missing ';' at end of statement; received reset",
    "./qasm_examples/syntax_errors/e59.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e60.qasm":
    "Missing ';' at end of statement; received if",
    "./qasm_examples/syntax_errors/e61.qasm":
    "Ill-formed IF statement.  Expected '==', received '<",
    "./qasm_examples/syntax_errors/e62.qasm":
    "Ill-formed IF statement.  Expected a number, received '<qiskit.qasm._node._id.Id object at 0x111bc5630>",
    "./qasm_examples/syntax_errors/e63.qasm":
    "Ill-formed IF statement, unmatched '('",
    "./qasm_examples/syntax_errors/e64.qasm":
    "Ill-formed IF statement. Perhaps a missing '('?",
    "./qasm_examples/syntax_errors/e65.qasm":
    "Missing ';' at end of statement; received U",
    "./qasm_examples/syntax_errors/e66.qasm":
    "Expected an ID, received ';'",
    "./qasm_examples/syntax_errors/e67.qasm":
    "Expected an ID, received 'U'",
    "./qasm_examples/syntax_errors/e68.qasm":
    "Expected an ID, received '0.3'",
    "./qasm_examples/syntax_errors/e69.qasm":
    "Expected an ID, received '0.1'",
    "./qasm_examples/syntax_errors/e70.qasm":
    "Missing ';' at end of statement; received CX",
    "./qasm_examples/syntax_errors/e71.qasm":
    "Expected an ID, received ';'",
    "./qasm_examples/syntax_errors/e72.qasm":
    "Expected an ID, received '1'",
    "./qasm_examples/syntax_errors/e73.qasm":
    "Expected an ID, received '2'",
    "./qasm_examples/symtab_errors/s1.qasm":
    "Duplicate declaration for qreg 'cin' at line 3, file ./qasm_examples/symtab_errors/s1.qasm.\nPrevious occurence at line 1, file ./qasm_examples/symtab_errors/s1.qasm"
}

for f, e in file_except.items():
    try:
        print("Try \"%s\" ..." % f)
        q = Qasm(filename=f)
        a = q.parse()
    except QasmException as err:
        print(err.msg)
        if err.msg == e:
            print("pass")
        else:
            print("FAIL")
