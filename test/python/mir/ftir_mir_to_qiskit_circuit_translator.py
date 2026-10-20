# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Translate ftir .mlir to qiskit QuantumCircuit.

currently we assume that each instruction is in a single line, each line contains at most single instruction.
"""


import argparse
import re
from circuit_emitter import InMemoryCircuitEmitter, PythonFileCircuitEmitter


class Tokenizer:
    _CIRCUIT_LINE_RE = re.compile(
        r"^\s*pbc\.circuit\s*"
        r'\{circuit_name\s*:\s*"(?P<circuit_name>[^"]*)"\s*,\s*'
        r"program_width\s*:\s*<(?P<program_width>\d+)\s*:\s*si64>\}\s*$"
    )
    _OUTLINED_ATTR_RE = re.compile(r"^\s*!(?P<id>\d+)\s*=\s*\[(?P<body>.*)\]\s*$")
    _SKIP_LINE_RE = re.compile(r"builtin\.module\b.*|\{|\}|\^\w+\(\)\:|pbc.return ()")
    _OP_LINE_RE = re.compile(
        r"^\s*(?:(?P<results>[\w, ]+?)\s*=\s*)?"
        r"(?P<op>[a-z][\w.]*)\s*"
        r"(?:\((?P<operands>[^)]*)\))?\s*"
        r"(?:\{(?P<attrs>[^}]*)\})?\s*"
        r"(?::\s*(?P<types>[^!;]*))?"
        r"(?:!(?P<attr_ref>\d+))?"
        r"\s*;\s*$"
    )
    _GIVEN_NAMES_RE = re.compile(
        r"^builtin_given_names\s*=\s*builtin\.given_names\s*\[(?P<names>[^\]]*)\]$"
    )
    _BRACKET_PAIRS = {"(": ")", "[": "]", "<": ">"}

    @staticmethod
    def _split_top_level(text, sep=","):
        parts = []
        current = []
        depth = 0
        closers = set(Tokenizer._BRACKET_PAIRS.values())
        for ch in text:
            if ch in Tokenizer._BRACKET_PAIRS:
                depth += 1
            elif ch in closers:
                depth -= 1
            if ch == sep and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail or parts:
            parts.append(tail)
        return parts

    @staticmethod
    def _parse_attr_value(text):
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            return Tokenizer._split_top_level(text[1:-1])
        try:
            return float(text)
        except ValueError:
            return text

    def parse_op(self, metcher):
        results = [r.strip() for r in metcher["results"].split(",")] if metcher["results"] else []
        operands = (
            [o.strip() for o in metcher["operands"].split(",")] if metcher["operands"] else []
        )
        types = Tokenizer._split_top_level(metcher["types"]) if metcher["types"] else []
        attrs = {}
        if metcher["attrs"]:
            for piece in Tokenizer._split_top_level(metcher["attrs"]):
                key, _, value = piece.partition(":")
                attrs[key.strip()] = Tokenizer._parse_attr_value(value)
        attr_ref = int(metcher["attr_ref"]) if metcher["attr_ref"] else None
        return {
            "kind": "op",
            "results": results,
            "op": metcher["op"],
            "operands": operands,
            "attrs": attrs,
            "types": types,
            "attr_ref": attr_ref,
        }

    def tokenize(self, lines):
        tokens = []
        in_outlined_attributes_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == "outlined_attributes:":
                in_outlined_attributes_section = True
                tokens.append({"kind": "skip", "text": line})
                continue
            if in_outlined_attributes_section:
                m = Tokenizer._OUTLINED_ATTR_RE.match(line)
                if not m:
                    raise ValueError(f"Unrecognized outlined-attribute line: {line}")
                m2 = Tokenizer._GIVEN_NAMES_RE.match(m["body"])
                names = []
                if m2:
                    names = [n.strip() for n in m2["names"].split(",")] if m2["names"] else []
                tokens.append(
                    {"kind": "outlined_attr", "id": int(m["id"]), "body": m["body"], "names": names}
                )
                continue
            if Tokenizer._SKIP_LINE_RE.match(line):
                tokens.append({"kind": "skip", "text": line})
                continue
            m = Tokenizer._CIRCUIT_LINE_RE.match(line)
            if m:
                tokens.append(
                    {
                        "kind": "op",
                        "op": "pbc.circuit",
                        "name": m["circuit_name"],
                        "width": m["program_width"],
                    }
                )
                continue
            m = Tokenizer._OP_LINE_RE.match(line)
            if m:
                tokens.append(self.parse_op(m))
                continue

            raise ValueError(f"Unrecognized line: {line}")

        return tokens


class PbcMirTranslator:

    _H_QUBIT_TYPE_RE = re.compile(r"pbc\.h_qubit\s*<\s*(?P<module>\d+)\s*,\s*(?P<slot>\d+)\s*>")

    def __init__(self, tokens, circuit_emitter=None):
        self._tokens = tokens
        self._address_to_qubit = {}
        self._ssa_name_to_address = {}
        self._value_to_clbit = {}
        self._clbit_by_name = {}
        if circuit_emitter is None:
            circuit_emitter = InMemoryCircuitEmitter()
        else:
            self._circuit_emitter = circuit_emitter
        self._op_tokens = [token for token in self._tokens if token["kind"] == "op"]
        self._op_handlers = {
            "pbc.circuit": self._handle_circuit,
            "pbc.h_qubit_decl": self._handle_h_qubit_decl,
            "pbc.bool_decl": self._handle_bool_decl,
            "pbc.pauli_measurement": self._handle_pauli_measurement,
            "pbc.pauli_rotation": self._handle_pauli_rotation,
        }
        self._given_names = {
            token["id"]: token["names"]
            for token in self._tokens
            if token["kind"] == "outlined_attr" and "names" in token
        }

    def get_circuit(self):
        return self._circuit_emitter.create_circuit()

    def translate(self):
        for token in self._op_tokens:
            op_handler = self._op_handlers[token["op"]]
            op_handler(token)

    def _extract_names(self, token):
        if token["attr_ref"] is None:
            return []
        return self._given_names.get(token["attr_ref"])

    def _handle_circuit(self, token):
        self._circuit_emitter.new_circuit(name=token["name"])

    def _handle_h_qubit_decl(self, token):
        m = PbcMirTranslator._H_QUBIT_TYPE_RE.match(token["types"][0])
        address = (int(m["module"]), int(m["slot"]))
        names = self._extract_names(token)
        name = names[0] if names else None

        self._address_to_qubit[address] = self._circuit_emitter.add_qubit(name)
        self._ssa_name_to_address[token["results"][0]] = address

    def _handle_bool_decl(self, token):
        names = self._extract_names(token)
        name = names[0] if names else None
        clbit_index = self._circuit_emitter.add_clbit(name)
        self._value_to_clbit[token["results"][0]] = clbit_index
        if name:
            self._clbit_by_name[name] = clbit_index

    def _handle_pauli_measurement(self, token):
        bases = token["attrs"]["pbc_pauli_bases"]
        operands = token["operands"]
        assert len(bases) == len(operands), "bases/operands length mismatch"  # noqa: S101
        addresses = [self._ssa_name_to_address[operand] for operand in operands]
        qubits = [self._address_to_qubit[address] for address in addresses]

        names = self._extract_names(token)
        *qubit_results, bool_result = token["results"]
        bool_name = names[-1] if len(names) == len(token["results"]) else None
        # NOTE: currently ftir has no SSA-level link between the result and the clbit. (seems to be a bug)
        # We will use the name hint.
        if bool_name and bool_name in self._clbit_by_name:
            clbit_index = self._clbit_by_name[bool_name]
        else:
            raise ValueError("Can't figure wich clbit use to store the result")

        self._circuit_emitter.pauli_measurment(
            qubits_indics=qubits, pauli_label=bases, clbit_index=clbit_index
        )

        for qubit_result, address in zip(qubit_results, addresses):
            self._ssa_name_to_address[qubit_result] = address
        self._value_to_clbit[bool_result] = clbit_index

    def _handle_pauli_rotation(self, token):
        bases = token["attrs"]["pbc_pauli_bases"]
        angle = token["attrs"]["angle"]
        operands = token["operands"]

        addresses = [self._ssa_name_to_address[operand] for operand in operands]
        qubits = [self._address_to_qubit[address] for address in addresses]

        self._circuit_emitter.pauli_rotation(qubits_indics=qubits, pauli_label=bases, angle=angle)

        for result, address in zip(token["results"], addresses):
            self._ssa_name_to_address[result] = address


def translate(source, circuit_emitter):
    with open(source) as file:
        lines = file.readlines()
    tokens = Tokenizer().tokenize(lines)
    translator = PbcMirTranslator(tokens, circuit_emitter=circuit_emitter)
    translator.translate()
    return translator.get_circuit()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("source", help="path to the ftir .mlir file")
    argparser.add_argument(
        "--output-mode",
        choices=["memory", "file"],
        default="memory",
        help="'memory' (default) build an in memory QuantumCircuit and prints it."
        "'file' writes a python program that builds the circuit.",
    )
    argparser.add_argument(
        "--target", help="path to the target python program (required for --output=file)"
    )
    args = argparser.parse_args()

    if args.output_mode == "file":
        if not args.target:
            argparser.error("target is required when --output-mode=file")
        emitter = PythonFileCircuitEmitter(args.target)
    else:
        emitter = InMemoryCircuitEmitter()

    circuit = translate(args.source, emitter)
    print(circuit)  # noqa: T201 -- this CLI's whole job is to print the circuit


if __name__ == "__main__":
    main()
