#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to update fake backends"""

import argparse
from datetime import datetime
import json
import os

from qiskit import IBMQ
from qiskit.circuit.parameterexpression import ParameterExpression


class BackendEncoder(json.JSONEncoder):
    """A json encoder for qobj"""

    def default(self, o):
        # Convert numpy arrays:
        if hasattr(o, "tolist"):
            return o.tolist()
        # Use Qobj complex json format:
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, ParameterExpression):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "qiskit",
    "providers",
    "fake_provider",
    "backends",
)


class _Summary:
    """
    In this context, as error is when error == 1.
    Args:
        properties (BackendProperty): BackendProperty to analyze.
    """

    def __init__(self, properties):
        self.properties = properties
        self.no_qubits = len(properties.qubits)
        self.no_gates = len(properties.gates)
        self._readout_errors = set()  # {qubits}
        self._gate_errors = dict()  # {gate_name: [qubits]}
        self._gate_total = dict()  # {gate_name: total_amount_of_them}
        self._readout_summary = None
        self._gate_summary = None
        self.defs_msg = "No defaults"
        self.conf_msg = "No configuration"
        self.props_msg = "No properties"
        self.defs_msg = "No defaults"

    @property
    def readout_errors(self):
        """[cached] A set with all the qubits with readout_error==1"""
        if not self._readout_errors:
            self._count_readout_errors()
        return self._readout_errors

    @property
    def gate_errors(self):
        """dict gate->list[qubits] with the gate_error==1 for each gate"""
        if not self._gate_errors:
            self._count_gate_errors()
        return self._gate_errors

    @property
    def gate_total(self):
        """dict gate->int with the total amount for each gate"""
        if not self._gate_total:
            self._count_gate_errors()
        return self._gate_total

    @property
    def readout_error_total(self) -> int:
        """
        total amount of readout errors
        """
        return len(self.readout_errors)

    @property
    def readout_summary(self):
        """[cached] the readout error == 1 summary"""
        if not self._readout_summary:
            self._readout_summary = []
            total_error = 0
            for qubit in self.readout_errors:
                self._readout_summary.append(f" * qubit {qubit}")
                total_error += 1
            if total_error != 0:
                self._gate_summary.append(
                    f" = rate readout error: {total_error}/{self.no_qubits} "
                    f"({100*total_error/self.no_qubits:.2f}%)"
                )
        return self._readout_summary

    @property
    def gate_summary(self):
        """[cached] the gate error == 1 summary"""
        if not self._gate_summary:
            self._gate_summary = []
            total_error = 0
            for gate, qubits in self.gate_errors.items():
                with_error = len(self.gate_errors[gate])
                total_error += with_error
                total_gate = self.gate_total[gate]
                self._gate_summary.extend(
                    [f" * qubit {', '.join(map(str,qubit))}" for qubit in qubits]
                )
                self._gate_summary.append(
                    f" - rate for {gate}: {with_error}/{total_gate} ({100*with_error/total_gate:.2f}%)"
                )
            if total_error != 0:
                self._gate_summary.append(
                    f" = rate for all gates: {total_error}/{self.no_gates} "
                    f"({100*total_error/self.no_gates:.2f}%)"
                )
        return self._gate_summary

    def generate_summary(self):
        """A list with lines of the summary output"""
        result = [f"{self.properties.backend_name}", "=" * len(self.properties.backend_name)]

        output = []
        if self.readout_summary:
            output.append("Readout errors == 1:")
            output += [f"{line}" for line in self.readout_summary]

        if self.gate_summary:
            output.append("Gate errors == 1:")
            output += [f"{line}" for line in self.gate_summary]

        if output:
            result += output
        else:
            result += [self.props_msg]

        result += [self.conf_msg, self.defs_msg]

        return "\n".join(result)

    def _count_readout_errors(self):
        for qubit in range(self.no_qubits):
            if self.properties.readout_error(qubit) == 1:
                self.readout_errors.add(qubit)

    def _count_gate_errors(self):
        for gate in self.properties.gates:
            if gate.gate == "reset":
                continue
            self._gate_total[gate.gate] = self._gate_total.get(gate.gate, 0) + 1
            if self.properties.gate_error(gate.gate, gate.qubits) == 1:
                self._gate_errors[gate.gate] = self._gate_errors.get(gate.gate, []) + [gate.qubits]


def _main():
    parser = argparse.ArgumentParser(description="Generate fake backend snapshots")
    parser.add_argument("--dir", "-d", type=str, default=DEFAULT_DIR)
    parser.add_argument("backends", type=str, nargs="*")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--hub", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    args = parser.parse_args()
    provider = IBMQ.load_account()
    if args.hub or args.group or args.project:
        provider = IBMQ.get_provider(hub=args.hub, group=args.group, project=args.project)
    ibmq_backends = provider.backends()
    for backend in ibmq_backends:
        raw_name = backend.name()
        if "sim" in raw_name:
            continue
        if raw_name == "ibmqx2":
            name = "yorktown"
        else:
            name = raw_name.split("_")[1]
            if name == "16":
                name = "melbourne"
        if not args.backends or (name in args.backends or raw_name in args.backends):
            if not os.path.isdir(os.path.join(args.dir, name)):
                print("Skipping, fake backend for %s does not exist yet" % name)
                continue
            config = backend.configuration()
            props = backend.properties()
            defs = backend.defaults()

            summary = _Summary(props)

            if config:
                config_path = os.path.join(args.dir, name, "conf_%s.json" % name)
                config_dict = config.to_dict()

                with open(config_path, "w") as fd:
                    fd.write(json.dumps(config_dict, cls=BackendEncoder))
                summary.conf_msg = f"conf_{name}.json okey"
            if props:
                props_path = os.path.join(args.dir, name, "props_%s.json" % name)
                with open(props_path, "w") as fd:
                    fd.write(json.dumps(props.to_dict(), cls=BackendEncoder))
                summary.props_msg = f"props_{name}.json okey"
            if defs:
                defs_path = os.path.join(args.dir, name, "defs_%s.json" % name)
                with open(defs_path, "w") as fd:
                    fd.write(json.dumps(defs.to_dict(), cls=BackendEncoder))
                summary.defs_msg = f"defs_{name}.json okey"

            print(summary.generate_summary())


if __name__ == "__main__":
    _main()
