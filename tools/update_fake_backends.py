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
import sys
from importlib import import_module
from collections import defaultdict
from tabulate import tabulate

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


class _PropertyStats:
    def __init__(self, properties):
        self.properties = properties
        self.no_qubits = len(properties.qubits)
        self.no_gates = len(properties.gates)
        self._gate_errors = None  # {gate_name: [qubits]}
        self._gate_total = None  # {gate_name: total_amount_of_them}
        self._readout_errors = None  # {qubit}

    @property
    def name(self):
        """backend name"""
        return self.properties.backend_name

    @property
    def date(self):
        """backend last update date"""
        return self.properties.last_update_date

    @property
    def readout_errors(self):
        """list[qubits] with the readout==1"""
        if self._readout_errors is None:
            self._readout_errors = set()
            for qubit in range(self.no_qubits):
                if self.properties.readout_error(qubit) == 1:
                    self._readout_errors.add(qubit)
        return self._readout_errors

    @property
    def total_readout_rate(self):
        """tuple(witherror/total/percentage"""
        return (
            len(self.readout_errors),
            self.no_qubits,
            100 * len(self.readout_errors) / self.no_qubits,
        )

    @property
    def gate_errors(self):
        """dict gate->list[qubits] with the gate_error==1 for each gate"""
        if self._gate_errors is None:
            self._count_gate_errors()
        return self._gate_errors

    def gate_rate(self, gate_name):
        """gate_name -> tuple(with_error, total, percentage)"""
        return (
            len(self.gate_errors[gate_name]),
            self.gate_total[gate_name],
            100 * len(self.gate_errors[gate_name]) / self.gate_total[gate_name],
        )

    @property
    def total_gate_rate(self):
        """tuple(with_error, total, percentage)"""
        with_error = sum([len(e) for e in self.gate_errors.values()])
        return (with_error, self.no_gates, 100 * with_error / self.no_gates)

    @property
    def gate_total(self):
        """dict gate->int with the total amount for each gate"""
        if not self._gate_total:
            self._count_gate_errors()
        return self._gate_total

    def _count_gate_errors(self):
        """calculates:
        * gate_total
        * gate_errors"""
        self._gate_total = defaultdict(lambda: 0)
        self._gate_errors = defaultdict(lambda: [])
        for gate in self.properties.gates:
            if gate.gate == "reset":
                continue
            self._gate_total[gate.gate] += 1
            if self.properties.gate_error(gate.gate, gate.qubits) == 1:
                self._gate_errors[gate.gate].append(gate.qubits)


class _Summary:
    """
    In this context, as error is when error == 1.
    Args:
        properties (BackendProperty): BackendProperty to analyze.
    """

    def __init__(self, properties, current_properties):
        self.properties = _PropertyStats(properties)
        self.current_properties = _PropertyStats(current_properties)
        self.name = properties.backend_name
        self.defs_msg = "No defaults"
        self.conf_msg = "No configuration"
        self.props_msg = "No properties"
        self._readout_summary = None
        self._gate_summary = None
        self._rate_table = None

    @staticmethod
    def _format_rate(rate):
        return f"{rate[0]}/{rate[1]} ({rate[2]:.2f}%)"

    @property
    def readout_summary(self):
        """[cached] the readout error == 1 summary"""
        if self._readout_summary is None:
            self._readout_summary = []
            for qubit in self.properties.readout_errors:
                self._readout_summary.append(f" * qubit {qubit}")

        return self._readout_summary

    @property
    def rate_table(self):
        """[cached] build rate_table information"""
        if self._rate_table is None:
            self._rate_table = []
            total_readout_rate = self.properties.total_readout_rate
            total_readout_rate_was = self.current_properties.total_readout_rate

            if total_readout_rate[0] != 0 or total_readout_rate_was[0] != 0:
                self._rate_table.append(
                    [
                        "readout error",
                        _Summary._format_rate(total_readout_rate),
                        _Summary._format_rate(total_readout_rate_was),
                    ]
                )
            new_gates = self.properties.gate_errors.keys()
            cur_gates = self.current_properties.gate_errors.keys()

            for gate in set(new_gates).union(cur_gates):
                rate = self.properties.gate_rate(gate)
                rate_was = self.current_properties.gate_rate(gate)
                self._rate_table.append(
                    [
                        f"{gate} error",
                        _Summary._format_rate(rate),
                        _Summary._format_rate(rate_was),
                    ]
                )
            total_gate_rate = self.properties.total_gate_rate
            total_gate_rate_was = self.current_properties.total_gate_rate
            if total_gate_rate[0] != 0 or total_gate_rate_was[0] != 0:
                self._rate_table.append(
                    [
                        "all gates",
                        _Summary._format_rate(total_gate_rate),
                        _Summary._format_rate(total_gate_rate_was),
                    ]
                )
        return self._rate_table

    @property
    def gate_summary(self):
        """[cached] the gate error == 1 summary"""
        if self._gate_summary is None:
            self._gate_summary = []
            for gate, qubits in self.properties.gate_errors.items():
                self._gate_summary.extend(
                    [f" * {gate} {', '.join(map(str,qubit))}" for qubit in qubits]
                )
        return self._gate_summary

    def generate_summary(self):
        """A list with lines of the summary output"""
        result = [f"\n{self.name}", "=" * len(self.name)]
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

        if self.rate_table:
            result.append(
                tabulate(
                    self.rate_table,
                    [
                        "error == 1",
                        f"new {self.properties.date}",
                        f"current {self.current_properties.date}",
                    ],
                    tablefmt="fancy_grid",
                    colalign=("right", "center", "center"),
                )
            )

        result += [self.conf_msg, self.defs_msg]

        return "\n".join(result)


def _main():
    parser = argparse.ArgumentParser(description="Generate fake backend snapshots")
    parser.add_argument("--dir", "-d", type=str, default=DEFAULT_DIR)
    parser.add_argument("backends", type=str, nargs="*")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--hub", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument(
        "--datetime", type=str, default=None, help="date in ISO format. Default: now"
    )
    args = parser.parse_args()
    provider = IBMQ.load_account()
    if args.hub or args.group or args.project:
        provider = IBMQ.get_provider(hub=args.hub, group=args.group, project=args.project)
    dt = datetime.fromisoformat(args.datetime) if args.datetime else None
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
            backend_dirname = os.path.join(args.dir, name)
            if not os.path.isdir(backend_dirname):
                print("Skipping, fake backend for %s does not exist yet" % name)
                continue

            sys.path.append(args.dir)
            backend_mod = import_module(name)
            fake_backend = getattr(backend_mod, "Fake" + name.capitalize())()
            config = backend.configuration()
            props = backend.properties(datetime=dt)
            defs = backend.defaults()

            summary = _Summary(props, fake_backend.properties())

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
