# This file is derived from Qiskit (https://github.com/Qiskit/qiskit-metapackage)
# 
# Original source:
# https://github.com/Qiskit/qiskit-metapackage/blob/0.43.3/docs/versionutils.py
#
# Copyright IBM 2023
# Licensed under the Apache License, Version 2.0

import re
import subprocess
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.util.nodes import nested_parse_with_titles

def setup(app):
    app.add_directive("qpy-version-history", QPYVersionHistory)

class QPYVersionHistory(Directive):
    has_content = False

    def run(self):
        data = self._build_data()
        if not data:
            return [nodes.paragraph(text="No QPY data found")]
        return [self._build_table(data)]

    def _get_tags(self):
        proc = subprocess.run(["git", "tag", "--sort=-creatordate"], capture_output=True, text=True)
        return proc.stdout.splitlines()

    def _extract_qpy_constants(self, tag):
        try:
            proc = subprocess.run(["git", "show", f"{tag}^:qiskit/qpy/common.py"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            return None

        content = proc.stdout
        version = re.search(r"QPY_VERSION\s*=\s*(\d+)", content)
        compat = re.search(r"QPY_COMPATIBILITY_VERSION\s*=\s*(\d+)", content)

        if not version:
            return None
        dump_version = int(version.group(1))

        # handling legacy versions
        if compat:
            load_version = int(compat.group(1))
        else:
            load_version = dump_version

        return dump_version, load_version

    def _version_key(self, v):
        return tuple(int(x) for x in v.split("."))

    def _build_data(self):
        data = {}

        for tag in self._get_tags():
            if not re.fullmatch(r"\d+\.\d+\.\d+", tag):
                continue

            version_str = tag
            result = self._extract_qpy_constants(tag)
            if not result:
                continue

            qpy_version, compat_version = result

            data[version_str] = {
                "dump": list(range(compat_version, qpy_version + 1)),
                "load": qpy_version,
            }

        # Manually adding legacy versions without proper tags available on github (< 0.20.0)
        legacy_data = {
            "0.19.2": {"dump": [4], "load": 4},
            "0.19.1": {"dump": [3], "load": 3},
            "0.19.0": {"dump": [2], "load": 2},
            "0.18.3": {"dump": [1], "load": 1},
            "0.18.2": {"dump": [1], "load": 1},
            "0.18.1": {"dump": [1], "load": 1},
            "0.18.0": {"dump": [1], "load": 1},
        }

        for k, v in legacy_data.items():
            if k not in data:
                data[k] = v
        
        sorted_data = dict(sorted(data.items(), key=lambda x: self._version_key(x[0]), reverse=True))

        return sorted_data

    def _build_table(self, data):
        table = nodes.table()
        table["classes"] += ["colwidths-auto"]
        tgroup = nodes.tgroup(cols=3)
        table += tgroup

        for _ in range(3):
            tgroup += nodes.colspec()

        thead = nodes.thead()
        tbody = nodes.tbody()
        tgroup += thead
        tgroup += tbody
        
        headers = [
            "Qiskit (qiskit-terra for < 1.0.0) version",
            ":func:`.dump` format(s) output versions",
            ":func:`.load` maximum supported version (older format versions can always be read)",
        ]

        row = nodes.row()
        for h in headers:
            entry = nodes.entry()

            vl = ViewList()
            vl.append(h, "<qpy-version-history>")

            node = nodes.paragraph()
            nested_parse_with_titles(self.state, vl, node)

            entry += node
            row += entry

        thead += row

        for tag, info in data.items():
            row = nodes.row()
            entry = nodes.entry()
            entry += nodes.paragraph(text=tag)
            row += entry
            entry = nodes.entry()
            entry += nodes.paragraph(
                text=", ".join(map(str, info["dump"]))
            )
            row += entry
            entry = nodes.entry()
            entry += nodes.paragraph(text=str(info["load"]))
            row += entry
            tbody += row

        return table