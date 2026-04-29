# This file is derived from Qiskit (https://github.com/Qiskit/qiskit-metapackage)
# 
# Original source:
# https://github.com/Qiskit/qiskit-metapackage/blob/0.43.3/docs/versionutils.py
#
# Copyright IBM 2023
# Licensed under the Apache License, Version 2.0

import re
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.util.nodes import nested_parse_with_titles
from dulwich.repo import Repo
from packaging import version
from pathlib import Path

from qiskit.qpy.common import QPY_VERSION_HISTORY


def setup(app):
    app.add_directive("qpy-version-history", QPYVersionHistory)

class QPYVersionHistory(Directive):
    has_content = False

    def run(self):
        data = self._build_data()
        if not data:
            return [nodes.paragraph(text="QPY data not found")]
        return [self._build_table(data)]

    def _get_repo(self):
        path = Path(__file__).resolve()
        for parent in path.parents:
            if (parent / ".git").exists():
                return Repo(str(parent))
        raise RuntimeError("Repository not found")
    
    def _get_tags(self):
        repo = self._get_repo()
        tags = []

        for ref in repo.refs.keys():
            if ref.startswith(b"refs/tags/"):
                tag = ref.decode().split("/")[-1]
                tags.append(tag)

        return tags
    
    def _is_valid_version(self, tag):
        return re.fullmatch(r"\d+\.\d+\.\d+", tag)
    
    def _build_data(self):
        tags = self._get_tags()
        versions = sorted((version.parse(t) for t in tags if self._is_valid_version(t)))
        history = sorted(
            [(max_, min_, version.parse(first)) for max_, min_, first in QPY_VERSION_HISTORY],
            key=lambda x: x[2]
        )

        def get_qpy_range(cur):
            for max_, min_, first in reversed(history):
                if cur >= first:
                    return list(range(min_, max_ + 1))
            return None

        data = {}
        for v in versions:
            qpy_range = get_qpy_range(v)
            if not qpy_range:
                continue

            data[str(v)] = {
                "dump": qpy_range,
                "load": max(qpy_range),
            }

        return dict(sorted(data.items(), key=lambda x: version.parse(x[0]), reverse=True))

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