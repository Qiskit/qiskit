#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""List deprecated decorators."""

from pathlib import Path
from collections import OrderedDict, defaultdict
import ast
from datetime import datetime
import requests


class Deprecation:
    """
    Deprecation node, representing a single deprecation decorator.

    Args:
        filename: where is the deprecation.
        decorator_node: AST node of the decorator call.
        func_node: AST node of the decorated call.
    """

    def __init__(self, filename, decorator_node, func_node):
        self.filename = filename
        self.decorator_node = decorator_node
        self.func_node = func_node
        self._since = None

    @property
    def since(self):
        """Version since the deprecation applies."""
        if not self._since:
            for kwarg in self.decorator_node.keywords:
                if kwarg.arg == "since":
                    self._since = ".".join(kwarg.value.value.split(".")[:2])
        return self._since

    @property
    def lineno(self):
        """Line number of the decorator."""
        return self.decorator_node.lineno

    @property
    def target(self):
        """Name of the decorated function/method."""
        return self.func_node.name

    @property
    def location_str(self):
        """String with the location of the deprecated decorator <filename>:<line number>"""
        return f"{self.filename}:{self.lineno}"


class DecoratorVisitor(ast.NodeVisitor):
    """
    Node visitor for finding deprecation decorator
    Args:
        filename: Name of the file to analyze
    """

    def __init__(self, filename=None):
        self.filename = filename
        self.deprecations = []

    @staticmethod
    def is_deprecation_decorator(node):
        """Check if a node is a deprecation decorator"""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("deprecate_")
        )

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name
        """Visitor for function declarations"""
        self.deprecations += [
            Deprecation(self.filename, d_node, node)
            for d_node in node.decorator_list
            if DecoratorVisitor.is_deprecation_decorator(d_node)
        ]
        ast.NodeVisitor.generic_visit(self, node)


class DeprecationCollection:
    """
    A collection of :class:~.Deprecation

    Args:
        dirname: Directory name that would be checked recursively for deprecations.
    """

    def __init__(self, dirname):
        self.dirname = dirname
        self._deprecations = None
        self.grouped = OrderedDict()

    @property
    def deprecations(self):
        """List of deprecation :class:~.Deprecation"""
        if not self._deprecations:
            self.collect_deprecations()
        return self._deprecations

    def collect_deprecations(self):
        """Run the :class:~.DecoratorVisitor on `self.dirname`"""
        self._deprecations = []
        files = [self.dirname] if self.dirname.is_file() else self.dirname.rglob("*.py")
        for filename in files:
            self._deprecations.extend(DeprecationCollection.find_deprecations(filename))

    def group_by(self, attribute_idx):
        """Group :class:~`.Deprecation` in self.deprecations based on the attribute attribute_idx"""
        grouped = defaultdict(list)
        for obj in self.deprecations:
            grouped[getattr(obj, attribute_idx)].append(obj)
        for key in sorted(grouped.keys()):
            self.grouped[key] = grouped[key]

    @staticmethod
    def find_deprecations(file_name):
        """Runs the deprecation finder on file_name"""
        with open(file_name, encoding="utf-8") as fp:
            code = fp.read()
        mod = ast.parse(code, file_name)
        decorator_visitor = DecoratorVisitor(file_name)
        decorator_visitor.visit(mod)
        return decorator_visitor.deprecations


if __name__ == "__main__":
    collection = DeprecationCollection(Path(__file__).joinpath("..", "..", "qiskit").resolve())
    # collection.collect_deprecations()
    collection.group_by("since")

    DATA_JSON = LAST_TIME_MINOR = DETAILS = None
    try:
        DATA_JSON = requests.get("https://pypi.org/pypi/qiskit-terra/json", timeout=5).json()
        LAST_MINOR = ".".join(DATA_JSON["info"]["version"].split(".")[:2])
        LAST_TIME_MINOR = datetime.fromisoformat(
            DATA_JSON["releases"][f"{LAST_MINOR}.0"][0]["upload_time"]
        )
    except requests.exceptions.ConnectionError:
        print("https://pypi.org/pypi/qiskit-terra/json timeout...")

    for since_version, deprecations in collection.grouped.items():
        if DATA_JSON and LAST_TIME_MINOR:
            try:
                release_minor_datetime = datetime.fromisoformat(
                    DATA_JSON["releases"][f"{since_version}.0"][0]["upload_time"]
                )
                release_minor_date = release_minor_datetime.strftime("%B %d, %Y")
                diff_days = (LAST_TIME_MINOR - release_minor_datetime).days
                DETAILS = (
                    f"Released in {release_minor_date} ({diff_days} days until last minor release)"
                )
            except KeyError:
                DETAILS = "Future release"
        print(f"\n{since_version}: {DETAILS}")
        for deprecation in deprecations:
            print(f" - {deprecation.location_str}")
