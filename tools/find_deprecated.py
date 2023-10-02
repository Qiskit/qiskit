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
from __future__ import annotations
from typing import cast, Optional
from pathlib import Path
from collections import OrderedDict, defaultdict
import ast
from datetime import datetime
import sys
import requests


def short_path(path: Path) -> Optional[Path]:
    """shorten the full path, when possible"""
    original_path = path
    while path is not None:
        if Path.cwd().is_relative_to(path):
            return original_path.relative_to(path)
        path = path.parent if path != path.parent else None
    return original_path


class Deprecation:
    """
    Root class for Deprecation classes.
    """

    @property
    def location_str(self) -> str:
        """String with the location of the deprecated decorator <filename>:<line number>"""
        return f"{self.filename}:{self.lineno}"


class DeprecationDecorator(Deprecation):
    """
    Deprecation decorator, representing a single deprecation decorator.

    Args:
        filename: where is the deprecation.
        decorator_node: AST node of the decorator call.
        func_node: AST node of the decorated call.
    """

    def __init__(
        self, filename: Path, decorator_node: ast.Call, func_node: ast.FunctionDef
    ) -> None:
        self.filename = filename
        self.decorator_node = decorator_node
        self.func_node = func_node
        self._since: str | None = None

    @property
    def since(self) -> str | None:
        """Version since the deprecation applies."""
        if not self._since:
            for kwarg in self.decorator_node.keywords:
                if kwarg.arg == "since":
                    self._since = ".".join(cast(ast.Constant, kwarg.value).value.split(".")[:2])
        return self._since

    @property
    def lineno(self) -> int:
        """Line number of the decorator."""
        return self.decorator_node.lineno

    @property
    def target(self) -> str:
        """Name of the decorated function/method."""
        return self.func_node.name


class DeprecationCall(Deprecation):
    """
    Deprecation call, representing a single deprecation call.

    Args:
        decorator_call: deprecation call.
    """

    def __init__(self, filename: Path, decorator_call: ast.Call) -> None:
        self.filename = filename
        self.decorator_node = decorator_call
        self.lineno = decorator_call.lineno
        self._target: str | None = None
        self._since: str | None = None

    @property
    def target(self) -> str | None:
        """what's deprecated."""
        if not self._target:
            arg = self.decorator_node.args.__getitem__(0)
            if isinstance(arg, ast.Attribute):
                self._target = f"{arg.value.id}.{arg.attr}"
            if isinstance(arg, ast.Name):
                self._target = arg.id
        return self._target

    @property
    def since(self) -> str | None:
        """Version since the deprecation applies."""
        if not self._since:
            for kwarg in self.decorator_node.func.keywords:
                if kwarg.arg == "since":
                    self._since = ".".join(cast(ast.Constant, kwarg.value).value.split(".")[:2])
        return self._since


class DecoratorVisitor(ast.NodeVisitor):
    """
    Node visitor for finding deprecation decorator
    Args:
        filename: Name of the file to analyze
    """

    def __init__(self, filename: Path):
        self.filename = short_path(filename)
        self.deprecations: list[Deprecation] = []

    @staticmethod
    def is_deprecation_decorator(node: ast.expr) -> bool:
        """Check if a node is a deprecation decorator"""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("deprecate_")
        )

    @staticmethod
    def is_deprecation_call(node: ast.expr) -> bool:
        """Check if a node is a deprecation call"""
        return (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Name)
            and node.func.func.id.startswith("deprecate_")
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pylint: disable=invalid-name
        """Visitor for function declarations"""
        self.deprecations += [
            DeprecationDecorator(self.filename, cast(ast.Call, d_node), node)
            for d_node in node.decorator_list
            if DecoratorVisitor.is_deprecation_decorator(d_node)
        ]
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Call(self, node: ast.Call) -> None:  # pylint: disable=invalid-name
        """Visitor for function call"""
        if DecoratorVisitor.is_deprecation_call(node):
            self.deprecations.append(DeprecationCall(self.filename, node))
        ast.NodeVisitor.generic_visit(self, node)


class DeprecationCollection:
    """
    A collection of :class:~.Deprecation

    Args:
        dirname: Directory name that would be checked recursively for deprecations.
    """

    def __init__(self, dirname: Path):
        self.dirname = dirname
        self._deprecations: list[Deprecation] | None = None
        self.grouped: OrderedDict[str, list[Deprecation]] = OrderedDict()

    @property
    def deprecations(self) -> list[Deprecation]:
        """List of deprecation :class:~.Deprecation"""
        if self._deprecations is None:
            self.collect_deprecations()
        return cast(list, self._deprecations)

    def collect_deprecations(self) -> None:
        """Run the :class:~.DecoratorVisitor on `self.dirname` (in place)"""
        self._deprecations = []
        files = [self.dirname] if self.dirname.is_file() else self.dirname.rglob("*.py")
        for filename in files:
            self._deprecations.extend(DeprecationCollection.find_deprecations(filename))

    def group_by(self, attribute_idx: str) -> None:
        """Group :class:~`.Deprecation` in self.deprecations based on the attribute attribute_idx"""
        grouped = defaultdict(list)
        for obj in self.deprecations:
            grouped[getattr(obj, attribute_idx)].append(obj)
        for key in sorted(grouped.keys()):
            self.grouped[key] = grouped[key]

    @staticmethod
    def find_deprecations(file_name: Path) -> list[Deprecation]:
        """Runs the deprecation finder on file_name"""
        code = Path(file_name).read_text()
        mod = ast.parse(code, file_name)
        decorator_visitor = DecoratorVisitor(file_name)
        decorator_visitor.visit(mod)
        return decorator_visitor.deprecations


if __name__ == "__main__":
    collection = DeprecationCollection(Path(__file__).joinpath("..", "..", "qiskit").resolve())
    collection.group_by("since")

    DATA_JSON = LAST_TIME_MINOR = DETAILS = None
    try:
        DATA_JSON = requests.get("https://pypi.org/pypi/qiskit-terra/json", timeout=5).json()
    except requests.exceptions.ConnectionError:
        print("https://pypi.org/pypi/qiskit-terra/json timeout...", file=sys.stderr)

    if DATA_JSON:
        LAST_MINOR = ".".join(DATA_JSON["info"]["version"].split(".")[:2])
        LAST_TIME_MINOR = datetime.fromisoformat(
            DATA_JSON["releases"][f"{LAST_MINOR}.0"][0]["upload_time"]
        )

    for since_version, deprecations in collection.grouped.items():
        if DATA_JSON and LAST_TIME_MINOR:
            try:
                release_minor_datetime = datetime.fromisoformat(
                    DATA_JSON["releases"][f"{since_version}.0"][0]["upload_time"]
                )
                release_minor_date = release_minor_datetime.strftime("%B %d, %Y")
                diff_days = (LAST_TIME_MINOR - release_minor_datetime).days
                DETAILS = f"Released in {release_minor_date}"
                if diff_days:
                    DETAILS += f" (wrt last minor release, {round(diff_days / 30.4)} month old)"
            except KeyError:
                DETAILS = "Future release"
        print(f"\n{since_version}: {DETAILS}")
        for deprecation in deprecations:
            print(f" - {deprecation.location_str} ({deprecation.target})")
