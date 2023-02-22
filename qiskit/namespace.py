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

# pylint: disable=unused-argument

"""Module for utilities to manually construct qiskit namespace"""

import sys
from importlib.abc import MetaPathFinder, Loader
import importlib


def _new_namespace(fullname, old_namespace, new_package):
    names = fullname.split(".")
    new_namespace_names = new_package.split(".")
    old_namespace_names = old_namespace.split(".")
    fullname = ".".join(new_namespace_names + names[len(old_namespace_names) :])
    return fullname


class QiskitLoader(Loader):
    """Load qiskit element as a namespace package."""

    def __init__(self, new_package, old_namespace):
        super().__init__()
        self.new_package = new_package
        self.old_namespace = old_namespace

    def module_repr(self, module):
        return repr(module)

    def create_module(self, spec):
        return self.load_module(spec.name)

    def exec_module(self, module):
        """Executes the module. Not needed in Qiskit."""
        pass

    def load_module(self, fullname):
        old_name = fullname
        fullname = _new_namespace(fullname, self.old_namespace, self.new_package)
        module = importlib.import_module(fullname)
        sys.modules[old_name] = module
        return module


class QiskitElementImport(MetaPathFinder):
    """Meta importer to enable unified qiskit namespace."""

    def __init__(self, old_namespace, new_package):
        super().__init__()
        self.old_namespace = old_namespace
        self.new_package = new_package

    def find_spec(self, fullname, path=None, target=None):
        """Return the ModuleSpec for Qiskit element."""
        if fullname.startswith(self.old_namespace):
            try:
                importlib.import_module(
                    _new_namespace(fullname, self.old_namespace, self.new_package)
                )
                return importlib.util.spec_from_loader(
                    fullname, QiskitLoader(self.new_package, self.old_namespace), origin="qiskit"
                )
            except ModuleNotFoundError:
                return None
        return None
