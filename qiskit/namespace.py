# -*- coding: utf-8 -*-

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


class QiskitLoader(Loader):
    """Load qiskit element as a namespace package."""
    def __init__(self, new_package, old_namespace):
        super().__init__()
        self.new_package = new_package
        self.old_namespace = old_namespace

    def module_repr(self, module):
        return repr(module)

    def load_module(self, fullname):
        old_name = fullname
        names = fullname.split(".")
        new_namespace_names = self.new_package.split('.')
        old_namespace_names = self.old_namespace.split('.')
        fullname = ".".join(
            new_namespace_names + names[len(old_namespace_names):])
        module = importlib.import_module(fullname)
        sys.modules[old_name] = module
        return module


class QiskitElementImport(MetaPathFinder):
    """Meta importer to enable unified qiskit namespace."""
    def __init__(self, new_package, old_namespace):
        super().__init__()
        self.new_package = new_package
        self.old_namespace = old_namespace

    def find_spec(self, fullname, path=None, target=None):
        """Return the ModuleSpec for Qiskit element."""
        if fullname.startswith(self.old_namespace):
            return importlib.util.spec_from_loader(
                fullname,
                QiskitLoader(self.new_package, self.old_namespace),
                origin='qiskit')
        return None
