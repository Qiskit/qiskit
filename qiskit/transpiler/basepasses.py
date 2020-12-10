# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base transpiler passes."""

from abc import abstractmethod
from collections.abc import Hashable
from inspect import signature
from .propertyset import PropertySet


class MetaPass(type):
    """Metaclass for transpiler passes.

    Enforces the creation of some fields in the pass while allowing passes to
    override ``__init__``.
    """

    def __call__(cls, *args, **kwargs):
        pass_instance = type.__call__(cls, *args, **kwargs)
        pass_instance._hash = hash(MetaPass._freeze_init_parameters(cls, args, kwargs))
        return pass_instance

    @staticmethod
    def _freeze_init_parameters(class_, args, kwargs):
        self_guard = object()
        init_signature = signature(class_.__init__)
        bound_signature = init_signature.bind(self_guard, *args, **kwargs)
        arguments = [('class_.__name__', class_.__name__)]
        for name, value in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)


class BasePass(metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        self.requires = []  # List of passes that requires
        self.preserves = []  # List of passes that preserves
        self.property_set = PropertySet()  # This pass's pointer to the pass manager's property set.
        self._hash = None

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def name(self):
        """Return the name of the pass."""
        return self.__class__.__name__

    @abstractmethod
    def run(self, dag):
        """Run a pass on the DAGCircuit. This is implemented by the pass developer.

        Args:
            dag (DAGCircuit): the dag on which the pass is run.
        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        raise NotImplementedError

    @property
    def is_transformation_pass(self):
        """Check if the pass is a transformation pass.

        If the pass is a TransformationPass, that means that the pass can manipulate the DAG,
        but cannot modify the property set (but it can be read).
        """
        return isinstance(self, TransformationPass)

    @property
    def is_analysis_pass(self):
        """Check if the pass is an analysis pass.

        If the pass is an AnalysisPass, that means that the pass can analyze the DAG and write
        the results of that analysis in the property set. Modifications on the DAG are not allowed
        by this kind of pass.
        """
        return isinstance(self, AnalysisPass)

    @classmethod
    def on(cls, circuit, property_set=None, conf=None):  # pylint: disable=invalid-name
        """Runs the pass on circuit.

        Args:
            circuit (QuantumCircuit): the dag on which the pass is run.
            property_set (PropertySet or dict or None): input/output property set.
            conf (dict): kwargs to the pass constructor. See `cls.__init__()` for details.

        Returns:
            QuantumCircuit or None: If on transformation pass, a QuantumCircuit. If analysis
                   pass, None.
        """
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        from qiskit.dagcircuit.dagcircuit import DAGCircuit

        if conf is None:
            conf = {}

        property_set_ = None
        if isinstance(property_set, dict):
            property_set_ = PropertySet(property_set)

        cls_pass = cls(**conf)

        if isinstance(property_set_, PropertySet):
            cls_pass.property_set = property_set_

        result = cls_pass.run(circuit_to_dag(circuit))

        if isinstance(property_set, (dict, PropertySet)):
            property_set.update(cls_pass.property_set)

        if isinstance(result, DAGCircuit):
            result = dag_to_circuit(result)

        return result


class AnalysisPass(BasePass):  # pylint: disable=abstract-method
    """An analysis pass: change property set, not DAG."""
    pass


class TransformationPass(BasePass):  # pylint: disable=abstract-method
    """A transformation pass: change DAG, not property set."""
    pass
