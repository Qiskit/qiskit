# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
======================================================
Circuit annotations (:mod:`qiskit.circuit.annotation`)
======================================================

.. currentmodule:: qiskit.circuit.annotation

This module contains the infrastructure for working with custom circuit annotations.

The main user-facing class is the base-class :class:`qiskit.circuit.Annotation`, which is also
re-exported from this module.

A stand-alone function allows iterating through namespaces and parent namespaces in priority order
from most specific to least specific.

.. autofunction:: iter_namespaces

The rest of this module defines interfaces for handling the serialization and deserialization of
these objects with QPY (see :mod:`qiskit.qpy`) and OpenQASM 3 (see :mod:`qiskit.qasm3`).

.. autoclass:: QPYSerializer
.. autoclass:: QPYFromOpenQASM3Serializer
.. autoclass:: OpenQASM3Serializer
"""

from __future__ import annotations

import abc
from typing import Literal, Iterator

from qiskit._accelerate.circuit import Annotation


__all__ = [
    "Annotation",  # Also exported in `qiskit.circuit`, but for convenience is here too.
    "QPYSerializer",
    "OpenQASM3Serializer",
    "QPYFromOpenQASM3Serializer",
    "iter_namespaces",
]


def iter_namespaces(namespace: str) -> Iterator[str]:
    """An iterator over all namespaces that can be used to lookup the given namespace.

    This includes the namespace and all parents, including the root empty-string namespace.

    Examples:

    .. code-block:: python

        from qiskit.circuit.annotation import iter_namespaces
        assert list(iter_namespaces("hello.world")) == ["hello.world", "hello", ""]
    """
    while namespace:
        yield namespace
        split = namespace.rsplit(".", 1)
        if len(split) == 1:
            break
        namespace = split[0]
    yield ""


class QPYSerializer(abc.ABC):
    """The interface for serializers and deserializers of :class:`.Annotation` objects to QPY.

    For more information on QPY, see :mod:`qiskit.qpy`.

    This interface-definition class is designed to be subclassed.  The individual methods describe
    their contracts, and how they will be called.

    The general structure of calls during serialization of a single circuit is:

    1. many calls to :meth:`dump_annotation`, potentially with different ``namespace`` arguments.
    2. one call to :meth:`dump_state` for each unique ``namespace`` in step 1.

    The general structure of calls during deserialization of a single circuit out of a QPY payload
    is:

    1. one call to :meth:`load_state` for each unique ``namespace`` that will be used in step 2.
    2. many calls to :meth:`load_annotation`.

    When subclassing this, recall that QPY is intended to have strict backwards-compatibility
    guarantees, and it is strongly recommended that annotation-serialisation subclasses maintain
    this.  In particular, it is suggested that any non-trivial serializer includes "version"
    information for the serializer in its total "state" (see :meth:`dump_state`), and the
    deserialization should make
    """

    @abc.abstractmethod
    def dump_annotation(
        self, namespace: str, annotation: Annotation
    ) -> bytes | Literal[NotImplemented]:
        """Serialize an annotation to a bytestream.

        This method may mutate the serializer's internal state (the object that will be serialized
        by :meth:`dump_state`).

        The ``namespace`` argument is the resolved key used to lookup this serializer.  It may not
        be identical to the :attr:`.Annotation.namespace` field of the ``annotation`` argument; it
        might be an ancestor, up to and including the empty string (the root namespace).

        The method can return :data:`NotImplemented` if the serializer cannot serialize a particular
        annotation.

        This method is the mirror of :meth:`load_annotation`.

        Args:
            namespace: the namespace that this serializer was accessed under.  This may be an
                ancestor of the annotation.
            annotation: the object to serialize to a bytestream.

        Returns:
            Either the serialized form of the annotation (optionally after mutating the
            serialization state of this class), or :data:`NotImplemented` if the annotation cannot
            be handled.
        """

    @abc.abstractmethod
    def load_annotation(self, namespace: str, payload: bytes) -> Annotation:
        """Load an annotation from a view of memory.

        A subclass can assume that :meth:`load_state` will have been called exactly once for each
        ``namespace`` argument that might appear, before any call to this method.

        The ``payload`` will be a read-only view of byte data.  The ``namespace`` will be the
        namespace context under which the payload was serialized during dumping; this might have
        been an ancestor of the annotation's.

        This method is the mirror of :meth:`dump_annotation`.

        Args:
            namespace: the namespace that was used as the lookup key when the payload was
                serialized.
            payload: the bytes to deserialized into an annotation.

        Returns:
            The deserialized annotation.  It may be considered an exceptional state if this class is
            asked to deserialized a payload that it cannot support (assuming the set of objects a
            serializer can deserialize is a non-strict superset of the set of objects it can
            serialize); this would indicate that a user has not supplied the same serializer
            configuration on both sides of the QPY load and dump.
        """

    def dump_state(self, namespace: str) -> bytes:  # pylint: disable=unused-argument
        """Serialize a state object for the given namespace context.

        This method will be called exactly once for each unique ``namespace`` that was given to a
        call to :meth:`dump_annotation`.  This method will only be called after all calls to
        :meth:`dump_annotation`.  The order in which the different ``namespace``\\ s will be dumped
        is arbitrary; a subclass should not depend in any way on this order.

        The default state is the empty bytestring; if your serializer is stateless, you do not need
        to override this method.

        This method is the mirror of :meth:`load_state`.
        """
        return b""

    def load_state(self, namespace: str, payload: bytes):  # pylint: disable=unused-argument
        """Initialize the state of the deserializer for a given ``namespace`` key.

        This method will be called exactly once for each unique ``namespace`` that will be in a
        subsequent call to :meth:`load_annotation`, and all calls to :meth:`load_state` will happen
        before any call to :meth:`load_annotation`.  The order of calls to :meth:`load_state` is
        arbitrary, and subclasses should not depend on it.

        The default implementation is a no-op; if you have not overridden :meth:`dump_state`, you do
        not need to override this method.

        This method is the mirror of :meth:`dump_state`.

        Args:
            namespace: the namespace key that the corresponding dump was resolved under.
            payload: the state payload that was dumped by the corresponding call to
                :meth:`dump_state`.
        """
        pass


class OpenQASM3Serializer(abc.ABC):
    """The interface for serializers and deserializers of :class:`.Annotation` objects to
    OpenQASM 3.

    For more information on OpenQASM 3 support in Qiskit, see :mod:`qiskit.qasm3`.

    This interface-definition class is designed to be subclassed.  OpenQASM 3 annotations are
    stateless within a program, therefore a subclass must not track state.
    """

    @abc.abstractmethod
    def dump(self, annotation: Annotation) -> str | Literal[NotImplemented]:
        """Serialize the paylaod of an annotation to a single line of UTF-8 text.

        The output of this method should not include the annotation's
        :attr:`~.Annotation.namespace` attribute; this is handled automatically by the OpenQASM 3
        exporter.

        The serialized form must not contain newline characters; it must be valid as the "arbitrary"
        component of the annotation as defined by OpenQASM 3.  If there is no data required, the
        method should return the empty string.  If this serializer cannot handle the particular
        annotation, it should return :data:`NotImplemented`.

        Args:
            annotation: the annotation object to serialize.

        Returns:
            the serialized annotation (without the namespace component), or the sentinel
            :data:`NotImplemented` if it cannot be handled by this object.
        """

    @abc.abstractmethod
    def load(self, namespace: str, payload: str) -> Annotation | Literal[NotImplemented]:
        """Load an annotation, if possible, from an OpenQASM 3 program.

        The two arguments will be the two components of an annotation, as defined by the OpenQASM 3
        specification.  The method should return :data:`NotImplemented` if it cannot handle the
        annotation.

        Args:
            namespace: the OpenQASM 3 "namespace" of the annotation.
            payload: the rest of the payload for the annotation.  This is arbitrary and free-form,
                and in general should have been serialized by a call to :meth:`dump`.

        Returns:
            the created :class:`.Annotation` object, whose :attr:`.Annotation.namespace` attribute
            should be identical to the incoming ``namespace`` argument.  If this class cannot handle
            the annotation, it can also return :data:`NotImplemented`.
        """

    def as_qpy(self) -> QPYFromOpenQASM3Serializer:
        """Derive a serializer/deserializer for QPY from this OpenQASM 3 variant.

        OpenQASM 3 serialization and deserialization is intended to be stateless and return single
        lines of UTF-8 encoded text.  This is a subset of the allowable serializations for QPY."""
        return QPYFromOpenQASM3Serializer(self)


class QPYFromOpenQASM3Serializer(QPYSerializer):
    """An adaptor that converts a :class:`OpenQASM3Serializer` into a :class:`QPYSerializer`.

    This works because OpenQASM 3 annotation serializers are required to be stateless and return
    UTF-8-encoded single lines of text, which is a subset of what QPY permits.

    Typically you create one of these using the :meth:`~OpenQASM3Serializer.as_qpy` method
    of an OpenQASM 3 annotation serializer.

    Examples:

    Instances of this class can be called like a zero-argument function and return themselves.  This
    lets you use them directly as a factory function to the QPY entry points, such as:

    .. code-block:: python

        import io
        from qiskit.circuit import OpenQASM3Serializer, Annotation
        from qiskit import qpy

        class MyAnnotation(Annotation):
            namespace = "my_namespace"

        class MySerializer(OpenQASM3Serializer):
            def dump(self, annotation):
                if not isinstance(annotation, MyAnnotation):
                    return NotImplemented
                return ""

            def load(self, namespace, payload):
                assert namespace == "my_namespace"
                assert payload == ""
                return MyAnnotation()

        qc = QuantumCircuit(2)
        with qc.box(annotations=[MyAnnotation()]):
            qc.cx(0, 1)

        with io.BytesIO() as fptr:
            qpy.dump(fptr, qc, annotation_serializers = {"my_namespace": MySerializer().as_qpy()})

    """

    def __init__(self, inner: OpenQASM3Serializer):
        """
        Args:
            inner: the OpenQASM 3 serializer that this is derived from.
        """
        self.inner = inner

    def dump_annotation(self, namespace, annotation):
        qasm3 = self.inner.dump(annotation)
        if qasm3 is NotImplemented:
            return NotImplemented
        return (
            # For OpenQASM 3 serialisation, we need the exact namespace the annotation claims.
            annotation.namespace.encode("utf-8")
            + b"\x00"
            + qasm3.encode("utf-8")
        )

    def load_annotation(self, namespace, payload):
        # The `namespace` we get called with might not be the full namespace of the object, so it's
        # useless for our purposes.
        namespace, payload = payload.split(b"\x00", maxsplit=2)
        out = self.inner.load(namespace.decode("utf-8"), payload.decode("utf-8"))
        if out is NotImplemented:
            raise ValueError(
                "asked to deserialize an object the provided OpenQASM deserializer cannot handle"
            )
        return out

    def __call__(self) -> QPYFromOpenQASM3Serializer:
        # Our internal object is stateless because it's an OpenQASM 3 exporter (which is a stateless
        # format).  Defining this method allows an instance of ourself to be used as a factory
        # function, simplifying the interface for creating a QPY serializer from an OpenQASM 3 one,
        # and attaching this to the class means that pickling would "just work".
        return self
