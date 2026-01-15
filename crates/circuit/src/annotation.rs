// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// An arbitrary annotation for instructions.
///
/// .. note::
///
///     The annotation framework is a new and evolving component of Qiskit.  We expect the
///     functionality of this and its first-class support within the transpiler to expand as we
///     get more evidence of how it is used.
///
/// This base class alone has very little prescribed behavior or semantics.  The primary interaction
/// is by user- or library subclassing.  See :ref:`circuit-annotation-subclassing` for more detail.
///
/// This is a framework for structuring additional metadata that can be attached to :class:`.BoxOp`
/// instructions within a :class:`.QuantumCircuit` and :class:`.DAGCircuit` in ways that can be
/// tracked and consumed by arbitrary transpiler passes, including custom passes that are not in
/// Qiskit core.
///
/// While the stateful :class:`.PropertySet` used during a compilation also supplies a way for
/// custom transpiler passes to store arbitrary "state" objects into the compilation workflow that
/// can be retrieved by later compiler passes, the :class:`.PropertySet` is stored next to the
/// circuit, and so is most suitable for analyses that relate to the circuit as a whole. An
/// :class:`Annotation` is intended to be more local in scope, applying to a box of instructions,
/// and further, may still be present in the output of :class:`.transpile`, if it is intended for
/// further consumption by a lower-level part of your backend's execution machinery (for example, an
/// annotation might include metadata instructing an error-mitigation routine to treat a particular
/// box in a special way).
///
/// The :class:`.PassManager` currently does not make any effort to track and validate
/// pre-conditions on the validity of an :class:`Annotation`.  That is, if you apply a custom
/// annotation to a box of instructions that would be invalidated by certain transformations (such
/// as routing, basis-gate decomposition, etc), it is currently up to you as the caller of
/// :func:`.transpile` or :func:`.generate_preset_pass_manager` to ensure that the compiler passes
/// selected will not invalidate the annotation.  We expect to have more first-class support for
/// annotations to declare their validity requirements in the future.
#[pyclass(module = "qiskit.circuit", name = "Annotation", subclass, frozen)]
pub struct PyAnnotation;
#[pymethods]
impl PyAnnotation {
    #[allow(unused_variables)]
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Self
    }

    /// The "namespace" the annotation belongs to.
    ///
    /// This can be standard Python identifier (e.g. ``my_namespace``), or a dot-separated list of
    /// identifiers (e.g. ``my_namespace.subnamespace``).  The namespace is used by all consumers of
    /// annotations to determine what handler should be invoked.
    ///
    /// This must be overridden by subclasses.
    ///
    /// The concept of the namespace corresponds to the `same concept in OpenQASM 3
    /// <https://openqasm.com/language/directives.html#annotations>`__.
    ///
    /// Typically during dispatch operations, first the entire :attr:`namespace` will be looked up,
    /// and dispatched if there is a match.  Failing that, each "parent" namespace (formed by
    /// removing everything from the last ``.`` onwards) will be tried.  See
    /// :func:`~.annotation.iter_namespaces` for access to the dispatch ordering.
    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "").clone().unbind()
    }
}
