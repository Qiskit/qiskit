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
///
///
/// .. _circuit-annotation-subclassing:
///
/// Subclassing
/// ===========
///
/// This class is intended to be subclassed.  Subclasses must set the :attr:`.namespace` field.
/// The namespace is used as part of the dispatch mechanism, as described in
/// :ref:`circuit-annotation-namespacing`.
///
/// If you intend your annotation to be able to be serialized via :ref:`QPY <qiskit-qpy>` or :ref:`
/// OpenQASM 3 <qiskit-qasm3>`, you must provide separate implementations of the serialization and
/// deserialization methods as discussed in :ref:`circuit-annotation-serialization`.
///
///
/// .. _circuit-annotation-namespacing:
///
/// Namespacing
/// -----------
///
/// TODO.
///
/// .. _circuit-annotation-serialization:
///
/// Serialization and deserialization
/// ---------------------------------
///
/// Annotations represent completely custom data, that may persist after compilation.  TODO.
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
    /// This must be overridden by subclasses.
    ///
    /// This can be standard Python identifier (e.g. ``my_namespace``), or a dot-separated list of
    /// identifiers (e.g. ``my_namespace.subnamespace``).  The namespace is primarily used to
    /// dispatch to the correct custom handler in serialization/deserialization contexts, such as
    /// QPY and OpenQASM 3.  The concept of the namespace corresponds to the `same concept in
    /// OpenQASM 3 <https://openqasm.com/language/directives.html#annotations>`__.
    ///
    /// During dispatch operations, first the entire :attr:`namespace` will be looked up, and
    /// dispatched if there is a match.  Failing that, each "parent" namespace (formed by removing
    /// everything from the last ``.`` onwards) will be tried.
    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "").clone().unbind()
    }
}
