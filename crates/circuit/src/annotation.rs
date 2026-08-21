// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;
use std::any::Any;
use std::fmt::Debug;

/// Private traits that allow [`Annotation`] trait objects to be cloned.
mod annotation_traits {
    use crate::annotation::Annotation;

    /// A trait which implements dynamically cloning [`Annotation`] dyn objects.
    ///
    /// If an annotation implements [`Clone`], this trait will be automatically implemented.
    #[diagnostic::on_unimplemented(
        message = "Clone is required to correctly implement Annotation on {Self}.",
        label = "This type needs an implementation of Clone",
        note = "Consider annotating {Self} with `#[derive(Clone)]`"
    )]
    pub trait ClonableAnnotation {
        fn clone_dyn(&self) -> Box<dyn Annotation>;
    }

    impl<A: Clone + Annotation> ClonableAnnotation for A {
        fn clone_dyn(&self) -> Box<dyn Annotation> {
            Box::new(self.clone())
        }
    }
}

use annotation_traits::ClonableAnnotation;

/// A native Rust annotation that can be attached to circuit instructions.
///
/// Implementors can use this trait to store annotation payloads directly in Rust-owned circuit
/// data.  Existing Python-space annotations are represented by [`PyAnnotationObject`].
pub trait Annotation: Any + Debug + Send + Sync + ClonableAnnotation {
    /// The namespace that consumers use to dispatch annotation handling.
    fn namespace(&self) -> &str;

    /// Compare this annotation with another annotation.
    fn equals(&self, other: &dyn Annotation) -> bool;

    /// Compare this annotation with another annotation while attached to Python.
    fn py_eq(&self, _py: Python<'_>, other: &dyn Annotation) -> PyResult<bool> {
        Ok(self.equals(other))
    }

    /// Convert this annotation to its Python representation.
    ///
    /// Native Rust annotations that do not have a Python representation should leave the default
    /// implementation in place.  Python-backed annotations implement this by cloning the stored
    /// Python object reference.
    fn to_python(&self, _py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "annotation '{}' does not have a Python representation",
            self.namespace()
        )))
    }
}

impl Clone for Box<dyn Annotation> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

impl PartialEq for dyn Annotation {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl dyn Annotation + '_ {
    /// Cast a reference to a concrete annotation type.
    pub fn downcast_ref<T: Annotation + 'static>(&self) -> Option<&T> {
        let self_as_any: &dyn Any = self;
        self_as_any.downcast_ref()
    }
}

/// Python-backed annotation payload.
#[derive(Clone, Debug)]
pub struct PyAnnotationObject {
    annotation: Py<PyAny>,
    namespace: String,
}

impl PyAnnotationObject {
    /// Build a Rust-owned annotation wrapper from a Python annotation object.
    pub fn new(py: Python<'_>, annotation: Py<PyAny>) -> Self {
        let namespace = annotation
            .bind(py)
            .getattr(intern!(py, "namespace"))
            .ok()
            .and_then(|namespace| namespace.extract().ok())
            .unwrap_or_default();
        Self {
            annotation,
            namespace,
        }
    }

    /// Borrow the wrapped Python object.
    pub fn as_python(&self) -> &Py<PyAny> {
        &self.annotation
    }
}

impl Annotation for PyAnnotationObject {
    fn namespace(&self) -> &str {
        self.namespace.as_str()
    }

    fn equals(&self, other: &dyn Annotation) -> bool {
        Python::attach(|py| self.py_eq(py, other).unwrap_or(false))
    }

    fn py_eq(&self, py: Python<'_>, other: &dyn Annotation) -> PyResult<bool> {
        let Some(other) = other.downcast_ref::<Self>() else {
            return Ok(false);
        };
        self.annotation.bind(py).eq(other.annotation.bind(py))
    }

    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.annotation.clone_ref(py))
    }
}

impl From<PyAnnotationObject> for Box<dyn Annotation> {
    fn from(value: PyAnnotationObject) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct NativeAnnotation {
        namespace: &'static str,
        value: u64,
    }

    impl Annotation for NativeAnnotation {
        fn namespace(&self) -> &str {
            self.namespace
        }

        fn equals(&self, other: &dyn Annotation) -> bool {
            other.downcast_ref::<Self>() == Some(self)
        }
    }

    #[test]
    fn native_annotations_clone_and_compare_as_trait_objects() {
        let annotation: Box<dyn Annotation> = Box::new(NativeAnnotation {
            namespace: "test.native",
            value: 5,
        });

        let cloned = annotation.clone();
        assert_eq!(annotation.namespace(), "test.native");
        assert!(annotation.equals(cloned.as_ref()));

        let different: Box<dyn Annotation> = Box::new(NativeAnnotation {
            namespace: "test.native",
            value: 8,
        });
        assert!(!annotation.equals(different.as_ref()));
    }
}

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
