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

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::OnceLock;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::annotation::custom_traits::ComparableAnnotation;

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
pub struct PyAnnotation {
    inner: Option<Arc<dyn Annotation>>,
}

#[pymethods]
impl PyAnnotation {
    #[allow(unused_variables)]
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(args: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Self { inner: None }
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

impl PyAnnotation {
    pub fn new(inner: Arc<dyn Annotation>) -> Self {
        Self { inner: Some(inner) }
    }

    pub fn inner(&self) -> Option<&Arc<dyn Annotation>> {
        self.inner.as_ref()
    }
}

mod custom_traits {
    use crate::annotation::Annotation;

    /// A trait which implements comparisons between [`Annotation`] instances.
    /// If the operation implements [`PartialEq`], this trait will be automatically implemented.
    /// Otherwise, the user is responsible for implementing this trait.
    #[diagnostic::on_unimplemented(
        message = "PartialEq is required to correctly implement ComparableAnnotation on {Self}.",
        label = "This type needs an implementation of PartialEq",
        note = "Consider annotating {Self} with `#[derive(PartialEq)]`"
    )]
    pub trait ComparableAnnotation {
        fn rich_eq(&self, other: &dyn Annotation) -> bool;
    }

    impl<Op: PartialEq + Annotation> ComparableAnnotation for Op {
        fn rich_eq(&self, other: &dyn Annotation) -> bool {
            let Some(other) = other.downcast_ref() else {
                return false;
            };
            self.eq(other)
        }
    }
}

/// Trait that implements methods for annotations.
///
/// This trait has an implicit requirement of [`PartialEq`] to allow for comparison
/// between opaque annotations.
pub trait Annotation: Any + Debug + Send + Sync + ComparableAnnotation {
    /// Return the namespace of the annotation.
    fn namespace(&self) -> &str;

    /// Return a Python representation of this annotation.
    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>>;
}

impl PartialEq for dyn Annotation {
    fn eq(&self, other: &Self) -> bool {
        ComparableAnnotation::rich_eq(self, other)
    }
}

impl dyn Annotation + 'static {
    /// Casts a reference to an Annotation to its original type if the correct
    /// type is specified.
    pub fn downcast_ref<T: Annotation + 'static>(&self) -> Option<&T> {
        let self_as_any: &dyn Any = self;
        self_as_any.downcast_ref()
    }
}

/// Internal representation of a Python annotation.
#[derive(Debug)]
pub struct PythonAnnotation {
    annotation: Py<PyAny>,
    namespace: OnceLock<String>,
}

impl PythonAnnotation {
    pub fn new(annotation: Py<PyAny>) -> Self {
        Self {
            annotation,
            namespace: OnceLock::new(),
        }
    }
}

impl Annotation for PythonAnnotation {
    /// Return the namespace of the annotation.
    ///
    /// On construction, the underlying namespace field is uninitialized. The first time this method is called,
    /// it sets the namespace from Python.
    fn namespace(&self) -> &str {
        if let Some(namespace) = self.namespace.get() {
            return namespace;
        }
        let namespace = Python::attach(|py| {
            self.annotation
                .getattr(py, "namespace")
                .and_then(|py_ctrl_state| py_ctrl_state.extract::<String>(py))
                .unwrap()
        });
        let _ = self.namespace.set(namespace);
        self.namespace.get().expect("Value was set.")
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(self.annotation.clone_ref(py))
    }
}

impl PartialEq for PythonAnnotation {
    fn eq(&self, other: &Self) -> bool {
        self.annotation.is(&other.annotation)
            || Python::attach(|py| {
                self.annotation
                    .bind(py)
                    .eq(other.annotation.bind(py))
                    .unwrap()
            })
    }
}

/// Return the internal representation of a Python annotation.
pub fn extract_annotation(ob: &Bound<'_, PyAny>) -> Arc<dyn Annotation> {
    if let Ok(base) = ob.cast::<PyAnnotation>()
        && let Some(native) = base.get().inner()
    {
        return Arc::clone(native);
    }

    Arc::new(PythonAnnotation::new(ob.clone().unbind()))
}

#[cfg(test)]
mod test_annotation {
    use crate::annotation::Annotation;
    use pyo3::prelude::*;
    use std::sync::Arc;

    macro_rules! impl_annotation {
        ($ty:ident; $namespace:expr,) => {
            impl $crate::annotation::Annotation for $ty {
                fn namespace(&self) -> &str {
                    $namespace
                }

                fn create_py_annotation(&self, _: Python) -> PyResult<Py<PyAny>> {
                    unimplemented!()
                }
            }
        };
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Tag(&'static str);
    impl_annotation!(Tag; "tag",);

    #[derive(Debug, Clone, PartialEq)]
    struct Mark;
    impl_annotation!(Mark; "mark",);

    #[test]
    fn test_namespace() {
        assert_eq!(Tag("my_tag").namespace(), "tag");
        assert_eq!(Mark.namespace(), "mark");
    }

    #[test]
    fn test_downcast() {
        let tag: Arc<dyn Annotation> = Arc::new(Tag("my_tag"));

        let tag = tag.downcast_ref::<Tag>().expect("Should be a Tag.");
        assert_eq!(tag, &Tag("my_tag"));
    }

    #[test]
    fn test_equality() {
        let mark = Mark;
        let mark_as_dyn: Arc<dyn Annotation> = Arc::new(Mark);
        let my_tag = Tag("my_tag");
        let my_other_tag = Tag("my_tag");
        let my_different_tag = Tag("different!");

        assert_eq!(&mark as &dyn Annotation, mark_as_dyn.as_ref());
        assert_eq!(my_tag, my_other_tag);
        assert_ne!(my_tag, my_different_tag);
        assert_ne!(&my_tag as &dyn Annotation, mark_as_dyn.as_ref());
    }

    #[test]
    fn test_arc_sharing() {
        let mark: Arc<dyn Annotation> = Arc::new(Mark);
        let mark_vec = vec![mark.clone(); 10];

        assert_eq!(11, Arc::strong_count(&mark));
        for a_mark in mark_vec {
            assert!(Arc::ptr_eq(&mark, &a_mark));
        }
    }
}
