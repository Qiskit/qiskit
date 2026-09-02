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
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::OnceLock;

use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use thiserror::Error;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::annotation::custom_traits::ComparableAnnotation;

/// Error conditions for the [Annotation] trait.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum AnnotationError {
    #[error("tried to recurse with annotation in namespace {0}")]
    WrappedPythonError(String),
}

impl From<AnnotationError> for PyErr {
    fn from(error: AnnotationError) -> Self {
        match error {
            AnnotationError::WrappedPythonError(e) => PyValueError::new_err(e.to_string()),
        }
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
    fn py_new(args: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyAny>>) -> Self {
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

/// An annotation native to Qiskit.
///
/// This subclass will be used natively in Qiskit and abides by the same "namespace" semantics as
/// its base class.
#[pyclass(name = "NativeAnnotation", module = "qiskit.circuit", extends = PyAnnotation, frozen)]
pub struct PyNativeAnnotation {
    inner: Arc<dyn Annotation>,
}
#[pymethods]
impl PyNativeAnnotation {
    /// The namespace the annotation belongs to.
    #[getter]
    pub fn namespace(&self) -> &str {
        self.inner.namespace()
    }
}

impl PyNativeAnnotation {
    /// Return a new instance.
    ///
    /// This method guards against [PythonAnnotation] to avoid recursion.
    pub fn new(inner: Arc<dyn Annotation>) -> Result<Self, AnnotationError> {
        match inner.downcast_ref::<PythonAnnotation>() {
            Some(py_ann) => Err(AnnotationError::WrappedPythonError(
                py_ann.namespace().to_string(),
            )),
            None => Ok(Self { inner }),
        }
    }

    pub fn inner(&self) -> &Arc<dyn Annotation> {
        &self.inner
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

    /// The payload.
    ///
    /// Note that there is no inverse method on the trait. This is deliberately omitted
    /// and is implemented with a [NativeLoader]. The [NativeLoader] could do this directly,
    /// or can defer to a per [Annotation] call. The payload can be borrowed when it's
    /// already on self and owned if it has to be built.
    fn payload(&self) -> Option<Cow<'_, str>> {
        None
    }
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

    pub fn annotation(&self, py: Python) -> Py<PyAny> {
        self.annotation.clone_ref(py)
    }
}

impl Annotation for PythonAnnotation {
    /// Return the namespace of the annotation.
    ///
    /// On construction, the underlying namespace field is uninitialized. The first time this method is called,
    /// it sets the namespace from Python.
    fn namespace(&self) -> &str {
        self.namespace.get_or_init(|| {
            Python::attach(|py| {
                self.annotation
                    .getattr(py, intern!(py, "namespace"))
                    .and_then(|namespace| namespace.extract::<String>(py))
                    .unwrap_or_default()
            })
        })
    }
}

impl PartialEq for PythonAnnotation {
    fn eq(&self, other: &Self) -> bool {
        self.annotation.is(&other.annotation)
            || Python::attach(|py| {
                self.annotation
                    .bind(py)
                    .eq(other.annotation.bind(py))
                    .is_ok_and(|b| b)
            })
    }
}

/// Iterate through namespaces from narrowest to broadest.
pub fn iter_namespaces(namespace: &str) -> impl Iterator<Item = &str> {
    std::iter::successors((!namespace.is_empty()).then_some(namespace), |ns| {
        ns.rsplit_once('.')
            .map(|(p, _)| p)
            .filter(|p| !p.is_empty())
    })
    .chain(std::iter::once(""))
}

/// A loader for an [Annotation].
///
/// This function takes a namespace and a payload, and returns an annotation.
pub type NativeLoader = fn(&str, &str) -> Option<Arc<dyn Annotation>>;

/// Loaders for annotations.
///
/// This structure contains a bank of loaders keyed by namespace. Note that this struct does not
#[derive(Debug, Default, Clone)]
pub struct NativeLoaders(HashMap<String, NativeLoader>);

impl NativeLoaders {
    pub fn insert(&mut self, namespace: &str, loader: NativeLoader) {
        self.0.insert(namespace.to_string(), loader);
    }

    /// Load an annotation from a payload.
    ///
    /// This method uses [iter_namespaces] to find the narrowest namespace contained in this loader
    /// that matches the namespace of a given payload, then uses the corresponding [NativeLoader].
    pub fn load(&self, namespace: &str, payload: &str) -> Option<Arc<dyn Annotation>> {
        if let Some(loader) = iter_namespaces(namespace).find_map(|ns| self.0.get(ns)) {
            loader(namespace, payload)
        } else {
            None
        }
    }
}

/// Create a Python annotation.
///
/// For a [PythonAnnotation], returns the underlying [PyAnnotation], while for other annotation types,
/// creates and returns a [PyNativeAnnotation].
pub fn create_py_annotation(annotation: &Arc<dyn Annotation>, py: Python) -> PyResult<Py<PyAny>> {
    if let Some(annotation) = annotation.downcast_ref::<PythonAnnotation>() {
        return Ok(annotation.annotation(py));
    }
    let init = match PyNativeAnnotation::new(Arc::clone(annotation)) {
        Ok(py_annotation) => PyClassInitializer::from(PyAnnotation).add_subclass(py_annotation),
        Err(e) => return Err(e.into()),
    };
    Ok(Py::new(py, init)?.into_any())
}

/// Used to extract an instance of [Annotation].
pub struct AnnotationFromPython(pub Arc<dyn Annotation>);

impl<'a, 'py> FromPyObject<'a, 'py> for AnnotationFromPython {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        match ob.cast::<PyAnnotation>() {
            Ok(base) => match base.cast::<PyNativeAnnotation>() {
                Ok(native) => Ok(Self(Arc::clone(native.get().inner()))),
                Err(..) => Ok(Self(Arc::new(PythonAnnotation::new(ob.into())))),
            },
            Err(e) => Err(Self::Error::from(e)),
        }
    }
}

#[cfg(test)]
mod test_annotation {
    use crate::annotation::{Annotation, NativeLoaders, iter_namespaces};
    use std::{assert_eq, sync::Arc};

    macro_rules! impl_annotation {
        ($ty:ident; $namespace:expr,) => {
            impl $crate::annotation::Annotation for $ty {
                fn namespace(&self) -> &str {
                    $namespace
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

    #[test]
    fn test_iter_namespaces() {
        assert_eq!(iter_namespaces("").collect::<Vec<_>>(), vec![""]);
        assert_eq!(iter_namespaces("a").collect::<Vec<_>>(), vec!["a", ""]);
        assert_eq!(
            iter_namespaces("hello.world").collect::<Vec<_>>(),
            vec!["hello.world", "hello", ""]
        );
        assert_eq!(
            iter_namespaces("a.b.c").collect::<Vec<_>>(),
            vec!["a.b.c", "a.b", "a", ""]
        );
        assert_eq!(
            iter_namespaces(".leading").collect::<Vec<_>>(),
            vec![".leading", ""]
        );
        assert_eq!(
            iter_namespaces("trailing.").collect::<Vec<_>>(),
            vec!["trailing.", "trailing", ""]
        );
        assert_eq!(iter_namespaces(".").collect::<Vec<_>>(), vec![".", ""]);
        assert_eq!(
            iter_namespaces("a..b").collect::<Vec<_>>(),
            vec!["a..b", "a.", "a", ""]
        );
    }

    #[test]
    fn test_load_native_annotation() {
        assert_eq!(NativeLoaders::default().load("a.b", "c"), None);
    }
}

#[cfg(test)]
mod test_annotated_boxes {
    use smallvec::smallvec;
    use std::sync::Arc;

    use crate::Qubit;
    use crate::annotation::Annotation;
    use crate::circuit_data::CircuitData;
    use crate::dag_circuit::DAGCircuit;
    use crate::instruction::Parameters;
    use crate::operations::{ControlFlow, ControlFlowInstruction, ControlFlowView, Param};
    use crate::packed_instruction::PackedOperation;
    use crate::standard_gate::StandardGate;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Twirl {
        twirl: String,
    }

    impl Annotation for Twirl {
        fn namespace(&self) -> &str {
            "randomization.twirl"
        }
    }

    /// Add a [`ControlFlow::Box`] with a twirl annotation with the given name around
    /// all two-qubit operations.
    pub fn twirl_2q(dag: &mut DAGCircuit, twirl: &str) {
        let twirl: Arc<dyn Annotation> = Arc::new(Twirl {
            twirl: twirl.to_string(),
        });
        let node_indices: Vec<_> = dag
            .two_qubit_ops()
            .map(|(node_idx, _)| node_idx)
            .collect::<Vec<_>>();
        for node_idx in node_indices {
            let instruction = dag[node_idx].unwrap_operation();
            let new_op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
                control_flow: ControlFlow::Box {
                    duration: None,
                    annotations: vec![twirl.clone()],
                },
                num_qubits: 2,
                num_clbits: 0,
            }));

            let mut body = DAGCircuit::new();
            _ = body.apply_operation_back(
                instruction.op.clone(),
                dag.get_qargs(instruction.qubits),
                dag.get_cargs(instruction.clbits),
                None,
                None,
            );
            let block = dag.add_block(body);
            _ = dag.substitute_op(
                node_idx,
                new_op,
                Some(Parameters::Blocks(vec![block])),
                None,
            );
        }
    }

    /// Remove any [`ControlFlow::Box`] with annotations in the given namespace.
    pub fn remove_namespace(dag: &mut DAGCircuit, namespace: &str) {
        let to_remove: Vec<_> = dag
            .op_nodes(false)
            .filter_map(|(node_idx, instr)| {
                if let Some(box_op) = dag.try_view_control_flow(instr) {
                    return match box_op {
                        ControlFlowView::Box { annotations, .. } => {
                            if annotations.iter().any(|annotation| {
                                annotation.namespace().starts_with(namespace)
                            }) {
                                return Some(node_idx);
                            }
                            None
                        }
                        _ => None,
                    };
                }
                None
            })
            .collect();

        for node_idx in to_remove {
            dag.remove_op_node(node_idx);
        }
    }

    #[test]
    fn test_box_annotations() {
        let circuit1 = CircuitData::from_packed_operations(
            2,
            1,
            vec![
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(1)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(1)],
                    vec![],
                )),
            ],
            Param::Float(0.),
        )
        .unwrap();

        let mut dag =
            DAGCircuit::from_circuit_data(&circuit1, false, None, None, None, None).unwrap();

        // Twirl both CXs, this pass replaces two-qubit gates with annotated box with the operation.
        twirl_2q(&mut dag, "twirl");

        // This just checks that the Arc::strong_count is the same on both annotations, it should be unique.
        // The Arc from the pass is out of scope at the assert.
        for op_node_idx in dag.op_node_indices(false) {
            let annotations = match dag
                .try_view_control_flow(dag[op_node_idx].unwrap_operation())
                .unwrap()
            {
                ControlFlowView::Box { annotations, .. } => Some(annotations),
                _ => None,
            }
            .unwrap();
            assert!(annotations.len() == 1);

            let annotation = &annotations[0];
            assert_eq!(2, Arc::strong_count(annotation));
        }

        // Remove every box with an annotation in the namespace.
        remove_namespace(&mut dag, "randomization");
        assert_eq!(0, dag.op_node_indices(false).collect::<Vec<_>>().len());
    }
}

#[cfg(test)]
mod test_annotation_loading {
    use crate::annotation::{Annotation, NativeLoaders};
    use std::sync::Arc;

    #[derive(Debug, PartialEq)]
    struct Twirl {
        twirl: String,
    }

    impl Twirl {
        pub fn from_payload(payload: &str) -> Self {
            let (_, twirl) = payload
                .rsplit_once("twirl:")
                .expect("Should be dispatched.");
            Twirl {
                twirl: twirl.to_string(),
            }
        }
    }

    impl Annotation for Twirl {
        fn namespace(&self) -> &str {
            "randomization.twirl"
        }

        fn payload(&self) -> Option<std::borrow::Cow<'_, str>> {
            Some(std::borrow::Cow::Owned(format!("twirl:{}", self.twirl)))
        }
    }

    #[derive(Debug, PartialEq)]
    struct InjectNoise(String);

    impl InjectNoise {
        pub fn from_payload(payload: &str) -> Self {
            InjectNoise(payload.to_string())
        }
    }

    impl Annotation for InjectNoise {
        fn namespace(&self) -> &str {
            "randomization.inject_noise"
        }

        fn payload(&self) -> Option<std::borrow::Cow<'_, str>> {
            Some(std::borrow::Cow::Borrowed(&self.0))
        }
    }

    #[test]
    fn test_native_loaders() {
        let mut loaders = NativeLoaders::default();

        // A loader than handles the randomization namespace and returns new instances with their corresponding payloads.
        loaders.insert("randomization", |ns, payload| match ns.rsplit_once(".") {
            Some((_, ns)) => match ns {
                "twirl" => Some(Arc::new(Twirl::from_payload(payload))),
                "inject_noise" => Some(Arc::new(InjectNoise::from_payload(payload))),
                _ => None,
            },
            None => None,
        });

        // A loader with a narrower namespace that returns an inject noise with a fixed payload.
        loaders.insert("randomization.inject_noise", |_, _| {
            Some(Arc::new(InjectNoise("different".to_string())))
        });

        let annotation: Arc<dyn Annotation> = Arc::new(Twirl {
            twirl: "pauli".to_string(),
        });
        let roundtrip = loaders
            .load(
                "randomization.twirl",
                &annotation.payload().expect("It's implemented."),
            )
            .expect("It's implemented.");
        assert_eq!(roundtrip.as_ref(), annotation.as_ref());

        let annotation: Arc<dyn Annotation> = Arc::new(InjectNoise("ok".to_string()));
        let roundtrip = loaders
            .load(
                "randomization.inject_noise",
                &annotation.payload().expect("It's implemented."),
            )
            .expect("It's implemented.");
        let expected: Arc<dyn Annotation> = Arc::new(InjectNoise("different".to_string()));
        assert_ne!(roundtrip.as_ref(), annotation.as_ref());
        assert_eq!(roundtrip.as_ref(), expected.as_ref());
    }
}
