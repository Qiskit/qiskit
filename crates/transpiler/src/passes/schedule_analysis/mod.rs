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

pub mod alap_schedule_analysis;
pub mod asap_schedule_analysis;

use std::ops::{Add, Deref, Sub};

use hashbrown::HashMap;
use pyo3::{
    exceptions::{PyKeyError, PyTypeError}, prelude::*, types::{PyDict, PyList}, IntoPyObjectExt
};
use qiskit_circuit::dag_node::{DAGNode, DAGOpNode};
use rustworkx_core::petgraph::graph::NodeIndex;

pub trait TimeOps: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> {
    fn zero() -> Self;
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self;
}

impl TimeOps for u64 {
    fn zero() -> Self {
        0
    }
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self {
        if a >= b { a } else { b }
    }
}

impl TimeOps for f64 {
    fn zero() -> Self {
        0.0
    }
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self {
        if a >= b { a } else { b }
    }
}

/*
For the following NoteDuration methods, we need a mapping between NodeIndices
and a node duration. Node durations may either an integer or a floating point
number depending on the duration's unit. How do we properly introduce this mapping?

We have a couple of things to consider:

1. The mapping must be accessible to both, python and rust, to avoid conversion
overhead as the other methods perform worse due to that.
2. The mapping should be able to have values that can be either floats or ints.
3. The mapping must be able to be used the same way in both ASAP and ALAP algorithms.

For the values, should we do an enum struct? Or do we introduce an enum for either type?
Since in a mapping of durations the type of duration stays stable between entries,
it would make sense for the reliability of the type to be stored as part of the struct
itself instead of making it something that the values store.

Let's picture those constraints below:

#[pyclass(mapping, class="foo",)]
pub struct PyNodeDurations (
    // the mapping between nodes and durations
    // We cannot represent this correctly using generics but with an enum
    NodeDurations,
)

pub enum NodeDurations {
    Dt(HashMap<NodeIndex<_>, u64>),
    Seconds(HashMap<NodeIndex<_>, f64>)
}

impl<'py> FromPyObject<'py> for NodeDurations {
    //...
}
*/

#[pyclass(
    mapping,
    name = "NodeDurations",
    module = "qiskit._accelerate.scheduling"
)]
#[derive(Debug, Clone)]
pub struct PyNodeDurations(NodeDurations);

impl Deref for PyNodeDurations {
    type Target = NodeDurations;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyNodeDurations {
    #[new]
    pub fn new(mapping: NodeDurations) -> Self {
        Self(mapping)
    }

    fn __getitem__<'py>(
        &'py self,
        node: Bound<'py, DAGOpNode>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let node_as_base: &Bound<DAGNode> = node.cast()?;
        let py = node.py();
        match &self.0 {
            NodeDurations::Dt(map) => map
                .get(&node_as_base.borrow().node.expect("Node index not found."))
                .ok_or(PyKeyError::new_err(format!(
                    "key '{}' not in mapping",
                    node.repr()?
                )))?
                .into_bound_py_any(py),
            NodeDurations::Seconds(map) => map
                .get(&node_as_base.borrow().node.expect("Node index not found."))
                .ok_or(PyKeyError::new_err(format!(
                    "key '{}' not in mapping",
                    node.repr()?
                )))?
                .into_bound_py_any(py),
        }
    }

    fn __setitem__<'py>(
        &'py mut self,
        node: Bound<'py, DAGOpNode>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let node_as_base: &Bound<DAGNode> = node.cast()?;
        match &mut self.0 {
            NodeDurations::Dt(map) => {
                let value = value.extract()?;
                map
                .entry(node_as_base.borrow().node.expect("Node index not found."))
                .and_modify(|val| *val = value)
                .or_insert(value);
                Ok(())
            }
            NodeDurations::Seconds(map) => {
                let value = value.extract()?;
                map
                .entry(node_as_base.borrow().node.expect("Node index not found."))
                .and_modify(|val| *val = value)
                .or_insert(value);
                Ok(())
            }
        }
    }

    fn items<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &**self {
            NodeDurations::Dt(map) => PyList::new(py, map.iter().map(|(k, v)| (k.index(), *v))),
            NodeDurations::Seconds(map) => PyList::new(py, map.iter().map(|(k, v)| (k.index(), *v))),
        }
    }

    fn get<'py>(
        &'py self,
        node: Bound<'py, DAGOpNode>,
        default: Bound<'py, PyAny>,
    ) -> Bound<'py, PyAny> {
        match self.__getitem__(node) {
            Ok(res) => res,
            Err(_) => default,
        }
    }

    fn __contains__<'py>(&'py self, node: Bound<'py, PyAny>) -> bool {
        node.cast_into()
            .map(|node| self.__getitem__(node).is_ok())
            .is_ok_and(|val| val)
    }

    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.0.clear();
    }

    fn copy(&self) -> Self {
        self.__copy__()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
}

#[derive(Debug, Clone)]
pub enum NodeDurations {
    Dt(HashMap<NodeIndex<u32>, u64>),
    Seconds(HashMap<NodeIndex<u32>, f64>),
}

impl NodeDurations {
    pub fn clear(&mut self) {
        match self {
            NodeDurations::Dt(map) => map.clear(),
            NodeDurations::Seconds(map) => map.clear(),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for NodeDurations {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let dict_downcast: Borrowed<'a, 'py, PyDict> = obj.cast()?;
        if let Some((_, key)) = dict_downcast.iter().next() {
            if key.extract::<u64>().is_ok() {
                // All durations are of type u64
                let mut op_durations = HashMap::default();
                for (py_node, py_duration) in dict_downcast.iter() {
                    let node_idx = py_node
                        .cast_into::<DAGOpNode>()?
                        .cast::<DAGNode>()?
                        .borrow()
                        .node
                        .expect("Node index not found.");
                    let val = py_duration.extract()?;
                    op_durations.insert(node_idx, val);
                }
                Ok(Self::Dt(op_durations))
            } else if key.extract::<f64>().is_ok() {
                // All durations are of type u64
                let mut op_durations = HashMap::default();
                for (py_node, py_duration) in dict_downcast.iter() {
                    let node_idx = py_node
                        .cast_into::<DAGOpNode>()?
                        .cast::<DAGNode>()?
                        .borrow()
                        .node
                        .expect("Node index not found.");
                    let val = py_duration.extract()?;
                    op_durations.insert(node_idx, val);
                }
                Ok(Self::Seconds(op_durations))
            } else {
                Err(PyTypeError::new_err(
                    "Only integer or float types allowed for durations",
                ))
            }
        } else {
            Ok(Self::Dt(Default::default()))
        }
    }
}

impl From<HashMap<NodeIndex<u32>, u64>> for NodeDurations {
    fn from(value: HashMap<NodeIndex<u32>, u64>) -> Self {
        Self::Dt(value)
    }
}

impl From<HashMap<NodeIndex<u32>, f64>> for NodeDurations {
    fn from(value: HashMap<NodeIndex<u32>, f64>) -> Self {
        Self::Seconds(value)
    }
}

pub fn scheduling_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeDurations>()?;
    Ok(())
}
