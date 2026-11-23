// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! The Python binding of a program function being built, and of the program it becomes.

use std::collections::BTreeMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::data_tree::{ObjectTree, PyDataTree};
use super::ops::PyProgramOp;
use super::tensor::{tensor, tensor_object};
use super::value_error;
use crate::data_tree::DataTree;
use crate::program::{ProgramFunction, QuantumProgram, Value};
use crate::tensor::TensorType;

/// One tensor value: an output slot of the instruction that produces it.
#[pyclass(
    name = "Value",
    module = "qiskit.quantum_program",
    frozen,
    eq,
    from_py_object,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyValue(Value);

#[pymethods]
impl PyValue {
    /// The instruction producing this value, by its position in the function holding it.
    #[getter]
    fn instruction(&self) -> usize {
        self.0.instruction().index()
    }

    /// Which of that instruction's results this value is.
    #[getter]
    fn slot(&self) -> usize {
        self.0.slot()
    }

    fn __repr__(&self) -> String {
        format!("Value({})", self.0)
    }
}

/// A dataflow graph being assembled, instruction by instruction.
///
/// Every instruction is type-checked as it is added, so the function cannot be malformed. Its
/// parameters and results are positional: they are added in the order the structures given to
/// [`seal`](Self::seal) name them.
#[pyclass(
    name = "FunctionBuilder",
    module = "qiskit._accelerate.quantum_program"
)]
pub struct PyFunctionBuilder(ProgramFunction);

#[pymethods]
impl PyFunctionBuilder {
    #[new]
    fn new() -> Self {
        Self(ProgramFunction::new())
    }

    /// Declare a parameter of type `ty`, returning its value.
    #[pyo3(signature = (ty, /))]
    fn add_parameter(&mut self, ty: TensorType) -> PyValue {
        PyValue(self.0.add_parameter(ty))
    }

    /// Apply `op` to `operands`, returning the values it produces.
    #[pyo3(signature = (op, operands, /))]
    fn add_op(&mut self, op: &PyProgramOp, operands: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        let operands: Vec<Value> = operands.into_iter().map(|value| value.0).collect();
        self.0
            .add_boxed_op(op.op.to_owned(), &operands)
            .map(|values| values.into_iter().map(PyValue).collect())
            .map_err(|error| value_error(&error))
    }

    /// Declare `value` as the next result.
    #[pyo3(signature = (value, /))]
    fn add_result(&mut self, value: PyValue) -> PyResult<()> {
        self.0
            .add_result(value.0)
            .map_err(|error| value_error(&error))
    }

    /// Seal this function into a program whose inputs and outputs are arranged as the given trees.
    ///
    /// Only the shape and the names of the trees are read; their leaves are discarded. The builder
    /// is left empty.
    #[pyo3(signature = (inputs, outputs, /))]
    fn seal(&mut self, inputs: &PyDataTree, outputs: &PyDataTree) -> PyResult<PyQuantumProgram> {
        let function = std::mem::take(&mut self.0);
        QuantumProgram::new(vec![function], inputs.0.structure(), outputs.0.structure())
            .map(PyQuantumProgram)
            .map_err(|error| value_error(&error))
    }
}

/// A hybrid quantum-classical computation, described rather than performed.
///
/// A program declares the type of every input it takes and every output it produces, so both are
/// known before it runs. Calling it supplies one keyword argument per input and gives back a
/// `DataTree` of results arranged as `output_types()` describes.
#[pyclass(name = "QuantumProgram", module = "qiskit.quantum_program", frozen)]
pub struct PyQuantumProgram(QuantumProgram);

#[pymethods]
impl PyQuantumProgram {
    /// Return the declared type of every input, arranged as the program's input structure.
    ///
    /// Returns:
    ///     A data tree of tensor types.
    fn input_types(&self, py: Python<'_>) -> PyResult<PyDataTree> {
        let types = self.0.input_types();
        Ok(PyDataTree(object_tree(&types, |ty| {
            Ok(Py::new(py, ty.clone())?.into_any())
        })?))
    }

    /// Return the type of every output, arranged as the program's output structure.
    ///
    /// Type inference ran as the program was built, so this needs no evaluation.
    ///
    /// Returns:
    ///     A data tree of tensor types.
    fn output_types(&self, py: Python<'_>) -> PyResult<PyDataTree> {
        let types = self.0.output_types();
        Ok(PyDataTree(object_tree(&types, |ty| {
            Ok(Py::new(py, ty.clone())?.into_any())
        })?))
    }

    /// Evaluate the program on one keyword argument per declared input.
    ///
    /// Each argument is read with `numpy.asarray` and must then match its declared type: the
    /// program is monomorphic, so nothing is promoted here.
    ///
    /// Args:
    ///     inputs: One value per declared input, by keyword.
    ///
    /// Returns:
    ///     A data tree of arrays, arranged as the program's output structure.
    ///
    /// Raises:
    ///     TypeError: If the keywords are not the declared inputs, if the input structure is not one
    ///         of named values, if an input is read as a dtype no tensor holds, or if an output has no
    ///         NumPy form.
    ///     ValueError: If a value does not match the type its input declares, or if the program
    ///         holds work Qiskit cannot perform in process.
    #[pyo3(signature = (**inputs))]
    fn __call__(&self, py: Python<'_>, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataTree> {
        let declared = self.0.input_types();
        let expected = keyword_inputs(&declared)?;

        let names: Vec<&str> = expected.iter().map(|&(name, _)| name).collect();
        let mut arguments = Vec::with_capacity(expected.len());
        let mut missing = Vec::new();
        for &(name, _) in &expected {
            match inputs.map(|inputs| inputs.get_item(name)).transpose()? {
                Some(Some(argument)) => arguments.push(argument),
                _ => missing.push(name),
            }
        }
        let mut unexpected = Vec::new();
        for key in inputs.iter().flat_map(|inputs| inputs.keys()) {
            let key = key.extract::<String>()?;
            if !names.contains(&key.as_str()) {
                unexpected.push(key);
            }
        }
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "this program takes inputs {}; missing {}, unexpected {}",
                name_list(&names),
                name_list(&missing),
                name_list(&unexpected),
            )));
        }

        let mut tree = DataTree::with_capacity(expected.len());
        for ((name, ty), argument) in expected.into_iter().zip(arguments) {
            let value = tensor(&argument)?;
            if !value.matches(ty) {
                return Err(PyValueError::new_err(format!(
                    "input '{name}': expected {ty}, got {}",
                    value.tensor_type()
                )));
            }
            tree.insert_leaf(name, value)
                .map_err(|err| value_error(&err))?;
        }

        let outputs = self.0.eval(tree).map_err(|error| value_error(&error))?;
        let structure = outputs.structure();
        let arrays = outputs
            .into_leaves()
            .map(|value| tensor_object(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyDataTree(structure.unflatten(arrays).expect(
            "a structure has one leaf per leaf of the tree it came from",
        )))
    }

    /// Return how many instructions the program holds of each op.
    fn _type_name_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for function in self.0.functions() {
            for instruction in function.iter_instructions() {
                *counts.entry(instruction.full_name()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumProgram(inputs={}, outputs={})",
            self.0.input_structure(),
            self.0.output_structure()
        )
    }
}

/// Format `names` as Python renders a list of strings.
fn name_list(names: &[impl AsRef<str>]) -> String {
    let names: Vec<String> = names
        .iter()
        .map(|name| format!("'{}'", name.as_ref()))
        .collect();
    format!("[{}]", names.join(", "))
}

/// Return the name and declared type of each input, in the order the program takes them.
fn keyword_inputs(declared: &DataTree<TensorType>) -> PyResult<Vec<(&str, &TensorType)>> {
    let refuse = || {
        Err(PyTypeError::new_err(format!(
            "calling a program by keyword needs an input structure of named values, and this one \
             is {}",
            declared.structure()
        )))
    };
    if matches!(declared, DataTree::Leaf(_)) {
        return refuse();
    }
    let mut inputs = Vec::with_capacity(declared.len());
    for (name, child) in declared.iter_children() {
        let (Some(name), DataTree::Leaf(ty)) = (name, child) else {
            return refuse();
        };
        inputs.push((name, ty));
    }
    Ok(inputs)
}

/// Return `tree` with each leaf converted to a Python object by `object`.
fn object_tree<T>(
    tree: &DataTree<T>,
    mut object: impl FnMut(&T) -> PyResult<Py<PyAny>>,
) -> PyResult<ObjectTree> {
    let leaves = tree
        .iter_leaves()
        .map(&mut object)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(tree
        .structure()
        .unflatten(leaves)
        .expect("a structure has one leaf per leaf of the tree it came from"))
}
