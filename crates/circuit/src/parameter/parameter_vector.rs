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

use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyString, PyTuple};
use uuid::Uuid;

use crate::parameter::parameter_expression::{uuid_from_py, uuid_to_py, PyParameterVectorElement};

#[pyclass(name = "ParameterVector", module = "qiskit.circuit.parametervector", subclass, sequence)]
pub struct PyParameterVector {
    #[pyo3(get)]
    name: String,
    root_uuid: u128,
    #[pyo3(get)]
    params: Py<PyList>,
}

#[pymethods]
impl PyParameterVector {
    #[new]
    #[pyo3(signature = (name, length=0, uuid=None))]
    fn __new__(py: Python, name: String, length: u32, uuid: Option<Py<PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let root_uuid = uuid_from_py(py, uuid)?.unwrap_or_else(Uuid::new_v4).as_u128();
        Ok(Self {
            name,
            root_uuid,
            params: PyList::empty(py).unbind(),
        }.into())
    }

    #[pyo3(signature = (name, length=0, uuid=None))]
    fn __init__(slf: &Bound<'_, Self>, py: Python, name: String, length: u32, uuid: Option<Py<PyAny>>) -> PyResult<()> {
        Self::resize(slf, py, length)
    }

    #[getter]
    fn _name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn uuid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_py(py, Uuid::from_u128(self.root_uuid)).map(|obj| obj.into_bound(py))
    }

    fn index(&self, py: Python, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.params.bind(py).call_method1("index", (value,))?.extract()
    }

    fn __getitem__(&self, py: Python, key: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(self.params.bind(py).as_any().get_item(key)?.unbind())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        pyo3::types::PyIterator::from_object(&self.params.bind(py).as_any())
    }

    fn __len__(&self, py: Python) -> usize {
        self.params.bind(py).len()
    }

    fn __str__(&self, py: Python) -> PyResult<String> {
        let mut elements = Vec::new();
        for param in self.params.bind(py).iter() {
            let s: String = param.call_method0("__str__")?.extract()?;
            elements.push(s);
        }
        let list_str = format!("[{}]", elements.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", "));
        Ok(format!("{}, {}", self.name, list_str))
    }

    fn __repr__(&self, py: Python) -> String {
        format!("ParameterVector(name='{}', length={})", self.name, self.params.bind(py).len())
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, vec![
            self.name.clone().into_pyobject(py)?.into_any(),
            self.params.bind(py).len().into_pyobject(py)?.into_any()
        ])
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let params_state = PyList::empty(py);
        for param in self.params.bind(py).iter() {
            params_state.append(param.call_method0("__getstate__")?)?;
        }
        let uuid_obj = uuid_to_py(py, Uuid::from_u128(self.root_uuid))?;
        PyTuple::new(py, vec![
            self.name.clone().into_pyobject(py)?.into_any(),
            params_state.into_any(),
            uuid_obj.into_bound(py)
        ])
    }

    fn __setstate__(slf: &Bound<'_, Self>, py: Python, state: Bound<'_, PyTuple>) -> PyResult<()> {
        let name = state.get_item(0)?.extract::<String>()?;
        let item1 = state.get_item(1)?;
        let params_list = item1.downcast::<PyList>()?;
        let uuid_obj = state.get_item(2)?;
        
        let root_uuid = uuid_from_py(py, Some(uuid_obj.into()))?.unwrap().as_u128();
        
        let sys = py.import("sys")?;
        let module = sys.getattr("modules")?.get_item("qiskit.circuit.parametervector")?;
        let pve_class = module.getattr("ParameterVectorElement")?;
        
        let new_params = PyList::empty(py);
        for param_state in params_list.iter() {
            let p_tuple = param_state.downcast::<PyTuple>()?;
            let vector = p_tuple.get_item(0)?;
            let index = p_tuple.get_item(1)?;
            let uuid = p_tuple.get_item(2)?;
            
            let element = pve_class.call1((vector, index, uuid))?;
            new_params.append(element)?;
        }
        
        let mut slf_mut = slf.borrow_mut();
        slf_mut.name = name;
        slf_mut.root_uuid = root_uuid;
        slf_mut.params = new_params.unbind();
        Ok(())
    }

    fn resize(slf: &Bound<'_, Self>, py: Python, length: u32) -> PyResult<()> {
        let current_len = slf.borrow().params.bind(py).len() as u32;
        let root_uuid = slf.borrow().root_uuid;
        if length > current_len {
            for i in current_len..length {
                let uuid = Uuid::from_u128(root_uuid + (i as u128));
                let uuid_py = uuid_to_py(py, uuid)?;
                
                let element_init = PyParameterVectorElement::py_new(
                    py,
                    slf.clone().into_any().unbind(),
                    i,
                    Some(uuid_py)
                )?;
                
                let element_obj = Py::new(py, element_init)?;
                slf.borrow().params.bind(py).append(element_obj)?;
            }
        } else {
            let params = slf.borrow().params.bind(py).clone();
            for _ in length..current_len {
                params.call_method0("pop")?;
            }
        }
        Ok(())
    }
}
