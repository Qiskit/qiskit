use pyo3::prelude::*;

#[pyclass]
pub struct Dummy {
    pub name: String,
}

#[pymethods]
impl Dummy {
    #[new]
    fn new(py: Python, name: String) -> PyResult<Bound<'_, Self>> {
        let instance = Py::new(py, Self { name })?.into_bound(py);
        Ok(instance)
    }
}
