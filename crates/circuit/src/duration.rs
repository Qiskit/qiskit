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

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::PyTypeInfo;

/// A length of time used to express circuit timing.
///
/// It defines a group of classes which are all subclasses of itself (functionally, an
/// enumeration carrying data).
///
/// In Python 3.10+, you can use it in a match statement::
///
///   match duration:
///      case Duration.dt(dt):
///          return dt
///      case Duration.s(seconds):
///          return seconds / 5e-7
///      case _:
///          raise ValueError("expected dt or seconds")
///
/// And in Python 3.9, you can use :meth:`Duration.unit` to determine which variant
/// is populated::
///
///   if duration.unit() == "dt":
///       return duration.value()
///   elif duration.unit() == "s":
///       return duration.value() / 5e-7
///   else:
///       raise ValueError("expected dt or seconds")
#[pyclass(eq, module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub enum Duration {
    dt(i64),
    ns(f64),
    us(f64),
    ms(f64),
    s(f64),
}

#[pymethods]
impl Duration {
    /// The corresponding ``unit`` of the duration.
    fn unit(&self) -> &'static str {
        match self {
            Duration::dt(_) => "dt",
            Duration::us(_) => "us",
            Duration::ns(_) => "ns",
            Duration::ms(_) => "ms",
            Duration::s(_) => "s",
        }
    }

    /// The ``value`` of the duration.
    ///
    /// This will be a Python ``int`` if the :meth:`~Duration.unit` is ``"dt"``,
    /// else a ``float``.
    #[pyo3(name = "value")]
    fn py_value(&self, py: Python) -> PyResult<PyObject> {
        match self {
            Duration::dt(v) => v.into_py_any(py),
            Duration::us(v) | Duration::ns(v) | Duration::ms(v) | Duration::s(v) => {
                v.into_py_any(py)
            }
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Duration::ns(t) => format!("Duration.ns({})", t),
            Duration::us(t) => format!("Duration.us({})", t),
            Duration::ms(t) => format!("Duration.ms({})", t),
            Duration::s(t) => format!("Duration.s({})", t),
            Duration::dt(t) => format!("Duration.dt({})", t),
        }
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (
            Duration::type_object(py).getattr(self.unit())?,
            (self.py_value(py)?,),
        )
            .into_py_any(py)
    }
}
