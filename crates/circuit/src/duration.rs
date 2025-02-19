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

#[pyclass(eq, module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub enum Duration {
    dt(u64),
    ns(f64),
    us(f64),
    ms(f64),
    s(f64),
}

impl Duration {
    fn __repr__(&self) -> String {
        match self {
            Duration::ns(t) => format!("Duration.ns({})", t),
            Duration::us(t) => format!("Duration.us({})", t),
            Duration::ms(t) => format!("Duration.ms({})", t),
            Duration::s(t) => format!("Duration.s({})", t),
            Duration::dt(t) => format!("Duration.dt({})", t),
        }
    }
}
