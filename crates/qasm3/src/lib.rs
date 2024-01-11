// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod build;
mod circuit;
mod error;
mod expr;

use hashbrown::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use oq3_semantics::syntax_to_semantics::parse_source_string;

use crate::error::QASM3ImporterError;

/// Load an OpenQASM 3 program from a string into a :class:`.QuantumCircuit`.
#[pyfunction]
#[pyo3(pass_module)]
pub fn loads(
    module: &PyModule,
    py: Python,
    source: String,
    custom_gates: Option<Vec<circuit::PyGate>>,
) -> PyResult<circuit::PyCircuit> {
    let result = parse_source_string(source, None);
    if result.any_errors() {
        result.print_errors();
        return Err(QASM3ImporterError::new_err(
            "errors during parsing; see printed errors",
        ));
    }
    let gates = match custom_gates {
        Some(gates) => gates
            .into_iter()
            .map(|gate| (gate.name().to_owned(), gate))
            .collect(),
        None => module
            .getattr("_STDGATES_INC_GATES")?
            .iter()?
            .map(|obj| {
                let gate = obj?.extract::<circuit::PyGate>()?;
                Ok((gate.name().to_owned(), gate))
            })
            .collect::<PyResult<HashMap<_, _>>>()?,
    };
    crate::build::convert_asg(py, result.program(), result.symbol_table(), gates)
}

fn stdgates_inc_gates(py: Python) -> PyResult<&PyTuple> {
    let library = PyModule::import(py, "qiskit.circuit.library")?;
    Ok(PyTuple::new(
        py,
        vec![
            circuit::PyGate::new(py, library.getattr("PhaseGate")?, "p".to_owned(), 1, 1)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("XGate")?, "x".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("YGate")?, "y".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("ZGate")?, "z".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("HGate")?, "h".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("SGate")?, "s".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("SdgGate")?, "sdg".to_owned(), 0, 1)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("TGate")?, "t".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("TdgGate")?, "tdg".to_owned(), 0, 1)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("SXGate")?, "sx".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("RXGate")?, "rx".to_owned(), 1, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("RYGate")?, "ry".to_owned(), 1, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("RZGate")?, "rz".to_owned(), 1, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("CXGate")?, "cx".to_owned(), 0, 2).into_py(py),
            circuit::PyGate::new(py, library.getattr("CYGate")?, "cy".to_owned(), 0, 2).into_py(py),
            circuit::PyGate::new(py, library.getattr("CZGate")?, "cz".to_owned(), 0, 2).into_py(py),
            circuit::PyGate::new(py, library.getattr("CPhaseGate")?, "cp".to_owned(), 1, 2)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CRXGate")?, "crx".to_owned(), 1, 2)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CRYGate")?, "cry".to_owned(), 1, 2)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CRZGate")?, "crz".to_owned(), 1, 2)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CHGate")?, "ch".to_owned(), 0, 2).into_py(py),
            circuit::PyGate::new(py, library.getattr("SwapGate")?, "swap".to_owned(), 0, 2)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CCXGate")?, "ccx".to_owned(), 0, 3)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CSwapGate")?, "cswap".to_owned(), 0, 3)
                .into_py(py),
            circuit::PyGate::new(py, library.getattr("CUGate")?, "cu".to_owned(), 4, 2).into_py(py),
            circuit::PyGate::new(py, library.getattr("CXGate")?, "CX".to_owned(), 0, 2).into_py(py),
            circuit::PyGate::new(py, library.getattr("PhaseGate")?, "phase".to_owned(), 1, 1)
                .into_py(py),
            circuit::PyGate::new(
                py,
                library.getattr("CPhaseGate")?,
                "cphase".to_owned(),
                1,
                2,
            )
            .into_py(py),
            circuit::PyGate::new(py, library.getattr("IGate")?, "id".to_owned(), 0, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("U1Gate")?, "u1".to_owned(), 1, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("U2Gate")?, "u2".to_owned(), 2, 1).into_py(py),
            circuit::PyGate::new(py, library.getattr("U3Gate")?, "u3".to_owned(), 3, 1).into_py(py),
        ],
    ))
}

#[pymodule]
fn _qasm3(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loads, module)?)?;
    module.add_class::<circuit::PyGate>()?;
    module.add(
        "QASM3ImporterError",
        py.get_type::<error::QASM3ImporterError>(),
    )?;
    module.add("_STDGATES_INC_GATES", stdgates_inc_gates(py)?)?;
    Ok(())
}
