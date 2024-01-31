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

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use hashbrown::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use oq3_semantics::syntax_to_semantics::parse_source_string;

use crate::error::QASM3ImporterError;

/// The name of a Python attribute to define on the given module where the default implementation
/// of the ``stdgates.inc`` custom instructions is located.
const STDGATES_INC_CUSTOM_GATES_ATTR: &str = "STDGATES_INC_GATES";

/// Load an OpenQASM 3 program from a string into a :class:`.QuantumCircuit`.
///
/// .. warning::
///
///     This native version of the OpenQASM 3 importer is currently experimental.  It is typically
///     much faster than :func:`~qiskit.qasm3.loads`, but has a reduced supported feature set,
///     which will expand over time.
///
/// Args:
///     source (str): the program source in a Python string.
///     custom_gates (Iterable[CustomGate]): Python constructors to use for particular named gates.
///         If not supplied, Qiskit will use its own standard-library constructors for gates
///         defined in the OpenQASM 3.0 standard-library file ``stdgates.inc``.
///     include_path (Iterable[str]): the path to search when resolving ``include`` statements.
///         If not given, Qiskit will arrange for this to point to a location containing
///         ``stdgates.inc`` only.  Paths are tried in the sequence order.
///
/// Returns:
///     :class:`.QuantumCircuit`: the constructed circuit object.
///
/// Raises:
///     :class:`.QASM3ImporterError`: if an error occurred during parsing or semantic analysis.
///         In the case of a parsing error, most of the error messages are printed to the terminal
///         and formatted, for better legibility.
#[pyfunction]
#[pyo3(pass_module, signature = (source, /, *, custom_gates=None, include_path=None))]
pub fn loads(
    module: &PyModule,
    py: Python,
    source: String,
    custom_gates: Option<Vec<circuit::PyGate>>,
    include_path: Option<Vec<OsString>>,
) -> PyResult<circuit::PyCircuit> {
    let default_include_path = || -> PyResult<Vec<OsString>> {
        Ok(vec![Path::new(module.filename()?)
            .parent()
            .unwrap()
            .join(["qasm", "libs", "dummy"].iter().collect::<PathBuf>())
            .into_os_string()])
    };
    let include_path = include_path.map(Ok).unwrap_or_else(default_include_path)?;
    let result = parse_source_string(source, None, Some(&include_path));
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
            .getattr(STDGATES_INC_CUSTOM_GATES_ATTR)?
            .iter()?
            .map(|obj| {
                let gate = obj?.extract::<circuit::PyGate>()?;
                Ok((gate.name().to_owned(), gate))
            })
            .collect::<PyResult<HashMap<_, _>>>()?,
    };
    crate::build::convert_asg(py, result.program(), result.symbol_table(), gates)
}

/// Load an OpenQASM 3 program from a source file into a :class:`.QuantumCircuit`.
///
/// .. warning::
///
///     This native version of the OpenQASM 3 importer is currently experimental.  It is typically
///     much faster than :func:`~qiskit.qasm3.load`, but has a reduced supported feature set, which
///     will expand over time.
///
/// Args:
///     pathlike_or_filelike (str | os.PathLike | io.TextIOBase): the program source.  This can
///         either be given as a filepath, or an open text stream object.  If the stream is already
///         opened it is consumed in Python space, whereas filenames are opened and consumed in
///         Rust space; there might be slightly different performance characteristics, depending on
///         your system and how the streams are buffered by default.
///     custom_gates (Iterable[CustomGate]): Python constructors to use for particular named gates.
///         If not supplied, Qiskit will use its own standard-library constructors for gates
///         defined in the OpenQASM 3.0 standard-library file ``stdgates.inc``.
///     include_path (Iterable[str]): the path to search when resolving ``include`` statements.
///         If not given, Qiskit will arrange for this to point to a location containing
///         ``stdgates.inc`` only.  Paths are tried in the sequence order.
///
/// Returns:
///     :class:`.QuantumCircuit`: the constructed circuit object.
///
/// Raises:
///     :class:`.QASM3ImporterError`: if an error occurred during parsing or semantic analysis.
///         In the case of a parsing error, most of the error messages are printed to the terminal
///         and formatted, for better legibility.
#[pyfunction]
#[pyo3(
    pass_module,
    signature = (pathlike_or_filelike, /, *, custom_gates=None, include_path=None),
)]
pub fn load(
    module: &PyModule,
    py: Python,
    pathlike_or_filelike: &PyAny,
    custom_gates: Option<Vec<circuit::PyGate>>,
    include_path: Option<Vec<OsString>>,
) -> PyResult<circuit::PyCircuit> {
    let source =
        if pathlike_or_filelike.is_instance(PyModule::import(py, "io")?.getattr("TextIOBase")?)? {
            pathlike_or_filelike
                .call_method0("read")?
                .extract::<String>()?
        } else {
            let path = PyModule::import(py, "os")?
                .getattr("fspath")?
                .call1((pathlike_or_filelike,))?
                .extract::<OsString>()?;
            ::std::fs::read_to_string(&path).map_err(|err| {
                QASM3ImporterError::new_err(format!("failed to read file '{:?}': {:?}", &path, err))
            })?
        };
    loads(module, py, source, custom_gates, include_path)
}

/// Create a suitable sequence for use with the ``custom_gates`` of :func:`load` and :func:`loads`,
/// as a Python object on the Python heap, so we can re-use it, and potentially expose it has a
/// data attribute to users.
fn stdgates_inc_gates(py: Python) -> PyResult<&PyTuple> {
    let library = PyModule::import(py, "qiskit.circuit.library")?;
    let stdlib_gate = |qiskit_class, name, num_params, num_qubits| -> PyResult<Py<PyAny>> {
        Ok(circuit::PyGate::new(
            py,
            library.getattr(qiskit_class)?,
            name,
            num_params,
            num_qubits,
        )
        .into_py(py))
    };
    Ok(PyTuple::new(
        py,
        vec![
            stdlib_gate("PhaseGate", "p", 1, 1)?,
            stdlib_gate("XGate", "x", 0, 1)?,
            stdlib_gate("YGate", "y", 0, 1)?,
            stdlib_gate("ZGate", "z", 0, 1)?,
            stdlib_gate("HGate", "h", 0, 1)?,
            stdlib_gate("SGate", "s", 0, 1)?,
            stdlib_gate("SdgGate", "sdg", 0, 1)?,
            stdlib_gate("TGate", "t", 0, 1)?,
            stdlib_gate("TdgGate", "tdg", 0, 1)?,
            stdlib_gate("SXGate", "sx", 0, 1)?,
            stdlib_gate("RXGate", "rx", 1, 1)?,
            stdlib_gate("RYGate", "ry", 1, 1)?,
            stdlib_gate("RZGate", "rz", 1, 1)?,
            stdlib_gate("CXGate", "cx", 0, 2)?,
            stdlib_gate("CYGate", "cy", 0, 2)?,
            stdlib_gate("CZGate", "cz", 0, 2)?,
            stdlib_gate("CPhaseGate", "cp", 1, 2)?,
            stdlib_gate("CRXGate", "crx", 1, 2)?,
            stdlib_gate("CRYGate", "cry", 1, 2)?,
            stdlib_gate("CRZGate", "crz", 1, 2)?,
            stdlib_gate("CHGate", "ch", 0, 2)?,
            stdlib_gate("SwapGate", "swap", 0, 2)?,
            stdlib_gate("CCXGate", "ccx", 0, 3)?,
            stdlib_gate("CSwapGate", "cswap", 0, 3)?,
            stdlib_gate("CUGate", "cu", 4, 2)?,
            stdlib_gate("CXGate", "CX", 0, 2)?,
            stdlib_gate("PhaseGate", "phase", 1, 1)?,
            stdlib_gate("CPhaseGate", "cphase", 1, 2)?,
            stdlib_gate("IGate", "id", 0, 1)?,
            stdlib_gate("U1Gate", "u1", 1, 1)?,
            stdlib_gate("U2Gate", "u2", 2, 1)?,
            stdlib_gate("U3Gate", "u3", 3, 1)?,
        ],
    ))
}

/// Internal module supplying the OpenQASM 3 import capabilities.  The entries in it should largely
/// be re-exposed directly to public Python space.
#[pymodule]
fn _qasm3(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loads, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    module.add_class::<circuit::PyGate>()?;
    module.add(STDGATES_INC_CUSTOM_GATES_ATTR, stdgates_inc_gates(py)?)?;
    Ok(())
}
