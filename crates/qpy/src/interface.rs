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

// QPY interface module providing high-level dump/load functions
//
// This module provides the main entry points for serializing and deserializing
// quantum circuits to/from QPY format. It handles the complete file structure
// including headers, circuit tables, and multiple circuits.

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use qiskit_circuit::converters::QuantumCircuitData;

use crate::bytes::Bytes;
use crate::circuit_writer::pack_circuit;
use crate::formats;
use crate::formats::{QPY17File, QPYFile};
use crate::value::ValueType;
use crate::value::serialize;

// TODO: use env!("CARGO_PKG_VERSION")
const QISKIT_VERSION: (u8, u8, u8) = (2, 4, 0); // TODO: placeholder; should be replaced with rust code reading VERSION.txt

pub fn dump_qpy(
    mut circuits: Vec<QuantumCircuitData>,
    metadata_serializer: Option<&Bound<PyAny>>,
    use_symengine: bool,
    qpy_version: u8,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<Bytes> {
    if qpy_version < 17 {
        return Err(PyValueError::new_err(
            "dump_qpy only supports QPY version 17 and above",
        ));
    }
    let packed_circuits: Vec<formats::QPYCircuitV17> = circuits
        .iter_mut()
        .map(|circuit| {
            pack_circuit(
                circuit,
                metadata_serializer,
                use_symengine,
                qpy_version as u32,
                annotation_factories,
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    let qpy_file = formats::QPYFile {
        qpy_version,
        qiskit_version: QISKIT_VERSION,
        symbolic_encoding: b'p',      // using symengine is obsolete
        type_key: ValueType::Circuit, //for now, no other value type is used
        circuits: packed_circuits,
    };
    let qpy_file_v17: QPY17File = qpy_file.try_into()?;
    Ok(serialize(&qpy_file_v17))
}

#[pyfunction]
pub fn py_dump_qpy(
    py: Python,
    file_obj: &Bound<PyAny>,
    circuits: &Bound<PyAny>,
    metadata_serializer: Option<&Bound<PyAny>>,
    use_symengine: bool,
    qpy_version: u8,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<()> {
    let serialized_qpy = dump_qpy(
        circuits.extract()?,
        metadata_serializer,
        use_symengine,
        qpy_version,
        annotation_factories,
    )?;
    file_obj.call_method1("write", (pyo3::types::PyBytes::new(py, &serialized_qpy),))?;
    Ok(())
}

impl TryFrom<QPYFile> for QPY17File {
    type Error = PyErr;
    fn try_from(qpy_file: QPYFile) -> PyResult<QPY17File> {
        // We should serialize the circuits and compute the offset table based on their lengths
        let circuits = qpy_file
            .circuits
            .iter()
            .map(serialize)
            .collect::<Vec<_>>();
        // the initial offset is the size of all the fields in QPY17File up to and not including circuits.
        // The fields up to circuit_table take QPY17_HEADER_SIZE bytes, and circuit_table take 64*circuits.len() bytes.
        let initial_offset = formats::QPY17_HEADER_SIZE + 64 * qpy_file.circuits.len();
        let mut circuit_table: Vec<u64> = Vec::with_capacity(qpy_file.circuits.len());
        let mut current_offset = initial_offset;
        for circuit in &circuits {
            circuit_table.push(current_offset as u64);
            current_offset += circuit.len();
        }
        Ok(QPY17File {
            qpy_version: qpy_file.qpy_version,
            qiskit_version: qpy_file.qiskit_version,
            symbolic_encoding: qpy_file.symbolic_encoding,
            type_key: qpy_file.type_key,
            circuit_table,
            circuits,
        })
    }
}
