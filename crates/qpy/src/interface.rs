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

use binrw::{BinRead, Endian, VecArgs};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use qiskit_circuit::converters::QuantumCircuitData;

use crate::bytes::Bytes;
use crate::circuit_reader::unpack_circuit;
use crate::circuit_writer::pack_circuit;
use crate::error::QpyError;
use crate::formats::{self, QPYCircuit};
use crate::formats::{QPY17File, QPYFile, QPYFileHeader};
use crate::value::{SymbolicEncoding, ValueType, deserialize, deserialize_with_args, serialize};

use std::io::{Cursor, Seek};
// parse the qiskit version
const fn parse_version() -> (u8, u8, u8) {
    let version_str = env!("CARGO_PKG_VERSION");
    let bytes = version_str.as_bytes();
    let mut major = 0u8;
    let mut minor = 0u8;
    let mut patch = 0u8;
    let mut i = 0;
    let mut part = 0; // 0=major, 1=minor, 2=patch

    while i < bytes.len() {
        let b = bytes[i];
        if b >= b'0' && b <= b'9' {
            let digit = b - b'0';
            match part {
                0 => major = major * 10 + digit,
                1 => minor = minor * 10 + digit,
                2 => patch = patch * 10 + digit,
                _ => {}
            }
        } else if b == b'.' {
            part += 1;
        } else {
            // Stop at any non-digit, non-dot character (e.g., '-' in "2.4.0-dev")
            break;
        }
        i += 1;
    }

    (major, minor, patch)
}

const QISKIT_VERSION: (u8, u8, u8) = parse_version();

pub fn dump_qpy(
    mut circuits: Vec<QuantumCircuitData>,
    metadata_serializer: Option<Bound<PyAny>>,
    use_symengine: bool,
    qpy_version: u8,
    annotation_factories: Bound<PyDict>,
) -> PyResult<Bytes> {
    if qpy_version < 17 {
        return Err(PyValueError::new_err(
            "Rust QPY only supports QPY version 17 and above",
        ));
    }
    let packed_circuits: Vec<formats::QPYCircuit> = circuits
        .iter_mut()
        .map(|circuit| {
            pack_circuit(
                circuit,
                metadata_serializer.as_ref(),
                use_symengine,
                qpy_version as u32,
                &annotation_factories,
            )
        })
        .collect::<Result<Vec<formats::QPYCircuit>, QpyError>>()?;
    let symbolic_encoding = match use_symengine {
        true => SymbolicEncoding::Symengine,
        false => SymbolicEncoding::Sympy,
    };
    let qpy_file = formats::QPYFile {
        qpy_version,
        qiskit_version: QISKIT_VERSION,
        symbolic_encoding,
        type_key: ValueType::Circuit, //for now, no other value type is used
        circuits: packed_circuits,
    };
    let qpy_file_v17: QPY17File = qpy_file.try_into()?;
    Ok(serialize(&qpy_file_v17)?)
}

#[pyfunction]
#[pyo3(name = "dump")]
#[pyo3(signature = (programs, file_obj, metadata_serializer, use_symengine, version, annotation_factories))]
pub fn py_dump_qpy(
    py: Python,
    programs: &Bound<PyAny>,
    file_obj: &Bound<PyAny>,
    metadata_serializer: Option<Bound<PyAny>>,
    use_symengine: Option<bool>,
    version: u8,
    annotation_factories: Option<Bound<PyDict>>,
) -> PyResult<()> {
    let annotation_factories = annotation_factories.unwrap_or(PyDict::new(py));
    let serialized_qpy = dump_qpy(
        programs.extract()?,
        metadata_serializer,
        use_symengine.unwrap_or(false),
        version,
        annotation_factories,
    )?;
    file_obj.call_method1("write", (pyo3::types::PyBytes::new(py, &serialized_qpy),))?;
    Ok(())
}

// reads the offset table and splits the raw bytes into circuits accordingly
pub fn read_raw_circuits(
    cursor: &mut Cursor<&[u8]>,
    num_programs: usize,
) -> Result<Vec<Bytes>, QpyError> {
    let circuit_table = Vec::<u64>::read_options(
        cursor,
        Endian::Big,
        VecArgs {
            count: num_programs,
            inner: (),
        },
    )?;

    // Read circuits using offset differences to determine sizes
    let mut circuits = Vec::with_capacity(num_programs);

    for i in 0..num_programs {
        let size = if i + 1 < circuit_table.len() {
            (circuit_table[i + 1] - circuit_table[i]) as usize
        } else {
            // Last circuit: read remaining bytes
            let current_pos = cursor.stream_position()?;
            let end_pos = cursor.seek(std::io::SeekFrom::End(0))?;
            cursor.seek(std::io::SeekFrom::Start(current_pos))?;
            (end_pos - current_pos) as usize
        };

        let circuit = Bytes::read_options(
            cursor,
            Endian::Big,
            VecArgs::<Vec<u8>> {
                count: size,
                inner: Vec::new(),
            },
        )?;
        circuits.push(circuit);
    }
    Ok(circuits)
}

pub fn load_qpy(
    py: Python,
    data: &Bytes,
    metadata_deserializer: Option<&Bound<PyAny>>,
    annotation_factories: &Bound<PyDict>,
) -> Result<Vec<Py<PyAny>>, QpyError> {
    // Deserialize the QPY17File structure using BinRead
    let (qpy_file_header, header_size) = deserialize::<QPYFileHeader>(data)?;
    // Verify the type key is for circuits
    if qpy_file_header.type_key != ValueType::Circuit {
        Err(PyValueError::new_err(format!(
            "Invalid payload format data kind '{}'",
            qpy_file_header.type_key
        )))?;
    }
    let num_programs = qpy_file_header.num_programs as usize;
    let use_symengine = matches!(
        qpy_file_header.symbolic_encoding,
        SymbolicEncoding::Symengine
    );
    let mut circuits = vec![py.None(); num_programs];
    let mut cursor = Cursor::new(data as &[u8]);
    cursor.seek(std::io::SeekFrom::Start(header_size as u64))?;
    if qpy_file_header.qpy_version >= 16 {
        // let qpy_raw_circuits = QPYFileRawCircuits::read_options(&mut cursor, Endian::Big, (qpy_file_header.num_programs, ))?;
        let qpy_raw_circuits = read_raw_circuits(&mut cursor, num_programs)?;
        for (index, raw_circuit) in qpy_raw_circuits.iter().enumerate() {
            let (packed_circuit, _) = deserialize_with_args::<QPYCircuit, (u32,)>(
                raw_circuit,
                (qpy_file_header.qpy_version as u32,),
            )?;
            circuits[index] = unpack_circuit(
                py,
                &packed_circuit,
                qpy_file_header.qpy_version as u32,
                metadata_deserializer,
                use_symengine,
                annotation_factories,
            )?;
        }
    } else {
        // QPY version < 16, no offset table
        // let packed_qpy_circuits = QPYFileCircuits::read_options(&mut cursor, Endian::Big, (qpy_file_header.num_programs, )).unwrap();
        let packed_qpy_circuits = Vec::<QPYCircuit>::read_options(
            &mut cursor,
            Endian::Big,
            VecArgs {
                count: num_programs,
                inner: (qpy_file_header.qpy_version as u32,),
            },
        )?;
        for (index, packed_circuit) in packed_qpy_circuits.iter().enumerate() {
            circuits[index] = unpack_circuit(
                py,
                packed_circuit,
                qpy_file_header.qpy_version as u32,
                metadata_deserializer,
                use_symengine,
                annotation_factories,
            )?;
        }
    }
    Ok(circuits)
}

#[pyfunction]
#[pyo3(name = "load")]
#[pyo3(signature = (file_obj, metadata_deserializer, annotation_factories))]
pub fn py_load_qpy(
    py: Python,
    file_obj: &Bound<PyAny>,
    metadata_deserializer: Option<Bound<PyAny>>,
    annotation_factories: Option<Bound<PyDict>>,
) -> Result<Vec<Py<PyAny>>, QpyError> {
    let annotation_factories = annotation_factories.unwrap_or(PyDict::new(py));

    // Read all data from file object
    let data: Bytes = file_obj.call_method0("read")?.extract()?;

    load_qpy(
        py,
        &data,
        metadata_deserializer.as_ref(),
        &annotation_factories,
    )
}

impl TryFrom<QPYFile> for QPY17File {
    type Error = PyErr;
    fn try_from(qpy_file: QPYFile) -> PyResult<QPY17File> {
        // We should serialize the circuits and compute the offset table based on their lengths
        let circuits = qpy_file
            .circuits
            .iter()
            .map(serialize)
            .collect::<Result<Vec<_>, QpyError>>()?;
        // the initial offset is the size of all the fields in QPY17File up to and not including circuits.
        // The fields up to circuit_table take QPY17_HEADER_SIZE bytes, and circuit_table take 8*circuits.len() bytes.
        let initial_offset =
            formats::QPY17_HEADER_SIZE + size_of::<u64>() * qpy_file.circuits.len();
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
