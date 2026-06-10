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
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use qiskit_circuit::converters::QuantumCircuitData;

use crate::bytes::Bytes;
use crate::circuit_reader::unpack_circuit;
use crate::circuit_writer::pack_circuit;
use crate::error::QpyError;
use crate::formats::{QPYCircuit, QPYFileHeader};
use crate::value::{ProgramType, SymbolicEncoding, deserialize, deserialize_with_args, serialize};

use std::io::{Cursor, Seek};

// helper function to parse int from ascii at compile time
const fn parse_u8_from_ascii(s: &str) -> u8 {
    let bytes = s.as_bytes();
    let mut result: u8 = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < b'0' || bytes[i] > b'9' {
            panic!("Invalid character in version string");
        }
        let digit = bytes[i] - b'0';
        result = result * 10 + digit;
        i += 1;
    }
    result
}

const fn parse_version() -> (u8, u8, u8) {
    let major = parse_u8_from_ascii(env!("CARGO_PKG_VERSION_MAJOR"));
    let minor = parse_u8_from_ascii(env!("CARGO_PKG_VERSION_MINOR"));
    let patch = parse_u8_from_ascii(env!("CARGO_PKG_VERSION_PATCH"));
    (major, minor, patch)
}

const QISKIT_VERSION: (u8, u8, u8) = parse_version();

pub fn dump_qpy(
    mut circuits: Vec<QuantumCircuitData>,
    metadata_serializer: Option<Bound<PyAny>>,
    use_symengine: bool,
    qpy_version: u8,
    annotation_factories: Bound<PyDict>,
) -> Result<Bytes, QpyError> {
    if qpy_version < 17 {
        Err(QpyError::UnsupportedFeatureForVersion {
            feature: "Rust QPY".to_string(),
            version: qpy_version,
            min_version: 17,
        })?;
    }
    let serialized_circuits: Vec<Bytes> = circuits
        .iter_mut()
        .map(|circuit| {
            serialize(&pack_circuit(
                circuit,
                metadata_serializer.as_ref(),
                use_symengine,
                qpy_version,
                &annotation_factories,
            )?)
        })
        .collect::<Result<Vec<Bytes>, QpyError>>()?;
    let symbolic_encoding = match use_symengine {
        true => SymbolicEncoding::Symengine,
        false => SymbolicEncoding::Sympy,
    };
    let qpy_header = QPYFileHeader {
        qpy_version,
        qiskit_version: QISKIT_VERSION,
        num_programs: serialized_circuits.len() as u64,
        symbolic_encoding,
        type_key: ProgramType::Circuit, //for now, no other value type is used
    };
    let serialized_qpy_header = serialize(&qpy_header)?;

    // At this point we have collected all the relevant data
    // But still need to create the offset table and put everything together
    let header_size = serialized_qpy_header.len();
    let offset_table_size = serialized_circuits.len() * 8; // 8 bytes per u64
    let circuits_start_offset = header_size + offset_table_size;
    // Build the offset table
    let mut offset_table: Vec<u64> = Vec::with_capacity(serialized_circuits.len());
    let mut current_offset = circuits_start_offset as u64;

    for circuit_bytes in &serialized_circuits {
        offset_table.push(current_offset);
        current_offset += circuit_bytes.len() as u64;
    }

    let mut output = Vec::<u8>::with_capacity(current_offset as usize);

    output.extend_from_slice(&serialized_qpy_header);
    for offset in offset_table {
        output.extend_from_slice(&offset.to_be_bytes());
    }
    for circuit_bytes in serialized_circuits {
        output.extend_from_slice(&circuit_bytes);
    }

    Ok(Bytes::from(output))
}

#[pyfunction]
#[pyo3(name = "dump")]
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
    let (qpy_file_header, header_size) = deserialize::<QPYFileHeader>(data)?;
    // Verify the type key is for circuits
    if qpy_file_header.type_key == ProgramType::Schedule {
        return Err(QpyError::PayloadTypeError(
            "Payloads of type `Schedule` cannot be loaded as of Qiskit 2.0. \nUse an earlier version of Qiskit if you want to load `Schedule` payloads.".to_string()
        ));
    }
    if qpy_file_header.type_key != ProgramType::Circuit {
        return Err(QpyError::PayloadTypeError(format!(
            "Invalid payload format data kind '{}'",
            qpy_file_header.type_key
        )));
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
        let qpy_raw_circuits = read_raw_circuits(&mut cursor, num_programs)?;
        for (index, raw_circuit) in qpy_raw_circuits.iter().enumerate() {
            let (packed_circuit, _) = deserialize_with_args::<QPYCircuit, (u8,)>(
                raw_circuit,
                (qpy_file_header.qpy_version,),
            )?;
            circuits[index] = unpack_circuit(
                py,
                &packed_circuit,
                qpy_file_header.qpy_version,
                metadata_deserializer,
                use_symengine,
                annotation_factories,
            )?;
        }
    } else {
        // QPY version < 16, no offset table
        let packed_qpy_circuits = Vec::<QPYCircuit>::read_options(
            &mut cursor,
            Endian::Big,
            VecArgs {
                count: num_programs,
                inner: (qpy_file_header.qpy_version,),
            },
        )?;
        for (index, packed_circuit) in packed_qpy_circuits.iter().enumerate() {
            circuits[index] = unpack_circuit(
                py,
                packed_circuit,
                qpy_file_header.qpy_version,
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
pub fn py_load_qpy(
    py: Python,
    file_obj: &Bound<PyAny>,
    metadata_deserializer: Option<Bound<PyAny>>,
    annotation_factories: Option<Bound<PyDict>>,
) -> Result<Vec<Py<PyAny>>, QpyError> {
    let annotation_factories = annotation_factories.unwrap_or(PyDict::new(py));

    // Read all data from file object
    // TODO: When we read from a rust native stream, maybe we can seek according to the circuit offsets instead of reading everything at once
    let data: Bytes = file_obj.call_method0("read")?.extract()?;

    load_qpy(
        py,
        &data,
        metadata_deserializer.as_ref(),
        &annotation_factories,
    )
}
