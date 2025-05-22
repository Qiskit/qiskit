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



use binrw::{BinRead, BinResult, BinWrite, Endian, VecArgs};
use pyo3::exceptions::PyValueError;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use std::fmt::Debug;
use std::io::{Cursor, Write, Read, Seek};
use std::ops::{Deref, DerefMut};

// Bytes are the format used to store serialized data which is not automatically handled by binrw
// It's a wrapper around Vec<u8> with extended serialization/deserialization capabilities
#[derive(Debug)]
pub struct Bytes(pub Vec<u8>);

impl Bytes {
    /// This method is used for debugging; it displays the data as a string of hexdecimal digits
    pub fn to_hex_string(&self) -> String {
        self.0
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
    }
    pub fn try_to_le_f64(&self) -> PyResult<f64>{
        let byte_array: [u8; 8] = self.0.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("Expected exactly 8 bytes"))?;
        Ok(f64::from_le_bytes(byte_array))
    }
    pub fn new() -> Self {
        Bytes(Vec::new())
    }
}

impl TryFrom<&Bytes> for f64 {
    type Error = PyErr;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let byte_array: [u8; 8] = bytes.0.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("Expected exactly 8 bytes"))?;
        Ok(f64::from_be_bytes(byte_array))
    }
}

impl TryFrom<&Bytes> for (f64, f64) {
    type Error = PyErr;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let byte_array: [u8; 16] = bytes.0.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("Expected exactly 8 bytes"))?;
        Ok((f64::from_be_bytes(byte_array[0..8].try_into()?), f64::from_be_bytes(byte_array[8..16].try_into()?)))
    }
}

impl TryFrom<&Bytes> for i64 {
    type Error = PyErr;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let byte_array: [u8; 8] = bytes.0.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("Expected exactly 8 bytes"))?;
        Ok(i64::from_be_bytes(byte_array))
    }
}

impl TryFrom<&Bytes> for String {
    type Error = PyErr;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        String::from_utf8(bytes.0.clone())
        .map_err(|_| PyValueError::new_err("Not a valid UTF-8 string"))
    }
}


impl<'a> TryFrom<&'a Bytes> for &'a str {
    type Error = PyErr;
    fn try_from(bytes: &'a Bytes) -> Result<Self, Self::Error> {
        std::str::from_utf8(&bytes.0)
        .map_err(|_| PyValueError::new_err("Not a valid UTF-8 string"))
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Bytes(v)
    }
}
impl From<&[u8]> for Bytes {
    fn from(s: &[u8]) -> Self {
        Bytes(s.to_vec())
    }
}
impl From<[u8; 8]> for Bytes {
    fn from(s: [u8; 8]) -> Self {
        Bytes(s.to_vec())
    }
}

impl From<String> for Bytes {
    fn from(s: String) -> Self {
        Bytes(s.into_bytes())
    }
}


impl From<Cursor<Vec<u8>>> for Bytes {
    fn from(cursor: Cursor<Vec<u8>>) -> Self {
        Bytes(cursor.into_inner())
    }
}

impl Deref for Bytes {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for Bytes {
    type Item = u8;
    type IntoIter = std::vec::IntoIter<u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<u8> for Bytes {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        Bytes(Vec::from_iter(iter))
    }
}

impl BinRead for Bytes {
    type Args<'a> = VecArgs<Vec<u8>>;

    fn read_options<R: Read + Seek>(
        reader: &mut R,
        _endian: Endian,
        args: Self::Args<'_>,
    ) -> BinResult<Self> {
        let mut buf = vec![0u8; args.count];
        reader.read_exact(&mut buf)?;
        Ok(Bytes(buf))
    }
}

impl BinWrite for Bytes {
    type Args<'a> = ();

    fn write_options<W: Write + Seek>(
        &self,
        writer: &mut W,
        _endian: Endian,
        _args: Self::Args<'_>,
    ) -> BinResult<()> {
        writer.write_all(&self.0)?;
        Ok(())
    }
}

impl<'py> FromPyObject<'py> for Bytes {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract::<Vec<u8>>()?))
    }
}