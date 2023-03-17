// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::import_exception;

use crate::lex::Token;

pub struct Position<'a> {
    filename: &'a std::ffi::OsStr,
    line: usize,
    col: usize,
}

impl<'a> Position<'a> {
    pub fn new(filename: &'a std::ffi::OsStr, line: usize, col: usize) -> Self {
        Self {
            filename,
            line,
            col,
        }
    }
}

impl<'a> std::fmt::Display for &Position<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{},{}",
            self.filename.to_string_lossy(),
            self.line,
            self.col
        )
    }
}

/// Create an error message that includes span data from the given [token][Token].  The base of the
/// message is `message`, and `filename` is the file the triggering OpenQASM 2 code came from.  For
/// string inputs, this can be a placeholder.
pub fn message_generic(position: Option<&Position>, message: &str) -> String {
    if let Some(position) = position {
        format!("{}: {}", position, message)
    } else {
        message.to_owned()
    }
}

/// Shorthand form for creating an error message when a particular type of token was required, but
/// something else was `received`.
pub fn message_incorrect_requirement(
    required: &str,
    received: &Token,
    filename: &std::ffi::OsStr,
) -> String {
    message_generic(
        Some(&Position::new(filename, received.line, received.col)),
        &format!(
            "needed {}, but instead saw {}",
            required,
            received.ttype.describe()
        ),
    )
}

/// Shorthand form for creating an error message when a particular type of token was required, but
/// the input ended unexpectedly.
pub fn message_bad_eof(position: Option<&Position>, required: &str) -> String {
    message_generic(
        position,
        &format!("unexpected end-of-file when expecting to see {}", required),
    )
}

// We define the exception in Python space so it can inherit from QiskitError; it's easier to do
// that from Python and wrap rather than also needing to import QiskitError to Rust to wrap.
import_exception!(qiskit.qasm2.exceptions, QASM2ParseError);
