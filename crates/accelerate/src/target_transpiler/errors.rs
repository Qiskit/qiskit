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

use std::{error::Error, fmt::Display};

/// Error thrown when operation key is not present in the Target
#[derive(Debug)]
pub struct TargetKeyError {
    pub message: String,
}

impl TargetKeyError {
    /// Initializes the new error
    pub fn new_err(message: String) -> Self {
        Self { message }
    }
}

impl Display for TargetKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for TargetKeyError {}
