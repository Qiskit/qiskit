#![allow(dead_code)]
#![allow(unused)]
// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Conventions in journal article and Python code
/// A "program" means a unitary opeator.

pub mod circuits;
pub mod decomposer;
pub mod utilities;
mod types;
mod weyl;
mod polytopes;
mod paths;
