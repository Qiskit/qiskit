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

use qiskit_util::py::ImportOnceCell;

pub static GENERIC_PASS: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "GenericPass");
pub static PASS: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "Pass");
pub static STATE_FROM_CONTEXT: ImportOnceCell = ImportOnceCell::new(
    "qiskit.passmanager.compilation_status",
    "state_from_passcontext",
);
pub static CONTEXT_FROM_STATE: ImportOnceCell = ImportOnceCell::new(
    "qiskit.passmanager.compilation_status",
    "passcontext_from_state",
);
