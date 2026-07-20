// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_bigint::BigUint;

use crate::expr::Expr;
use crate::parse::{ClbitId, CregId, GateId, QubitId};

/// An internal representation of the bytecode that will later be converted to the more free-form
/// [Bytecode] Python-space objects.  This is fairly tightly coupled to Python space; the intent is
/// just to communicate to Python as concisely as possible what it needs to do.  We want to have as
/// little work to do in Python space as possible, since everything is slower there.
///
/// In various enumeration items, we use zero-indexed numeric keys to identify the object rather
/// than its name.  This is much more efficient in Python-space; rather than needing to build and
/// lookup things in a hashmap, we can just build Python lists and index them directly, which also
/// has the advantage of not needing to pass strings to Python for each gate.  It also gives us
/// consistency with how qubits and clbits are tracked; there is no need to track both the register
/// name and the index separately when we can use a simple single index.
pub enum InternalBytecode {
    Gate {
        id: GateId,
        arguments: Vec<f64>,
        qubits: Vec<QubitId>,
    },
    ConditionedGate {
        id: GateId,
        arguments: Vec<f64>,
        qubits: Vec<QubitId>,
        creg: CregId,
        value: BigUint,
    },
    Measure {
        qubit: QubitId,
        clbit: ClbitId,
    },
    ConditionedMeasure {
        qubit: QubitId,
        clbit: ClbitId,
        creg: CregId,
        value: BigUint,
    },
    Reset {
        qubit: QubitId,
    },
    ConditionedReset {
        qubit: QubitId,
        creg: CregId,
        value: BigUint,
    },
    Barrier {
        qubits: Vec<QubitId>,
    },
    DeclareQreg {
        name: String,
        size: usize,
    },
    DeclareCreg {
        name: String,
        size: usize,
    },
    DeclareGate {
        name: String,
        num_qubits: usize,
    },
    GateInBody {
        id: GateId,
        arguments: Vec<Expr>,
        qubits: Vec<QubitId>,
    },
    EndDeclareGate {},
    DeclareOpaque {
        name: String,
        num_qubits: usize,
    },
    SpecialInclude {
        indices: Vec<usize>,
    },
}

#[cfg(feature = "py")]
mod py;
#[cfg(feature = "py")]
pub use py::*;
