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

use indexmap::IndexMap;
use qiskit_cext_vtable::ExportedFunction;

pub struct SlotsLists {
    pub slots: IndexMap<String, SlotsList>,
}
impl SlotsLists {
    /// The slots lists that are defined in the depended-on `qiskit_cext_vtable`.
    pub fn ours() -> Self {
        let names = |funcs: Vec<Option<ExportedFunction>>| {
            SlotsList(
                funcs
                    .into_iter()
                    .map(|f| f.map(|f| f.name.to_owned()))
                    .collect(),
            )
        };
        Self {
            slots: [
                ("circuit", &qiskit_cext_vtable::FUNCTIONS_CIRCUIT),
                ("transpile", &qiskit_cext_vtable::FUNCTIONS_TRANSPILE),
                ("qi", &qiskit_cext_vtable::FUNCTIONS_QI),
            ]
            .iter()
            .map(|(name, funcs)| (String::from(*name), names(funcs.slots())))
            .collect(),
        }
    }
}

pub struct SlotsList(Vec<Option<String>>);
impl SlotsList {
    /// Iterate over the slot indices and the name stored there in order.
    pub fn iter_names(&self) -> impl Iterator<Item = (usize, &str)> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, fname)| fname.as_deref().map(|fname| (i, fname)))
    }
}
