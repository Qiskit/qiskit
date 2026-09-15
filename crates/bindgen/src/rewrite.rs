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

//! Rewriting rules for the `cbindgen` representation.

use cbindgen::ir;
use hashbrown::HashMap;

fn iter_direct_type_uses(bindings: &cbindgen::Bindings) -> impl Iterator<Item = &ir::Type> {}

/// Rewrite any ``ItemContainer::Typedef`` objects that are only ever used as opaque pointers to
/// instead be declared as ``ItemContainer::OpaqueItem``.
pub fn lift_typedefs_to_opaque_structs(bindings: &mut cbindgen::Bindings) {
    let mut typedefs = bindings
        .items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            let ir::ItemContainer::Typedef(item) = item else {
                return None;
            };
            Some((item.export_name.to_owned(), i))
        })
        .collect::<HashMap<_, _>>();
    for ty in iter_direct_type_uses(bindings) {
        typedefs.remove
    }
}
