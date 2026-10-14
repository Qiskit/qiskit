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

//! Implementation of the Symbol Table for Qiskit Midle-End Intermediate Representation.

use std::fmt;
use std::ops::{Add, AddAssign};

/// Errors encountered when dealing with Symbols and the SymbolTable.
#[derive(thiserror::Error, Clone, Debug)]
#[error("{kind}")]
pub struct SymbolTableError {
    kind: SymbolTableErrorKind,
    /// Symbol is optional because not all errors may be related to specific symbols.
    /// But when an error *is* related to a specific symbol, it will be good to include it
    /// in the error message to help with debugging.
    sym: Option<Symbol>,
}

impl SymbolTableError {
    /// View the [`Symbol`] attached to this error (if there is one).
    #[inline]
    pub fn symbol(&self) -> &Option<Symbol> {
        &self.sym
    }
    /// Retrieve the ownership of the Symbol.
    #[inline]
    pub fn into_symbol(self) -> Option<Symbol> {
        self.sym
    }
    /// Get the underlying error.
    #[inline]
    pub fn kind(&self) -> SymbolTableErrorKind {
        self.kind.clone() // Clone is necessary because Symbol contains a String, and cannot derive Copy.
    }
}

/// The kind of error that triggered a [`AddSymbolError`]
#[derive(thiserror::Error, Clone, Debug, PartialEq)]
pub enum SymbolTableErrorKind {
    #[error("Symbol ID already set: {0}")]
    AlreadyInitialized(SymbolId),
}

impl SymbolTableErrorKind {
    pub fn with_symbol(self, sym: Symbol) -> SymbolTableError {
        SymbolTableError {
            kind: self,
            sym: Some(sym),
        }
    }
}

/// Symbol represents various types of classical information. This will start
/// with variables, but likely grow to include other things over time.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    /// Symbol ID used in the IR. This field should be set when adding to the
    /// Symbol Table. This field is intentionally private and does not have a
    /// set method because it should not be set outside of this module.
    sym_id: Option<SymbolId>,
    name: String,
}

impl Symbol {
    pub const fn new(name: String) -> Self {
        Self { sym_id: None, name }
    }

    /// Get the name for this symbol.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Get the symbol id field for this symbol.
    ///
    /// Note that it is expected for the SymbolId field to be None before being
    /// added to the symbol table, and contain a valid Id after being added to
    /// the table (i.e., the Symbol Table manages and provides the Symbol Ids).
    pub fn get_symbol_id(&self) -> Option<SymbolId> {
        self.sym_id
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol: {:?}, {}", self.sym_id, self.name)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SymbolId(usize);

impl AddAssign for SymbolId {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0
    }
}

impl Add for SymbolId {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        SymbolId(self.0 + other.0)
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Default, Debug)]
pub struct SymbolTable {
    sym_table: Vec<Symbol>,
    next_id: SymbolId,
}

impl SymbolTable {
    /// Add a symbol to the symbol table.
    ///
    /// It is assumed that the provided symbol does *not* have an Id set. If it
    /// does, it is assumed to be already in the symbol table and an error is
    /// returned.
    pub fn add_symbol(&mut self, mut sym: Symbol) -> Result<SymbolId, SymbolTableError> {
        if let Some(sym_id) = sym.sym_id {
            return Err(SymbolTableErrorKind::AlreadyInitialized(sym_id).with_symbol(sym));
        }

        let this_sym_id = self.next_id;
        sym.sym_id = Some(this_sym_id);

        self.next_id += SymbolId(1);

        self.sym_table.push(sym);

        debug_assert_eq!(
            self.next_id,
            SymbolId(self.sym_table.len()),
            "Symbol table inconsistent!"
        );
        Ok(this_sym_id)
    }

    /// Get a reference to a symbol in the symbol table.
    ///
    /// This will panic if the specified SymbolId is *not* in the table.
    pub fn get_symbol(&self, sym_id: SymbolId) -> &Symbol {
        &self.sym_table[sym_id.0]
    }

    /// Try to get a symbol from the symbol table.
    ///
    /// This will safely attempt to return a reference to specified Symbol. If
    /// the specified SymbolId is not in the table, return None.
    pub fn try_get_symbol(&self, sym_id: SymbolId) -> Option<&Symbol> {
        self.sym_table.get(sym_id.0)
    }

    /// Get a mutable reference to the specified symbol.
    ///
    /// This will panic if the SymbolId is not in the table.
    pub fn get_symbol_mut(&mut self, sym_id: SymbolId) -> &mut Symbol {
        &mut self.sym_table[sym_id.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_symbol() {
        let new_sym = Symbol::new(String::from("test"));

        assert_eq!(new_sym.sym_id, None);
        assert_eq!(new_sym.name, "test");
    }

    #[test]
    fn test_sym_table_add_symbols() {
        let new_sym = Symbol::new(String::from("test"));
        let mut new_sym2 = Symbol::new(String::from("bad"));

        new_sym2.sym_id = Some(SymbolId(1));

        let mut sym_table = SymbolTable::default();

        let sym_id = sym_table.add_symbol(new_sym).unwrap();

        assert_eq!(sym_id, SymbolId(0));

        assert_eq!(
            sym_table.add_symbol(new_sym2).unwrap_err().kind(),
            SymbolTableErrorKind::AlreadyInitialized(SymbolId(1))
        );
    }

    #[test]
    fn test_get_symbol() {
        let new_sym = Symbol::new(String::from("test"));
        let mut copy_sym = new_sym.clone();

        let mut sym_table = SymbolTable::default();

        let sym_id = sym_table.add_symbol(new_sym).unwrap();
        copy_sym.sym_id = Some(sym_id);

        assert_eq!(*sym_table.get_symbol(sym_id), copy_sym);

        assert!(sym_table.try_get_symbol(SymbolId(1)).is_none());
    }

    #[test]
    #[should_panic]
    fn test_get_symbol_panics() {
        let new_sym = Symbol::new(String::from("test"));
        let mut sym_table = SymbolTable::default();

        _ = sym_table.add_symbol(new_sym).unwrap();
        sym_table.get_symbol(SymbolId(1));
    }
}
