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

use std::fmt;
use std::sync::Arc;

use hashbrown::HashMap;
use thiserror::Error;

use super::{RzRx, Synth};

/// The name of a synth.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SynthName(String);

impl SynthName {
    /// Return the name `name`.
    pub fn new(name: &str) -> Self {
        Self(
            name.chars()
                .filter(|&c| c != '_')
                .flat_map(char::to_lowercase)
                .collect(),
        )
    }

    /// The name of `synth`.
    pub fn of(synth: &dyn Synth) -> Self {
        Self::new(synth.name())
    }
}

/// Render in lowercase without underscores.
impl fmt::Display for SynthName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a synth could not be registered.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SynthRegistryError {
    #[error("a synth named {0} is already registered")]
    Duplicate(SynthName),
}

/// A registry of synths by name.
#[derive(Debug, Clone)]
pub struct SynthRegistry {
    synths: HashMap<SynthName, Arc<dyn Synth>>,
}

impl SynthRegistry {
    /// Return a registry with no synths.
    pub fn empty() -> Self {
        Self {
            synths: HashMap::new(),
        }
    }

    /// Register `synth` under its [`SynthName`].
    pub fn register(&mut self, synth: Arc<dyn Synth>) -> Result<(), SynthRegistryError> {
        let name = SynthName::of(synth.as_ref());
        if self.synths.contains_key(&name) {
            return Err(SynthRegistryError::Duplicate(name));
        }
        self.synths.insert(name, synth);
        Ok(())
    }

    /// Return the synth registered under `name`, or `None` if there is none.
    pub fn get(&self, name: &SynthName) -> Option<&dyn Synth> {
        self.synths.get(name).map(Arc::as_ref)
    }
}

impl Default for SynthRegistry {
    fn default() -> Self {
        let mut registry = Self::empty();
        registry
            .register(Arc::new(RzRx))
            .expect("Qiskit's synths have distinct names");
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_ignores_case_and_underscores() {
        let name = SynthName::new("rzrx");
        assert_eq!(SynthName::new("RzRx"), name);
        assert_eq!(SynthName::new("rz_rx"), name);
        assert_eq!(SynthName::new("RZ_RX"), name);
        assert_ne!(SynthName::new("rzsx"), name);
    }

    #[test]
    fn test_name_display() {
        assert_eq!(SynthName::new("Rz_Rx").to_string(), "rzrx");
    }

    #[test]
    fn test_name_of_synth() {
        assert_eq!(SynthName::of(&RzRx), SynthName::new("rzrx"));
    }

    #[test]
    fn test_default_registry_has_rzrx() {
        let registry = SynthRegistry::default();
        let synth = registry.get(&SynthName::new("rz_rx")).unwrap();
        assert_eq!(synth.name(), "rzrx");
        assert!(registry.get(&SynthName::new("rzsx")).is_none());
    }

    #[test]
    fn test_register_rejects_a_duplicate() {
        let mut registry = SynthRegistry::empty();
        assert!(registry.get(&SynthName::new("rzrx")).is_none());
        registry.register(Arc::new(RzRx)).unwrap();
        assert!(registry.get(&SynthName::new("rzrx")).is_some());
        assert_eq!(
            registry.register(Arc::new(RzRx)).unwrap_err(),
            SynthRegistryError::Duplicate(SynthName::new("rzrx"))
        );
    }
}
