use std::fmt::Debug;

/// Keeps information about where a bit is located within the circuit.
///
/// This information includes whether the bit was added by a register,
/// which register it belongs to and where it is located within it.
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct BitInfo {
    added_by_reg: bool,
    registers: Vec<BitLocation>,
}

impl BitInfo {
    pub fn new(orig_reg: Option<(u32, u32)>) -> Self {
        // If the instance was added by a register, add it and prefil its locator
        if let Some((reg_idx, idx)) = orig_reg {
            Self {
                added_by_reg: true,
                registers: vec![BitLocation::new(reg_idx, idx)],
            }
        } else {
            Self {
                added_by_reg: false,
                registers: vec![],
            }
        }
    }

    /// Add a register to the bit instance
    pub fn add_register(&mut self, register: u32, index: u32) {
        self.registers.push(BitLocation(register, index))
    }

    /// Returns a list with all the [BitLocation] instances
    pub fn get_registers(&self) -> &[BitLocation] {
        &self.registers
    }

    /// Returns the index of the original register if any exists
    pub fn orig_register_index(&self) -> Option<&BitLocation> {
        if self.added_by_reg {
            Some(&self.registers[0])
        } else {
            None
        }
    }
}

/// Keeps information about where a qubit is located within a register.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct BitLocation(u32, u32);

impl BitLocation {
    pub fn new(register_idx: u32, index: u32) -> Self {
        Self(register_idx, index)
    }

    pub fn register_index(&self) -> u32 {
        self.0
    }

    pub fn index(&self) -> u32 {
        self.1
    }
}
