// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::Block;

/// Internal entry in the block list.
#[derive(Clone, Debug)]
pub enum Entry<T> {
    Occupied {
        block: T,
        refcount: u32,
    },
    /// The index of the next free entry in the stack, if any.
    Vacant(Option<Block>),
}
impl<T> Entry<T> {
    #[inline]
    pub fn block(&self) -> Option<&T> {
        match self {
            Self::Occupied { block, refcount: _ } => Some(block),
            Self::Vacant(_) => None,
        }
    }
    #[inline]
    pub fn block_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Occupied { block, refcount: _ } => Some(block),
            Self::Vacant(_) => None,
        }
    }
    #[inline]
    pub fn refcount(&self) -> Option<u32> {
        match self {
            Self::Occupied { block: _, refcount } => Some(*refcount),
            Self::Vacant(_) => None,
        }
    }

    #[inline]
    pub fn refcount_mut(&mut self) -> Option<&mut u32> {
        match self {
            Self::Occupied { block: _, refcount } => Some(refcount),
            Self::Vacant(_) => None,
        }
    }
    #[inline]
    pub fn clone_without_references(&self) -> Self
    where
        T: Clone,
    {
        match self {
            Self::Occupied { block, refcount: _ } => Self::Occupied {
                block: block.clone(),
                refcount: 0,
            },
            Self::Vacant(next) => Self::Vacant(*next),
        }
    }
    pub fn try_map_without_reference<B, E>(
        &self,
        map_fn: impl FnOnce(&T) -> Result<B, E>,
    ) -> Result<Entry<B>, E> {
        match self {
            Self::Occupied { block, refcount: _ } => {
                map_fn(block).map(|block| Entry::Occupied { block, refcount: 0 })
            }
            Self::Vacant(free) => Ok(Entry::Vacant(*free)),
        }
    }
}

/// Storage for the blocks used by control-flow objects in a circuit representation.
///
/// Blocks are stored and associated with indices (a [Block]) that is stable with respect to removal
/// of blocks and usable as an index.  The [Block] indices are _not_ guaranteed to be contiguous for
/// a given storage, since removal of blocks may leave holes.
///
/// The structure tracks the holes in its distributed [Block] values using a free list, so adding
/// more blocks after removals will re-use previous [Block] values.
#[derive(Clone, Debug)]
pub struct ControlFlowBlocks<T> {
    /// The backing storage for the blocks.
    entries: Vec<Entry<T>>,
    /// The top of the free-list stack.
    free: Option<Block>,
    /// How many of the `entries` are vacant.
    num_free: usize,
}

impl<T> ControlFlowBlocks<T> {
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
            free: None,
            num_free: 0,
        }
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free = None;
        self.num_free = 0;
    }

    /// Clone this tracker, retaining the exact structure of it including the block ids and free
    /// list, but set the refcount of every block to 0.
    pub fn clone_without_references(&self) -> Self
    where
        T: Clone,
    {
        Self {
            entries: self
                .entries
                .iter()
                .map(Entry::clone_without_references)
                .collect(),
            free: self.free,
            num_free: self.num_free,
        }
    }
    /// Create a new tracker that has the same keys and free list, but with the blocks mapped to
    /// another type.
    ///
    /// The converter function can be fallible.
    pub fn try_map_without_references<B, E>(
        &self,
        mut map_fn: impl FnMut(&T) -> Result<B, E>,
    ) -> Result<ControlFlowBlocks<B>, E> {
        let entries = self
            .entries
            .iter()
            .map(|entry| entry.try_map_without_reference(&mut map_fn))
            .collect::<Result<_, _>>()?;
        Ok(ControlFlowBlocks {
            entries,
            free: self.free,
            num_free: self.num_free,
        })
    }
    /// Create a new tracker that has the same keys and free list, but with the blocks mapped to
    /// another type.
    pub fn map_without_references<B>(
        &self,
        mut map_fn: impl FnMut(&T) -> B,
    ) -> ControlFlowBlocks<B> {
        let Ok(entries) = self
            .entries
            .iter()
            .map(|entry| entry.try_map_without_reference(|b| Ok(map_fn(b))))
            .collect::<Result<_, ::std::convert::Infallible>>();
        ControlFlowBlocks {
            entries,
            free: self.free,
            num_free: self.num_free,
        }
    }

    /// How many blocks are present (whether or not they have references).
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len() - self.num_free
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.len() == self.num_free
    }

    /// Get a block, if it exists.  For the (more usual) panicking variant, use the `Index` or
    /// `IndexMut` implementations like `blocks[block]`.
    pub fn get(&self, block: Block) -> Option<&T> {
        self.entries
            .get(block.index())
            .and_then(|entry| entry.block())
    }
    /// Get a block, if it exists.  For the (more usual) panicking variant, use the `Index` or
    /// `IndexMut` implementations like `blocks[block]`.
    pub fn get_mut(&mut self, block: Block) -> Option<&mut T> {
        self.entries
            .get_mut(block.index())
            .and_then(|entry| entry.block_mut())
    }

    fn push_with_refcount(&mut self, block: T, refcount: u32) -> Block {
        let entry = Entry::Occupied { block, refcount };
        match self.free {
            Some(out) => {
                let Entry::Vacant(slot) =
                    ::std::mem::replace(&mut self.entries[out.index()], entry)
                else {
                    panic!("only empty slots should be in the free list");
                };
                self.free = slot;
                self.num_free -= 1;
                out
            }
            None => {
                debug_assert_eq!(self.num_free, 0);
                let out = Block::new(self.entries.len());
                self.entries.push(entry);
                out
            }
        }
    }
    /// Add a new block with a reference already set.  Returns the new id.
    ///
    /// Use [push] if you want to push a block to the system without adding a reference to it.
    #[inline]
    pub fn push_with_reference(&mut self, block: T) -> Block {
        self.push_with_refcount(block, 1)
    }
    /// Add a new block, but leave its refcount at 0.  Returns the new id.
    ///
    /// Use [push_with_reference] if your intention is to immediately reference the block.
    #[inline]
    pub fn push(&mut self, block: T) -> Block {
        self.push_with_refcount(block, 0)
    }

    /// Decrement the refcount of a block, removing and returning if it this was the last reference.
    ///
    /// Panics if the block is not present, or already had no references.
    pub fn decrement(&mut self, block: Block) -> Option<T> {
        match &mut self.entries[block.index()] {
            Entry::Occupied {
                block: _,
                refcount: 0,
            }
            | Entry::Vacant(_) => panic!("should not attempt decrement of unreferenced block"),
            node @ Entry::Occupied {
                block: _,
                refcount: 1,
            } => {
                self.num_free += 1;
                let Entry::Occupied { block, refcount: 1 } =
                    ::std::mem::replace(node, Entry::Vacant(self.free.replace(block)))
                else {
                    unreachable!("pattern is the same as in the 'match' arm, but owned");
                };
                Some(block)
            }
            Entry::Occupied { block: _, refcount } => {
                *refcount -= 1;
                None
            }
        }
    }

    /// Increment the refcount of a block.
    ///
    /// Panics if the block is not present.
    pub fn increment(&mut self, block: Block) {
        let refcount = self.entries[block.index()]
            .refcount_mut()
            .expect("should not attempt increment of absent block");
        *refcount += 1;
    }

    /// Iterator over the blocks in the tracker.
    pub fn blocks(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().filter_map(Entry::block)
    }
    /// Iterator over mutable blocks in the tracker.
    pub fn blocks_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.entries.iter_mut().filter_map(Entry::block_mut)
    }

    /// Iterator over the blocks and their indices.
    pub fn items(&self) -> impl Iterator<Item = (Block, &T)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| entry.block().map(|block| (Block::new(idx), block)))
    }
    /// Iterator over mutable blocks and their indices.
    pub fn items_mut(&mut self) -> impl Iterator<Item = (Block, &mut T)> {
        self.entries
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, entry)| entry.block_mut().map(|block| (Block::new(idx), block)))
    }

    pub fn iter_raw(&self) -> impl ExactSizeIterator<Item = (Block, &Entry<T>)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (Block::new(i), entry))
    }
}

impl<T> ::std::ops::Index<Block> for ControlFlowBlocks<T> {
    type Output = T;
    fn index(&self, index: Block) -> &T {
        self.get(index)
            .expect("caller should only use live block references")
    }
}
impl<T> ::std::ops::IndexMut<Block> for ControlFlowBlocks<T> {
    fn index_mut(&mut self, index: Block) -> &mut T {
        self.get_mut(index)
            .expect("caller should only use live block references")
    }
}

// We can't derive `Default` because the derivation logic only triggers if `T: Default`, even though
// that's not actually required.
impl<T> Default for ControlFlowBlocks<T> {
    fn default() -> Self {
        Self::new()
    }
}
