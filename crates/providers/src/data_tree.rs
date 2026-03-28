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

use std::marker::PhantomData;

use hashbrown::HashMap;

/// An item stored in a `DataTree`
///
/// This can either be a Leaf which is a concrete item of type `T` or a subtree.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeEntry<'a, T> {
    Leaf(T),
    // TODO: Box this to reduce memory consumption
    Tree(DataTree<'a, T>),
}

impl<'a, T> TreeEntry<'a, T> {
    /// Return true if the entry is a leaf
    pub fn is_leaf(&self) -> bool {
        match self {
            Self::Leaf(_) => true,
            Self::Tree(_) => false,
        }
    }

    /// Consume the entry and return the leaf value otherwise panic
    pub fn unwrap_leaf(self) -> T {
        match self {
            Self::Leaf(data) => data,
            Self::Tree(_) => panic!("called TreeEntry::unwrap_leaf() on a `Tree` value"),
        }
    }

    /// Consume the entry and return the data tree otherwise panic
    pub fn unwrap_tree(self) -> DataTree<'a, T> {
        match self {
            Self::Leaf(_) => panic!("called TreeEntry::unwrap_tree() on a `Leaf` value"),
            Self::Tree(data) => data,
        }
    }

    /// Return a reference to the underlying tree
    ///
    /// This will be None if the `TreeEntry` is a `Leaf`
    pub fn as_tree_ref(&self) -> Option<&DataTree<'a, T>> {
        match *self {
            Self::Leaf(_) => None,
            Self::Tree(ref tree) => Some(tree),
        }
    }

    /// Return a reference to the underlying tree
    ///
    /// This will be None if the `TreeEntry` is a `Tree`
    pub fn as_leaf_ref(&self) -> Option<&T> {
        match *self {
            Self::Leaf(ref val) => Some(val),
            Self::Tree(_) => None,
        }
    }
}

/// A generic tree that is addressable either by either indices or string keys
#[derive(Debug, Clone)]
pub struct DataTree<'a, T> {
    data: Vec<TreeEntry<'a, T>>,
    keys: HashMap<String, usize>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Default for DataTree<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> DataTree<'a, T> {
    /// Create a new empty data tree
    pub fn new() -> Self {
        DataTree {
            data: Vec::new(),
            keys: HashMap::new(),
            _marker: PhantomData,
        }
    }

    /// Create a new empty data tree with an underlying allocation of a given size.
    ///
    /// The specified capacity is the number of items of type T stored in the `DataTree`
    /// along with an associated `String` key for each element in the tree. This does not
    /// account for nesting in the allocation as each layer in the tree is a separate
    /// `DataTree` object.
    pub fn with_capacity(capacity: usize) -> Self {
        DataTree {
            data: Vec::with_capacity(capacity),
            keys: HashMap::with_capacity(capacity),
            _marker: PhantomData,
        }
    }

    /// The number of items in this `DataTree`. This length is just the number of items in this
    /// local tree object and will not recurse through the tree to compute the total number of
    /// leaves. If you want to do that you should use [`DataTree::iter_leaves`].
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut inner_tree = DataTree::with_capacity(5);
    /// inner_tree.insert_leaf("y", 10);
    /// inner_tree.insert_leaf("z", 11);
    /// inner_tree.insert_leaf("a", 12);
    /// inner_tree.insert_leaf("b", 13);
    /// inner_tree.push_leaf(15);
    ///
    /// let mut tree = DataTree::new();
    /// tree.insert_tree("x", inner_tree);
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return whether this `DataTree` has an items in it.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Take a string key and return the entry at the given key.
    ///
    /// The "." character is reserved in keys and used to indicate a path
    /// through the graph.
    ///
    /// This will return `None` if the string key can not be found. This includes
    /// an invalid path, such as a path containing component or a leaf node in the
    /// middle.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut inner_tree = DataTree::new();
    /// inner_tree.insert_leaf("y", 10);
    /// let mut tree = DataTree::new();
    /// tree.insert_tree("x", inner_tree);
    /// let result = tree.get_by_str_key("x.y").unwrap().as_leaf_ref();
    /// assert_eq!(*result.unwrap(), 10);
    /// ```
    pub fn get_by_str_key(&self, key: &str) -> Option<&TreeEntry<'_, T>> {
        if key.contains(".") {
            let mut components = key.split(".");
            let first = self.get_by_str_key(components.next().unwrap());
            components.fold(first, |tree, key| match tree {
                Some(entry) => {
                    match entry {
                        // If we encounter a leaf in the accumulated tree than
                        // that means we have an incorrect path and there is no
                        // match
                        TreeEntry::Leaf(_) => None,
                        TreeEntry::Tree(tree) => tree.get_by_str_key(key),
                    }
                }
                None => None,
            })
        } else {
            self.keys.get(key).map(|value| &self.data[*value])
        }
    }

    /// Get an item from the `DataTree` by index.
    ///
    /// This will return `None` if the index is not valid.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut inner_tree = DataTree::new();
    /// inner_tree.insert_leaf("y", 10);
    /// let mut tree = DataTree::new();
    /// tree.insert_tree("x", inner_tree);
    /// tree.push_leaf(124);
    /// let Some(result) = tree.get(1).unwrap().as_leaf_ref() else {
    ///     panic!("Encountered an unexpected Tree");
    /// };
    /// assert_eq!(*result, 124);
    /// let subtree = tree.get(0).unwrap().as_tree_ref().unwrap();
    /// let subtree_result = subtree.get(0).unwrap().as_leaf_ref();
    /// assert_eq!(*subtree_result.unwrap(), 10);
    /// ```
    pub fn get(&self, index: usize) -> Option<&TreeEntry<'_, T>> {
        self.data.get(index)
    }

    /// Insert a new leaf node with an associated string key
    ///
    /// If a key is provided that is already in the tree the new value will be associated with
    /// with the key and the old value will remain in the tree but without a string key.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("y", 10);
    /// tree.insert_leaf("y", 1000);
    /// let result = tree.get_by_str_key("y").unwrap().as_leaf_ref();
    /// assert_eq!(*result.unwrap(), 1000);
    /// ```
    pub fn insert_leaf(&mut self, key: &str, value: T) {
        self.data.push(TreeEntry::Leaf(value));
        self.keys.insert(key.to_string(), self.data.len() - 1);
    }

    /// Add a new leaf to the tree
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.push_leaf(10);
    /// tree.push_leaf(1000);
    /// assert_eq!(vec![10, 1000], tree.iter_leaves().copied().collect::<Vec<_>>());
    /// ```
    pub fn push_leaf(&mut self, value: T) {
        self.data.push(TreeEntry::Leaf(value));
    }

    /// Add a subtree to the tree with an associated string key
    ///
    /// If a key is provided that is already in the tree the new value will be associated with
    /// with the key and the old value will remain in the tree but without a string key.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("y", 10);
    /// let mut subtree = DataTree::with_capacity(2);
    /// subtree.push_leaf(123);
    /// subtree.push_leaf(456);
    /// tree.insert_tree("y", subtree);
    /// let result = tree.get_by_str_key("y").unwrap().as_tree_ref().unwrap();
    /// let leaves: Vec<_> = result.iter_leaves().copied().collect();
    /// assert_eq!(leaves, vec![123, 456]);
    /// ```
    pub fn insert_tree(&mut self, key: &str, value: DataTree<'a, T>) {
        self.data.push(TreeEntry::Tree(value));
        self.keys.insert(key.to_string(), self.data.len() - 1);
    }

    pub fn push_tree(&mut self, value: DataTree<'a, T>) {
        self.data.push(TreeEntry::Tree(value));
    }

    /// Return an iterator over the leaves in the `DataTree`
    ///
    /// This method will return an iterator over all leave nodes in the tree by traversing the tree
    /// in a DFS order.
    ///
    /// # Example
    ///
    /// Traversing this tree:
    ///
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut subsubsubtree = DataTree::new();
    /// subsubsubtree.push_leaf(3);
    /// subsubsubtree.push_leaf(4);
    /// let mut subsubtree = DataTree::new();
    /// subsubtree.push_tree(subsubsubtree);
    /// subsubtree.insert_leaf("b", 5);
    /// let mut subsubtree_prime = DataTree::new();
    /// subsubtree_prime.push_leaf(7);
    /// let mut subtree = DataTree::new();
    /// subtree.insert_tree("c", subsubtree);
    /// subtree.insert_leaf("d", 6);
    /// subtree.push_tree(subsubtree_prime);
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("a", 0);
    /// tree.insert_tree("root", subtree);
    /// tree.insert_leaf("z", 26);
    /// let leaves: Vec<_> = tree.iter_leaves().copied().collect();
    /// let expected = vec![0, 3, 4, 5, 6, 7, 26];
    /// assert_eq!(leaves, expected);
    /// ```
    pub fn iter_leaves(&self) -> IterLeaves<'_, T> {
        IterLeaves {
            tree: self,
            index: 0,
            inner: None,
            inner_next: None,
        }
    }
}

#[derive(Debug)]
pub struct IterLeaves<'a, T> {
    tree: &'a DataTree<'a, T>,
    index: usize,
    inner: Option<Box<IterLeaves<'a, T>>>,
    inner_next: Option<&'a T>,
}

impl<'a, T: std::fmt::Debug> Iterator for IterLeaves<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.tree.len() {
            return None;
        }
        let entry = &self.tree.data[self.index];
        match entry {
            TreeEntry::Leaf(val) => {
                self.index += 1;
                Some(val)
            }
            TreeEntry::Tree(subtree) => match self.inner {
                Some(ref mut inner) => {
                    if let Some(val) = inner.next() {
                        let return_val = self.inner_next;
                        self.inner_next = Some(val);
                        return_val
                    } else {
                        self.inner = None;
                        self.index += 1;
                        let return_val = self.inner_next;
                        self.inner_next = None;
                        return_val
                    }
                }
                None => {
                    self.inner = Some(Box::new(subtree.iter_leaves()));
                    let val = self.inner.as_mut().map(|x| x.next().unwrap());
                    self.inner_next = self.inner.as_mut().and_then(|x| x.next());
                    if self.inner_next.is_none() {
                        self.index += 1;
                        self.inner = None;
                        self.inner_next = None;
                    }
                    val
                }
            },
        }
    }
}

impl<'a, T: PartialEq> PartialEq for DataTree<'a, T> {
    fn eq(&self, other: &DataTree<T>) -> bool {
        self.data == other.data && self.keys == other.keys
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_data_leaf() {
        let mut tree = DataTree::new();
        tree.push_leaf(42);
        let result = tree.get(0).unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 42);
    }

    #[test]
    fn test_flat_dict() {
        let mut tree = DataTree::with_capacity(3);
        tree.insert_leaf("a", 1);
        tree.insert_leaf("b", 2);
        let result = tree.get_by_str_key("b").unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 2);
        let result = tree.get_by_str_key("a").unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 1);
    }

    #[test]
    fn test_nested_dict() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        let mut tree = DataTree::new();
        tree.insert_tree("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        assert_eq!(None, tree.get_by_str_key("z.y"));
        let expected = TreeEntry::Tree(inner_tree);
        assert_eq!(Some(&expected), tree.get_by_str_key("x"));
    }

    #[test]
    fn test_nested_dict_iter() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.push_leaf(4);
        inner_inner_tree.push_leaf(5);
        inner_tree.push_tree(inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_tree("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        assert_eq!(
            vec![10, 1, 2, 3, 4, 5, 100],
            tree.iter_leaves().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_get_by_str() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.insert_leaf("a", 4);
        inner_inner_tree.push_leaf(5);
        let inner_inner_tree_expected = inner_inner_tree.clone();
        inner_tree.insert_tree("yyy", inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_tree("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        let result = tree.get_by_str_key("x.yyy.a");
        assert_eq!(result, Some(&TreeEntry::Leaf(4)));
        assert_eq!(tree.get_by_str_key("z"), Some(&TreeEntry::Leaf(100)));
        assert_eq!(
            tree.get_by_str_key("x.yyy"),
            Some(&TreeEntry::Tree(inner_inner_tree_expected))
        );
        assert_eq!(tree.get_by_str_key("x.yy"), Some(&TreeEntry::Leaf(1)));
    }

    #[test]
    fn test_get_by_str_no_match() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.insert_leaf("a", 4);
        inner_inner_tree.push_leaf(5);
        inner_tree.insert_tree("yyy", inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_tree("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        assert_eq!(None, tree.get_by_str_key("a"));
        assert_eq!(None, tree.get_by_str_key("x.yyyy"));
        assert_eq!(None, tree.get_by_str_key("x.yy.a"));
        assert_eq!(None, tree.get_by_str_key("🎩"));
        assert_eq!(None, tree.get_by_str_key("z.yyy.a"));
    }
}
