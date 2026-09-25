/// An identifier for a type that may have additional runtime-defined components in it, such as a
/// dynamic trait implementer that comes from C or Python.
#[derive(Copy, Clone, Debug)]
pub struct DynTypeId<'a> {
    static_id: any::TypeId,
    static_name: &'static str,
    /// The dynamic components of the type information.
    ///
    /// The payload and name are logically tied to some object that creates them.  The pointer, if
    /// used, is valid for the same lifetime as `'a`.
    dynamic: Option<(*mut (), &'a str)>,
}
impl DynTypeId<'_> {
    /// Produce a representation of a [`DynTypeId`] for a type whose implementation is completely
    /// known at Rust compile time.
    ///
    /// Use [`Self::with_dynamic`] to add dynamic context afterwards.
    pub fn of<T: 'static>() -> Self {
        Self {
            static_id: any::TypeId::of::<T>(),
            static_name: any::type_name::<T>(),
            dynamic: None,
        }
    }

    /// Discard the dynamic components of the type, leaving only the static part.
    pub fn to_static(self) -> DynTypeId<'static> {
        DynTypeId {
            dynamic: None,
            ..self
        }
    }

    /// A key object that subsets the fields to define equality and hashing.
    #[inline]
    fn compare_key(&self) -> impl Eq + hash::Hash {
        (self.static_id, self.dynamic.map(|(addr, _)| addr))
    }

    /// Describe this type.
    pub fn describe(&self) -> borrow::Cow<'_, str> {
        match self.dynamic {
            Some((_addr, dyn_name)) => {
                borrow::Cow::Owned(format!("{}[{}]", self.static_name, dyn_name))
            }
            None => borrow::Cow::Borrowed(self.static_name),
        }
    }
}
impl<'a> DynTypeId<'a> {
    /// Set the dynamic components of the type identifier.
    ///
    /// The combination of the Rust type `T` and the address of `payload` is what uniquely defines
    /// the "dynamic type".  If you are using this object to represent a pure type from (say)
    /// Python, you might want to use the pointer to the object's Python `type`.  If you are
    /// representing a dynamic implementation of some trait coming in from C, you probably want to
    /// use a pointer to the vtable of the trait methods.
    ///
    /// Note that the `name` is purely for human inspectability and plays no part in hashing or
    /// comparisons.
    pub fn with_dynamic(self, payload: *mut (), name: &'a str) -> Self {
        Self {
            dynamic: Some((payload, name)),
            ..self
        }
    }
}
impl PartialEq for DynTypeId<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.compare_key() == other.compare_key()
    }
}
impl Eq for DynTypeId<'_> {}
impl hash::Hash for DynTypeId<'_> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.compare_key().hash(state)
    }
}

/// Trait for types that interact with the Qiskit dynamic-typing system ([`DynTyped`]) as static
/// Rust objects.
///
/// This trait is not object safe; use the blanket implementation of [`DynTyped`] for that.
pub trait StaticDynTyped {
    fn static_dyn_type_id() -> DynTypeId<'static>;
}
/// Declare a static Rust type as directly usable with the Qiskit dynamic-typing system.
#[macro_export]
macro_rules! static_dyn_typed {
    ($ty:ty) => {
        impl $crate::StaticDynTyped for $ty {
            fn static_dyn_type_id() -> $crate::DynTypeId<'static> {
                $crate::DynTypeId::of::<$ty>()
            }
        }
    };
}
