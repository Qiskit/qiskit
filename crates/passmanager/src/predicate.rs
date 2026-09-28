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

use std::{
    any::{self, Any},
    marker,
};

use crate::{IR, PassContext, PassError};
use qiskit_util::dyn_types::DynTypeId;

/// The base behavior for a [`Task`] predicate written in first-party Rust code.
pub trait StaticPredicate<In>: Send + Sync + Sized + 'static
where
    In: IR,
{
    /// Evaluate the predicate.
    fn evaluate(&self, ir: &In, context: &PassContext) -> anyhow::Result<bool>;

    /// Turn this object into the full type-erased version.
    fn into_predicate(self) -> Box<dyn Predicate> {
        Box::new(StaticPredicateOb {
            ob: self,
            phantom: marker::PhantomData,
        })
    }
}

impl<In, F> StaticPredicate<In> for F
where
    In: IR,
    F: Fn(&In, &PassContext) -> anyhow::Result<bool> + Send + Sync + 'static,
{
    fn evaluate(&self, ir: &In, context: &PassContext) -> anyhow::Result<bool> {
        self(ir, context)
    }
}

/// Type-system wrapper object to move the `In` IR type of `StaticPredicate` into a concrete
/// object.
struct StaticPredicateOb<C, In> {
    ob: C,
    phantom: marker::PhantomData<In>,
}
impl<C, In> Predicate for StaticPredicateOb<C, In>
where
    In: IR,
    C: StaticPredicate<In>,
{
    fn ir_id(&self) -> DynTypeId<'_> {
        DynTypeId::of::<In>()
    }
    fn name(&self) -> &str {
        any::type_name::<C>()
    }
    fn evaluate(&self, ir: &dyn IR, context: &PassContext) -> Result<bool, PassError> {
        let ir = (ir as &dyn Any)
            .downcast_ref::<In>()
            .ok_or(PassError::Conversion)?;
        self.ob.evaluate(ir, context).map_err(PassError::Runtime)
    }
}

/// The trait for objects that can be used as predicates in some [`Task`] variants.
pub trait Predicate: Send + Sync {
    /// Return the type ID of the IR that this predicate reads.
    fn ir_id(&self) -> DynTypeId<'_>;
    /// A human-readable name for the predicate.
    ///
    /// This is primarily for debugging purposes and may generally depend on the crate the predicate
    /// is defined in. No part of the name should be relied on as being stable.
    fn name(&self) -> &str;
    /// Evaluate the predicate.
    ///
    /// In general, the [`PassManager`](crate::PassManager) construction logic will have validated
    /// that `ir` matches [`ir_id`](Self::ir_id), so `ir` should typically cast correctly into the
    /// desired object.  However, this trait may be called outside the context of a validated
    /// pipeline.
    fn evaluate(&self, ir: &dyn IR, context: &PassContext) -> Result<bool, PassError>;
}

/// Stop as soon as the loop body reports that it left the IR alone.
pub struct UntilStable;
impl UntilStable {
    pub fn is_stable(context: &PassContext) -> bool {
        !context.ir_modified
    }
}
impl<In: IR> StaticPredicate<In> for UntilStable {
    fn evaluate(&self, _ir: &In, context: &PassContext) -> anyhow::Result<bool> {
        Ok(Self::is_stable(context))
    }
}
