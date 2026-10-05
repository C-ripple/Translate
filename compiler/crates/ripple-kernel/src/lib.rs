//! Compiler markers for extraction experiments, not a device runtime.
//!
//! No marker has an executable implementation. Even bypassing the cfg guard will
//! leave unresolved symbols if ordinary code generation attempts to link them.
#![no_std]
#![feature(rustc_attrs)]
#![allow(internal_features)]

#[cfg(not(ripple_frontend))]
compile_error!(
    "ripple-kernel requires the experimental Ripple compiler; normal compilation is unsupported"
);

unsafe extern "C" {
    /// Returns the logical lane in the selected one-dimensional block.
    ///
    /// # Safety
    /// Only meaningful to a Ripple compiler which replaces this marker. There is
    /// no implementation that can be invoked by an ordinary Rust executable.
    #[rustc_diagnostic_item = "ripple_lane_id_v1"]
    pub fn lane_id() -> u32;
}
