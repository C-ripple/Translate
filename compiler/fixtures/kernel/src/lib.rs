#![no_std]

/// Exercise target-width wrapping arithmetic and unsigned comparisons.
/// # Safety
/// The caller must satisfy the same contract as [`copy`].
pub unsafe fn copy_arithmetic(src: *const f32, dst: *mut f32, n: u32) {
    let lane = unsafe { ripple_kernel::lane_id() };
    let added = (lane as usize + usize::MAX) as u32;
    let multiplied = added * 0x8000_0001;
    let selected = multiplied - 3;
    if lane < n {
        if lane == 2 {
            unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
        } else if selected < n {
            unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
        }
    }
}

/// Negative control-flow fixture.
/// # Safety
/// The caller must satisfy the same contract as [`copy`].
pub unsafe fn unsupported_loop(src: *const f32, dst: *mut f32, n: u32) {
    let mut lane = unsafe { ripple_kernel::lane_id() };
    while lane < n {
        unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
        lane += 32;
    }
}

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {}
}
#[inline(never)]
fn helper_with_drop() -> u32 {
    let _guard = Guard;
    unsafe { ripple_kernel::lane_id() }
}
pub fn unsupported_drop_helper() -> u32 {
    helper_with_drop()
}

/// Initial kernel fixture, compiled through the owned IR and Ripple C backend.
///
/// # Safety
/// For active lanes, src and dst must each hold at least n valid aligned floats
/// in target memory; writes must be exclusive and source/destination disjoint.
pub unsafe fn copy(src: *const f32, dst: *mut f32, n: u32) {
    let lane = unsafe { ripple_kernel::lane_id() };
    if lane < n {
        unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
    }
}

#[inline(never)]
fn helper_lane() -> u32 {
    unsafe { ripple_kernel::lane_id() }
}

/// Same extraction fixture with the marker behind a helper.
///
/// # Safety
/// The caller must satisfy the same contract as [`copy`].
pub unsafe fn copy_via_helper(src: *const f32, dst: *mut f32, n: u32) {
    let lane = helper_lane();
    if lane < n {
        unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
    }
}

#[cfg(feature = "borrow-error")]
pub fn borrow_error() -> u32 {
    let mut value = 0;
    let a = &mut value;
    let b = &mut value;
    *a += 1;
    *b
}

#[cfg(feature = "type-error")]
pub fn type_error() -> u32 {
    false
}

#[cfg(feature = "unavailable-body")]
unsafe extern "C" {
    fn unknown_device_call() -> u32;
}

#[cfg(feature = "unavailable-body")]
pub fn missing_body() -> u32 {
    unsafe { unknown_device_call() }
}

#[cfg(feature = "unsupported-type")]
pub fn wide_lane() -> u64 {
    unsafe { ripple_kernel::lane_id() as u64 }
}

/// Deliberately lacks a length guard: imported IR must be rejected by codegen.
/// # Safety
/// Test-only fixture; requires a full block of valid disjoint buffer elements.
pub unsafe fn unguarded(src: *const f32, dst: *mut f32, _n: u32) {
    let lane = unsafe { ripple_kernel::lane_id() };
    unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
}

/// Deliberately has a scalar output address: codegen must reject it.
/// # Safety
/// Test-only negative fixture; not safe to launch as a parallel kernel.
pub unsafe fn scalar_store(src: *const f32, dst: *mut f32, n: u32) {
    let lane = unsafe { ripple_kernel::lane_id() };
    if lane < n {
        unsafe { *dst = *src.add(lane as usize) };
    }
}

/// Exercise local values assigned across divergent branches before a join.
/// # Safety
/// The caller must satisfy the same contract as [`copy`].
pub unsafe fn copy_via_join(src: *const f32, dst: *mut f32, n: u32) {
    let lane = unsafe { ripple_kernel::lane_id() };
    if lane < n {
        let index;
        if lane < 16 {
            index = lane;
        } else {
            index = lane;
        }
        unsafe { *dst.add(index as usize) = *src.add(index as usize) };
    }
}

/// A selective mask leaves the lower half of every output block untouched.
/// # Safety
/// The caller must satisfy the same contract as [`copy`].
pub unsafe fn copy_upper_half(src: *const f32, dst: *mut f32, n: u32) {
    let lane = unsafe { ripple_kernel::lane_id() };
    if lane < n {
        if lane < 16 {
            // No output effect for these lanes.
        } else {
            unsafe { *dst.add(lane as usize) = *src.add(lane as usize) };
        }
    }
}

/// Deliberately unsupported integer operation, even with a safe divisor.
pub fn unsupported_division() -> u32 {
    let lane = unsafe { ripple_kernel::lane_id() };
    lane / 2
}

/// Deliberately unsupported integer operation, even with an in-range shift.
pub fn unsupported_shift() -> u32 {
    let lane = unsafe { ripple_kernel::lane_id() };
    lane << 1
}
