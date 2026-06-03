//! Device + precision selection (ADR-0010). The 3090 Ti (Ampere) is the v0
//! training target; CPU is the always-available fallback for builds/tests.

use candle_core::Device;

/// The best available device: CUDA/Metal when the feature is on, else CPU.
pub fn best_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0);
    }
    #[cfg(feature = "metal")]
    {
        return Device::new_metal(0);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}
