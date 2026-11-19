//! Device + precision selection. The 3090 Ti (Ampere) is the v0
//! training target; CPU is the always-available fallback for builds/tests.
//! And what the CPU at hand can do that changes how a base is served on it.

use candle_core::Device;

/// Whether this CPU multiplies a quantized base through Antumbra's own tiles
/// (`models::qwen::tiles`), which is what puts a decoded token at the rate
/// memory is read: AVX-512 F, BW and VNNI, and F16C. Intel's Ice Lake and
/// Tiger Lake laptop CPUs and its Xeons from Cascade Lake have them, as do
/// AMD's CPUs from Zen 4; Intel's desktop and laptop CPUs since Alder Lake
/// do not.
pub fn tiles() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether this CPU multiplies six-bit tiles as well: their values are
/// unpacked with AVX-512 VBMI, which the first CPUs with VNNI (Cascade Lake)
/// lack.
pub fn six_bit_tiles() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        tiles() && is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

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
