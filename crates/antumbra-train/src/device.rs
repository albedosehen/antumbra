//! Device + precision selection. The 3090 Ti (Ampere) is the v0
//! training target; CPU is the always-available fallback for builds/tests.
//! And what the CPU at hand can do that changes how a base is served on it.

use candle_core::Device;

/// Whether this CPU multiplies a quantized base through Antumbra's own tiles
/// (`models::qwen::tiles`), which is what puts a decoded token at the rate
/// memory is read. On x86-64: AVX2 with FMA and F16C, which every Intel CPU
/// has since Haswell and every AMD CPU since Excavator, or the AVX-512 of
/// [`wide_tiles`]. On aarch64: NEON with the dot-product instructions, which
/// every Apple silicon CPU has and ARM's server cores from Neoverse N1.
pub fn tiles() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        wide_tiles()
            || (is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c"))
    }
    #[cfg(target_arch = "aarch64")]
    {
        // The dot-product instructions: every Apple silicon CPU, and ARM's
        // server cores from Neoverse N1.
        std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Whether this x86-64 CPU multiplies through the AVX-512 kernels: AVX-512
/// F, BW and VNNI, and F16C, which Intel's Ice Lake and Tiger Lake laptop
/// CPUs and its Xeons from Cascade Lake have, as do AMD's CPUs from Zen 4;
/// Intel's desktop and laptop CPUs since Alder Lake do not. The others that
/// tile multiply through AVX2.
///
/// `ANTUMBRA_CPU_KERNELS=avx2` holds a CPU that has AVX-512 to the AVX2
/// kernels, read once when first asked: for measuring one tier against the
/// other on one machine, and for testing the AVX2 kernels on a runner that
/// would not otherwise reach them.
#[cfg(target_arch = "x86_64")]
pub fn wide_tiles() -> bool {
    static WIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WIDE.get_or_init(|| {
        let held_to_avx2 = std::env::var("ANTUMBRA_CPU_KERNELS")
            .is_ok_and(|kernels| kernels.trim().eq_ignore_ascii_case("avx2"));
        !held_to_avx2
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("f16c")
    })
}

/// Whether this CPU multiplies six-bit tiles as well: every CPU that tiles
/// does. On x86-64 their values are cut out of their bytes with AVX-512 VBMI
/// where the CPU has it, and with AVX2 where it does not (the first CPUs
/// with VNNI, Cascade Lake, among them); on aarch64 with a table and a
/// shift.
pub fn six_bit_tiles() -> bool {
    tiles()
}

/// Whether six bits is the form this CPU serves from when nothing says
/// otherwise: wherever it multiplies six-bit tiles, since Q6_K was measured
/// to decode faster than Q8_0 through each set of kernels. Through AVX-512
/// VBMI and through NEON it decodes about a third more tokens a second.
/// Through AVX2, with a 1.5B base on a laptop CPU held to those kernels, it
/// decoded 25.7 tokens a second where Q8_0 decoded 22.6.
pub fn six_bit_by_default() -> bool {
    six_bit_tiles()
}

/// Whether this build serves from the CPU's tiles though it has a GPU
/// backend: a Metal build on a CPU that tiles. Measured on an Apple M4 Pro
/// with a 1.5B base and one expert, the dense base on Metal decodes at 55
/// tokens a second and the CPU's tiles at 135 from Q6_K and 174 from Q4_K;
/// Metal reads a prompt faster than Q6_K (523 tokens a second against 380)
/// and about as fast as Q4_K (562). An answer is mostly decoding. A CUDA
/// build serves from its GPU.
pub fn serves_from_cpu() -> bool {
    cfg!(all(feature = "metal", not(feature = "cuda"))) && tiles()
}

/// The device a model is loaded on: the CPU for one served from a quantized
/// base where [`serves_from_cpu`], and otherwise [`best_device`], which is
/// where every model that trains goes.
pub fn device_for(served_quantized: bool) -> candle_core::Result<Device> {
    if served_quantized && serves_from_cpu() {
        return Ok(Device::Cpu);
    }
    best_device()
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
