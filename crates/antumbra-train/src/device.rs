//! Device + precision selection. The 3090 Ti (Ampere) is the v0
//! training target; CPU is the always-available fallback for builds/tests.
//! And what the CPU at hand can do that changes how a base is served on it.

use candle_core::Device;

/// Whether this CPU multiplies a quantized base through Antumbra's own tiles
/// (`models::qwen::tiles`), which is what puts a decoded token at the rate
/// memory is read. On x86-64: AVX-512 F, BW and VNNI, and F16C, which
/// Intel's Ice Lake and Tiger Lake laptop CPUs and its Xeons from Cascade
/// Lake have, as do AMD's CPUs from Zen 4; Intel's desktop and laptop CPUs
/// since Alder Lake do not. On aarch64: NEON with the dot-product
/// instructions, which every Apple silicon CPU has and ARM's server cores
/// from Neoverse N1.
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
/// F, BW and VNNI, and F16C. The others that tile multiply through AVX2.
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

/// Whether this CPU multiplies six-bit tiles as well. On x86-64 their values
/// are unpacked with AVX-512 VBMI, which the first CPUs with VNNI (Cascade
/// Lake) lack; on aarch64 every CPU that tiles does.
pub fn six_bit_tiles() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // Through AVX-512 VBMI where the CPU has it, and through AVX2 where
        // it does not.
        tiles()
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON cuts six-bit values out with a table and a shift.
        tiles()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Whether six bits is the form this CPU serves from when nothing says
/// otherwise: where it was measured to decode faster than Q8_0, which is
/// through AVX-512 VBMI and through NEON. Through AVX2 the six-bit tiles
/// multiply, and `--serve-quant q6_k` serves from them, but their rate
/// beside Q8_0's is not yet measured there, so the default stays Q8_0.
pub fn six_bit_by_default() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        wide_tiles() && is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        six_bit_tiles()
    }
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
