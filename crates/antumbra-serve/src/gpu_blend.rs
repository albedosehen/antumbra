//! Live GPU check that the resident engine, built at the blend rank, serves an
//! expert exactly as a model at the expert's own rank does, and that a blend
//! of the expert with itself at half weight each comes out the same.
//!
//! Ignored, and asserts `ANTUMBRA_GPU_BLEND_ADAPTER` (an adapter trained on
//! the default base) is set. On the GPU host, from the `d2-probe` image, which
//! carries this crate's library tests:
//!
//!   docker run --rm --device nvidia.com/gpu=all -v antumbra-gpu-test-weights:/weights \
//!     -v "$ADAPTER:/adapter.safetensors:ro" -e ANTUMBRA_GPU_BLEND_ADAPTER=/adapter.safetensors \
//!     --entrypoint /usr/local/bin/antumbra-d2 antumbra-d2 --ignored --nocapture gpu_blend

use antumbra_core::ports::{ActRequest, Serve};
use antumbra_core::ExpertId;
use antumbra_train::{CandleModelLoader, CausalLm, ModelLoader, RaftConfig};

use crate::MultiAdapterServe;

const PROMPTS: [&str; 6] = [
    "Extract the text of manual.pdf into manual.txt. Command only.",
    "Create a git branch for issue #12, \"Add search\". Command only.",
    "Open a pull request for fix/a. I branched it from feat/b, which has not merged. Command only.",
    "Show disk usage on this machine. Command only.",
    "Count the lines in notes.txt. Command only.",
    "Write a one-line Rust function that adds two i64 values.",
];

// MultiAdapterServe generates under block_in_place: a multi-threaded runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "live GPU: set ANTUMBRA_GPU_BLEND_ADAPTER and run with --ignored"]
async fn gpu_blend_serves_an_expert_as_its_own_rank_does() {
    let adapter = std::env::var("ANTUMBRA_GPU_BLEND_ADAPTER")
        .expect("ANTUMBRA_GPU_BLEND_ADAPTER names the adapter to serve");
    let cfg = RaftConfig {
        temperature: 0.0,
        max_new_tokens: 48,
        ..RaftConfig::default()
    };
    let base = cfg.base_model.clone();

    let mut own = ModelLoader::load(&CandleModelLoader::new(cfg.clone()), &base, None)
        .await
        .expect("the base loads at the trained rank");
    own.load_adapter(&adapter).expect("the adapter loads");
    let mut want = Vec::new();
    for p in PROMPTS {
        want.push(own.generate(p, 1).await.expect("generates").remove(0));
    }
    drop(own);

    let expert = ExpertId::new("expert:blend-probe");
    let engine = MultiAdapterServe::new(&base, cfg).with_adapter(expert.clone(), &adapter);
    let (mut same, mut halves_same) = (0, 0);
    for (p, w) in PROMPTS.iter().zip(&want) {
        let one = engine
            .act(ActRequest::new("one", *p, vec![expert.clone()]))
            .await
            .expect("serves one expert")
            .final_output;
        let mut halves = ActRequest::new("halves", *p, vec![expert.clone(), expert.clone()]);
        halves.weights = vec![0.5, 0.5];
        let two = engine
            .act(halves)
            .await
            .expect("serves a blend")
            .final_output;
        eprintln!("PROMPT {p}\n  own rank: {w:?}\n  blend rank: {one:?}\n  halves: {two:?}");
        same += usize::from(one == *w);
        halves_same += usize::from(two == *w);
    }
    eprintln!(
        "RESULT: {same}/{n} the same at the blend rank, {halves_same}/{n} for halves",
        n = PROMPTS.len()
    );
    assert_eq!(same, PROMPTS.len(), "the blend rank changed an answer");
}
