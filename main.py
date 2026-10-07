import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

from src.patching import (
    run_sae_patching_experiment,
    rank_causal_latents,
    evaluate_circuit_sufficiency,
    run_latent_ablation_experiment
)


def main():
    print("--- Loading GPT-2 Small & Layer 8 SAE ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device
    )

    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device
    )

    clean_prompt = "When John and Mary went to the store, John gave a book to Mary"
    corrupted_prompt = "When John and Mary went to the store, Mary gave a book to John"

    print("\n--- Executing SAE Causal Interventions ---")

    results = run_sae_patching_experiment(
        model=model,
        sae=sae,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        clean_io_name=" Mary",
        corrupted_io_name=" John",
        layer=8
    )

    print("\n================ EXPERIMENTAL RESULTS ================")
    print(f"Clean Logit Difference:         {results['clean_logit_diff']:.4f}")
    print(f"Corrupted Logit Difference:     {results['corrupted_logit_diff']:.4f}")
    print(f"SAE-Patched Logit Difference:   {results['sae_patched_logit_diff']:.4f}")
    print(f"Logit Diff Recovered by SAE:    {results['pct_logit_diff_recovered']:.2f}%")
    print("=======================================================")

    print("\n--- Ranking Top Causal Latents ---")

    top_latents = rank_causal_latents(
        model=model,
        sae=sae,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        clean_io_name=" Mary",
        corrupted_io_name=" John",
        layer=8,
        top_k=5
    )

    print("\nTop 5 Causal SAE Latents:")

    for rank, (latent_id, recovery) in enumerate(top_latents, 1):
        print(
            f"Rank {rank}: Latent #{latent_id:<6} -> "
            f"{recovery:.2f}% recovery"
        )

    top_indices = [latent_id for latent_id, _ in top_latents]

    print("\n--- Testing Combined SAE Latents ---")

    circuit_results = evaluate_circuit_sufficiency(
        model=model,
        sae=sae,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        clean_io_name=" Mary",
        corrupted_io_name=" John",
        top_latent_indices=top_indices,
        layer=8
    )

    print("\nCombined Latent Results:")

    for k, recovery in circuit_results.items():
        print(
            f"Top {k} Latents Combined -> "
            f"{recovery:.2f}% recovery"
        )

    print("\n--- Testing Latent Ablation ---")

    ablation_indices = [13484, 15707, 16308, 8876, 12354]

    ablation_results = run_latent_ablation_experiment(
        model=model,
        sae=sae,
        clean_prompt=clean_prompt,
        clean_io_name=" Mary",
        corrupted_io_name=" John",
        latent_indices=ablation_indices,
        layer=8
    )

    print("\nLatent Ablation Results:")

    for latent_id, change in ablation_results.items():
        print(
            f"Latent #{latent_id} ablated -> "
            f"{change:.4f} logit difference change"
        )


if __name__ == "__main__":
    main()