import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.patching import run_sae_patching_experiment, rank_causal_latents

def main():
    print("--- Loading GPT-2 Small & Layer 8 SAE ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
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
        print(f"Rank {rank}: Latent #{latent_id:<6} -> {recovery:.2f}% recovery")

if __name__ == "__main__":
    main()