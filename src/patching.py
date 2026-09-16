import torch
from typing import List, Dict, Tuple
from transformer_lens import HookedTransformer

def compute_logit_diff(
    logits: torch.Tensor, 
    clean_io_tok: int, 
    corrupted_io_tok: int
) -> torch.Tensor:
    """
    Computes logit difference: logit(correct_target) - logit(incorrect_target)
    at the final token position.
    """
    final_logits = logits[0, -1, :]
    return final_logits[clean_io_tok] - final_logits[corrupted_io_tok]


def get_sae_reconstruction_hook(sae, latent_indices: List[int] = None):
    """
    Creates a PyTorch forward hook that replaces dense model layer activations
    with SAE reconstructions (or selected latent subsets).
    """
    def hook_fn(activations: torch.Tensor, hook):
        # Flatten sequence length to encode via SAE: [batch, pos, d_model] -> [batch * pos, d_model]
        shape = activations.shape
        flat_acts = activations.view(-1, shape[-1])
        
        # 1. Encode activations into sparse latents
        feature_acts = sae.encode(flat_acts)
        
        # 2. Optional: Filter activations down to specific causal features
        if latent_indices is not None:
            mask = torch.zeros_like(feature_acts)
            mask[:, latent_indices] = 1.0
            feature_acts = feature_acts * mask
            
        # 3. Decode back into dense model space
        reconstructed_acts = sae.decode(feature_acts)
        
        # Reshape back to model expectations [batch, pos, d_model]
        return reconstructed_acts.view(shape)
        
    return hook_fn


def run_sae_patching_experiment(
    model: HookedTransformer,
    sae,
    clean_prompt: str,
    corrupted_prompt: str,
    clean_io_name: str,
    corrupted_io_name: str,
    layer: int = 8
) -> Dict[str, float]:
    """
    Runs baseline and SAE-patched causal interventions on IOI prompts.
    """
    # Tokenize IO target names
    clean_io_tok = model.to_single_token(clean_io_name)
    corrupted_io_tok = model.to_single_token(corrupted_io_name)

    # 1. Clean Run Baseline
    clean_logits = model(clean_prompt)
    clean_ld = compute_logit_diff(clean_logits, clean_io_tok, corrupted_io_tok).item()

    # 2. Corrupted Run Baseline
    corrupted_logits = model(corrupted_prompt)
    corrupted_ld = compute_logit_diff(corrupted_logits, clean_io_tok, corrupted_io_tok).item()

    # 3. Full SAE Patching Run (Replace residual stream with full SAE reconstruction)
    sae_hook = get_sae_reconstruction_hook(sae, latent_indices=None)
    hook_point = f"blocks.{layer}.hook_resid_pre"
    
    with model.hooks(fwd_hooks=[(hook_point, sae_hook)]):
        patched_logits = model(clean_prompt)
        
    patched_ld = compute_logit_diff(patched_logits, clean_io_tok, corrupted_io_tok).item()

    # Calculate logit difference recovery percentage
    ld_recovered = ((patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)) * 100

    results = {
        "clean_logit_diff": clean_ld,
        "corrupted_logit_diff": corrupted_ld,
        "sae_patched_logit_diff": patched_ld,
        "pct_logit_diff_recovered": ld_recovered
    }
    
    return results


if __name__ == "__main__":
    print("SAE Activation Patching module compiled successfully.")


def rank_causal_latents(
    model: HookedTransformer,
    sae,
    clean_prompt: str,
    corrupted_prompt: str,
    clean_io_name: str,
    corrupted_io_name: str,
    layer: int = 8,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Patches latents one-by-one to measure individual causal contributions.
    """
    clean_io_tok = model.to_single_token(clean_io_name)
    corrupted_io_tok = model.to_single_token(corrupted_io_name)

    # 1. Extract active latents on clean prompt
    hook_point = f"blocks.{layer}.hook_resid_pre"
    _, cache = model.run_with_cache(clean_prompt)
    clean_acts = cache[hook_point]
    
    # Encode last token activations
    flat_acts = clean_acts[0, -1, :].unsqueeze(0)
    feature_acts = sae.encode(flat_acts)[0]
    active_indices = torch.nonzero(feature_acts > 0).squeeze(-1).tolist()

    # 2. Measure baseline corrupted logit diff
    corrupted_logits = model(corrupted_prompt)
    corrupted_ld = compute_logit_diff(corrupted_logits, clean_io_tok, corrupted_io_tok).item()
    clean_logits = model(clean_prompt)
    clean_ld = compute_logit_diff(clean_logits, clean_io_tok, corrupted_io_tok).item()

    latent_effects = []

    # 3. Test each active latent individually
    for idx in active_indices:
        sae_hook = get_sae_reconstruction_hook(sae, latent_indices=[idx])
        with model.hooks(fwd_hooks=[(hook_point, sae_hook)]):
            patched_logits = model(corrupted_prompt)
        
        patched_ld = compute_logit_diff(patched_logits, clean_io_tok, corrupted_io_tok).item()
        recovery = ((patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)) * 100
        latent_effects.append((idx, recovery))

    # Sort descending by recovery percentage
    latent_effects.sort(key=lambda x: x[1], reverse=True)
    return latent_effects[:top_k]