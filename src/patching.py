import torch
from typing import List, Dict, Tuple
from transformer_lens import HookedTransformer


def compute_logit_diff(
    logits: torch.Tensor,
    clean_io_tok: int,
    corrupted_io_tok: int
) -> torch.Tensor:
    """
    Computes the logit difference at the final token position.
    """
    final_logits = logits[0, -1, :]
    return final_logits[clean_io_tok] - final_logits[corrupted_io_tok]


def get_clean_and_corrupted_latents(
    model: HookedTransformer,
    sae,
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int = 8
):
    """
    Gets the SAE latent activations for the clean and corrupted prompts.
    """
    hook_point = f"blocks.{layer}.hook_resid_pre"

    _, clean_cache = model.run_with_cache(clean_prompt)
    _, corrupted_cache = model.run_with_cache(corrupted_prompt)

    clean_acts = clean_cache[hook_point]
    corrupted_acts = corrupted_cache[hook_point]

    clean_final_act = clean_acts[0, -1, :].unsqueeze(0)
    corrupted_final_act = corrupted_acts[0, -1, :].unsqueeze(0)

    clean_latents = sae.encode(clean_final_act)
    corrupted_latents = sae.encode(corrupted_final_act)

    return (
        clean_final_act,
        corrupted_final_act,
        clean_latents,
        corrupted_latents
    )


def get_causal_latent_patch_hook(
    sae,
    clean_latents: torch.Tensor,
    corrupted_latents: torch.Tensor,
    latent_indices: List[int]
):
    """
    Creates a hook that transfers selected clean SAE features
    into the corrupted residual stream.
    """
    latent_delta = clean_latents - corrupted_latents

    selected_delta = torch.zeros_like(latent_delta)
    selected_delta[:, latent_indices] = latent_delta[:, latent_indices]

    residual_delta = selected_delta @ sae.W_dec

    def hook_fn(activations: torch.Tensor, hook):
        modified_activations = activations.clone()
        modified_activations[:, -1, :] += residual_delta
        return modified_activations

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

    clean_io_tok = model.to_single_token(clean_io_name)
    corrupted_io_tok = model.to_single_token(corrupted_io_name)

    hook_point = f"blocks.{layer}.hook_resid_pre"

    clean_logits = model(clean_prompt)
    clean_ld = compute_logit_diff(
        clean_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    corrupted_logits = model(corrupted_prompt)
    corrupted_ld = compute_logit_diff(
        corrupted_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    (
        clean_final_act,
        corrupted_final_act,
        clean_latents,
        corrupted_latents
    ) = get_clean_and_corrupted_latents(
        model,
        sae,
        clean_prompt,
        corrupted_prompt,
        layer
    )

    all_latents = list(range(clean_latents.shape[-1]))

    intervention_hook = get_causal_latent_patch_hook(
        sae,
        clean_latents,
        corrupted_latents,
        all_latents
    )

    with model.hooks(
        fwd_hooks=[(hook_point, intervention_hook)]
    ):
        patched_logits = model(corrupted_prompt)

    patched_ld = compute_logit_diff(
        patched_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    denominator = clean_ld - corrupted_ld

    if abs(denominator) < 1e-8:
        recovery = float("nan")
    else:
        recovery = (
            (patched_ld - corrupted_ld) / denominator
        ) * 100

    return {
        "clean_logit_diff": clean_ld,
        "corrupted_logit_diff": corrupted_ld,
        "sae_patched_logit_diff": patched_ld,
        "pct_logit_diff_recovered": recovery
    }


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

    clean_io_tok = model.to_single_token(clean_io_name)
    corrupted_io_tok = model.to_single_token(corrupted_io_name)

    hook_point = f"blocks.{layer}.hook_resid_pre"

    clean_logits = model(clean_prompt)
    clean_ld = compute_logit_diff(
        clean_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    corrupted_logits = model(corrupted_prompt)
    corrupted_ld = compute_logit_diff(
        corrupted_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    denominator = clean_ld - corrupted_ld

    if abs(denominator) < 1e-8:
        raise ValueError(
            "Clean and corrupted logit differences are too close."
        )

    (
        clean_final_act,
        corrupted_final_act,
        clean_latents,
        corrupted_latents
    ) = get_clean_and_corrupted_latents(
        model,
        sae,
        clean_prompt,
        corrupted_prompt,
        layer
    )

    active_indices = torch.nonzero(
        clean_latents[0] > 0
    ).squeeze(-1).tolist()

    print(f"Found {len(active_indices)} active clean SAE latents.")

    latent_effects = []

    for idx in active_indices:

        intervention_hook = get_causal_latent_patch_hook(
            sae,
            clean_latents,
            corrupted_latents,
            [idx]
        )

        with model.hooks(
            fwd_hooks=[(hook_point, intervention_hook)]
        ):
            patched_logits = model(corrupted_prompt)

        patched_ld = compute_logit_diff(
            patched_logits,
            clean_io_tok,
            corrupted_io_tok
        ).item()

        recovery = (
            (patched_ld - corrupted_ld) / denominator
        ) * 100

        latent_effects.append((idx, recovery))

    latent_effects.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return latent_effects[:top_k]
def evaluate_circuit_sufficiency(
    model: HookedTransformer,
    sae,
    clean_prompt: str,
    corrupted_prompt: str,
    clean_io_name: str,
    corrupted_io_name: str,
    top_latent_indices: List[int],
    layer: int = 8
) -> Dict[int, float]:
    """
    Tests how much logit recovery is obtained as more SAE
    latents are patched into the corrupted prompt.
    """

    clean_io_tok = model.to_single_token(clean_io_name)
    corrupted_io_tok = model.to_single_token(corrupted_io_name)

    hook_point = f"blocks.{layer}.hook_resid_pre"

    clean_logits = model(clean_prompt)
    clean_ld = compute_logit_diff(
        clean_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    corrupted_logits = model(corrupted_prompt)
    corrupted_ld = compute_logit_diff(
        corrupted_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    denominator = clean_ld - corrupted_ld

    if abs(denominator) < 1e-8:
        raise ValueError(
            "Clean and corrupted logit differences are too close."
        )

    (
        clean_final_act,
        corrupted_final_act,
        clean_latents,
        corrupted_latents
    ) = get_clean_and_corrupted_latents(
        model,
        sae,
        clean_prompt,
        corrupted_prompt,
        layer
    )

    results = {}

    for k in range(1, len(top_latent_indices) + 1):

        selected_latents = top_latent_indices[:k]

        intervention_hook = get_causal_latent_patch_hook(
            sae,
            clean_latents,
            corrupted_latents,
            selected_latents
        )

        with model.hooks(
            fwd_hooks=[(hook_point, intervention_hook)]
        ):
            patched_logits = model(corrupted_prompt)

        patched_ld = compute_logit_diff(
            patched_logits,
            clean_io_tok,
            corrupted_io_tok
        ).item()

        recovery = (
            (patched_ld - corrupted_ld) / denominator
        ) * 100

        results[k] = recovery

    return results
def ablate_latent(
    model: HookedTransformer,
    sae,
    prompt: str,
    clean_io_tok: int,
    corrupted_io_tok: int,
    latent_index: int,
    layer: int = 8
) -> float:
    """
    Removes one SAE latent from the model activation
    and returns the resulting logit difference.
    """

    hook_point = f"blocks.{layer}.hook_resid_pre"

    _, cache = model.run_with_cache(prompt)
    activations = cache[hook_point]

    final_act = activations[0, -1, :].unsqueeze(0)
    latents = sae.encode(final_act)

    latent_value = latents[:, latent_index]

    latent_delta = torch.zeros_like(latents)
    latent_delta[:, latent_index] = -latent_value

    residual_delta = latent_delta @ sae.W_dec

    def hook_fn(activations: torch.Tensor, hook):
        modified_activations = activations.clone()
        modified_activations[:, -1, :] += residual_delta
        return modified_activations

    with model.hooks(
        fwd_hooks=[(hook_point, hook_fn)]
    ):
        logits = model(prompt)

    logit_diff = compute_logit_diff(
        logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    return logit_diff
def run_latent_ablation_experiment(
    model: HookedTransformer,
    sae,
    clean_prompt: str,
    clean_io_name: str,
    corrupted_io_name: str,
    latent_indices: List[int],
    layer: int = 8
) -> Dict[int, float]:

    clean_io_tok = model.to_single_token(clean_io_name)
    corrupted_io_tok = model.to_single_token(corrupted_io_name)

    clean_logits = model(clean_prompt)

    clean_ld = compute_logit_diff(
        clean_logits,
        clean_io_tok,
        corrupted_io_tok
    ).item()

    results = {}

    for latent_index in latent_indices:

        ablated_ld = ablate_latent(
            model=model,
            sae=sae,
            prompt=clean_prompt,
            clean_io_tok=clean_io_tok,
            corrupted_io_tok=corrupted_io_tok,
            latent_index=latent_index,
            layer=layer
        )

        change = clean_ld - ablated_ld

        results[latent_index] = change

    return results

if __name__ == "__main__":
    print("SAE causal intervention module compiled successfully.")