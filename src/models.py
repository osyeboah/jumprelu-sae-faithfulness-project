"""Model and SAE loader utilities using TransformerLens and SAELens."""

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def load_model_and_sae(layer: int = 8, device: str = None):
    """Loads gpt2-small and the corresponding Layer SAE."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading gpt2-small on device: {device}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    sae_hook_point = f"blocks.{layer}.hook_resid_pre"
    print(f"Loading pre-trained SAE for: {sae_hook_point}...")
    
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_hook_point,
        device=device
    )
    
    return model, sae, device
