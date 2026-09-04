"""Main entry point for SAE fidelity vs faithfulness experiments."""

import torch
from src.models import load_model_and_sae
from src.prompts import get_ioi_prompts

def run_pipeline():
    # 1. Load Model and Layer 8 SAE
    model, sae, device = load_model_and_sae(layer=8)
    
    # 2. Grab Benchmark Prompt
    prompts = get_ioi_prompts()
    sample = prompts[0]
    
    print(f"\n--- Testing Prompt ---")
    print(f"Clean: '{sample['clean']}'")
    
    # 3. Run model & extract Layer 8 activations
    _, cache = model.run_with_cache(sample['clean'], names_filter="blocks.8.hook_resid_pre")
    raw_activations = cache["blocks.8.hook_resid_pre"]
    
    # 4. Encode via SAE
    feature_acts = sae.encode(raw_activations)
    active_count = (feature_acts[0, -1, :] > 0).sum().item()
    
    print(f"Layer 8 Residual Vector Dim: {raw_activations.shape[-1]}")
    print(f"SAE Latent Dimension: {sae.cfg.d_sae}")
    print(f"Active Features on Last Token: {active_count}")
    print("\nSetup verified successfully!")

if __name__ == "__main__":
    run_pipeline()
