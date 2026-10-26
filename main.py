
"""Check baseline next-token preferences on a controlled IOI prompt pair."""

from src.models import load_model_and_sae
from src.prompts import get_ioi_prompts


def main():
    model, sae, device = load_model_and_sae(layer=8)
    sample = get_ioi_prompts()[0]

    correct_id = model.to_single_token(sample["correct_target"])
    incorrect_id = model.to_single_token(sample["incorrect_target"])

    print(f"Device: {device}")
    print("SAE hook: blocks.8.hook_resid_pre")

    for label in ("clean", "corrupted"):
        prompt = sample[label]
        logits = model(prompt)

        logit_diff = (
            logits[0, -1, correct_id]
            - logits[0, -1, incorrect_id]
        )

        print(f"\n{label.capitalize()} prompt: {prompt!r}")
        print(f"Mary - John logit difference: {logit_diff.item():.6f}")

    print("\nBaseline check complete. No SAE intervention was run.")


if __name__ == "__main__":
    main()
