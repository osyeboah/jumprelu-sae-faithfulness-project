"""Dataset and prompt utilities for mechanistic interpretability experiments."""

def get_ioi_prompts():
    """Returns clean and corrupted prompts for the Indirect Object Identification (IOI) task."""
    return [
        {
            "clean": "When Kofi and Ama went to school, Kofi gave a book to",
            "corrupted": "When John and Ama went to school, John gave a book to",
            "correct_target": " Ama",
            "incorrect_target": " Kofi"
        },
        {
            "clean": "When Mary and John went to the store, Mary gave a drink to",
            "corrupted": "When Alice and John went to the store, Alice gave a drink to",
            "correct_target": " John",
            "incorrect_target": " Mary"
        }
    ]
