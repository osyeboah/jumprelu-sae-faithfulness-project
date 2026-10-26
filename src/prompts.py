
"""Controlled prompts for IOI next-token experiments."""


def get_ioi_prompts():
    """Return a matched clean and corrupted IOI prompt pair."""
    return [
        {
            "clean": (
                "When John and Mary went to the store, "
                "John gave a book to"
            ),
            "corrupted": (
                "When John and Mary went to the store, "
                "Mary gave a book to"
            ),
            "correct_target": " Mary",
            "incorrect_target": " John",
        }
    ]
