"""Small input checks and score normalization shared by Jacek's rerankers."""

from collections.abc import Mapping
from math import isclose, isfinite

from ..contracts import ScoreTable


class MissingMetadataError(ValueError):
    """The method cannot run with the supplied training profiles or features."""


def normalized_scores(scores: ScoreTable, k: int, strength: float) -> dict[str, dict[str, float]]:
    scores.validate()
    if type(k) is not int or k <= 0 or any(len(row) < k for row in scores.scores.values()):
        raise ValueError("K must be a positive integer within every candidate pool")
    if isinstance(strength, bool) or not isinstance(strength, (float, int)) or not isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Strength must be finite and between zero and one")
    result = {}
    for user, row in scores.scores.items():
        low, high = min(row.values()), max(row.values())
        span = high - low
        if not isfinite(span):
            raise ValueError("Score range is too large to normalize")
        result[user] = {item: (value - low) / span if span else 0.0 for item, value in row.items()}
    return result


def validate_profile(profile: Mapping[str, float], user: str) -> None:
    if not profile:
        raise MissingMetadataError(f"No training profile for user {user}")
    if any(not isinstance(key, str) or not key or not isfinite(value) or value < 0
           for key, value in profile.items()) or not isclose(sum(profile.values()), 1.0, abs_tol=1e-8):
        raise ValueError(f"Training profile for user {user} must be a probability distribution")
