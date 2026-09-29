"""PERSON-3: Tasks 1.3-1.5. Reuse shared scores; keep RecBole changes separate."""

from .contracts import ScoreTable, require_aligned


def fit_weighted_hybrid(development_scores, targets):
    """Fit regression coefficients and score transforms on development data.

    TODO: agree regression target, fitting subset/cross-fitting, regularization,
    normalization, and constraints. Separate coefficient fitting from validation
    used for selecting hyperparameters; never fit on test labels. Save component
    model/run IDs, fitted transforms, coefficients, and target definition.
    """
    require_aligned(development_scores)
    if any(table.partition == "test" for table in development_scores):
        raise ValueError("Test scores must not be used to fit a hybrid.")
    raise NotImplementedError("PERSON-3: fit the regression-based hybrid (1.3).")


def predict_weighted_hybrid(component_scores, fitted_hybrid) -> ScoreTable:
    """TODO: apply saved transforms/coefficients to aligned candidates, then rank."""
    require_aligned(component_scores)
    raise NotImplementedError("PERSON-3: implement weighted hybrid inference.")


def predict_other_hybrid(component_scores, settings) -> ScoreTable:
    # TODO Task 1.4: add the alternatives from lectures, each with its own name
    # and configuration. Document how missing candidates or rank fusion work.
    require_aligned(component_scores)
    raise NotImplementedError("PERSON-3: implement the agreed alternative hybrids.")
