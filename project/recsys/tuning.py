"""PERSON-2/3: Tasks 1.2 and 1.5; PERSON-1 reviews selection records."""


def run_search(search_space, evaluate_on_validation, budget, seed):
    """TODO: evaluate trial configurations on the approved development protocol.

    Save every trial's config, split ID, seed, validation metrics, failure status,
    and artifact directory. Select by the agreed primary validation metric and
    direction, not test scores or execution order. Keep individual-model tuning
    and hybrid-coefficient fitting on their assigned development subsets.
    Reuse run_hyper.py only after checking its split/selection behavior matches.
    """
    raise NotImplementedError("PERSON-2/3: implement bounded validation search.")
