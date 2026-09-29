"""Optional shared search wrapper for PERSON-2/3; existing scripts may suffice."""


def run_search(search_space, evaluate_on_validation, budget, seed):
    """TODO: evaluate trial configurations on the approved development protocol.

    Use this only if a shared wrapper makes both tuning tasks easier. The
    required handoff is the search space, trial outcomes, and selected run;
    PERSON-4 reviews individual tuning and PERSON-1 reviews hybrid tuning.
    Save every trial's config, split ID, seed, validation metrics, failure status,
    and artifact directory. Select by the agreed primary validation metric and
    direction, not test scores or execution order. Keep individual-model tuning
    and hybrid-coefficient fitting on their assigned development subsets.
    Reuse run_hyper.py only after checking its split/selection behavior matches.
    """
    raise NotImplementedError("PERSON-2/3: implement bounded validation search.")
