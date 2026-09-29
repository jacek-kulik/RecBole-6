"""Tasks 3.1-3.4: PERSON-4 leads reranking, PERSON-3 orders, PERSON-5 groups."""


def rerank_then_combine(component_scores, reranker, hybrid, context, settings):
    # TODO PERSON-3 (3.3 order 1): rerank each individual model, then combine.
    # Rerankers return ordered lists, while hybrids consume full candidate scores.
    # Agree and record a rank-to-score/fusion rule, truncation depth, and missing
    # candidate treatment. Do not accidentally combine the original score tables.
    raise NotImplementedError("PERSON-3: implement rerank-then-combine.")


def combine_then_rerank(component_scores, reranker, hybrid, context, settings):
    # TODO PERSON-3 (3.3 order 2): combine aligned component scores first, then
    # rerank the hybrid. Match models, candidates, K, and strengths to order 1.
    raise NotImplementedError("PERSON-3: implement combine-then-rerank.")


def run():
    # TODO: 3.1 integrate the four methods; 3.2 evaluate strength sweeps on selected
    # individual models; 3.3 call BOTH orders above; 3.4 compare group outcomes.
    # Select strengths on development data, freeze them, then report test effects.
    raise NotImplementedError("Task 3 orchestration awaits rerankers and pipeline orders.")
