"""Task 2.1: PERSON-4 implements, PERSON-1 reviews; Recall is a working example."""


def recall_at_k(rankings, relevant, k):
    """Macro recall, with one equally weighted contribution per evaluation user.

    This example requires nonempty relevance sets and complete user coverage.
    Decide the real cohort/relevance rule before calling it on MovieLens outputs.
    """
    if k <= 0 or not relevant or rankings.keys() != relevant.keys():
        raise ValueError("Require positive K and exactly the evaluation users.")
    recalls = []
    for user, positives in relevant.items():
        items = rankings[user][:k]
        if not positives or len(items) != k or len(set(items)) != k:
            raise ValueError("Require nonempty relevance and K distinct ranked items.")
        recalls.append(len(set(items) & set(positives)) / len(set(positives)))
    return sum(recalls) / len(recalls)


def additional_accuracy_metrics(rankings, relevant, k):
    # TODO PERSON-4: independently implement the agreed Precision/MRR/NDCG/etc.
    # State gain, discount, cutoff, averaging, and empty-user behavior. Add small
    # known-answer checks before using these in model selection or final tables.
    raise NotImplementedError("PERSON-4: implement the remaining accuracy metrics.")
