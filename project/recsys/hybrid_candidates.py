"""Shared frozen-model test masking for full-catalogue score tables."""

from .contracts import ScoreTable


def test_candidates(table, training, validation, testing, catalogue):
    if table.partition != 'valid' or table.scores.keys() != validation.keys():
        raise ValueError('Validation scores and relevance users differ')
    if validation.keys() != testing.keys() or training.keys() != testing.keys():
        raise ValueError('Expected identical user cohorts')
    scores = {}
    for user, row in table.scores.items():
        if training[user] & validation[user] or training[user] & testing[user] or validation[user] & testing[user]:
            raise ValueError('Outer partitions overlap')
        if set(row) != catalogue - training[user]:
            raise ValueError('Validation candidates differ from catalogue minus training history')
        candidates = catalogue - training[user] - validation[user]
        if not testing[user] <= candidates:
            raise ValueError('Excluded test positives')
        scores[user] = {i: s for i, s in row.items() if i in candidates}
    result = ScoreTable(table.model, table.split_id, 'test', scores)
    result.validate()
    return result
