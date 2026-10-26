"""Known-answer rank conversion and matched-order controls for Task 3.3."""

import unittest
from copy import deepcopy

from project.recsys.contracts import RerankContext, ScoreTable
from project.recsys.experiments.task3 import (
    combine_standardized_scores,
    combine_then_rerank,
    rerank_then_combine,
    score_preserving_permutation,
)
from project.recsys.rerankers import calibration, item_fairness


class RerankingOrderTests(unittest.TestCase):
    def setUp(self):
        self.tables = [ScoreTable(n,'s','valid',{'u':{'a':3.,'b':2.,'c':1.}}) for n in ('one','two')]
        self.fitted = {'components':['one','two'],'coefficients':[.7,.3],'intercept':3.}
        self.context = RerankContext(item_features={'a':('Action',),'b':('Action',),'c':('Drama',)},
                                     user_profiles={'u':{'Action':.5,'Drama':.5}})

    def test_permutation_preserves_scores_and_original_remainder(self):
        table=self.tables[0]; old=deepcopy(table)
        transformed=score_preserving_permutation(table,{'u':['c']})
        self.assertEqual(transformed.scores,{'u':{'c':3.,'a':2.,'b':1.}})
        self.assertEqual(sorted(transformed.scores['u'].values()),sorted(table.scores['u'].values()))
        self.assertEqual(table,old)

    def test_fusion_does_not_renormalize_or_refit(self):
        result=combine_standardized_scores(self.tables,self.fitted)
        self.assertAlmostEqual(result.scores['u']['a'],6.)
        self.assertAlmostEqual(result.scores['u']['b'],5.)
        self.assertAlmostEqual(result.scores['u']['c'],4.)

    def test_both_orders_zero_exactly_match_original_without_metadata(self):
        expected=combine_standardized_scores(self.tables,self.fitted).top_k(2)
        for method in (calibration.rerank,item_fairness.rerank):
            for pipeline in (rerank_then_combine,combine_then_rerank):
                self.assertEqual(pipeline(self.tables,method,self.fitted,RerankContext(),
                                          {'k':2,'strength':0}),expected)

    def test_pre_order_uses_reranked_outputs(self):
        result=rerank_then_combine(self.tables,calibration.rerank,self.fitted,self.context,
                                   {'k':2,'strength':1})
        self.assertEqual(result,{'u':['a','c']})
        self.assertNotEqual(result,combine_standardized_scores(self.tables,self.fitted).top_k(2))

    def test_invalid_rankings_and_candidate_mismatch_fail(self):
        for ranked in ({'u':['missing']},{'u':['a','a']},{'other':['a']},{'u':[]}):
            with self.assertRaises(ValueError): score_preserving_permutation(self.tables[0],ranked)
        mismatch=ScoreTable('two','s','valid',{'u':{'a':3.,'b':2.,'d':1.}})
        for pipeline in (rerank_then_combine,combine_then_rerank):
            with self.assertRaises(ValueError):
                pipeline([self.tables[0],mismatch],calibration.rerank,self.fitted,self.context,
                         {'k':2,'strength':0})


if __name__=='__main__':unittest.main()
