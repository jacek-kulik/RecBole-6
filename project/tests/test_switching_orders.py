"""Switching commutes with local calibration, but not cohort exposure allocation."""

import unittest
from functools import partial

from project.recsys.contracts import RerankContext, ScoreTable
from project.recsys.experiments.switching_orders import (
    rerank_then_switch,
    switch_scores,
    switch_then_rerank,
)
from project.recsys.rerankers import calibration, item_fairness


class SwitchingOrderTests(unittest.TestCase):
    def setUp(self):
        self.tables = [
            ScoreTable('one','s','valid',{'u1':{'a':2.,'b':1.},'u2':{'a':2.,'b':1.}}),
            ScoreTable('two','s','valid',{'u1':{'b':2.,'a':1.},'u2':{'a':2.,'b':1.}}),
        ]
        self.routes = {'u1':'one','u2':'two'}
        self.context = RerankContext(item_features={'a':('Action',),'b':('Drama',)},
            user_profiles={u:{'Action':.5,'Drama':.5} for u in self.routes},
            item_groups={'a':'head','b':'tail'},
            popularity_profiles={u:{'head':.5,'tail':.5} for u in self.routes})

    def test_local_calibration_commutes_at_every_strength(self):
        for method in (calibration.rerank, partial(calibration.rerank,attribute='popularity')):
            for strength in (0,.25,.5,.75,1):
                for k in (1,2):
                    self.assertEqual(rerank_then_switch(self.tables,method,self.routes,self.context,k,strength),
                                     switch_then_rerank(self.tables,method,self.routes,self.context,k,strength))

    def test_exposure_uses_component_cohorts_before_routing(self):
        pre=rerank_then_switch(self.tables,item_fairness.rerank,self.routes,self.context,1,1)
        post=switch_then_rerank(self.tables,item_fairness.rerank,self.routes,self.context,1,1)
        self.assertEqual(pre,{'u1':['a'],'u2':['a']})
        self.assertEqual(post,{'u1':['a'],'u2':['b']})
        self.assertNotEqual(pre,post)
        self.assertEqual(pre,rerank_then_switch(self.tables,item_fairness.rerank,self.routes,self.context,1,1))

    def test_zero_strength_preserves_switched_ranking(self):
        expected=switch_scores(self.tables,self.routes).top_k(2)
        for pipeline in (rerank_then_switch,switch_then_rerank):
            self.assertEqual(pipeline(self.tables,item_fairness.rerank,self.routes,RerankContext(),2,0),expected)

    def test_routes_and_candidate_alignment_are_checked(self):
        for routes in ({'u1':'one'},{'u1':'one','u2':'unknown'}):
            with self.assertRaises(ValueError): switch_scores(self.tables,routes)
        bad=ScoreTable('two','s','valid',{'u1':{'a':2.,'b':1.},'u2':{'a':2.,'c':1.}})
        with self.assertRaises(ValueError): switch_scores([self.tables[0],bad],self.routes)


if __name__=='__main__': unittest.main()
