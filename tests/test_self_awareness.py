import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_awareness import reflect, grounded_self_reply

class SelfAwarenessTests(unittest.TestCase):
    def test_report_is_read_only_and_preserves_distinctions(self):
        current=torch.tensor([.7,.3]); previous=torch.tensor([.2,.5]); fresh=torch.tensor([.4,.4])
        turn=SimpleNamespace(state=SimpleNamespace(emotion=current), before=SimpleNamespace(emotion=previous),
                             proposal=fresh,policy=torch.tensor([.7,.3]),reflex_share=torch.tensor(.2))
        report=reflect(turn,('stress','calm'),('ANSWER','CLARIFY'))
        self.assertAlmostEqual(report['changes']['stress'],.5)
        self.assertAlmostEqual(report['state_minus_fresh_appraisal']['stress'],.3)
        self.assertIn('Elevated fear/stress', ' '.join(report['guidance']))
        self.assertTrue(torch.equal(current,torch.tensor([.7,.3])))
        reply=grounded_self_reply('But I have programmed you to feel emotions',report)
        self.assertIn('Yes, V14 implements',reply)
        self.assertIn('stress=0.70',reply)
        self.assertIn('not proof of subjective feeling',reply)
        self.assertIsNone(grounded_self_reply('How do you feel about this poem?',report))
        self.assertIsNone(grounded_self_reply('My friend is sad. What should I say?',report))

if __name__=='__main__': unittest.main()
