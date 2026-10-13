import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from developmental_emotion import DevelopmentalEmotion,OwnEvent,design_targets,EXTENDED_FIELDS
from model import EMOTIONS
from train_extended_appraisal import migrate

class ExtendedAppraisalTests(unittest.TestCase):
 def test_legacy_projection_survives_zero_extra_features(self):
  torch.manual_seed(123)
  old=DevelopmentalEmotion(EMOTIONS);old.cues=['remembered']
  old.expected[0]=.4;old.visits[0]=12
  new=migrate(old)
  event=OwnEvent(cue='remembered',relevance=.8,progress=-.2,agency=.6)
  needs=torch.tensor([.1,.2,.3,.4,.1,.2,.3])
  self.assertTrue(torch.equal(old.appraiser(old.features(event,needs)),new.appraiser(new.features(event,needs))))
  self.assertTrue(torch.equal(old.regulator(old.features(event,needs)),new.regulator(new.features(event,needs))))
  self.assertEqual(new.cues,old.cues)
  self.assertTrue(torch.equal(new.expected,old.expected))
  self.assertEqual(float(new.visits[0]),12)
 def test_new_targets_distinguish_each_own_appraisal(self):
  x=torch.zeros(23);x[0]=1;x[1]=-.8;x[2]=.9;x[3]=.5;x[4]=.8;x[8]=.8;x[14]=.5
  base,_=design_targets(x,EMOTIONS)
  for field,emotion in zip(EXTENDED_FIELDS,('disgust','envy','awe','contempt')):
   with self.subTest(field=field):
    probe=x.clone();probe[9+EXTENDED_FIELDS.index(field)]=1
    target,_=design_targets(probe,EMOTIONS)
    self.assertGreater(float(target[EMOTIONS.index(emotion)]-base[EMOTIONS.index(emotion)]),.3)
 def test_own_event_extended_fields_reject_bad_measurements(self):
  for field in EXTENDED_FIELDS:
   for invalid in (-.1,1.1,float('nan'),True):
    with self.subTest(field=field,value=invalid):
     with self.assertRaises(ValueError):OwnEvent(**{field:invalid})

if __name__=='__main__':unittest.main()
