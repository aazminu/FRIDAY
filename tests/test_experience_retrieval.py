import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experience_model import ExperienceModel

class RetrievalTests(unittest.TestCase):
 def episode(self,i,text,delta=.1):
  return {'id':i,'input':text,'response':'I understood that detail.',
   'interpretation':{'kind':'ordinary','scope':'none','subject':'user'},
   'before':{'sadness':0},'after':{'sadness':delta}}
 def test_specific_older_topic_beats_new_irrelevant_topics(self):
  memory=ExperienceModel()
  memory.data['episodes']=[self.episode(1,'My holiday in Spain was restful.'),
       self.episode(2,'I repaired my car.'),self.episode(3,'My dog died yesterday.')]
  rows=memory.retrieve('What did I tell you about my holiday in Spain?')
  self.assertEqual(rows[0]['user_report'],'My holiday in Spain was restful.')
  self.assertEqual(rows[0]['source_type'],'recorded_user_message')
  self.assertTrue(rows[0]['agent_reply_is_not_verified_world_fact'])
 def test_dog_loss_is_not_a_cat_memory(self):
  memory=ExperienceModel();memory.data['episodes']=[self.episode(1,'My dog died yesterday.')]
  self.assertEqual(memory.retrieve('What did I say about losing my cat?'),[])
 def test_emotion_prioritizes_relevant_memories_without_creating_them(self):
  memory=ExperienceModel();memory.data['episodes']=[self.episode(1,'The repair was a relief.',.5),
                                                self.episode(2,'The repair was a relief.',.01)]
  self.assertEqual(memory.retrieve('That repair was a relief.')[0]['emotional_changes'][0]['direction'],'increased')
  self.assertEqual(memory.retrieve('Did we attend school together?'),[])
 def test_old_snapshot_without_timestamp_stays_compatible(self):
  memory=ExperienceModel();memory.data['episodes']=[self.episode(1,'My cat was quiet today.')]
  restored=ExperienceModel(memory.snapshot())
  self.assertEqual(restored.retrieve('My cat')[0]['time_utc'],None)
 def test_external_claim_does_not_enter_world_fact_retrieval(self):
  memory=ExperienceModel();episode=self.episode(1,'The Moon is made of cheese.')
  episode['interpretation']['scope']='external_fact';memory.data['episodes']=[episode]
  self.assertEqual(memory.retrieve('What is the Moon made of?'),[])
  memory.data['last_significant']={'kind':'correction','scope':'external_fact','summary':'The Moon is made of cheese.',
    'input':'The Moon is made of cheese.'}
  self.assertEqual(memory.context()['last_significant']['source_status'],'unverified_external_claim')

if __name__=='__main__':unittest.main()
