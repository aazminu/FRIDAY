import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from conversation_evidence import belief_revision,interpersonal_report
from response_monitor import trim_routine_followup

class ConversationOwnershipTests(unittest.TestCase):
 def test_action_actor_and_target_are_distinct(self):
  report=interpersonal_report('I snapped at you because I was tired, and that was unfair.')
  self.assertEqual(report['actor'],'user');self.assertEqual(report['target'],'Friday')
  self.assertFalse(interpersonal_report('I did not snap at you.'))
  self.assertFalse(interpersonal_report('My friend said "I snapped at you."'))
  self.assertFalse(interpersonal_report('Imagine I snapped at you.'))
  self.assertFalse(interpersonal_report('If I snapped at you, would you be upset?'))
  self.assertFalse(interpersonal_report('A line of dialogue: "I snapped at you."'))
 def test_belief_revision_belongs_to_speaker(self):
  result=belief_revision('I thought you were ignoring me, but I now understand the delay was technical.')
  self.assertEqual(result['holder'],'user')
  self.assertIn('Friday',result['previous_reported_belief'])
  self.assertIn('technical',result['revised_reported_belief'])
  self.assertFalse(belief_revision('My friend said "I thought you were ignoring me, but I now understand the delay."'))
 def test_question_trim_preserves_substantive_questions(self):
  self.assertEqual(trim_routine_followup('That sounds frustrating. What’s on your mind?', 'My laptop crashed.'),'That sounds frustrating.')
  self.assertEqual(trim_routine_followup('What’s on your mind?', 'Hello.'),'What’s on your mind?')
  self.assertEqual(trim_routine_followup('I understand. Which file is missing?', 'My file vanished.'),'I understand. Which file is missing?')
  self.assertEqual(trim_routine_followup('I understand. How can I help?', 'Ask me questions.'),'I understand. How can I help?')

if __name__=='__main__': unittest.main()
