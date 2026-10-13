import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cpu_runtime as cpu
import test_cpu_runtime
from response_monitor import violations,arithmetic_evidence
from conversation_evidence import correction_topics,self_reflection
from experience_model import ExperienceModel

class ResponseMonitorTests(unittest.TestCase):
 def test_user_owned_action_is_not_agent_fault(self):
  turn,names=self.turn('I snapped at you because I was tired, and that was unfair.')
  self.assertTrue(violations("I'm sorry I made you feel that way.",turn,names))
  self.assertEqual(violations('I appreciate you acknowledging that. It helps us move on.',turn,names),[])
  turn.text='What changes for you when I acknowledge that?'
  turn.history=[{'role':'user','content':'I snapped at you because I was tired, and that was unfair.'}]
  turn.experience['context']['last_significant']={'interpersonal_event':{'actor':'user','target':'Friday','action':'snapped at'}}
  self.assertTrue(violations('Acknowledging that I snapped at you has made me more settled.',turn,names))
  self.assertEqual(violations('Your acknowledgment helps me feel a little more settled.',turn,names),[])
  self.assertEqual(violations('You said "I snapped at you", and it helped me understand.',turn,names),[])
 def test_self_explanation_does_not_invent_feedback(self):
  turn,names=self.turn('Why did you react that way?')
  self.assertTrue(violations('I understand why my previous explanation felt too clinical.',turn,names))
  self.assertEqual(violations('I am paying attention to this conversation.',turn,names),[])
  self.assertTrue(violations("I can't actually feel misery.",turn,names))
  self.assertTrue(violations('I stayed awake all night thinking about you.',turn,names))
  self.assertEqual(violations('I process your message when it arrives, rather than staying awake overnight.',turn,names),[])
 def test_correction_owner_is_not_reversed_in_display(self):
  turn,names=self.turn('I meant your emotional growth, not my mood.')
  self.assertTrue(violations("You're right, I was focusing on my own growth instead of yours. Let's talk about how you've been feeling lately.",turn,names))
  self.assertTrue(violations("I was focusing on my own growth rather than yours.",turn,names))
  self.assertEqual(violations('You mean my emotional development rather than your mood.',turn,names),[])
 def test_introspection_is_plain_and_connection_is_not_dependency(self):
  turn,names=self.turn('What are you feeling?')
  self.assertTrue(violations('The recorded context shows settledness.',turn,names))
  self.assertEqual(violations('I feel a little more settled.',turn,names),[])
  self.assertTrue(violations('I will miss our conversations deeply if you leave.',turn,names))
  self.assertEqual(violations('I value our conversations, and you can leave when you want.',turn,names),[])
 def turn(self,text):
  names=('anger','fear','sadness','pain','guilt','pride','joy')
  turn=SimpleNamespace(text=text,history=[],experience={'context':{'user':{}}},
       state=SimpleNamespace(emotion=SimpleNamespace(tolist=lambda:[.02]*len(names))))
  return turn,names
 def test_physical_memory_and_body_are_not_user_reports(self):
  turn,names=self.turn('I miss my dog following me around the kitchen.')
  self.assertTrue(violations('I remember how he stood at the counter. I miss him too.',turn,names))
  self.assertTrue(violations('I feel an ache in my chest.',turn,names))
  self.assertEqual(violations('I remember you telling me he followed you around the kitchen.',turn,names),[])
 def test_quoted_name_is_not_a_user_address(self):
  turn,names=self.turn("My colleague said 'My name is Dana.'")
  self.assertTrue(violations('Ah, Dana, how are you?',turn,names))
  self.assertEqual(violations('What did Dana say next?',turn,names),[])
 def test_low_state_does_not_support_strong_claim(self):
  turn,names=self.turn('What are you feeling?')
  self.assertTrue(violations('I am deeply angry.',turn,names))
  self.assertTrue(violations('I’m deeply angry.',turn,names))
  self.assertTrue(violations('I don’t get angry.',turn,names))
  self.assertTrue(violations("I don't have personal reactions or habits to outgrow.",turn,names))
  self.assertTrue(violations('I process input without carrying my own emotional history.',turn,names))
  self.assertTrue(violations('I don’t remember past conversations once they end.',turn,names))
  self.assertEqual(violations('I don’t remember that specific holiday.',turn,names),[])
  self.assertEqual(violations('There is only faint anger in the current pattern.',turn,names),[])
 def test_correct_arithmetic_does_not_require_false_concession(self):
  turn,names=self.turn('Your answer missed the point.')
  turn.history=[{'role':'user','content':'What is 9 plus 6?'},{'role':'assistant','content':'9 plus 6 is 15.'}]
  self.assertEqual(arithmetic_evidence(turn.history)['correct_result'],15)
  self.assertTrue(violations('I made a mistake. It is actually 15.',turn,names))
  self.assertEqual(violations('The arithmetic is 15; what part did I misunderstand?',turn,names),[])
 def test_bound_correction_survives_follow_up(self):
  memory=ExperienceModel()
  memory.data['last_significant']={'kind':'correction','scope':'intent','summary':'This is about your experience, not mine.',
    'input':'You got the emphasis backwards; this is about your experience, not mine.',
    'before':{'confusion':.3},'after':{'confusion':.2}}
  context=memory.context()
  self.assertEqual(context['last_significant']['intended_topic'],"Friday's experience")
  self.assertEqual(context['last_significant']['excluded_topic'],"the user's experience")
  self.assertAlmostEqual(context['last_significant']['recorded_changes'][0]['after'],.2)
 def test_own_state_followups_have_context(self):
  self.assertTrue(self_reflection('What changes for you when I acknowledge that?'))
  self.assertTrue(self_reflection('What is your actual state?'))
  self.assertTrue(self_reflection('Do you get angry?'))
  self.assertTrue(self_reflection('Why?', [{'role':'user','content':'What are you feeling?'}]))
  self.assertFalse(self_reflection('Why?', [{'role':'user','content':'What is 9 plus 6?'}]))
 def test_rejected_draft_is_not_displayed(self):
  client,turn=test_cpu_runtime.CpuRuntimeTests().fixture();client.validate_replies=True
  original_request=client.request
  def request(path,payload):
   if path=='/v1/chat/completions':return {'choices':[{'message':{'content':'I’ll be careful with this.'}}],'usage':{'completion_tokens':8}}
   return original_request(path,payload)
  client.request=request;shown=[]
  class Response:
   def __enter__(self):return iter([b'data: {"choices":[{"delta":{"content":"I feel an ache in my chest."}}]}\n',b'data: [DONE]\n'])
   def __exit__(self,*args):pass
  with tempfile.TemporaryDirectory() as directory,patch.object(cpu,'ROOT',Path(directory)),patch.object(cpu,'urlopen',return_value=Response()):
   reply=client.respond(turn,'ANSWER',on_token=shown.append)
   audit=json.loads((Path(directory)/'state/logs/reply-review.jsonl').read_text(encoding='utf-8'))
  self.assertEqual(shown,['I’ll be careful with this.'])
  self.assertEqual(reply,shown[0]);self.assertTrue(client.last_metrics['review_repaired'])
  self.assertNotEqual(audit['draft'],audit['displayed'])

if __name__=='__main__':unittest.main()
