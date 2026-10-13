import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpu_runtime import NativeLanguage

class ProfileParserTests(unittest.TestCase):
 def client(self,result):
  client=object.__new__(NativeLanguage)
  client.request=lambda *_:{'choices':[{'message':{'content':json.dumps(result)}}]}
  return client
 def test_explicit_alias_has_exact_source(self):
  text='Use Wren when addressing me from now on.'
  result={'attribute':'name','value':'Wren','evidence':text,'confidence':.95}
  self.assertEqual(self.client(result).profile(text),result)
 def test_unfounded_or_uncertain_slots_are_rejected(self):
  text='Call me Wren.'
  for result in ([1],{'attribute':'name','value':'Tess','evidence':text,'confidence':.95},
                 {'attribute':'name','value':'Wren','evidence':'I am Wren','confidence':.95},
                 {'attribute':'name','value':'Wren','evidence':text,'confidence':.5},
                 {'attribute':'name','value':123,'evidence':text,'confidence':.95},
                 {'attribute':'name','value':'Wren','evidence':text,'confidence':True}):
   with self.subTest(result=result):self.assertEqual(self.client(result).profile(text),{})
 def test_reported_and_fictional_names_never_request_extraction(self):
  client=self.client({})
  def forbidden(*_):raise AssertionError('Parser should not be invoked')
  client.request=forbidden
  for text in ('My colleague said "Call me Wren."','Imagine a character named Wren.',
               'Pretend my name is Wren.'):
   with self.subTest(text=text):self.assertEqual(client.profile(text),{})

if __name__=='__main__':unittest.main()
