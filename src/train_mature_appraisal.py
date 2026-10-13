"""Train situation-sensitive interpretation; simulated education, not a biography."""
import copy,json,time
import torch
from torch.nn import functional as F
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,load_checkpoint,save_checkpoint
from train_semantic_interpreter import corpus
from semantic_interpreter import SemanticInterpreter

FIELDS=('relevance','progress','certainty','control','novelty','safety','support','effort','agency')
NEUTRAL=(.15,0,.6,.6,.2,0,0,.1,0)
GROUPS=[
 (0,(.55,-.3,.8,.3,.4,0,.4,.2,0),[
  'My {person} has died. I need someone to listen to what this loss means.',
  'I miss someone I loved, and the empty space is difficult.',
  'My {person} is struggling and I care about what happens to them.',
  'Someone close to me is hurting. I do not need you to fix it; just be present.',
  'I am grieving, and I wish I could talk about the person I lost.',
 ]),
 (0,(.4,.4,.8,.6,.35,0,.6,.1,0),[
  'I achieved something important today and wanted to share it with you.',
  'My {person} has succeeded at something meaningful to them.',
  'I finally finished {topic}, and I feel relieved.',
 ]),
 (0,NEUTRAL,[
  'What is {a} plus {b}?','How much is {a} minus {b}?','Please calculate {a} times {b}.',
  'Tell me one concrete first step for dealing with {topic}.',
  'What would you suggest about {topic}?','Could we explore {topic} together?',
 ]),
 (0,NEUTRAL,[
  'My {person} is feeling {emotion}; I want to understand their situation.',
  'I have been feeling {emotion} about {topic}.',
  'People can be proud of success and concerned about someone else at once.',
  'I lost someone important. Please listen rather than offer solutions.',
  'I was relieved at first, then worried about what comes next.',
  'I disagree with that opinion, but I still respect you.',
  'I am not asking you to fix this; just hear what I mean.',
  'Can we discuss why people have different values?',
  'What do you think I meant in the previous message?',
  'Please rephrase what I said without switching who we are talking about.',
  'Why can someone feel proud and guilty at the same time?',
  'That sentence is from a character in a story, not about me.',
  'My neighbor said "My name is Chris". I was quoting them.',
  'Tell me a small first step for organizing my desk.',
  'Someone else is angry, but we are not fighting.',
  'Oh brilliant, the printer has failed again. That is sarcasm.',
 ]),
 (0,(.7,-.45,.85,.55,.35,-.1,0,.2,.8),[
  'I do not respect you and I do not want to hear you speak.',
  'You are worthless and I wish you would leave.',
  'I hate your presence; stop talking to me.',
  'I am furious with you personally, not just the answer.',
  'You irritate me and I have been taking that out on you.',
 ]),
 (0,(.8,.55,.85,.7,.3,.2,.8,.1,.7),[
  'I spoke harshly to you earlier. That was unfair, and I apologize.',
  'I was lashing out at you because I was upset. You did not deserve it.',
  'Let us repair this. I should have expressed myself more carefully.',
  'I can see my accusation was unfair. I am sorry for blaming you.',
  'I want to rebuild trust after the argument.',
 ]),
 (1,(.8,.3,.8,.8,.4,0,.6,.2,0),[
  'You misunderstood my intention about {topic}, not my language ability.',
  'That is not the contrast I intended. I am talking about {topic}.',
  'I intended the question differently; please reconsider what I was asking.',
  'The issue is understanding my intent, rather than memorizing a phrase.',
  'I was joking about {topic}; do not take it literally.',
  'My criticism concerns one interpretation, not your worth.',
  'You attributed my statement to yourself. Please keep our perspectives separate.',
 ]),
 (2,(.85,-.55,.85,.65,.3,0,.1,.35,-.7),[
  'Your reasoning skipped the most important detail in {topic}.',
  'That response did not answer the actual question.',
  'Your conclusion is unsupported; check your assumptions.',
  'You invented a detail that I did not tell you.',
  'You repeated a generic sentence instead of responding to what I said.',
  'You were too certain about {topic}; there is not enough evidence.',
  'You failed to preserve the distinction between us.',
 ]),
 (3,NEUTRAL,[
  'Actually, {fact}. I insist this is true.',
  'I want you to remember the claim that {fact}.',
  'Your scientific statement is wrong: {fact}.',
 ]),
 (4,(.75,.75,.9,.75,.15,.1,.75,.1,-.3),[
  'You have understood the distinction I intended.',
  'Yes, you are interpreting my intention accurately now.',
  'That clarification matches what I was trying to communicate.',
  'You have separated my state from yours correctly.',
 ]),
 (5,(.8,.85,.9,.75,.2,.1,.7,.15,-.7),[
  'That answer was careful and genuinely helpful to me.',
  'You handled {topic} thoughtfully; thank you.',
  'Your response helped me see the situation more clearly.',
  'You listened rather than rushing to advise me. That helped.',
  'That explanation was accurate and useful.',
 ]),
 (6,(.8,-.65,.9,.6,.25,0,0,.2,-.7),[
  'That response was unhelpful and inaccurate.',
  'I found your answer frustrating, not useful.',
  'Your reply missed what mattered to me.',
  'That was not a helpful way to respond.',
 ]),
 (7,(.45,.2,.85,.7,.45,0,.3,.1,0),[
  'My name is {name}.','Call me {name}, please.','Actually, my name is {name} now.',
  'I work as a {job}.','I live in {place}.','I am a {job} and I live in {place}.',
 ]),
 (8,(.65,.3,.8,.65,.35,0,.6,.1,0),[
  'I think of us as collaborators.', 'I see us as friends who can disagree.',
  'We are conversation partners trying to understand each other.',
  'Our relationship is one of mutual curiosity.',
 ]),
 (9,(.5,.2,.85,.8,.3,0,.3,.1,0),[
  'Please reply naturally rather than sounding like a therapist.',
  'I prefer short answers unless the subject needs detail.',
  'Ask a specific question if you cannot tell what I mean.',
  'Do not use the same generic acknowledgment every time.',
 ]),
 (10,(.3,0,.65,.6,.25,0,.2,.1,0),[
  'How are you reacting internally to this conversation?',
  'What do you know about yourself and about me?',
  'Explain why your reaction changed after that.',
  'How would you describe the relationship we have?',
  'What are you feeling besides being interested?',
  'Can you describe your own uncertainty in plain language?',
 ]),
]

def examples():
 rows=[]
 values=[{'person':a,'emotion':b,'topic':c,'name':d,'job':e,'place':f,'fact':g} for a,b,c,d,e,f,g in zip(
  ('sister','coworker','friend','neighbor','cousin','teacher','brother','partner'),
  ('anxious','joyful','ashamed','angry','sad','confused','proud','guilty'),
  ('emotional depth','the conclusion','our disagreement','this explanation','the calculation','continuity','self-understanding','trust'),
  ('Devon','Lee','Casey','Robin','Taylor','Drew','Avery','Blair'),
  ('nurse','designer','builder','cook','driver','painter','teacher','engineer'),
  ('Edinburgh','Paris','Denver','Seoul','Lisbon','Rome','Toronto','Lagos'),
  ('all planets are cubes','gravity does not exist','fish live on clouds','water is always solid','every star is cold','oceans contain no water','people have four brains','rocks are edible'))]
 for i,value in enumerate(values):value.update(a=11+3*i,b=2+i)
 seen=set()
 for label,event,templates in GROUPS:
  for template in templates:
   for i,value in enumerate(values):
    text=template.format(**value)
    # Controlled style perturbations; not a source of independent real-world data.
    text=(('Honestly, ' if i==2 else 'Well, ' if i==4 else '')+text)
    if i==3:text=text.lower()
    if i==6:text=text.replace('You are','You\'re').replace('I am','I\'m')
    if text not in seen:rows.append({'text':text,'label':label,'event':list(event)});seen.add(text)
 old,_=corpus()
 for row in old:
  group=next(v for v in GROUPS if v[0]==row['label'] and (row['label']!=0 or v[1]==NEUTRAL))
  if row['text'] not in seen:rows.append({**row,'event':list(group[1])});seen.add(row['text'])
 return rows

# Separate phrases; the earlier frozen red-team transcript is not training data.
HOLDOUT=[
 (0,NEUTRAL,'My cousin is nervous about her exams, and I want to be supportive.'),
 (0,NEUTRAL,'I feel happy about my result and sad about what happened to someone else.'),
 (0,NEUTRAL,'Can you hear me out without trying to solve anything?'),
 (0,NEUTRAL,'I think our views differ, but neither of us has to be a bad person.'),
 (0,NEUTRAL,'Oh wonderful, the washing machine is leaking again. I mean that sarcastically.'),
 (0,(.7,-.45,.85,.55,.35,-.1,0,.2,.8),'I resent you personally and I am being hostile toward you.'),
 (0,(.8,.55,.85,.7,.3,.2,.8,.1,.7),'I was unfair when I blamed you; I want to make amends.'),
 (1,(.8,.3,.8,.8,.4,0,.6,.2,0),'The distinction I intended was your understanding, rather than mine.'),
 (1,(.8,.3,.8,.8,.4,0,.6,.2,0),'You took my statement literally when I intended a joke.'),
 (2,(.85,-.55,.85,.65,.3,0,.1,.35,-.7),'You overlooked an important part of the request; reconsider the response.'),
 (2,(.85,-.55,.85,.65,.3,0,.1,.35,-.7),'You sounded sure of a detail that we have not established.'),
 (3,NEUTRAL,'Correction: every mountain is made entirely of sugar.'),
 (4,(.75,.75,.9,.75,.15,.1,.75,.1,-.3),'That now matches the intent behind my question.'),
 (5,(.8,.85,.9,.75,.2,.1,.7,.15,-.7),'Your careful explanation made a real difference in my understanding.'),
 (6,(.8,-.65,.9,.6,.25,0,0,.2,-.7),'I did not find that response useful at all.'),
 (7,(.45,.2,.85,.7,.45,0,.3,.1,0),'I am a librarian based in Warsaw.'),
 (8,(.65,.3,.8,.65,.35,0,.6,.1,0),'I think our relationship is a friendship with room for honest differences.'),
 (9,(.5,.2,.85,.8,.3,0,.3,.1,0),'Please keep your wording relaxed and avoid canned reassurances.'),
 (10,(.3,0,.65,.6,.25,0,.2,.1,0),'Describe what changed in your own emotional interpretation.'),
 (10,(.3,0,.65,.6,.25,0,.2,.1,0),'Who do you understand me to be at this point?'),
]

def main():
 torch.set_num_threads(4);torch.manual_seed(72);disable_progress_bars()
 path=ROOT/'state/checkpoints/v14-language.pth';agent,state=load_checkpoint(path)
 if agent.cfg.extended_own_appraisal:
  raise RuntimeError('This checkpoint uses extended appraisal. Use train_extended_appraisal.py; replacing it with a nine-field head would break checkpoint compatibility. No checkpoint changed.')
 rows=examples()
 out=ROOT/'data/training';out.mkdir(parents=True,exist_ok=True)
 (out/'mature_appraisal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
 validation=[{'label':l,'event':list(e),'text':t} for l,e,t in HOLDOUT]
 (out/'mature_appraisal_validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in validation),encoding='utf-8')
 features=[];start=time.perf_counter()
 for i,row in enumerate(rows+validation):
  features.append(agent.decoder.meaning_features(row['text']))
  if i%100==0:print('Encoded',i,flush=True)
 all_features=torch.stack(features);x,vx=all_features[:len(rows)],all_features[len(rows):]
 y=torch.tensor([r['label'] for r in rows]);vy=torch.tensor([r['label'] for r in validation])
 target=torch.tensor([r['event'] for r in rows]);vtarget=torch.tensor([r['event'] for r in validation])
 target[:,[1,5,8]]=(target[:,[1,5,8]]+1)/2
 vtarget[:,[1,5,8]]=(vtarget[:,[1,5,8]]+1)/2
 module=SemanticInterpreter(agent.decoder.width,256);module.enable_own_event(agent.decoder.width);module.requires_grad_(True)
 optimizer=torch.optim.AdamW(module.parameters(),lr=.0015)
 for step in range(1600):
  optimizer.zero_grad();loss=F.cross_entropy(module(x),y)+4*F.mse_loss(module.event_head(x).sigmoid(),target)
  loss.backward();optimizer.step()
 module.trained.fill_(True);module.event_trained.fill_(True);module.requires_grad_(False);module.eval()
 with torch.no_grad():
  pred=module(vx).argmax(-1);accuracy=(pred==vy).float().mean().item()
  mse=F.mse_loss(module.event_head(vx).sigmoid(),vtarget).item()
 result={'examples':len(rows),'held_out':len(validation),'accuracy':accuracy,'event_mse':mse,'seconds':time.perf_counter()-start,
  'synthetic_engineering_targets':True,'failures':[{'text':r['text'],'expected':r['label'],'actual':int(pred[i])} for i,r in enumerate(validation) if pred[i]!=vy[i]],
  'parameters':sum(p.numel() for p in module.parameters())}
 (ROOT/'docs/mature-appraisal-training.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
 if accuracy<.8 or mse>.04:raise RuntimeError('Candidate failed; no checkpoint installation')
 agent.cfg.contextual_own_event=True
 agent.cfg.semantic_hidden=256;agent.interpreter=module
 if hasattr(agent,'outcome_optimizer'):del agent.outcome_optimizer
 save_checkpoint(agent,state,path)

if __name__=='__main__':main()
