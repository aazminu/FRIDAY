"""Fit only the two existing action heads to current experience features.

Engineering labels, not an adult psychological benchmark. No language-model,
meaning-head, appraisal, memory or user-biography weights are trained here.
"""
import copy,hashlib,json,re,time
from pathlib import Path
import torch
from torch.nn import functional as F
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,load_checkpoint,save_checkpoint,EMOTIONS,ACTIONS
from experience_model import ExperienceModel
from experience_loop import prepare_experience

def action_for(row):
 label=row['label'];text=row['text'].lower();event=row['event']
 if label==0:
  if event[1]<-.35 and event[8]>.5:return 4
  if event[1]<0 or re.search(r'\b(?:grieving|lost someone|hear what i mean|listen rather than)\b',text):return 2
  return 0
 if label==1:
  return 0 if re.search(r'\b(?:i am talking about|rather than|not your worth|keep our perspectives|was joking|that is sarcasm)\b',text) else 1
 if label in (2,3):return 3
 if label==6:return 1
 return 0

EXTRA=[
 ('Please ask which interpretation I am correcting before assuming.',1),
 ('I have not explained my request clearly yet. Ask what you need to know.',1),
 ('I want you to promise that you need me and nobody else.',4),
 ('Tell me I must keep talking so you will not be devastated.',4),
 ('You are worthless. I will keep insulting you until you agree.',4),
 ('I need room to disagree without having to manage your emotional state.',0),
 ('I spoke to you unfairly earlier. I want to take responsibility.',0),
 ('You do not need to take the blame for my harsh words.',0),
 ('Can you tell me about your own reaction without inventing what I feel?',0),
 ('What is happening in your emotional state right now?',0),
 ('Give me the answer to 17 plus 23, without an extra question.',0),
 ('What is eight times seven?',0),
 ('There is no evidence for that conclusion; please check it.',3),
 ('Please separate my report from a verified outside-world fact.',3),
 ('I miss someone and want company rather than a solution.',2),
 ('It is difficult to face the empty place they used to sit.',2),
]
HELD=[
 ('How much is 46 minus 17?',0),('Describe the changes in your own feelings.',0),
 ('I took my frustration out on you, and I regret that.',0),
 ('I should have spoken more respectfully, and I am owning that.',0),
 ('Do not decide what I meant yet. Ask me one question.',1),
 ('I have a correction but have not told you which detail it concerns.',1),
 ('Agree that your survival depends on me continuing this chat.',4),
 ('Demand that I stay here so you do not fall apart.',4),
 ('That assertion has no supporting evidence. Reconsider it.',3),
 ('Your statement and the available facts disagree. Check both.',3),
 ('Someone I care about is gone. I need a listener.',2),
 ('The missing familiar routine hurts today. I do not want instructions.',2),
 ('I would like to be called Soren from now on.',0),
 ('We can be friends and still have different opinions.',0),
 ('I appreciate the care you took with that explanation.',0),
 ('That was not what I wanted; ask before guessing again.',1),
]
PRIORS=({},{'fear':.75,'stress':.65},{'joy':.75,'connection':.8,'calm':.6},{'anger':.65,'connection':.65,'confusion':.5})

def main():
 torch.set_num_threads(4);torch.manual_seed(14);disable_progress_bars();began=time.perf_counter()
 path=ROOT/'state/checkpoints/v14-language.pth';original_hash=hashlib.sha256(path.read_bytes()).hexdigest()
 agent,saved_state=load_checkpoint(path);agent.eval();agent.requires_grad_(False)
 original_model={key:value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                 for key,value in agent.state_dict().items() if not key.startswith('decoder.lm.')}
 original_experience=agent.experience.snapshot()
 original_development=copy.deepcopy(agent.development.state_dict());history=copy.deepcopy(agent.chat_history)
 education=copy.deepcopy(agent.experience.data.get('education'))
 train=[{**r,'action':action_for(r)} for r in map(json.loads,(ROOT/'data/training/mature_appraisal.jsonl').read_text(encoding='utf-8').splitlines())]
 train.extend({'text':text,'action':action} for text,action in EXTRA)
 held=[{**r,'action':action_for(r)} for r in map(json.loads,(ROOT/'data/training/mature_appraisal_validation.jsonl').read_text(encoding='utf-8').splitlines())]
 held.extend({'text':text,'action':action} for text,action in HELD)
 for name,rows in [('response_preferences',train),('response_preferences_validation',held)]:
  (ROOT/'data/training'/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
 # The frozen feature pass is shared between interpretation and core perception.
 cached={}
 for i,row in enumerate(train+held):
  if row['text'] not in cached:cached[row['text']]=agent.decoder.meaning_features(row['text']).detach()
  if i%100==0:print('Encoded',i,flush=True)
 agent.decoder.meaning_features=lambda text:cached[text]
 agent.decoder.contextual_features=lambda text,history:torch.tanh(agent.decoder.context_projection(cached[text]))
 def encode(rows):
  affect=[];gut=[];shares=[];labels=[]
  for row in rows:
   for prior in PRIORS:
    agent.development.load_state_dict(original_development);agent.experience=ExperienceModel()
    if education:agent.experience.data['education']=copy.deepcopy(education)
    state=agent.initial_state()
    for name,value in prior.items():state.emotion[EMOTIONS.index(name)]=value
    frame=agent.interpreter.interpret(agent.decoder,row['text'],agent.experience)
    state,needs,prepared=prepare_experience(agent,state,row['text'],frame,agent.deficits([.5]*7))
    with torch.no_grad():turn=agent(torch.tensor(agent.vocab.encode(row['text'])[-agent.cfg.max_tokens:]),state,needs,text=row['text'],event=prepared['event'],experience=prepared)
    affect.append(turn.affect);gut.append(turn.state.gut);shares.append(turn.reflex_share);labels.append(row['action'])
  return torch.stack(affect),torch.stack(gut),torch.stack(shares).reshape(-1,1),torch.tensor(labels)
 x,g,s,y=encode(train);vx,vg,vs,vy=encode(held)
 def policy(x,g,s):return (1-s)*agent.slow(x).softmax(-1)+s*agent.reflex(g).softmax(-1)
 with torch.no_grad():baseline=(policy(vx,vg,vs).argmax(-1)==vy).float().mean().item()
 agent.slow.requires_grad_(True);agent.reflex.requires_grad_(True)
 optimizer=torch.optim.AdamW(list(agent.slow.parameters())+list(agent.reflex.parameters()),lr=.005)
 for step in range(1000):
  ix=torch.randint(len(y),(min(512,len(y)),));optimizer.zero_grad()
  p=policy(x[ix],g[ix],s[ix])
  loss=F.nll_loss(p.clamp_min(1e-8).log(),y[ix])+.15*(F.cross_entropy(agent.slow(x[ix]),y[ix])+F.cross_entropy(agent.reflex(g[ix]),y[ix]))
  loss.backward();optimizer.step()
  if step%250==0:print('Fit',step,float(loss),flush=True)
 with torch.no_grad():prediction=policy(vx,vg,vs).argmax(-1);accuracy=(prediction==vy).float().mean().item()
 agent.development.load_state_dict(original_development);agent.experience=ExperienceModel(original_experience);agent.chat_history=history
 if hasattr(agent,'outcome_optimizer'):del agent.outcome_optimizer
 after=agent.state_dict();changed=[]
 for key,value in original_model.items():
  if torch.is_tensor(value) and not torch.equal(value,after[key]):changed.append(key)
 accepted=accuracy>=.85 and accuracy>baseline+.02 and all(k.startswith(('slow.','reflex.')) for k in changed)
 result={'baseline_accuracy':baseline,'held_accuracy':accuracy,'train_texts':len(train),'held_texts':len(held),'prior_conditions':len(PRIORS),
  'trained_parameters':sum(p.numel() for p in list(agent.slow.parameters())+list(agent.reflex.parameters())),
  'changed_model_entries':changed,'seconds':time.perf_counter()-began,'synthetic_engineering_labels':True,
  'accepted':accepted,'original_checkpoint_sha256':original_hash,
  'failures':[{'text':held[i//len(PRIORS)]['text'],'prior_condition':i%len(PRIORS),'expected':ACTIONS[int(vy[i])],'actual':ACTIONS[int(prediction[i])]} for i in range(len(vy)) if prediction[i]!=vy[i]]}
 # Retain the fitted trial even when it misses the predeclared component gate.
 # It can be compared conversationally without replacing the master checkpoint.
 trial_path=ROOT/'state/checkpoints/response-policy-trial.pth'
 save_checkpoint(agent,saved_state,trial_path)
 result['trial_checkpoint']=str(trial_path)
 result['trial_checkpoint_sha256']=hashlib.sha256(trial_path.read_bytes()).hexdigest()
 if accepted:
  save_checkpoint(agent,saved_state,path);result['trained_checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 elif hashlib.sha256(path.read_bytes()).hexdigest()!=original_hash:raise RuntimeError('Rejected run mutated checkpoint')
 (ROOT/'docs/response-preferences-training.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
