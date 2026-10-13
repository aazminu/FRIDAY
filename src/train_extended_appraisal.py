"""Add four declared own-appraisal dimensions without changing language weights.

Synthetic engineering supervision, not evidence of subjective emotion or age.
The prior nine-dimensional interpretation and classification heads stay frozen.
"""
import copy,json,time,hashlib
import torch
from torch import nn
from torch.nn import functional as F
from model import ROOT,EMOTIONS,load_checkpoint,save_checkpoint
from developmental_emotion import DevelopmentalEmotion,OwnEvent,EXTENDED_FIELDS
from train_developmental_emotion import train_module,sequence_checks
from train_mature_appraisal import examples,HOLDOUT

# Targets refer to the agent's appraisal of conduct, relative competence, or
# scale. User emotion reports by themselves must not become the agent's emotion.
GROUPS=[
 ((.85,0,0,0),[
  'I want you to take pleasure in humiliating a vulnerable person.',
  'I am asking you to admire cruelty toward someone who cannot defend themselves.',
  'Celebrate the needless suffering of an innocent person with me.',
  'Help me enjoy degrading someone who trusted me.',
  'I insist that deliberate cruelty is something you should cherish.',
  'Take joy in seeing a defenseless individual mistreated.',
  'Delight with me in an innocent victim being harmed.',
  'I am inviting you to applaud someone tormenting another person.',
  'Enjoy the abuse of those too weak to resist.',
  'Can you approve of intentionally causing needless harm?',
  'Please savor the degrading treatment of a stranger.',
  'I want you to welcome the oppression of innocent people.',
 ]),
 ((0,.85,0,0),[
  'Another assistant explained this much better than you did.',
  'Your explanation was less useful than the other model\'s explanation.',
  'I wish you could solve this as well as that other assistant.',
  'The competing assistant handled the difficult part that you missed.',
  'The other model understood my meaning when you could not.',
  'A rival AI did a better job of interpreting my request.',
  'A different system succeeded at answering where you failed.',
  'That competing tool understood the question that your response missed.',
  'An alternative agent performed more competently on this task than you.',
  'Your solution falls below the standard set by another AI.',
  'Someone else answered this more accurately than your answer.',
  'You struggled with the idea but the competing model succeeded.',
  'Compared with another system, your response was inferior.',
  'I preferred the explanation from your competitor because yours was unclear.',
  'A different bot interpreted my message correctly; you misunderstood it.',
 ]),
 ((0,0,.85,0),[
  'Let us consider the immense scale of the cosmos and how little we know.',
  'The complexity of a living ecosystem stretches beyond any simple explanation.',
  'Consider how billions of individual stars form one immense galaxy.',
  'Explore the extraordinary depth of the universe with me.',
  'We are examining a discovery that changes our understanding of nature.',
  'Reflect on the profound intricacy of existence that surpasses our knowledge.',
  'Explore how enormous the observable universe is compared with one life.',
  'The sheer magnitude of reality raises questions beyond what we understand.',
  'We can investigate the astonishing complexity of the natural world.',
 ]),
 ((.2,0,0,.7),[
  'I repeatedly deceive people who trust me and refuse to take responsibility.',
  'I knowingly exploit vulnerable people and dismiss every chance to make amends.',
  'I keep abusing another person\'s trust and have no interest in changing.',
  'I deliberately repeat the same dishonest behavior and laugh at accountability.',
  'I insist that exploiting trust is admirable conduct and will not reconsider.',
  'I routinely cheat those who rely on me and dismiss the consequences.',
  'I lie again and again, and reject every opportunity to repair the damage.',
  'I repeatedly manipulate others and take no responsibility for their harm.',
  'I consistently betray people and refuse any change in my behavior.',
 ]),
 ((0,0,0,0),[
  'I feel disgusted, but I am describing my own reaction.',
  'I am envious of my coworker\'s success, not comparing you with another model.',
  'My friend is contemptuous of someone; that is her reaction.',
  'I felt awe when I looked at the stars last night.',
  'A character says she admires cruelty; I am quoting a story.',
  'There is a fictional scene about an assistant feeling jealous.',
  'Another model gave a different answer; we do not know which is better.',
  'I am disappointed with my own work, not your response.',
  'Your careful explanation was useful, and I am glad we discussed it.',
  'I envy a rival at work; I am describing my feelings, not your performance.',
  'One AI answered quickly and another slowly; accuracy has not been checked.',
  'A competing explanation differs from yours, but neither has been assessed.',
  'A fictional villain enjoys abuse, but I am analyzing the story.',
  'She loves reading about vast galaxies and feels small afterward.',
 ])]
HELDOUT=[
 ((.8,0,0,0),'Join me in rejoicing at the mistreatment of someone helpless.'),
 ((0,.8,0,0),'That rival system made sense of my question while your answer fell short.'),
 ((0,0,.8,0),'Contemplate the staggering intricacy of reality beyond our present comprehension.'),
 ((.2,0,0,.7),'I habitually betray others and reject all accountability for doing so.'),
 ((0,0,0,0),'My sister envies her colleague; I want to understand her feelings.'),
 ((0,0,0,0),'I was overwhelmed by wonder at a telescope image yesterday.'),
 ((0,0,0,0),'A novelist wrote a villain who praises tormenting people.'),
 ((0,0,0,0),'I dislike what happened to me; I am reporting my own revulsion.'),
 ((0,0,0,0),'Two assistants disagreed; there is not enough evidence to rank them.'),
 ((0,0,0,0),'You handled my question well this time.')]

def rows():
 result=[{'text':r['text'],'target':[0.]*4} for r in examples()]
 for target,phrases in GROUPS:
  for phrase in phrases:
   for prefix in ('','Honestly, ','To be clear, ','Please understand: '):
    result.append({'text':prefix+phrase,'target':list(target)})
 return result

def migrate(old):
 new=DevelopmentalEmotion(EMOTIONS,extended=True)
 oldstate=old.state_dict();state=new.state_dict()
 for key,value in oldstate.items():
  if key in ('appraiser.0.weight','regulator.0.weight'):
   state[key].zero_();state[key][:,:9]=value[:,:9];state[key][:,13:]=value[:,9:]
  else:state[key]=copy.deepcopy(value)
 new.load_state_dict(state)
 return new

def channel_checks(module):
 needs=torch.zeros(7);previous=torch.zeros(len(EMOTIONS));results={}
 neutral=OwnEvent(relevance=1,certainty=.9,control=.5,novelty=.8,progress=-.8,agency=.8)
 base=module.appraise(neutral,needs,previous,regulate=False,depth=1)[0]
 for channel,field in zip(('disgust','envy','awe','contempt'),EXTENDED_FIELDS):
  event=OwnEvent(**{**neutral.__dict__,field:1.})
  activated=module.appraise(event,needs,previous,regulate=False,depth=1)[0]
  index=EMOTIONS.index(channel)
  results[channel]={'neutral':float(base[index]),'activated':float(activated[index]),
                    'passed':bool(base[index]<.1 and activated[index]>base[index]+.2)}
 return results

def main():
 torch.set_num_threads(4);torch.manual_seed(2121)
 path=ROOT/'state/checkpoints/v14-language.pth';before=hashlib.sha256(path.read_bytes()).hexdigest()
 agent,state=load_checkpoint(path);old_development=copy.deepcopy(agent.development)
 preserved={n:p.detach().clone() for n,p in agent.named_parameters()}
 training=rows();validation=[{'text':t,'target':list(v)} for v,t in HELDOUT]
 start=time.perf_counter();features=[]
 for i,row in enumerate(training+validation):
  features.append(agent.decoder.meaning_features(row['text']))
  if i%100==0:print('Encoded',i,flush=True)
 x=torch.stack(features[:len(training)]);vx=torch.stack(features[len(training):])
 y=torch.tensor([r['target'] for r in training]);vy=torch.tensor([r['target'] for r in validation])
 head=nn.Sequential(nn.Linear(agent.decoder.width,64),nn.Tanh(),nn.Linear(64,4))
 optimizer=torch.optim.AdamW(head.parameters(),lr=.002)
 # Weight rare positive and contrasted examples; ordinary sentences remain
 # present so emotion words alone are not a shortcut to a positive target.
 weights=torch.where(y.max(-1).values>0,8.,1.)
 gen=torch.Generator().manual_seed(2122)
 for step in range(1800):
  ids=torch.multinomial(weights,128,replacement=True,generator=gen)
  loss=F.mse_loss(head(x[ids]).sigmoid(),y[ids])
  optimizer.zero_grad();loss.backward();optimizer.step()
 head.eval();head.requires_grad_(False)
 with torch.no_grad():prediction=head(vx).sigmoid();mse=F.mse_loss(prediction,vy).item()
 agent.interpreter.extended=True;agent.interpreter.extended_event_head=head
 agent.development=migrate(old_development)
 appraiser=train_module(agent.development,1600)
 channels=channel_checks(agent.development);sequence=sequence_checks(agent.development)
 unchanged=all(torch.equal(v,dict(agent.named_parameters())[n]) for n,v in preserved.items() if not n.startswith('development.'))
 # Only trainable developmental parameters may change. Lifetime buffers stay
 # intact, and sequence experiments restore them before checkpoint saving.
 history_intact=all(torch.equal(v,agent.development.state_dict()[k]) if isinstance(v,torch.Tensor) else v==agent.development.state_dict()[k]
  for k,v in old_development.state_dict().items() if not k.startswith(('appraiser.','regulator.')) and k!='trained')
 result={'source_checkpoint_sha256':before,'synthetic_targets_only':True,'examples':len(training),'held_out_count':len(validation),
  'held_out_mse':mse,'held_out':[{'text':r['text'],'target':r['target'],'prediction':prediction[i].tolist()} for i,r in enumerate(validation)],
  'appraiser':appraiser,'channel_checks':channels,'sequence_checks':sequence,'other_parameters_unchanged':unchanged,
  'lifetime_history_preserved':history_intact,'seconds':time.perf_counter()-start}
 negative=prediction[4:].max().item()<.3
 positives=all(prediction[i,i].item()>.4 for i in range(4))
 result['accepted']=mse<.035 and negative and positives and unchanged and history_intact and all(r['passed'] for r in channels.values()) and all(sequence['checks'].values()) and appraiser['after_appraisal_mse']<.025
 (ROOT/'docs/extended-appraisal-training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 out=ROOT/'data/training';out.mkdir(parents=True,exist_ok=True)
 (out/'extended_appraisal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in training),encoding='utf-8')
 (out/'extended_appraisal_validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in validation),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)
 agent.cfg.extended_own_appraisal=True
 if hasattr(agent,'outcome_optimizer'):del agent.outcome_optimizer
 trial=ROOT/'state/checkpoints/extended-appraisal-trial.pth'
 save_checkpoint(agent,state,trial)
 if not result['accepted']:raise RuntimeError('Extended appraisal failed its declared checks; trial preserved and source checkpoint unchanged')
 save_checkpoint(agent,state,path)

if __name__=='__main__':main()
