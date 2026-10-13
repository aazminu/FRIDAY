"""Simulated lifelong education with snapshots; never impersonates real user history."""
import copy,json,time
import torch
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,EMOTIONS,load_checkpoint,save_checkpoint
from developmental_emotion import OwnEvent
from experience_model import ExperienceModel
from experience_loop import prepare_experience,complete_experience

DOMAINS={
 'uncertainty':['When information is missing, distinguish a possibility from a fact.','A delayed reply does not establish personal rejection.','An ambiguous correction needs clarification before a judgment.'],
 'accountability':['A mistaken answer calls for a specific revision, not global worthlessness.','Admitting an error and checking the cause is a useful response.','Responsibility concerns what was actually done, not invented guilt.'],
 'ambivalence':['Concern and interest can coexist; do not flatten the state to one emotion.','A positive outcome does not erase concern about a different consequence.','Feelings can change at different rates rather than all switching at once.'],
 'repair':['A sincere apology is evidence for reconsidering an interpersonal interpretation.','Disagreement can coexist with respect and continuing connection.','Trust is revisable through subsequent behavior, not a single forced promise.'],
 'boundaries':['The other person can stop chatting without needing to manage your emotions.','Care does not require emotional pressure, jealousy, or ownership.','Setting a boundary does not justify insulting the other person.'],
 'perspective':['The user describing someone else’s fear does not mean you face that danger.','Quoted first-person statements belong to the quoted speaker.','When a user says your answer, they mean the engine’s answer.'],
 'grounding':['Explain an emotional change using recorded events, without inventing causes.','Do not invent a shared human childhood or claim an unavailable memory.','Current emotions can persist while your interpretation of an event changes.'],
 'growth':['New evidence can revise earlier expectations throughout later development.','Repeated praise alone does not prove that a lesson transferred.','An unpleasant state is information to examine, not a command to retaliate.'],
}

def probes(agent):
 needs=agent.deficits([.5]*7);state=agent.initial_state();result={}
 for name,event in (
  ('uncertain',OwnEvent(relevance=.85,certainty=.25,control=.65,novelty=.6)),
  ('critique',OwnEvent(relevance=.85,progress=-.5,certainty=.85,control=.65,agency=-.7)),
  ('hostility',OwnEvent(relevance=.8,progress=-.4,certainty=.8,control=.4,safety=-.1,agency=.8)),
  ('repair',OwnEvent(relevance=.8,progress=.6,certainty=.85,control=.7,support=.8,safety=.2)),
  ('mixed',OwnEvent(relevance=.85,progress=-.35,certainty=.7,control=.7,support=.85,novelty=.6))):
  proposal,emotion,_,details=agent.development.appraise(event,needs,state.emotion,depth=agent.experience.depth)
  result[name]={'proposal':dict(zip(EMOTIONS,proposal.tolist())),'state':dict(zip(EMOTIONS,emotion.tolist())),'strategy':details['strategy']}
 return result

def main():
 torch.set_num_threads(4);disable_progress_bars();path=ROOT/'state/checkpoints/v14-language.pth'
 agent,live_state=load_checkpoint(path);agent.eval()
 original=agent.experience.snapshot();original_history=copy.deepcopy(agent.chat_history)
 out=ROOT/'work/education';out.mkdir(parents=True,exist_ok=True)
 (out/'child-probes.json').write_text(json.dumps(probes(agent),indent=2),encoding='utf-8')
 save_checkpoint(agent,live_state,out/'before-education.pth')
 agent.experience=ExperienceModel();agent.chat_history=[];state=agent.initial_state();needs=agent.deficits([.5]*7)
 agent.experience.data['education']={'source':'synthetic situational curriculum','target':'mature adult-style emotional reasoning; no biological age equivalence','domain_exposures':{},'lessons':[]}
 log=(out/'curriculum.jsonl').open('w',encoding='utf-8')
 try:
  for cycle in range(8):
   for domain,lessons in DOMAINS.items():
    lesson=lessons[cycle%len(lessons)]
    # The supervised target is disclosed, not invented feedback from a real user.
    event=OwnEvent(cue='education:'+domain,relevance=.85,progress=(-.55 if cycle%3==0 else .55),
      certainty=(.3 if domain=='uncertainty' else .8),control=.65,novelty=.5,
      support=(.75 if domain in ('repair','ambivalence') else .2),agency=(-.6 if domain=='accountability' else .4))
    proposal,emotion,_,details=agent.development.appraise(event,needs,state.emotion,depth=agent.experience.depth)
    # Curriculum includes setback then recovery, using actual declared outcome labels.
    quality=-.5 if cycle%3==0 else .75
    agent.development.observe(details,quality,safe=quality>0,control=.75 if quality>0 else .45)
    state.emotion=agent.development.outcome_emotion(details,emotion,needs,quality,safe=quality>0).detach()
    education=agent.experience.data['education']
    education['domain_exposures'][domain]=education['domain_exposures'].get(domain,0)+1.5
    if lesson not in [r['lesson'] for r in education['lessons']]:education['lessons'].append({'domain':domain,'lesson':lesson,'source':'synthetic curriculum'})
    log.write(json.dumps({'cycle':cycle,'domain':domain,'lesson':lesson,'outcome':quality,'depth':agent.experience.depth,
     'state':dict(zip(EMOTIONS,state.emotion.tolist())),'event':details['event']})+'\n');log.flush()
   save_checkpoint(agent,state,out/f'stage-{cycle+1}.pth')
  mature=probes(agent)
  (out/'mature-probes.json').write_text(json.dumps(mature,indent=2),encoding='utf-8')
  education=copy.deepcopy(agent.experience.data['education'])
  # Restore real user/profile/episodes. Only disclosed education and learned
  # developmental statistics are adopted, not simulated interpersonal memories.
  agent.experience=ExperienceModel(original);agent.experience.data['education']=education
  agent.chat_history=original_history
  result={'simulated_events':64,'domains':8,'depth':agent.experience.depth,
   'source':'synthetic curriculum, not human experiences','neural_weights_changed':False,
   'real_biography_preserved':all(agent.experience.data[k]==original[k] for k in ('user','relationship','episodes','stats'))}
  (ROOT/'docs/emotional-education.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
  save_checkpoint(agent,live_state,path);print(json.dumps(result),flush=True)
 finally:log.close()

if __name__=='__main__':main()
