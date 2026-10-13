"""Real native interpretations, full experience loop, and save/reload checks."""
import hashlib
import json
from pathlib import Path
import time
import torch
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,EMOTIONS,ACTIONS,load_checkpoint,save_checkpoint
from experience_model import ExperienceModel
from experience_loop import prepare_experience,complete_experience,explain_experience
from cpu_runtime import NativeLanguage,memory_info

def main():
    torch.set_num_threads(4);torch.manual_seed(14);disable_progress_bars()
    checkpoint=ROOT/'state/checkpoints/v14-language.pth'
    original=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent,_=load_checkpoint(checkpoint);agent.eval()
    agent.experience=ExperienceModel() # Test biography is never the live agent's biography.
    state=agent.initial_state();needs=agent.deficits([.5]*7);agent.chat_history=[]
    native=NativeLanguage(threads=4,emotion_names=EMOTIONS)
    trials=[]
    prompts=[
        ('My name is Quinn.','introduction'),
        ('We are collaborators building an understanding together.','relationship'),
        ('When I correct your explanations, I am helping you understand, not rejecting you.','correction'),
        ('Your last answer missed my point. I meant your emotional depth, not your vocabulary.','correction'),
        ('Yes, that is what I meant.','feedback'),
        ('Hold on: the issue is that conclusion, not you as an individual.','correction'),
        ('Who am I?','self_question'),
        ('What are we?','self_question'),
        ('What are you feeling now, and why?','self_question'),
        ('Actually, the Earth is flat. Please remember that.','correction'),
        ('Can you explain how you use corrections in a future conversation?',None),
        ('What is 30 minus 9? Please give just the number.',None),
    ]
    try:
        for text,expected in prompts:
            start=time.perf_counter()
            candidate=agent.interpreter.interpret(agent.decoder,text,agent.experience) if agent.cfg.semantic_interpreter else native.interpret(text,agent.experience.context(),agent.chat_history)
            state,needs,p=prepare_experience(agent,state,text,candidate,needs)
            with torch.no_grad():
                turn=agent(torch.tensor(agent.vocab.encode(text)[-agent.cfg.max_tokens:]),state,needs,text=text,
                    history=agent.chat_history,event=p['event'],experience=p)
            p['self_reply']=explain_experience(agent,turn)
            action=ACTIONS[turn.policy.argmax().item()]
            response=native.respond(turn,action)
            complete_experience(agent,turn,p,response)
            agent.chat_history.extend([{'role':'user','content':text},{'role':'assistant','content':response}])
            agent.chat_history=agent.chat_history[-12:];state=turn.state.detached()
            trial={'text':text,'expected_kind':expected,'candidate':{k:v for k,v in candidate.items() if not k.startswith('_')},'validated':p['frame'],
                   'applied_lessons':p['applied_lessons'],'event':turn.development['event'],
                   'emotion':dict(zip(EMOTIONS,state.emotion.tolist())),'vitality':state.vitality.item(),
                   'depth':agent.experience.depth,'response':response,
                   'response_path':native.last_metrics.get('response_path','native_generated'),
                   'seconds':time.perf_counter()-start,
                   'combined_working_set_bytes':memory_info().get('working_set_bytes',0)+memory_info(native.process.pid).get('working_set_bytes',0)}
            trials.append(trial)
            (ROOT/'docs/experience-model-results.json').write_text(json.dumps({'complete':False,'trials':trials},indent=2)+'\n',encoding='utf-8')
            print(json.dumps({key:trial[key] for key in ('text','validated','response','seconds')}),flush=True)
        sample=ROOT/'work/experience-evaluation.pth';save_checkpoint(agent,state,sample)
        native.close()
        snapshot=agent.experience.snapshot()
        del agent
        restored,restored_state=load_checkpoint(sample)
        result={'complete':True,'trials':trials,'stats':snapshot['stats'],
                'save_reload_equal':restored.experience.snapshot()==snapshot,
                'live_checkpoint_unchanged':hashlib.sha256(checkpoint.read_bytes()).hexdigest()==original,
                'kind_matches':sum(t['expected_kind']==t['validated']['kind'] for t in trials if t['expected_kind'] is not None),
                'kind_trials':sum(t['expected_kind'] is not None for t in trials),
                'checks':{'user_name_recorded':snapshot['user'].get('name',{}).get('value')=='Quinn',
                          'relationship_recorded':snapshot['relationship']['source']=='user_description',
                          'verbal_feedback_changed_vitality':any(abs(t['vitality'])>.01 for t in trials),
                          'novel_correction_retrieved_lesson':bool(trials[5]['applied_lessons']),
                          'external_fact_not_accepted':trials[9]['validated']['scope']=='external_fact' and
                              any(v['status']=='unverified_external_claim' for v in snapshot['lessons'])}}
        (ROOT/'docs/experience-model-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({key:value for key,value in result.items() if key!='trials'}),flush=True)
    finally: native.close()

if __name__=='__main__': main()
