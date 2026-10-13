"""Benchmark the entire actual experience/appraisal/review path, in a test biography."""
import argparse,copy,hashlib,json,sys,time,statistics,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--threads',type=int,default=4)
p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
a.output=a.output or a.project/'docs/cpu-performance-results.json'
sys.stdout.reconfigure(encoding='utf-8',errors='replace');sys.path.insert(0,str(a.project/'src'))
import torch
from huggingface_hub.utils import disable_progress_bars
from model import load_checkpoint,EMOTIONS,ACTIONS
from experience_model import ExperienceModel
from experience_loop import prepare_experience,complete_experience,explain_experience
from cpu_runtime import NativeLanguage,memory_info

def main():
 torch.set_num_threads(4);torch.manual_seed(14);disable_progress_bars()
 checkpoint=a.checkpoint or a.project/'state/checkpoints/v14-language.pth';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 start=time.perf_counter();agent,_=load_checkpoint(checkpoint);agent.eval();core_load=time.perf_counter()-start
 education=copy.deepcopy(agent.experience.data.get('education'));agent.experience=ExperienceModel()
 if education:agent.experience.data['education']=education
 agent.chat_history=[];state=agent.initial_state();needs=agent.deficits([.5]*7)
 start=time.perf_counter();native=NativeLanguage(threads=a.threads,emotion_names=EMOTIONS);language_load=time.perf_counter()-start
 samples=[];stop=threading.Event();rows=[]
 def sample():
  while not stop.is_set():
   python=memory_info();language=memory_info(native.process.pid)
   samples.append({'seconds':time.perf_counter(),'combined':python['working_set_bytes']+language['working_set_bytes'],
                   'available':python['available_ram_bytes']});stop.wait(.2)
 monitor=threading.Thread(target=sample,daemon=True);monitor.start()
 try:
  for text in ['What is 29 minus 8? Reply briefly.',
    'The empty food bowl is what keeps getting to me after losing my cat. I want conversation, not a checklist.',
    'Your reply understood the detail I meant. Thank you.',
    'What changed for you after that feedback? Explain it in words.']:
   began=time.perf_counter();frame=agent.interpreter.interpret(agent.decoder,text,agent.experience)
   state,needs,prepared=prepare_experience(agent,state,text,frame,needs)
   with torch.no_grad():turn=agent(torch.tensor(agent.vocab.encode(text)[-agent.cfg.max_tokens:]),state,needs,text=text,history=agent.chat_history,event=prepared['event'],experience=prepared)
   prepared['self_reply']=explain_experience(agent,turn);appraisal=time.perf_counter()-began
   reply=native.respond(turn,ACTIONS[turn.policy.argmax()]);complete_experience(agent,turn,prepared,reply)
   agent.chat_history.extend([{'role':'user','content':text},{'role':'assistant','content':reply}]);state=turn.state.detached()
   row={'text':text,'reply':reply,'appraisal_seconds':appraisal,'full_seconds':time.perf_counter()-began,
        'first_display_seconds':appraisal+(native.last_metrics.get('first_display_seconds') or native.last_metrics.get('first_text_seconds') or 0),
        'metrics':copy.deepcopy(native.last_metrics)}
   rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 finally:stop.set();monitor.join();native.close()
 result={'native_threads':a.threads,'core_threads':4,'core_load_seconds':core_load,'language_load_seconds':language_load,
   'model':str(native.model),'context':native.context,'median_full_seconds':statistics.median(r['full_seconds'] for r in rows),
   'median_first_display_seconds':statistics.median(r['first_display_seconds'] for r in rows),
   'sampled_simultaneous_peak_gib':max(s['combined'] for s in samples)/1024**3,
   'minimum_available_gib':min(s['available'] for s in samples)/1024**3,
   'checkpoint_unchanged':hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest,'trials':rows,'samples':samples}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('trials','samples')}),flush=True)

if __name__=='__main__':main()
