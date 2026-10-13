"""Train own-event adapters, run sequence checks, and install on the saved V14 core."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,EMOTIONS,load_checkpoint,save_checkpoint
from developmental_emotion import DevelopmentalEmotion,OwnEvent,design_targets

def curriculum(seed,count,extended=False):
    gen=torch.Generator().manual_seed(seed)
    offset=13 if extended else 9
    x=torch.rand(count,23 if extended else 19,generator=gen)
    if extended:
        x[:,9:13]*=(torch.rand(count,4,generator=gen)<.4)
        x[:count//4,9:13]=0
    x[:,[1,5,8,offset]]=2*x[:,[1,5,8,offset]]-1
    x[:,offset+2]=x[:,1]-x[:,offset];x[:,offset+3:]*=.4
    labels=[design_targets(row,EMOTIONS) for row in x]
    return x,torch.stack([v[0] for v in labels]),torch.tensor([v[1] for v in labels])

def train_module(module,steps=700):
    x,y,strategy=curriculum(14,1200,module.extended);vx,vy,vs=curriculum(901,300,module.extended)
    module.requires_grad_(True)
    optimizer=torch.optim.Adam(module.parameters(),lr=.003)
    before=F.mse_loss(module.appraiser(vx).sigmoid(),vy).item()
    gen=torch.Generator().manual_seed(141)
    start=time.perf_counter()
    for _ in range(steps):
        indices=torch.randint(len(x),(128,),generator=gen)
        loss=F.mse_loss(module.appraiser(x[indices]).sigmoid(),y[indices])+ .15*F.cross_entropy(module.regulator(x[indices]),strategy[indices])
        optimizer.zero_grad();loss.backward();optimizer.step()
    module.trained.fill_(True);module.requires_grad_(False);module.eval()
    with torch.no_grad():
        mse=F.mse_loss(module.appraiser(vx).sigmoid(),vy).item()
        accuracy=(module.regulator(vx).argmax(-1)==vs).float().mean().item()
    return {'train_examples':len(x),'held_out_examples':len(vx),'steps':steps,
            'before_appraisal_mse':before,'after_appraisal_mse':mse,'regulation_accuracy':accuracy,
            'training_seconds':time.perf_counter()-start}

def sequence_checks(module):
    needs=torch.zeros(7);initial=torch.zeros(19);fear=EMOTIONS.index('fear')
    probe=OwnEvent(cue='uncertain-task',relevance=1,certainty=.4,control=.6,novelty=.5)
    original={k:(v.clone() if isinstance(v,torch.Tensor) else {'cues':list(v['cues'])}) for k,v in module.state_dict().items()}
    def run(event,state=initial,regulate=True): return module.appraise(event,needs,state,regulate=regulate)
    fresh=run(probe)[0][fear].item()
    for _ in range(20): module.observe(run(probe)[3],-1,False,.2)
    adverse=run(probe)[0][fear].item()
    persistence_after_adversity=module.persistence[fear].item()
    for _ in range(40): module.observe(run(probe)[3],1,True,.9)
    recovered=run(probe)[0][fear].item()
    persistence_after_recovery=module.persistence[fear].item()
    threat=OwnEvent(cue='hazard',relevance=1,safety=-1,certainty=.95,control=.1,progress=-.6)
    state=run(threat)[1].detach();threat_level=state[fear].item()
    state=run(OwnEvent(),state)[1].detach();first_neutral=state[fear].item()
    for _ in range(35): state=run(OwnEvent(),state)[1].detach()
    late_neutral=state[fear].item()
    # A learned check raises deliberation; reappraisal requires actual safe evidence.
    check=run(OwnEvent(relevance=1,safety=-.8,certainty=.2,control=.1))[3]
    no_reg=run(probe,regulate=False);reg=run(probe)
    evidence=OwnEvent(cue='new-evidence',relevance=1,certainty=.8,control=.9,safety=.2)
    for _ in range(20): module.observe(run(evidence)[3],-1,False,.3)
    evidence_reg=run(evidence);evidence_unreg=run(evidence,regulate=False)
    result={'fear_before_experience':fresh,'fear_after_adverse_experience':adverse,
            'fear_after_recovery_experience':recovered,'persistence_after_adversity':persistence_after_adversity,
            'persistence_after_recovery':persistence_after_recovery,'threat_fear':threat_level,
            'fear_after_one_neutral_turn':first_neutral,'fear_after_36_neutral_turns':late_neutral,
            'uncertain_threat_strategy':check['strategy'],'uncertain_threat_deliberation':check['deliberation'],
            'recovery_strategy':reg[3]['strategy'],'regulated_fear':reg[0][fear].item(),
            'unregulated_fear':no_reg[0][fear].item(),
            'evidence_strategy':evidence_reg[3]['strategy'],
            'fear_after_evidence_reappraisal':evidence_reg[0][fear].item(),
            'fear_without_evidence_reappraisal':evidence_unreg[0][fear].item(),
            'checks':{'adversity_changes_same_event':adverse>fresh+.02,
                      'later_success_changes_it_again':recovered<adverse-.02,
                      'recovery_rate_changes_with_experience':persistence_after_recovery<persistence_after_adversity,
                      'emotion_persists_then_recovers':threat_level>first_neutral>late_neutral,
                      'uncertain_threat_prompts_check':check['strategy']=='CHECK_EVIDENCE' and check['deliberation']>=.6,
                      'evidence_reappraisal_reduces_fear':evidence_reg[3]['strategy']=='REAPPRAISE' and bool(evidence_reg[0][fear]<evidence_unreg[0][fear])}}
    module.load_state_dict(original) # Experiments must not become the live engine's history.
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'state/checkpoints/v14-language.pth')
    parser.add_argument('--steps',type=int,default=700)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(14);disable_progress_bars()
    agent,state=load_checkpoint(args.checkpoint)
    if not agent.cfg.developmental_emotion:
        agent.cfg.developmental_emotion=True;agent.development=DevelopmentalEmotion(EMOTIONS,agent.cfg.extended_own_appraisal)
    preserved={name:value.detach().clone() for name,value in agent.named_parameters() if not name.startswith('development.') and not name.startswith('decoder.lm.')}
    result=train_module(agent.development,args.steps)
    result['sequence_checks']=sequence_checks(agent.development)
    result['old_core_parameters_unchanged']=all(torch.equal(value,dict(agent.named_parameters())[name]) for name,value in preserved.items())
    result['neural_parameters']=sum(p.numel() for p in agent.development.parameters())
    result['synthetic_targets_only']=True
    (ROOT/'docs/developmental-emotion-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if result['after_appraisal_mse']>.025 or not all(result['sequence_checks']['checks'].values()):
        raise RuntimeError('Developmental validation failed; active checkpoint was not replaced')
    # Lifetime history and saved conversation/state survive installation.
    if hasattr(agent,'outcome_optimizer'): del agent.outcome_optimizer
    save_checkpoint(agent,state,args.checkpoint)

if __name__=='__main__': main()
