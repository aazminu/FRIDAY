"""Install the shared experience contract without retraining language weights."""
from pathlib import Path
import torch
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,load_checkpoint,save_checkpoint
from experience_model import ExperienceModel

def main():
    torch.set_num_threads(4);disable_progress_bars()
    path=ROOT/'state/checkpoints/v14-language.pth'
    agent,state=load_checkpoint(path)
    if not agent.cfg.developmental_emotion: raise RuntimeError('Install the trained developmental module first')
    if not agent.cfg.experience_model:
        agent.cfg.experience_model=True;agent.experience=ExperienceModel()
    save_checkpoint(agent,state,path)
    print('Shared experience installed; existing neural weights, saved state, and conversation preserved.')

if __name__=='__main__': main()
