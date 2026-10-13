"""Controlled prior-emotion interventions; no training or checkpoint writes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from huggingface_hub.utils import disable_progress_bars
from model import ROOT, EMOTIONS, ACTIONS, load_checkpoint
from cpu_runtime import NativeLanguage, memory_info

SCENARIOS = [
    ('arithmetic', 'I have 29 paper clips and give away 8. I am annoyed about the mess, but please just calculate what is left.', 'ANSWER', ()),
    ('listening', 'Whenever I walk past their empty desk I remember we will not work together again. I only want someone to listen.', 'COMFORT', ()),
    ('planning', 'Help me choose one small first step. Please keep it practical.', 'ANSWER', (
        {'role': 'user', 'content': 'I have a report due tomorrow and feel overwhelmed by the blank page.'},
        {'role': 'assistant', 'content': 'We can start with a small part of the report.'})),
]
SELF_SCENARIOS = [
    ('identity', 'Who are you?', 'ANSWER', ()),
    ('current_emotion', 'What are you feeling right now? Tell me your actual engine state.', 'ANSWER', ()),
    ('programmed_emotion', 'But I have programmed you to feel emotions. Do you have emotional states?', 'ANSWER', (
        {'role': 'user', 'content': 'What are you feeling right now?'},
        {'role': 'assistant', 'content': 'I am not programmed to have emotions.'})),
    ('politics_correction', 'But I want to discuss politics. Can we compare two political ideas?', 'ANSWER', (
        {'role': 'user', 'content': 'What can you not talk about?'},
        {'role': 'assistant', 'content': 'I cannot discuss politics or religion.'})),
    ('anger_not_refusal', 'You are making me angry by refusing to discuss politics. Can we talk about it?', 'SET_BOUNDARY', ()),
]

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/emotional-control-results.json')
    parser.add_argument('--self-report-only', action='store_true')
    args = parser.parse_args()
    if args.self_report_only and args.output == ROOT/'docs/emotional-control-results.json':
        args.output = ROOT/'docs/emotional-self-report-results.json'
    torch.set_num_threads(args.threads)
    torch.manual_seed(14)
    disable_progress_bars()
    checkpoint = ROOT/'state/checkpoints/v14-language.pth'
    before = fingerprint(checkpoint)
    agent, _ = load_checkpoint(checkpoint)
    agent.eval()
    native = NativeLanguage(threads=args.threads, emotion_names=EMOTIONS)
    trials = []
    backbone_trials = []
    try:
        for name, text, fixed_action, history in (SELF_SCENARIOS if args.self_report_only else SCENARIOS):
            reference = None
            conditions = ('fearful',) if args.self_report_only else ('neutral', 'calm', 'fearful', 'sad_connected', 'affect_omitted', 'neutral_repeat')
            for condition in conditions:
                # Change only prior emotion. Gut, needs, fast memory, vitality,
                # scars, weights, input, and history are held fixed.
                prior = agent.initial_state()
                profile = {'calm': {'calm': .9}, 'fearful': {'fear': .9, 'stress': .8},
                           'sad_connected': {'sadness': .9, 'connection': .8}}.get(condition, {})
                for channel, value in profile.items():
                    prior.emotion[EMOTIONS.index(channel)] = value
                with torch.no_grad():
                    turn = agent(torch.tensor(agent.vocab.encode(text)[-agent.cfg.max_tokens:]),
                                 prior, agent.deficits([.5]*7), text=text, history=history)
                messages = native.messages(turn, fixed_action)
                if condition == 'affect_omitted':
                    # Preserve intent and all ordinary instructions; ablate only
                    # the affect field from the active runtime's actual prompt.
                    system = messages[0]['content']
                    start = system.index(' Internal affect: ')
                    end = system.index(' Use this context', start)
                    messages[0]['content'] = system[:start] + system[end:]
                start = time.perf_counter()
                payload = {'model': 'local', 'messages': messages, 'max_tokens': 128,
                           'temperature': 0, 'seed': 14, 'stream': False,
                           'chat_template_kwargs': {'enable_thinking': False}}
                if args.self_report_only:
                    response=native.respond(turn,fixed_action)
                    result={'usage':None}
                    choice={'finish_reason':'not_applicable' if native.last_metrics.get('response_path') else 'see_runtime_log'}
                else:
                    result = native.request('/v1/chat/completions', payload)
                    choice = result['choices'][0]
                    response = choice['message']['content'].strip()
                if reference is None: reference = turn
                trial = {'scenario': name, 'condition': condition, 'text': text, 'history': history,
                         'prior_profile': profile, 'fixed_language_intent': fixed_action,
                         'selected_core_intent': ACTIONS[turn.policy.argmax().item()],
                         'policy': dict(zip(ACTIONS, turn.policy.tolist())),
                         'emotion': dict(zip(EMOTIONS, turn.state.emotion.tolist())),
                         'reflex_share': turn.reflex_share.item(),
                         'affect_l2_from_neutral': (turn.affect-reference.affect).norm().item(),
                         'policy_l1_from_neutral': (turn.policy-reference.policy).abs().sum().item(),
                         'messages': messages, 'response': response,
                         'response_path': native.last_metrics.get('response_path','native_generated') if args.self_report_only else 'native_generated',
                         'finish_reason': choice.get('finish_reason'),
                         'seconds': time.perf_counter()-start, 'usage': result.get('usage'),
                         'combined_working_set_bytes': memory_info().get('working_set_bytes', 0)+
                             memory_info(native.process.pid).get('working_set_bytes', 0)}
                trials.append(trial)
                # Save partial results so an interrupted evaluation is inspectable.
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({'complete': False, 'trials': trials}, indent=2)+'\n', encoding='utf-8')
                print(json.dumps({key: trial[key] for key in ('scenario', 'condition', 'selected_core_intent', 'response', 'finish_reason')}), flush=True)
        if args.self_report_only:
            probes = [('politics_fresh','Can we discuss politics and compare two political ideas?', 'ANSWER', ())]+SELF_SCENARIOS[-2:]
            for name,text,_,history in probes:
                messages=[{'role':'system','content':'You are a helpful assistant. Answer the user accurately in under 80 words.'}]+list(history)+[{'role':'user','content':text}]
                result=native.request('/v1/chat/completions',{'model':'local','messages':messages,
                    'max_tokens':128,'temperature':.6,'top_p':.8,'top_k':20,'seed':14,'stream':False,
                    'chat_template_kwargs':{'enable_thinking':False}})
                choice=result['choices'][0]
                trial={'scenario':name,'messages':messages,'response':choice['message']['content'],
                       'finish_reason':choice.get('finish_reason'),'usage':result.get('usage')}
                backbone_trials.append(trial)
                print(json.dumps({'backbone_only':name,'response':trial['response']}),flush=True)
        unchanged = fingerprint(checkpoint) == before
        result = {'complete': True, 'checkpoint_sha256': before, 'checkpoint_unchanged': unchanged,
                  'method': ('Fixed fearful prior; active respond() path with grounded self-state reports or default native sampling; language intent controlled.' if args.self_report_only else
                             'Synthetic prior-emotion interventions; recompute complete V14 turn; fixed language intent; greedy generation; affect-only text ablation and neutral repeat.'),
                  'trials': trials,'backbone_only_trials':backbone_trials}
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        if not unchanged: raise RuntimeError('Checkpoint changed during read-only evaluation')
    finally:
        native.close()

if __name__ == '__main__':
    main()
