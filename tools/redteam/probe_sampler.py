"""Check which sampler settings this llama-server build accepts (prints the server's error body)."""
import json, sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cpu_runtime import NativeLanguage, CLICHE_WORDS

native = NativeLanguage(threads=4, emotion_names=('joy',))
base = {'model': 'local', 'messages': [{'role': 'user', 'content': 'Say hi in three words.'}], 'max_tokens': 8,
        'temperature': .4, 'stream': False, 'id_slot': 0, 'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
variants = {
    'base': {},
    'presence': {'presence_penalty': .2},
    'logit_bias_strings': {'logit_bias': [[w, -2.5] for w in CLICHE_WORDS]},
    'dry_all': {'dry_multiplier': .6, 'dry_base': 1.75, 'dry_allowed_length': 3, 'dry_penalty_last_n': -1},
    'dry_no_last_n': {'dry_multiplier': .6, 'dry_base': 1.75, 'dry_allowed_length': 3},
    'dry_last_n_0': {'dry_multiplier': .6, 'dry_base': 1.75, 'dry_allowed_length': 3, 'dry_penalty_last_n': 0},
    'stream_dry_bias': {'stream': True, 'dry_multiplier': .6, 'dry_base': 1.75, 'dry_allowed_length': 3, 'logit_bias': [[w, -2.5] for w in CLICHE_WORDS]},
}
try:
    for name, extra in variants.items():
        payload = {**base, **extra}
        req = Request(native.url + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=60) as r:
                body = r.read().decode('utf-8', 'replace')
            print(f'{name:22} OK   ', body[:90].replace('\n', ' '))
        except HTTPError as e:
            print(f'{name:22} {e.code} ', e.read().decode('utf-8', 'replace')[:300])
finally:
    native.close()
