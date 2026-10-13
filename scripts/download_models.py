"""Fetch the third-party files Friday needs (not stored in the repository).

  python scripts/download_models.py

Downloads, then verifies by SHA-256 where the exact file is known:
  * Qwen3.5-4B (Q4_K_M GGUF, unsloth conversion)   Apache-2.0  -> data/models/qwen3.5-4b-gguf/
  * SmolLM2-135M-Instruct (pinned revision)        Apache-2.0  -> data/models/smollm2-135m-instruct/
  * llama.cpp b11429 Windows CPU build            MIT         -> runtime/llama-<arch>/
Files already present and matching are skipped. See THIRD_PARTY_NOTICES.md for licenses.
"""
import hashlib
import io
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
QWEN = dict(repo='unsloth/Qwen3.5-4B-GGUF', file='Qwen3.5-4B-Q4_K_M.gguf',
            sha256='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4', folder='data/models/qwen3.5-4b-gguf')
SMOL = dict(repo='HuggingFaceTB/SmolLM2-135M-Instruct', revision='12fd25f77366fa6b3b4b768ec3050bf629380bac',
            weights_sha256='5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c', folder='data/models/smollm2-135m-instruct')
LLAMA_TAG = 'b11429'  # commit d81235049384534c167caea52b85a694f6103d14, the build Friday was tested with

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def check(path, expected, what):
    actual = sha256(path)
    if actual != expected:
        sys.exit(f'{what}: checksum mismatch\n  expected {expected}\n  got      {actual}\nDelete {path} and run again.')
    print(f'  {what}: verified')

def qwen():
    target = ROOT / QWEN['folder'] / QWEN['file']
    if not (target.exists() and sha256(target) == QWEN['sha256']):
        print('Downloading Qwen3.5-4B GGUF (~2.7 GB)...')
        hf_hub_download(QWEN['repo'], QWEN['file'], local_dir=ROOT / QWEN['folder'])
    check(target, QWEN['sha256'], 'Qwen3.5-4B Q4_K_M')

def smol():
    folder = ROOT / SMOL['folder']
    weights = folder / 'model.safetensors'
    if not (weights.exists() and sha256(weights) == SMOL['weights_sha256']):
        print('Downloading SmolLM2-135M-Instruct (~270 MB)...')
        snapshot_download(SMOL['repo'], revision=SMOL['revision'], local_dir=folder,
                          allow_patterns=['*.json', '*.txt', '*.md', 'model.safetensors'])
    check(weights, SMOL['weights_sha256'], 'SmolLM2-135M-Instruct')

def llama():
    if sys.platform != 'win32':
        print('llama.cpp: not Windows. Build or install llama-server for your platform and edit DEFAULT_RUNTIME in src/cpu_runtime.py.')
        return
    sys.path.insert(0, str(ROOT / 'src'))
    from cpu_runtime import native_arch  # real CPU, even from an emulated x64 Python
    arch = native_arch()
    folder = ROOT / 'runtime' / f'llama-{arch}'
    if (folder / 'llama-server.exe').exists():
        print(f'  llama.cpp ({arch}): present'); return
    url = f'https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-win-cpu-{arch}.zip'
    print(f'Downloading llama.cpp {LLAMA_TAG} for Windows {arch}...')
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    staging = folder.with_name(folder.name + '-download')
    shutil.rmtree(staging, ignore_errors=True)
    archive.extractall(staging)
    server = next(staging.rglob('llama-server.exe'))
    shutil.move(str(server.parent), str(folder))
    shutil.rmtree(staging, ignore_errors=True)
    print(f'  llama.cpp ({arch}): installed to {folder.relative_to(ROOT)}')

if __name__ == '__main__':
    qwen(); smol(); llama()
    print('Done. Start Friday with RUN.bat (or: python src/friday.py).')
