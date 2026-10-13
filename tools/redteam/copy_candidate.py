"""Create an isolated source/state copy without copying shared model assets."""
import argparse, shutil, subprocess, json, hashlib, datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path);a=p.parse_args()
source=a.source.resolve();target=a.target.resolve()
if target.exists():raise SystemExit('Target already exists; refusing to overwrite a saved experiment')
target.mkdir(parents=True)
for folder in ('src','tests','docs','state','data/training'):
 if (source/folder).exists():shutil.copytree(source/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','logs'))
root=Path(__file__).resolve().parents[2]
for folder in ('data/models','runtime'):
 link=target/folder;link.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['powershell','-NoProfile','-Command',f"New-Item -ItemType Junction -Path '{str(link).replace(chr(39),chr(39)*2)}' -Target '{str(root/folder).replace(chr(39),chr(39)*2)}' | Out-Null"],check=True)
manifest={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(source),'copy':str(target),
 'checkpoint_sha256':hashlib.sha256((target/'state/checkpoints/v14-language.pth').read_bytes()).hexdigest(),
 'files':{str(f.relative_to(target)):hashlib.sha256(f.read_bytes()).hexdigest() for f in target.glob('src/*.py')}}
(target/'snapshot.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
