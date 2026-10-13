"""Append real completed chat turns, preserving inspectable controller evidence."""
import json,time
from pathlib import Path

def append_turn(root,text,response,action,turn,names,metrics,measured):
    now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    path=Path(root)/'state/logs'/('chat-'+now[:10]+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    experience=getattr(turn,'experience',None) or {}
    row={'time_utc':now,'input':text,'reply':response,'action':action,
         'emotion_before':dict(zip(names,turn.before.emotion.tolist())),
         'emotion_after':dict(zip(names,turn.state.emotion.tolist())),
         'vitality':turn.state.vitality.item(),'needs_measured':measured,
         'interpretation':experience.get('frame'),'emotional_depth':experience.get('depth'),
         'development':turn.development,'language_metrics':metrics}
    with path.open('a',encoding='utf-8') as log:log.write(json.dumps(row,ensure_ascii=False)+'\n')
    return path
