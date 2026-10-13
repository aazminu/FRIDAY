"""Shared conversational preparation and completion used by chat and evaluation."""
import re
import torch

PROFILE_CUE=re.compile(r"(?i:(?<!\w)my name(?!\w)|(?<!\w)name(?:'|’)?s(?!\w)|(?<!\w)call me(?!\w)|(?<!\w)i work(?!\w)|(?<!\w)work(?:ing)? (?:as|at|in)(?!\w)|(?<!\w)i live(?!\w)|(?<!\w)living in(?!\w)|(?<!\w)(?:i'?m|i am|i)(?:\s+\w+)? from(?!\w)|(?<!\w)(?:i'?m|i am) an?(?!\w))|(?<!\w)(?:I'?m|I am|i'?m|i am) [A-Z][a-z]+")

def interpret_turn(agent,native,text):
    frame=agent.interpreter.interpret(agent.decoder,text,agent.experience) if agent.cfg.semantic_interpreter else native.interpret(text,agent.experience.context(),agent.chat_history)
    from conversation_evidence import declarations
    known=agent.experience.data['user'].get('name',{}).get('value')
    # The profile model call costs seconds; only make it when the message could declare a name, job or place.
    if not declarations(text,known) and PROFILE_CUE.search(text):
        profile=native.profile(text)
        if profile:
            frame.update(kind='introduction',scope='identity',subject='user',attribute=profile['attribute'],
                         value=profile['value'],evidence=profile['evidence'],confidence=profile['confidence'],
                         valence=0.,summary=profile['evidence'],_profile_parse=profile)
    return frame

def prepare_experience(agent,state,text,frame,needs,measured=False):
    prepared=agent.experience.prepare(text,frame)
    if frame.get('_profile_parse'):prepared['profile_parse']=frame['_profile_parse']
    feedback=prepared['feedback']
    if feedback:
        episode=next((v for v in agent.experience.data['episodes'] if v['id']==feedback['episode_id']),None)
        if episode and episode.get('development'):
            quality=feedback['quality']*feedback['confidence']
            agent.development.observe(episode['development'],quality)
            state=state.detached()
            # Explicit conversational evaluation is a signed own-functioning proxy,
            # distinct from externally measured homeostatic levels.
            state.vitality=.8*state.vitality+.2*state.vitality.new_tensor(quality)
    agent.experience.learn(prepared)
    if not measured: needs=agent.deficits(prepared['levels'])
    prepared['emotion_names']=tuple(agent.development.names)
    # Verify memory claims against the stored record (which survives restarts) before the writer sees them.
    from conversation_evidence import memory_check
    episodes=agent.experience.data['episodes']
    record=[row for e in episodes for row in ({'role':'user','content':e['input']},{'role':'assistant','content':e.get('response','')})]
    current=sum(r['role']=='user' for r in getattr(agent,'chat_history',[]))
    check=memory_check(text,record,earlier_conversations=max(0,len(episodes)-current))
    if check:prepared['memory_check']=check
    # How much history really exists, so a first conversation can't "remember" an earlier one.
    prepared['record_counts']={'exchanges_from_earlier_conversations':max(0,len(episodes)-current),
                               'earlier_messages_in_this_conversation':current}
    return state,needs,prepared

def complete_experience(agent,turn,prepared,response):
    agent.experience.record(turn.text,response,turn,prepared)

def explain_experience(agent,turn):
    return agent.experience.explanation(turn.text,turn,agent.development.names)
