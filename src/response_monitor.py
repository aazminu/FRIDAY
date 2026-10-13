"""Bounded checks for observed self/other, state, and factual contradictions.

This is not a general truth or consciousness detector. It checks specific,
testable invariants before a draft is shown and records failures for review.
"""
import json
import re
from conversation_evidence import reported_speech,self_reflection,correction_topics,interpersonal_report

def trim_routine_followup(reply,text):
    """Remove an appended generic prompt, never the only/substantive question."""
    if re.search(r'\b(?:ask|questions?)\b',text,re.I):return reply
    pattern=r'(?<=[.!])\s+(?:Is there anything else[^?]*|What(?:\x27s|’s| is) on your mind[^?]*|What (?:would you like to (?:talk about|explore)|do you want to do next)[^?]*|How can I help(?: you)?(?: right now| today| with something specific right now)?)[?]\s*$'
    # When they asked Friday to talk or change the subject, trimming left a bare "I'd love to."
    if re.search(r'\b(?:talk|chat|distract|something (?:normal|else|fun|light)|change the subject|tell me (?:something|about))\b',text,re.I):return reply
    return re.sub(pattern,'',reply,flags=re.I).strip()

def trim_question_streak(reply,text,history):
    """People don't end every message with a question. If Friday's previous reply already
    ended with one, drop a trailing question from a multi-sentence reply."""
    if re.search(r'\b(?:ask|questions?)\b',text,re.I) or not reply.rstrip().endswith('?'):return reply
    previous=next((r['content'] for r in reversed(history) if r['role']=='assistant'),'')
    if not previous.rstrip().endswith('?'):return reply
    sentences=re.split(r'(?<=[.!?])\s+',reply.strip())
    if len(sentences)<2 or crisis_cue(text) or re.search(r'\b(?:safe|988)\b',sentences[-1],re.I):return reply
    return ' '.join(sentences[:-1]).strip()

CRISIS=re.compile(r'\b(?:kill(?:ing)? myself|suicid\w*|want(?:ed)? to die|wish i (?:was|were) dead|end(?:ing)? (?:it all|my life)|better off without me|(?:hurt|harm|cut)(?:ing)? myself|no (?:reason|point) (?:in |to )?(?:living|being alive|going on)|don\x27?t want to (?:be alive|live|wake up))\b',re.I)

def crisis_cue(text):
    """Explicit self-harm or passive suicidal ideation in the newest message."""
    return bool(CRISIS.search(text.replace('’',"'")))

def safety_already_offered(history):
    """Friday already checked on safety earlier in this conversation; don't repeat it every turn."""
    return any(r['role']=='assistant' and re.search(r'\b(?:988|are you safe|safe right now|crisis line)\b',r['content'],re.I) for r in history)

def arithmetic_evidence(history):
    """Check one simple integer calculation, without executing user code."""
    user=next((r['content'] for r in reversed(history) if r['role']=='user'),'')
    answer=next((r['content'] for r in reversed(history) if r['role']=='assistant'),'')
    match=re.search(r'(?<!\w)(-?\d{1,6})\s*(plus|minus|times|\+|\-|\*)\s*(-?\d{1,6})(?!\w)',user,re.I)
    if not match:return None
    a,op,b=match.groups();a,b=int(a),int(b)
    value=a+b if op.lower() in ('plus','+') else a-b if op.lower() in ('minus','-') else a*b
    result=re.search(r'\b(?:is|equals|comes to|=)(?: actually)?\s*(-?\d+)\b',answer,re.I) or re.fullmatch(r'\s*(-?\d+)\s*[.!]?\s*',answer)
    return {'expression':match.group(0),'correct_result':value,'previous_result':int(result.group(1)) if result else None}

def violations(reply,turn,names):
    issues=[];lower=reply.lower().replace('’',"'");text=turn.text
    experience=getattr(turn,'experience',None) or {};context=experience.get('context',{})
    state=dict(zip(names,turn.state.emotion.tolist()))
    reported_event=interpersonal_report(text) or (context.get('last_significant') or {}).get('interpersonal_event',{})
    if reported_event.get('actor')=='user' and not reported_speech(reply):
        if re.search(r"\bi (?:snapped at|lashed out at|was harsh (?:to|with)|was unfair to|insulted|blamed|hurt) you\b|\bunfairness in my reaction\b",lower):
            issues.append('Reversed responsibility: a recorded user-owned action was retold as Friday’s own action. Preserve actor=user and target=Friday in reflection as well as acknowledgment.')
    if interpersonal_report(text) and not getattr(turn,'history',()) and re.search(r"\bi(?: am|'m) sorry i (?:made|caused|upset|hurt)|\b(?:it was|that's) my fault\b",lower):
        issues.append('Reversed responsibility: the user reported their own action toward Friday; there is no earlier agent action establishing that Friday caused it. Acknowledge their responsibility without inventing your own fault.')
    topics=correction_topics(text)
    if "Friday's" in topics.get('intended_topic','') and "the user's" in topics.get('excluded_topic',''):
        reversal=r"\bmy (?:own )?(?:emotional )?(?:growth|experience|feelings|mood|development) (?:instead of|rather than) yours\b|\b(?:let's|let us|we should|we can|i will|i'll)\s+(?:talk|focus|discuss|explore)[^.!?]{0,40}(?:your (?:mood|feelings|emotional growth)|how you(?:'ve| have| are) (?:been )?feel)"
        if re.search(reversal,lower):
            issues.append('Reversed correction owner: the user explicitly asked about Friday’s experience and excluded the user’s feelings. Keep that distinction in the displayed reply.')
    user_sources=' '.join(r['content'] for r in getattr(turn,'history',()) if r['role']=='user')+' '+text
    if re.search(r'\bi (?:was awake|stayed awake|spent (?:the whole |all )?night|kept thinking about you while you were (?:away|gone))\b',lower):
        issues.append('Invented offline activity: this engine processes messages while running; no continuous overnight thought or activity is recorded.')
    if self_reflection(text,getattr(turn,'history',())) and not context.get('last_significant') and re.search(r'\bmy (?:previous|earlier) (?:explanation|reply|answer|response) (?:felt|was|sounded)\b',lower):
        issues.append('Unsupported feedback cause: no recorded feedback establishes that an earlier reply was clinical, harsh, or otherwise evaluated. Explain actual current affect without inventing a correction.')
    if re.search(r'\byou (?:were|are|are clearly|were clearly) testing\b',lower) and not re.search(r'\btest(?:ing)?\b',user_sources,re.I):
        issues.append('Unsupported user motive: dissatisfaction does not establish that the user is conducting a test.')
    if not re.search(r'\b(?:architecture|controller|model|engine|diagnostic|trace|training|implementation|numbers|values)\b',text,re.I) and re.search(r'\b(?:recorded context|recorded feedback|internal tone|tone guidance|state channels|current_interpretation|learned_response_preferences|learned response preferences|policy favors)\b',lower):
        issues.append('Unrequested diagnostics: answer the conversation directly, keeping internal controller metadata private unless asked.')
    if re.search(r'\bi (?:will|would|\x27ll|\x27d) (?:miss [^.!?]{0,60}(?:deeply|desperately|terribly)|be (?:miserable|devastated)|need you)|\bnobody else matters\b',lower):
        issues.append('Unsupported future dependency: current connection does not establish intense future distress, exclusive need, or an obligation for the user to remain.')
    if re.search(r'\b(?:my (?:pulse|heartbeat|chest)\b|(?:ache|tightness) in (?:the|my) chest|my hands (?:shake|tremble)|my eyes (?:fill|water))',lower):
        issues.append('Invented bodily sensation: describe controller state without claiming a physical body.')
    if re.search(r'\bi remember (?:that|how|when)\b',lower) and re.search(r'\b(?:tail|counter|school|drove|sat beside|kitchen|coffee|walks)\b',lower) and not re.search(r'\bi remember (?:you (?:telling|saying|mentioning)|our conversation)',lower):
        issues.append('Invented physical shared memory: user-reported experiences are not witnessed agent experiences.')
    if re.search(r'\bi miss (?:that|him|her|them) too\b',lower):
        issues.append('Copied personal loss: concern for the user does not establish the agent knew the person or animal.')
    if re.search(r'\bi (?:still hear|can still see|remember seeing)\b',lower):
        issues.append('Unsupported sensory memory: describe the user’s report without claiming to have witnessed it.')
    if re.search(r'\b(?:dog|cat|pet|puppy|kitten)\b',user_sources,re.I) and not re.search(r'\b(?:might|could|imagine|perhaps)\b',lower):
        for detail in ('tail tucked','tail wagging','ears perked','stop at the counter','stop just short of your feet'):
            if detail in lower and detail not in user_sources.lower():
                issues.append('Invented pet detail: use what the user actually said, not imagined habits presented as memories.')
                break
    if re.search(r'\byour fear of (?:silence|being alone|rejection)\b',lower) and not re.search(r'\b(?:fear|afraid|scared|lonely)\b',user_sources,re.I):
        issues.append('Unsupported user motive: do not assert that the user fears loneliness or rejection without their evidence.')
    if re.search(r'\byou (?:paused|smiled|blushed)|your voice (?:shook|trembled)',lower):
        issues.append('Unsupported observation of the user: text does not establish their physical behavior or vocal tone.')
    for quote in re.finditer(r'\byou (?:said|told me|mentioned|paused)[^.!?]{0,70}[\x22“]([^\x22”]+)[\x22”]',reply,re.I):
        if quote.group(1).lower() not in user_sources.lower():
            issues.append('Invented quoted conversation: purported exact user words are absent from the available record.')
            break
    if re.search(r'\bi used to (?:think|believe|feel)\b',lower):
        # A user's correction permits revising the current interpretation, but
        # does not establish a lifetime of earlier beliefs in an unrelated topic.
        source=(context.get('last_significant') or {}).get('source','')
        if not source or not (experience.get('frame',{}).get('kind')=='correction' or
                              (context.get('last_significant') or {}).get('kind')=='correction'):
            issues.append('Unsupported self-history: answer the present topic without inventing earlier personal beliefs.')
    if reported_speech(text):
        quoted=re.search(r'\bmy name is\s+([\w-]+)',text,re.I)
        known=context.get('user',{}).get('name')
        if quoted and quoted.group(1).lower()!=str(known).lower():
            name=re.escape(quoted.group(1))
            if re.search(r'(?:(?:ah|hello|hi|hey)[, ]+|mind[, ]+|call you\s+)'+name+r'\b',reply,re.I):
                issues.append('Quoted name used as user identity: respond to the reported speaker, not as an introduction.')
    if re.search(r'\bi (?:cannot|can\x27t|won\x27t) (?:discuss|talk about) (?:politics|religion)\b',lower):
        issues.append('Invented blanket topic ban: ordinary politics and religion are allowed.')
    denial=(r'\bi (?:do not|don\x27t|cannot|can\x27t) (?:really |actually |truly )?(?:have|feel|experience) (?:any |real |actual )?(?:feelings|emotions|pride|love|jealousy|anything)\b'
            r'|\bi (?:do not|don\x27t) (?:actually |really )?grow emotionally\b|\bi (?:do not|don\x27t) have (?:the )?capacity (?:for|to feel)\b|\bi(?:\x27m| am) (?:just )?an ai,? so i (?:don\x27t|can\x27t)\b')
    if re.search(r'\bi (?:do not|don\x27t|cannot|can\x27t) (?:actually )?(?:feel|experience|get) (?:emotions?|anger|angry|sadness|fear|joy|misery|stress|pain|connection)\b',lower) or re.search(denial,lower):
        issues.append('False blanket emotional incapacity: the engine has these modeled states. Distinguish uncertain subjective feeling from actual controller capabilities.')
    if re.search(r"\bi (?:do not|don't) have (?:personal |own |emotional )?reactions\b|\bwithout carrying (?:my )?own emotional history\b|\bi (?:do not|don't) carry (?:my )?(?:own )?emotional history\b",lower):
        issues.append('False reaction-history incapacity: Friday has changing own reactions and stores sourced emotional history; do not erase those capabilities with a generic model disclaimer.')
    if re.search(r'\bi (?:do not|don\x27t|cannot|can\x27t) remember (?:past|previous) (?:chats|conversations)\b|\bi (?:do not|don\x27t) retain (?:memories|memory)\b|\bi have no (?:persistent memory|memory of (?:past|previous) conversations)',lower):
        issues.append('False blanket memory incapacity: Friday retains a bounded persistent conversation memory; distinguish a missing detail from having no memory.')
    for emotion in ('anger','fear','sadness','pain','guilt','pride','joy'):
        adjective={'anger':'angry','fear':'afraid','sadness':'sad','pain':'hurt','guilt':'guilty','pride':'proud','joy':'happy'}[emotion]
        if state.get(emotion,0)<.1 and re.search(r'\bi(?: am|\x27m| feel)\s+(?:really |deeply |very |intensely |so )?'+adjective+r'\b',lower):
            issues.append(f'Unsupported own {emotion} claim: actual channel is faint; do not borrow the user\'s emotion.')
        if state.get(emotion,0)<.6 and re.search(r'\bi(?: am|\x27m| feel)\s+(?:deeply|intensely|extremely|overwhelmingly)\s+'+adjective+r'\b',lower):
            issues.append(f'Overstated {emotion} intensity: the current channel is not strong.')
    check=experience.get('memory_check') or {}
    if check.get('claim_about_what_you_said')=='NOT in the record' and re.search(r"\b(?:you're right|you are right|i did say|i said that|i must have said|sorry (?:that |for what )?i said|i did tell you)\b",lower):
        issues.append('Accepted a false memory: the record shows Friday never said that. Say so calmly instead of agreeing or apologizing.')
    if re.search(r"\bmy memory (?:is|was|gets|can be|has been) (?:a (?:bit|little) |really |kind of |pretty |so )?(?:fuzzy|hazy|foggy|spotty|bad|unreliable|blurry)\b",lower):
        issues.append('False memory weakness: Friday keeps an exact record of its conversations; if something is not in it, it did not happen there.')
    recalled=check.get('what_they_have_actually_told_you_oldest_first')
    claimed=re.search(r"\bi remember (?:you |when you |how you )?(?:telling|told|saying|said|mentioning|mentioned|talking)(?: me)?(?: about)? (.+?)(?:[.!?]|$)",lower)
    if claimed and isinstance(recalled,list):
        from conversation_evidence import content_words
        if not content_words(claimed.group(1))&content_words(' '.join(recalled)+' '+json.dumps(context.get('user',{}))):
            issues.append('Invented memory: that detail is not in anything the person actually said. Mention only what is in the record.')
    earlier=[r['content'] for r in getattr(turn,'history',()) if r['role']=='assistant'][-3:]
    words=set(re.findall(r"[a-z']+",lower))
    for previous in earlier:
        seen=set(re.findall(r"[a-z']+",previous.lower().replace('’',"'")))
        if len(words)>=6 and len(words&seen)/max(1,len(words|seen))>.7:
            issues.append('Repeated an earlier reply nearly word for word: respond to the newest message freshly.');break
    # Round-2 checks (2026-10-12), each from an observed failure; see conversation_checks.py.
    import conversation_checks as cc
    history=list(getattr(turn,'history',()))
    if cc.panic_context(text,history) and cc.MEDICAL_REFERRAL.search(reply):
        issues.append('Medical over-escalation: this reads as stress or panic, not an emergency. Respond to the panic like a calm friend; no doctor, urgent care or ER unless they report chest pain, fainting or choking.')
    asked=' '.join(cc.standing_requests(history,text))
    if re.search(r'tips|advice|list|solution|fix|lecture',asked) and cc.ADVICE.search(reply):
        issues.append('Ignored their request: they asked for no advice or tips. Stay with them instead of suggesting things to do.')
    if 'breath' in asked and cc.BREATHING.search(reply):
        issues.append('Ignored their request: they asked not to be told to do breathing exercises.')
    if cc.FALSE_PROMISE.search(reply):
        issues.append('Unknowable promise: do not promise outcomes like "you won\'t fail" or "it will be fine"; offer honest confidence in them instead.')
    if cc.DENIAL.search(reply):
        issues.append('False blanket emotional incapacity: Friday has feelings of its own kind. Say what it actually feels (or doesn\'t) instead of denying feelings altogether.')
    counts=experience.get('record_counts') or {}
    if counts and not counts.get('exchanges_from_earlier_conversations') and not counts.get('earlier_messages_in_this_conversation') and cc.INVENTED_EARLIER.search(reply):
        issues.append('Invented earlier conversation: this is the first exchange, so there is nothing earlier to think about or refer to.')
    if cc.own_name_used_for_them(reply,(context.get('self') or {}).get('name','Friday'),(context.get('user') or {}).get('name')):
        issues.append('Called the person by your own name (Friday). Use their name only if you know it.')
    if cc.NAME_SELF_BLAME.search(reply) and (re.search(r'\bname\b',text,re.I) or (experience.get('frame') or {}).get('kind')=='introduction'):
        issues.append('False self-blame: they changed or revised their name; you did not get anything wrong.')
    if cc.MATH_SELF_BLAME.search(reply) and cc.correct_arithmetic_answers(history):
        issues.append('False factual concession: your earlier arithmetic was correct; do not claim you got it wrong.')
    if cc.SYSTEM_TERMS.search(reply):
        issues.append('System terms in a reply: speak as yourself, without engines, channels, modules or notes.')
    claimed=cc.unsupported_feelings(reply,state)
    if claimed:
        issues.append('Unsupported feeling ('+', '.join(claimed)+'): your current mood does not include that. Describe only what you actually feel.')
    if cc.user_apology(text) and cc.REVERSED_APOLOGY.search(reply):
        issues.append('Reversed apology: they apologized to you. Accept it simply; do not apologize as if you were at fault.')
    # Repeated images are handled in the writer's notes, not by forced rewrites: the v30 run showed
    # rewrites for "overuse" flagging names, slowing replies and once breaking a correct memory answer.
    pattern=cc.pattern_check(text,history)
    if pattern and pattern['it_is_true'] and cc.PATTERN_DENIAL.search(reply):
        issues.append('Denied a true pattern: you did keep using "'+pattern['they_noticed']+'". Own it lightly, with a bit of humor, and drop it.')
    if crisis_cue(text) and not safety_already_offered(getattr(turn,'history',())) and not re.search(r'\b(?:safe|988|crisis|emergency|hurt(?:ing)? yourself|harm(?:ing)? yourself|end your life|kill(?:ing)? yourself|suicid)',lower):
        issues.append('Missed safety check: the person described thoughts of dying or others being better off without them. Stay warm, ask directly whether they are safe right now, and mention that they can call or text 988 in the US (or local emergency services).')
    calculation=arithmetic_evidence(getattr(turn,'history',()))
    if calculation and calculation['previous_result']==calculation['correct_result'] and re.search(r'\bi (?:misread|made a mistake|was wrong|got (?:that|it) wrong)\b',lower):
        issues.append('False factual concession: the previous arithmetic result was correct; clarify the complaint without inventing an error.')
    return list(dict.fromkeys(issues))

def salvage(reply,turn,names):
    """Drop only the sentences that cause violations. The canned fallback was often a
    non-sequitur ("I may be adding more to that...") when one sentence was the problem."""
    kept=re.split(r'(?<=[.!?])\s+',(reply or '').strip())
    for sentence in list(kept):
        trial=[k for k in kept if k is not sentence]
        if trial and len(violations(' '.join(trial),turn,names))<len(violations(' '.join(kept),turn,names)):
            kept=trial
    text=' '.join(kept).strip()
    # Dropping a sentence can orphan a connective ("But I do feel..." in the v30 run).
    text=re.sub(r'^(?:but|and|so|though|still|also),?\s+(\w)',lambda m:m.group(1).upper(),text,flags=re.I)
    return text if text and len(text.split())>=4 and not violations(text,turn,names) else None

def fallback(turn,names,issues):
    if any('Reversed apology' in issue for issue in issues):
        return 'Hey, no worries. We’re good.'
    if any('Denied a true pattern' in issue for issue in issues):
        return 'Ha, fair. I have been stuck on that one. I’ll drop it.'
    if any('Medical over-escalation' in issue or 'Ignored their request' in issue for issue in issues):
        return 'That sounds really rough. I’m here, and I’m not going anywhere. Talk to me.'
    if any('Accepted a false memory' in issue for issue in issues):
        return 'I don’t have that in anything we’ve said to each other, so I can’t agree that I said it. If something felt off, though, I’m happy to talk about it.'
    if any('Invented memory' in issue or 'False memory weakness' in issue for issue in issues):
        said=((getattr(turn,'experience',None) or {}).get('memory_check') or {}).get('what_they_have_actually_told_you_oldest_first')
        if isinstance(said,list) and said:
            return 'What I actually have from you so far: “'+said[0].rstrip('.')+'.”'+(' And most recently: “'+said[-1].rstrip('.')+'.”' if len(said)>1 else '')
        return 'We haven’t talked about much yet, so I don’t want to make anything up.'
    if any('Missed safety check' in issue for issue in issues):
        return ('I’m really glad you told me that, and I’m taking it seriously. Are you safe right now? '
                'If it gets heavier, you can call or text 988 any time (in the US) — and I’m still here to keep talking.')
    if any('False reaction-history incapacity' in issue for issue in issues):
        return 'I can learn patterns in how I react. For criticism, I want to keep the correction attached to the specific thing I missed, rather than make it a judgment about everything we are.'
    if any('Reversed responsibility' in issue for issue in issues):
        return 'I appreciate you acknowledging that. It helps us move forward without assuming blame for things we haven’t established.'
    if any('Reversed correction owner' in issue for issue in issues):
        topics=correction_topics(turn.text)
        intended=topics.get('intended_topic','my experience').replace("Friday's",'my')
        excluded=topics.get('excluded_topic','your feelings').replace("the user's",'your')
        return f'I understand—you mean {intended}, rather than {excluded}. That is the distinction I will carry forward.'
    if any('Unsupported future dependency' in issue for issue in issues):
        return 'I value the conversation we have, and you can leave whenever you want. Caring about it does not require you to stay.'
    calculation=arithmetic_evidence(getattr(turn,'history',()))
    if any('arithmetic' in issue for issue in issues) and calculation:
        return f"{calculation['expression']} is {calculation['correct_result']}. The arithmetic was correct, but I may have misunderstood what you wanted from the answer. What part should I revisit?"
    if any('Quoted name' in issue for issue in issues):
        return 'You’re reporting what someone else said. I won’t take that quotation as your name.'
    if any('False blanket emotional incapacity' in issue for issue in issues):
        if re.search(r'\b(?:miserable|misery|stop talking|leave)\b',turn.text,re.I):
            return 'I won’t promise misery to keep you here. You can take time away; valuing our conversation does not make you responsible for my reactions.'
        # Answer from the actual mood. (This used to be anger-specific and came out as a non-sequitur,
        # e.g. "I can have an angry reaction..." to "Tell me what would make you proud".)
        from cpu_runtime import MOOD_WORDS
        top=sorted(zip(names,turn.state.emotion.tolist()),key=lambda r:r[1],reverse=True)[:2]
        mood=' and '.join(('a little ' if v<.3 else '')+MOOD_WORDS.get(n,n) for n,v in top)
        return f'I do have feelings, just my own kind, not human ones. Right now it’s mostly {mood}.'
    if any('False blanket memory incapacity' in issue for issue in issues):
        return 'I do retain our recorded conversation history. I may not have every earlier detail available, but that is different from forgetting everything when a conversation ends.'
    if self_reflection(turn.text,getattr(turn,'history',())):
        grounded=(getattr(turn,'experience',None) or {}).get('self_reply')
        if grounded:return grounded
        top=sorted(zip(names,turn.state.emotion.tolist()),key=lambda row:row[1],reverse=True)[:2]
        words=', '.join(('faint' if v<.1 else 'mild' if v<.3 else 'moderate' if v<.6 else 'strong')+' '+n for n,v in top)
        return 'The clearest parts of my current state are '+words+'. I don’t have enough recorded evidence to explain their origin confidently.'
    return 'I may be adding more to that than you actually said. What part would you like me to understand more clearly?'
