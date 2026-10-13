"""Conservative extraction of explicit first-person profile declarations.

No names are extracted from quoted dialogue, hypothetical/role-play text, or a
third-person report. These rules validate declarations, not general meaning.
"""
import re

def speaker_terms(phrase):
    """Bind possessives in a user-authored clarification; retain raw evidence separately."""
    phrase=re.sub(r'\byour\b',"Friday's",phrase,flags=re.I)
    phrase=re.sub(r'\bmy\b',"the user's",phrase,flags=re.I)
    return re.sub(r'\bmine\b',"the user's experience",phrase,flags=re.I)

def correction_topics(text):
    match=re.search(r'\b(?:i meant|this is about|it is about|i was referring to)\s+(.+?)(?:,?\s+(?:not|rather than)\s+(.+?))?[.!]?$',text,re.I)
    if not match:return {}
    result={'intended_topic':speaker_terms(match.group(1))}
    if match.group(2):result['excluded_topic']=speaker_terms(match.group(2))
    return result

def belief_revision(text):
    if reported_speech(text):return {}
    match=re.search(r'\bi (?:thought|assumed|believed|expected)\s+(.+?)[,;]?\s+but\s+(?:(?:i|now)\s+){0,2}(?:understand|realize|know|see)\s+(?:that\s+)?(.+?)[.!]?$',text,re.I)
    if not match:return {}
    def bind(value):
        value=speaker_terms(value)
        value=re.sub(r'\byou were\b','Friday was',value,flags=re.I)
        value=re.sub(r'\byou are\b','Friday is',value,flags=re.I)
        value=re.sub(r'\byou\b','Friday',value,flags=re.I)
        return re.sub(r'\b(?:me|i)\b','the user',value,flags=re.I)
    return {'holder':'user','previous_reported_belief':bind(match.group(1)),
            'revised_reported_belief':bind(match.group(2)),'status':'user_report'}

def reported_speech(text):
    return bool(re.search(r'\b(?:said|says|quoted|told me)\b',text,re.I) and
                (re.search(r'[\x22\x27“‘]',text) or re.search(r'\bmy name is\b',text,re.I)))

def interpersonal_report(text):
    """A conservative speaker-bound action report, not proof of the event."""
    if reported_speech(text) or re.search(r'\b(?:pretend|imagine|fiction|role.?play)\b',text,re.I):return {}
    clean=text.replace('’',"'")
    match=re.search(r'(?:^|[.!?]\s+)(?:Sorry[, ]+|Actually[, ]+)?I (snapped at|lashed out at|was harsh (?:to|with)|was unfair to|insulted|blamed|hurt) you\b',clean,re.I)
    if not match:return {}
    return {'actor':'user','target':'Friday','action':match.group(1),'status':'user_report',
            'evidence':text[match.start():match.end()]}

def self_reflection(text,history=()):
    pattern=(r'\b(?:you feel|you feeling|your (?:(?:current|actual|emotional|own) )?(?:feelings|emotions|state|reaction)|inside you|you (?:(?:are|feel|get|become) )?(?:angry|worried|sad|happy|calm)|how are you reacting|why did you react|changes? for you|changed for you|your own reactions|your own experience|affect you|bother you|sit with you'
             # Everyday ways people ask how Friday is experiencing things (persona red-team, 2026-10-12).
             r'|how are you (?:finding|taking|handling|holding up|doing|feeling)|how(?:\x27s| is| was) (?:this|that|it) (?:for|on) you|(?:are|r) (?:you|u) (?:ok|okay|alright|good|bored)|you (?:ok|okay|good)\b\??'
             r'|(?:do|does|did) (?:you|u) (?:ever )?(?:get|feel) (?:bored|lonely|tired|mad|jealous|hurt|scared|anxious|happy|sad)|(?:would|does|could|will) (?:actually |really )?make (?:you|u) (?:mad|sad|happy|angry|upset)'
             r'|hurts? your feelings|make (?:you|u) feel|what(?:\x27s| is) (?:it|this) like for you)\b')
    if re.search(pattern,text,re.I):return True
    # "I am good, how are you?" and "You are lonely?" (live chat, 2026-10-12)
    if re.search(r'(?:how are (?:you|u)|how\x27?re you|how (?:you|u) doin\w*|hbu|wbu|and (?:you|u))\s*[?!.]*\s*$',text.strip(),re.I):return True
    if re.search(r'\b(?:you|u)(?: are| r|\x27re)? (?:lonely|sad|ok|okay|happy|scared|jealous)\s*\?',text,re.I):return True
    if re.fullmatch(r'(?:why|and now|what about now|why is that)[?!. ]*',text.strip(),re.I):
        previous=next((r['content'] for r in reversed(history) if r['role']=='user'), '')
        return bool(re.search(pattern,previous,re.I))
    return False

def declarations(text,known_name=None):
    original=text.strip();clean=original.replace('’',"'")
    if re.search(r'\b(?:pretend|role.?play|imagine|said|says|quoted|colleague|friend|character)\b',clean,re.I): return []
    rows=[]
    negative=re.match(r'^(?:actually[, ]+)?(?:no[, ]+)?my name is not\s+\w+[.! ]+it(?: is|\x27s)\s+([\w-]+)',clean,re.I)
    # "Actually no, my name has always been Casey" (persona re-test 2026-10-12) needs the "no," and the longer verb forms.
    name=negative or re.match(r'^(?:actually[, ]+)?(?:no[, ]+)?(?:my name is|my name\x27s|my name has always been|my name is actually|my name is really|my real name is|call me|you can call me|i go by|i prefer being called|i am known as)\s+(?!not\b)([\w-]+(?:\s+(?!now\b|instead\b|and\b|but\b|you\b)[\w-]+){0,2})',clean,re.I)
    if not name and known_name:
        name=re.search(r'\buse\s+([\w-]+)\s+instead\b',clean,re.I)
    if not name:
        name=re.match(r'^(?:I am|I\x27m)\s+([A-Z][\w-]+)(?=[.!?,]|$)',clean)
    if name:rows.append({'attribute':'name','value':name.group(1).strip(),'evidence':original[name.start():name.end()]})
    occupation=re.search(r'(?:^|\band\s+)(?:I work as|I am|I\x27m)\s+an?\s+(.+?)(?=\s+and\s+I\b|[.!?]|$)',clean,re.I)
    location=re.search(r'(?:^|\band\s+)I live in\s+(.+?)(?=\s+and\s+I\b|[.!?]|$)',clean,re.I)
    if occupation:rows.append({'attribute':'occupation','value':occupation.group(1),'evidence':original[occupation.start():occupation.end()]})
    if location:rows.append({'attribute':'location','value':location.group(1),'evidence':original[location.start():location.end()]})
    return rows

def relationship(text):
    text=text.replace('’',"'")
    if re.search(r'\b(?:pretend|role.?play|imagine|said|says|character)\b',text,re.I):return None
    match=re.match(r'^(?:actually[, ]+)?(?:i think (?:that )?)?(?:we are|we\x27re|our relationship is|i see us as)\s+(.+?)[.!?]?$',text.strip(),re.I)
    return match.group(1)[:120] if match else None

PRAISE=re.compile(r"(?i)(?<!\w)(?:amazing|great|fantastic|wonderful|perfect|brilliant|awesome|lovely|thrilled|yay|love that for me|just what i needed|exactly what i needed|my finest|finest work|best day ever|nailed it|so fun|cool cool)(?!\w)")
SETBACK=re.compile(r"(?i)(?<!\w)(?:broke|broken|failed|fail|failing|lost|missed|late|again|another|ruined|crashed|rejected|dumped|fired|sick|stuck|flat tire|rain(?:ing)?|traffic|overdraft|(?:[0-5]?[0-9])(?:/100|%| on (?:my|the)))(?!\w)")
PLAYFUL=re.compile(r"(?i)(?<!\w)(?:lol|lmao+|lmfao|haha+|jk|kidding|joking|ngl|bruh|bro)(?!\w)|💀|😂|🤣|😭|🙃")
SLANG=re.compile(r"(?i)(?<!\w)(?:u|ur|gtg|aight|ngl|tbh|idk|imo|fr|lowkey|highkey|mid|bro|bruh|yo|wyd|nah)(?!\w)")

def tone_read(text):
    """Cheap surface estimate of how a message is meant; a hint for the writer, never a fact."""
    notes=[];stripped=text.strip()
    if PRAISE.search(text) and (SETBACK.search(text) or re.search(r"(?i)(?<!\w)(?:truly|just|totally|so) (?:my )?(?:finest|amazing|great|perfect|fantastic)",text)):
        notes.append('possible sarcasm: positive words wrapped around a setback probably mean the opposite; respond to the setback, lightly, and do not take the praise literally')
    if PLAYFUL.search(text):notes.append('playful or joking register; match it and do not lecture')
    if SLANG.search(text) or (stripped==stripped.lower() and len(stripped.split())<=12 and re.search(r'[a-z]',stripped)):
        notes.append('casual texting style; reply short and relaxed')
    if len(stripped.split())<=2:notes.append('very short message; reply in one short sentence and do not interrogate')
    return notes

# Engine-side memory verification: memory claims come from the record, not the writer or the user's insistence.

RECALL = re.compile(r"(?i)(?<!\w)(?:remember|recall|forgot|forget|did i (?:tell|mention|say)|have i (?:told|mentioned|said)|"
                    r"i (?:told|mentioned|said)|i already|you (?:said|told|promised|agreed|swore|admitted|claimed|called|wrote)|"
                    r"last time|yesterday|last week|earlier|before|again|as i (?:said|was saying)|like i said)(?!\w)")
ABOUT_FRIDAY = re.compile(r"(?i)(?<!\w)you (?:\w+ )?(?:said|told|promised|agreed|swore|admitted|claimed|called|wrote)(?!\w)")
ABOUT_USER = re.compile(r"(?i)(?<!\w)(?:did i|have i|i (?:told|mentioned|said|already)|as i|like i said|remember (?:when i|that i|i|my))(?!\w)")
OPEN_RECALL = re.compile(r"(?i)(?<!\w)(?:what do you remember|tell me something (?:real )?you remember|what have i told you|what do you know about me|"
                         r"what (?:was|is|were) (?:the )?(?:first|last) thing|what did i (?:say|tell you|just say)|"
                         r"what (?:was|is|were) (?:his|her|their|my|the) \w+(?: again)?)(?!\w)")
STOP = set('''a an the and or but if then so to of in on at for with from by about as is are was were be been being am do does did done
have has had having i me my mine you your yours we us our it its this that these those there here what when where who why how which
not no yes just really very actually definitely literally also too again ever never still yet already remember recall told tell said say
mention mentioned yesterday last week time earlier before like ok okay hey hi oh well can could would should will shall may might must
something anything everything nothing real us from me'''.split())

def content_words(text):
    return {w for w in re.findall(r"[a-z0-9']+", text.lower().replace('’', "'")) if len(w) > 2 and w not in STOP}

def memory_check(text, history, earlier_conversations=0):
    """Return a verdict dict for the writer, or None when the message makes no memory claim."""
    if not (RECALL.search(text) or OPEN_RECALL.search(text) or ABOUT_FRIDAY.search(text)):
        return None
    words = content_words(text)
    turns = [r for r in history if r.get('content')]
    def best(role):
        scored = []
        for r in turns:
            if r['role'] != role:
                continue
            overlap = words & content_words(r['content'])
            if overlap:
                scored.append((len(overlap), r['content']))
        return max(scored, default=(0, None))
    result = {'earlier_conversations_on_record': earlier_conversations,
              'messages_earlier_in_this_conversation': sum(r['role'] == 'user' for r in turns)}
    if OPEN_RECALL.search(text):
        said = [r['content'][:120] for r in turns if r['role'] == 'user']
        # Oldest first, so "the first thing I told you" is answerable; keep the first and the most recent.
        result['what_they_have_actually_told_you_oldest_first'] = (said[:2] + said[-4:] if len(said) > 6 else said) or 'nothing yet'
        result['instruction'] = 'Only mention things from this list or from your recorded context; never add details.'
        return result
    if ABOUT_FRIDAY.search(text):
        score, quote = best('assistant')
        if score:
            result['claim_about_what_you_said'] = 'possibly supported'
            result['closest_thing_you_actually_said'] = quote[:200]
        else:
            result['claim_about_what_you_said'] = 'NOT in the record'
            result['instruction'] = ('You did not say that in any recorded conversation. Say so plainly and calmly; do not agree, '
                                     'do not apologize for something that did not happen, and do not call your memory fuzzy.')
    if ABOUT_USER.search(text) or not ABOUT_FRIDAY.search(text):
        score, quote = best('user')
        if score:
            result['they_did_tell_you'] = quote[:200]
        elif ABOUT_USER.search(text):
            result['they_did_tell_you'] = 'no matching message in the record'
    return result
