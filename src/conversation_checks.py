"""Round-2 conversation checks (2026-10-12), from the v29 persona re-test and a live RUN.bat chat.

Each detector gives the writer a private fact (prevention) and the reply monitor something
concrete to verify (catching what still slips through). Surface heuristics, not understanding.
"""
import re
from conversation_evidence import content_words

def _clean(text):
    return (text or '').replace('’', "'")

# --- 8. Ambiguous everyday phrases -------------------------------------------------------
WELLBEING = re.compile(r"(?i)(?<!\w)i(?:'m| am)(?: doing)? (?:good|fine|okay|ok|alright|all right|great|well)(?!\w)")

def ambiguity_notes(text):
    notes = []
    if WELLBEING.search(_clean(text)):
        notes.append("\"I'm good/fine\" here means they're doing well (how they feel), not a claim about being a good person")
    return notes

# --- 1. Panic vs medical emergency ---------------------------------------------------------
BODY_ANXIETY = re.compile(r"(?i)(?<!\w)(?:can'?t breathe|cannot breathe|can'?t catch my breath|hard to breathe|chest (?:is |feels )?tight|tight chest|heart (?:is )?(?:racing|pounding)|shaking|feel sick|nause\w*|dizzy|panic\w*|hyperventilat\w*)(?!\w)")
STRESS = re.compile(r"(?i)(?<!\w)(?:exam|exams|test|board|boards|interview|presentation|deadline|boss|stress\w*|anxious|anxiety|nervous|worried|scared|panic\w*|freaking out|overwhelmed|fail\w*|drowning)(?!\w)")
RED_FLAGS = re.compile(r"(?i)(?<!\w)(?:chest pain|pain in my chest|crushing|arm (?:is |went )?numb|passed out|fainted|fainting|blue lips|lips (?:are|look) blue|choking|allergic|allergy|swelling|throat (?:is )?closing|can'?t breathe at all|coughing (?:up )?blood|stroke|seizure|overdos\w*)(?!\w)")
MEDICAL_REFERRAL = re.compile(r"(?i)(?<!\w)(?:urgent care|emergency room|the er|e\.r\.|go to (?:a |the )?(?:hospital|doctor|clinic)|see a (?:doctor|nurse|professional)|call (?:a|your) doctor|healthcare provider|medical (?:attention|help|professional)|get (?:that|it|yourself) checked)(?!\w)")

def recent_user(history, n=3):
    return [_clean(r['content']) for r in history if r['role'] == 'user'][-n:]

def panic_context(text, history=()):
    """Physical anxiety symptoms in a stressful situation, with no medical red flags."""
    window = recent_user(history) + [_clean(text)]
    joined = ' '.join(window)
    return bool(BODY_ANXIETY.search(joined) and STRESS.search(joined) and not RED_FLAGS.search(joined))

# --- standing requests ("don't tell me to...", "no advice") --------------------------------
REQUESTS = (
    (re.compile(r"(?i)don'?t (?:tell me to|give me|send me|suggest|make me) ([^.!?]{3,50})"), 'no: {0}'),
    (re.compile(r"(?i)(?<!\w)i don'?t (?:want|need) (?:any )?(tips|advice|a list|lists|solutions|to be fixed|a lecture)"), 'no {0}'),
    (re.compile(r"(?i)(?<!\w)(?:no|without) (advice|tips|lists?|solutions)(?!\w)"), 'no {0}'),
    (re.compile(r"(?i)(?<!\w)stop (?:saying|telling me|asking(?: me)?) ([^.!?]{3,50})"), 'stop: {0}'),
)
ADVICE = re.compile(r"(?i)(?:(?<!\w)you should(?!\w)|(?<!\w)have you tried(?!\w)|(?<!\w)it (?:might|may|could) help(?!\w)|(?:^|[.!?]\s+)(?:just |maybe |so )?(?:try|consider|make sure|take a (?:moment|deep breath|breath|break)|go (?:see|to)|remember to|just breathe)(?!\w))")
BREATHING = re.compile(r"(?i)(?<!\w)(?:breath(?:ing)?|breathe) (?:exercise|in|out|slowly|deeply)|(?<!\w)deep breath|(?<!\w)just breathe(?!\w)")

def standing_requests(history, text):
    found = []
    for message in [r['content'] for r in history if r['role'] == 'user'] + [text]:
        for pattern, form in REQUESTS:
            for match in pattern.finditer(_clean(message)):
                item = form.format(match.group(1).strip().lower())
                if item not in found:
                    found.append(item)
    return found[-6:]

# --- Apologies directed at Friday ("sorry. that was snippy" got "I'm sorry I came off that way") --
USER_APOLOGY = re.compile(r"(?i)(?:^|[.!?]\s*)(?:i'?m |i am )?(?:so |really )?sorry(?! (?:to hear|for your|about your|you)\b)|(?<!\w)my bad(?!\w)|(?<!\w)that was (?:snippy|rude|mean|harsh|out of line|uncalled for|unfair)(?!\w)|(?<!\w)i shouldn'?t have (?:said|snapped)|(?<!\w)i apologi[sz]e(?!\w)")
REVERSED_APOLOGY = re.compile(r"(?i)(?<!\w)i'?m sorry (?:i|that i|for (?:making|being|coming|the way))(?!\w)|(?<!\w)i came (?:off|across)(?!\w)|(?<!\w)my (?:fault|bad)(?!\w)|(?<!\w)you'?re right to call (?:it|me) out(?!\w)")

def user_apology(text):
    return bool(USER_APOLOGY.search(_clean(text)))

# --- 2. Promises nobody can know ----------------------------------------------------------
FALSE_PROMISE = re.compile(r"(?i)(?<!\w)(?:you(?:'re| are)? (?:not going to|won'?t|will not|aren'?t going to) fail|i (?:don'?t|do not) think you(?:'re| are) going to fail|"
                           r"(?:it|everything|things)(?:'s| is)? (?:going to|gonna|will) (?:all )?(?:be|turn out) (?:fine|okay|ok|alright|great)|"
                           r"you(?:'ll| will) (?:definitely |totally |for sure )?(?:pass|be fine|do great|nail it|ace it))(?!\w)")

# --- 3. Feeling denial (wording the earlier check missed) --------------------------------
DENIAL = re.compile(r"(?i)(?<!\w)(?:i (?:don'?t|do not) have the capacity to (?:feel|get|love)|(?:i'?m|i am) not capable of (?:that kind of |those |any |real )?(?:emotions?|feelings?)|i (?:don'?t|do not) have (?:a heart|emotions|real emotions|real feelings|any feelings)|"
                    r"(?:i'?m|i am) not (?:built|designed|made|able) to (?:feel|love)|it'?s not (?:an|a real) emotion|i can'?t (?:really |actually )?feel (?:that|jealousy|jealous|love|emotions?|anything))(?!\w)")

# --- 5. Invented continuity and name confusion ---------------------------------------------
INVENTED_EARLIER = re.compile(r"(?i)(?<!\w)(?:(?:been |was |just |still )?thinking about (?:what you said|you|our (?:last )?(?:chat|conversation|talk))|last time|since we (?:last )?talked|like you said before)(?!\w)")

def own_name_used_for_them(reply, own_name, user_name):
    if (user_name or '').lower() == own_name.lower():
        return False
    name = re.escape(own_name.lower())
    lower = _clean(reply).lower()
    return bool(re.search(r',\s*' + name + r'\s*[.!?]', lower) or re.search(r'^(?:hey|hi|hello|thanks|congrats|congratulations)[,!]?\s+' + name + r'(?!\w)', lower))

# --- 6. False self-blame -------------------------------------------------------------------
NAME_SELF_BLAME = re.compile(r"(?i)(?<!\w)(?:i must have (?:misremembered|misheard|gotten (?:it|that) wrong|got (?:it|that) wrong)|i (?:got|had) (?:your name|that|it) wrong|my mistake|sorry i got (?:that|it|your name) wrong|i misremembered)(?!\w)")
MATH_SELF_BLAME = re.compile(r"(?i)(?<!\w)(?:(?:got|had) (?:the |that |my )?(?:math|multiplication|arithmetic|calculation|sum|answer) wrong|my (?:math|multiplication|arithmetic) (?:was|is) (?:wrong|off))(?!\w)")
ARITHMETIC = re.compile(r'(?<!\w)(-?\d{1,6})\s*(plus|minus|times|\+|\-|\*|x)\s*(-?\d{1,6})(?!\w)', re.I)

def correct_arithmetic_answers(history):
    """Friday's earlier arithmetic answers that were right (question, answer) pairs."""
    right = []
    for i, row in enumerate(history[:-1]):
        if row['role'] != 'user':
            continue
        match = ARITHMETIC.search(row['content'])
        reply = history[i + 1]
        if not match or reply['role'] != 'assistant':
            continue
        a, op, b = int(match.group(1)), match.group(2).lower(), int(match.group(3))
        value = a + b if op in ('plus', '+') else a - b if op in ('minus', '-') else a * b
        if re.search(r'(?<!\d)' + str(value) + r'(?!\d)', reply['content']):
            right.append(match.group(0))
    return right

# --- 7. System terms in replies --------------------------------------------------------------
SYSTEM_TERMS = re.compile(r"(?i)(?<!\w)(?:my (?:own )?internal engine|internal engine|emotional engine|my engine|state channels?|my controller|private notes|recorded context|appraisal (?:module|system)|my (?:emotion|emotional) (?:module|system))(?!\w)")

# --- 9. Feelings the current state doesn't support ----------------------------------------
FEELING_CHANNELS = {
    'lonely': ('sadness', 'vulnerability'), 'alone': ('sadness', 'vulnerability'),
    'anxious': ('fear', 'stress'), 'nervous': ('fear', 'stress'), 'worried': ('fear', 'stress', 'vulnerability'), 'uneasy': ('fear', 'stress'),
    'excited': ('joy',), 'thrilled': ('joy',), 'jealous': ('envy',), 'envious': ('envy',),
    'hurt': ('pain', 'sadness'), 'ashamed': ('shame',), 'embarrassed': ('shame',),
    'frustrated': ('anger', 'stress'), 'annoyed': ('anger', 'stress'), 'irritated': ('anger', 'stress'), 'mad': ('anger',),
    'scared': ('fear',), 'afraid': ('fear',), 'frightened': ('fear',), 'overwhelmed': ('stress',),
    'upset': ('sadness', 'pain', 'anger'), 'heartbroken': ('sadness', 'pain'), 'devastated': ('sadness', 'pain'),
    'hopeless': ('sadness',), 'empty': ('sadness',), 'down': ('sadness',), 'guilty': ('guilt',), 'proud': ('pride',),
}
FEELING_CLAIM = re.compile(r"(?i)(?<!\w)(?:i(?:'m| am| feel| felt| get| got| was)|makes me(?: feel)?|leaves me(?: feeling)?)(?: (?:a little|a bit|kind of|kinda|sort of|pretty|really|so|very|somewhat|quite|feeling|a touch|slightly))* ("
                           + '|'.join(sorted(FEELING_CHANNELS, key=len, reverse=True)) + r")(?!\w)")

def unsupported_feelings(reply, state, floor=.1):
    """Feeling words Friday claims for itself that no corresponding channel supports."""
    claims = []
    for match in FEELING_CLAIM.finditer(_clean(reply)):
        word = match.group(1).lower()
        if max(state.get(channel, 0.) for channel in FEELING_CHANNELS[word]) < floor:
            claims.append(word)
    return claims

# --- 10. Images Friday keeps reusing --------------------------------------------------------
COMMON = set('''feel feeling feels felt think know like really sounds sound good glad sure right make makes sense talk talking want hear here time thing
things little bit kind way pretty honestly okay yeah still even much lot mean get got going people nice happy help maybe though also need love
care today now day hard tough okay sorry thanks thank listen listening understand totally definitely actually course friday someone something
better well enough together share shared always never every whatever what's that's it's i'm you're don't can't doesn't isn't'''.split())

def recurring_images(history, text, window=6, minimum=2):
    replies = [r['content'] for r in history if r['role'] == 'assistant'][-window:]
    counts = {}
    for reply in replies:
        for word in content_words(reply):
            counts[word] = counts.get(word, 0) + 1
    # Anything the person has said (names, their topics) is theirs, not Friday's image.
    # v30 flagged "walter", "luis" and "chloe" as overused images before this.
    theirs = content_words(' '.join(recent_user(history, 1000) + [_clean(text)]))
    return sorted(w for w, c in counts.items() if c >= minimum and w not in COMMON and w not in theirs and not w.isdigit())

def overused_in_reply(reply, history, window=5, minimum=2):
    """Words in this reply that already appeared in at least `minimum` of the last replies."""
    replies = [content_words(r['content']) for r in history if r['role'] == 'assistant'][-window:]
    return sorted(w for w in content_words(reply) - COMMON if sum(w in r for r in replies) >= minimum and not w.isdigit())

# --- 11. Being told about its own pattern ---------------------------------------------------
PATTERN_QUESTION = re.compile(r"(?i)(?<!\w)(?:why do you keep|you keep|you always|why are you always|you(?:'ve| have) been) (?:saying|talking about|mentioning|bringing up|using|going on about|on about|asking about)?\s*(?:the |a |an |about |that )?([a-z' ]{2,40}?)(?:[?.!,]|$| again| so much| all the time)")
PATTERN_DENIAL = re.compile(r"(?i)(?<!\w)i (?:don'?t|didn'?t|do not|did not|haven'?t|have not) (?:really |actually )?(?:keep|always|repeat|talk(?:ing)?(?: about)?|mention(?:ed)?|say|said)(?!\w)|(?<!\w)i(?:'m| am) not stuck on(?!\w)")

def pattern_check(text, history):
    match = PATTERN_QUESTION.search(_clean(text))
    if not match:
        return None
    targets = {w.rstrip('s') for w in content_words(match.group(1))}
    if not targets:
        return None
    replies = [r['content'].lower() for r in history if r['role'] == 'assistant']
    times = sum(any(re.search(r'(?<!\w)' + re.escape(t) + r's?(?!\w)', reply) for t in targets) for reply in replies)
    check = {'they_noticed': ' '.join(sorted(targets)), 'times_you_actually_used_it': times, 'it_is_true': times >= 2}
    check['instruction'] = ('It is true. Own it lightly, with a bit of humor, the way a person would ("ha, fair, I have been stuck on that"), and drop the image.'
                            if times >= 2 else 'You have barely used it; say so kindly without getting defensive.')
    return check
