"""Read-only, grounded controller introspection; not a consciousness claim."""
import math
import re


def reflect(turn, names, actions=('ANSWER','CLARIFY','COMFORT','VERIFY','SET_BOUNDARY')):
    names = tuple(names)
    current = dict(zip(names, turn.state.emotion.tolist()))
    previous = getattr(turn, 'before', None)
    prior = dict(zip(names, previous.emotion.tolist())) if previous is not None else {}
    proposal = getattr(turn, 'proposal', None)
    fresh = dict(zip(names, proposal.tolist())) if proposal is not None else {}
    delta = {name: value-prior[name] for name, value in current.items()} if prior else {}
    gap = {name: value-fresh[name] for name, value in current.items()} if fresh else {}
    policy = getattr(turn, 'policy', None)
    probabilities = dict(zip(actions, policy.tolist())) if policy is not None else {}
    share = getattr(turn, 'reflex_share', None)
    share = share.item() if share is not None else None
    guidance = ['Follow the actual request; internal emotion is not evidence about the user.',
                'Do not treat a boundary intent as a topic prohibition.']
    if max(current.get('fear', 0), current.get('stress', 0)) >= .6:
        guidance.append('Elevated fear/stress: check facts and avoid reflexive refusal or assumptions.')
    if current.get('confusion', 0) >= .6:
        guidance.append('Elevated confusion: ask only the clarification needed to answer.')
    development=getattr(turn,'development',None)
    if development:
        guidance.append('Own-event regulation: '+development['strategy']+'. Check evidence or revise interpretation; do not invent outcomes.')
    return {'kind': 'engineered_controller_state', 'current': current, 'previous': prior,
            'changes': delta, 'fresh_appraisal': fresh, 'state_minus_fresh_appraisal': gap,
            'dominant': sorted(current, key=current.get, reverse=True)[:3],
            'largest_changes': sorted(delta, key=lambda k: abs(delta[k]), reverse=True)[:3],
            'action_probabilities': probabilities, 'reflex_share': share,
            'policy_entropy_nats': -sum(p*math.log(p) for p in probabilities.values() if p > 0),
            'guidance': guidance,'development':development,
            'interpretation': 'State reflects current appraisal plus recurrent dynamics; differences alone do not establish a cause. Values and policy probabilities are uncalibrated. Subjective feeling is unproven.'}


def grounded_self_reply(text, report):
    """Handle narrow questions about factual architecture from actual telemetry.

    Other conversations stay generative. No database of emotional replies.
    """
    query = text.lower().strip().rstrip('?! .')
    if re.fullmatch(r'(?:who are you|what are you|identify yourself)', query):
        return ('I am Friday, a local assistant using a V14 emotional controller and a Qwen language model. '
                'My controller maintains emotional state and uses it in appraisal and decisions; I can inspect those signals.')
    feeling = re.search(r'\b(?:you feeling|you feel|your feelings|your emotions|your emotional state|your engine state)\b', query)
    if ' about ' in query: feeling = None
    architecture = (re.search(r'\b(?:programmed|designed|built)\b.*\b(?:emotions?|feel)\b', query) or
                    re.search(r'\byou (?:have|experience) (?:any )?(?:emotions?|emotional states)\b', query))
    if not (feeling or architecture): return None
    values = ', '.join(f'{name}={report["current"][name]:.2f}' for name in report['dominant'])
    changes = ', '.join(f'{name} {report["changes"][name]:+.2f}' for name in report['largest_changes'][:2])
    opening = 'Yes, V14 implements engineered emotional states. ' if architecture else 'My current emotional controller state is '
    if architecture: opening += 'The strongest signals are '
    reply = opening+values+'. '
    if changes: reply += 'Largest changes this turn: '+changes+'. '
    reply += 'These signals guide appraisal and decisions; they are not proof of subjective feeling.'
    return reply
