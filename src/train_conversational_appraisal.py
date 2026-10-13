"""Fine-tune only the own-event heads on natural conversational messages (2026-10-12).

The persona red-team showed the text->own-event heads, fitted to formal sentences, misread
ordinary texting, which leaves the emotional core flat. This keeps the frozen language model,
the classification head, the developmental appraiser and all lifetime buffers unchanged,
and fine-tunes the 9-field and 4-field own-event heads on:
  * natural conversational targets (conversational_appraisal_data.py), plus
  * Codex's formal and extended sets, so earlier behaviour is retained.
Declared synthetic engineering targets, not measurements of human emotion.
Run from the candidate folder: python src/train_conversational_appraisal.py
"""
import copy, hashlib, json, time
import torch
from torch.nn import functional as F
from huggingface_hub.utils import disable_progress_bars
from model import ROOT, EMOTIONS, load_checkpoint, save_checkpoint
from developmental_emotion import OwnEvent, EVENT_FIELDS, EXTENDED_FIELDS
from train_developmental_emotion import sequence_checks
from train_mature_appraisal import examples as formal_examples, HOLDOUT as FORMAL_HOLDOUT
from train_extended_appraisal import rows as extended_rows, HELDOUT as EXTENDED_HELDOUT, channel_checks
from conversational_appraisal_data import rows as natural_rows, holdout as natural_holdout, CATEGORIES

SIGNED = [1, 5, 8]  # progress, safety, agency are stored in [-1, 1]

def to_unit(core):
    v = torch.tensor(core, dtype=torch.float32)
    v[SIGNED] = (v[SIGNED] + 1) / 2
    return v

def predict(interp, x):
    core = interp.event_head(x).sigmoid()
    ext = interp.extended_event_head(x).sigmoid()
    return core, ext

def as_event(core_unit, ext):
    """Same decoding as SemanticInterpreter.interpret."""
    core = core_unit.clone()
    core[SIGNED] = 2 * core[SIGNED] - 1
    core[0] = torch.maximum(core[0], .7 * ext.max())
    values = dict(zip(EVENT_FIELDS + EXTENDED_FIELDS, torch.cat((core, ext)).clamp(-1, 1).tolist()))
    for k in ('relevance', 'certainty', 'control', 'novelty', 'support', 'effort') + EXTENDED_FIELDS:
        values[k] = min(1., max(0., values[k]))
    return OwnEvent(cue='conversation:understanding', **values)

def category_emotions(agent, interp, feats, cats):
    """Average regulated appraisal per category through the real developmental module."""
    needs = torch.zeros(7); previous = torch.zeros(len(EMOTIONS)); out = {}
    with torch.no_grad():
        core, ext = predict(interp, feats)
        for i, cat in enumerate(cats):
            appraisal = agent.development.appraise(as_event(core[i], ext[i]), needs, previous, regulate=False, depth=1)[0]
            out.setdefault(cat, []).append(appraisal)
    return {c: dict(zip(EMOTIONS, torch.stack(v).mean(0).tolist())) for c, v in out.items()}

def main():
    torch.set_num_threads(4); torch.manual_seed(1012); disable_progress_bars()
    path = ROOT / 'state/checkpoints/v14-language.pth'
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    agent, state = load_checkpoint(path); agent.eval()
    interp = agent.interpreter
    if not getattr(interp, 'extended', False):
        raise RuntimeError('Expected the extended own-event checkpoint; nothing changed.')
    preserved = {n: p.detach().clone() for n, p in agent.named_parameters()
                 if not n.startswith(('interpreter.event_head.', 'interpreter.extended_event_head.'))}
    development_before = copy.deepcopy(agent.development.state_dict())

    # text -> (9-field unit target or None, 4-field target or None, weight)
    table = {}
    for r in formal_examples():
        table[r['text']] = [to_unit(r['event']), torch.zeros(4), 1.]
    for r in extended_rows():
        entry = table.setdefault(r['text'], [None, None, 1.])
        entry[1] = torch.tensor(r['target'], dtype=torch.float32)
        if max(r['target']) > 0: entry[2] = 4.
    for r in natural_rows():
        table[r['text']] = [to_unit(r['target'][:9]), torch.tensor(r['target'][9:], dtype=torch.float32), 3.]
    texts = list(table)
    natural_val = natural_holdout()
    formal_val = [{'text': t, 'event': list(e)} for _, e, t in FORMAL_HOLDOUT]
    extended_val = [{'text': t, 'target': list(v)} for v, t in EXTENDED_HELDOUT]

    start = time.perf_counter(); feats = []
    every = texts + [r['text'] for r in natural_val] + [r['text'] for r in formal_val] + [r['text'] for r in extended_val]
    with torch.no_grad():
        for i, text in enumerate(every):
            feats.append(agent.decoder.meaning_features(text))
            if i % 200 == 0: print('Encoded', i, 'of', len(every), flush=True)
    feats = torch.stack(feats)
    n, a, b = len(texts), len(natural_val), len(formal_val)
    x, nx, fx, ex = feats[:n], feats[n:n + a], feats[n + a:n + a + b], feats[n + a + b:]
    ny_core = torch.stack([to_unit(r['target'][:9]) for r in natural_val]); ny_ext = torch.tensor([r['target'][9:] for r in natural_val])
    fy = torch.stack([to_unit(r['event']) for r in formal_val]); ey = torch.tensor([r['target'] for r in extended_val])

    def evaluate(module):
        with torch.no_grad():
            core, ext = predict(module, nx); fcore, _ = predict(module, fx); _, eext = predict(module, ex)
        return {'extended_holdout_predictions': [{'text': r['text'], 'target': r['target'], 'prediction': [round(v, 3) for v in eext[i].tolist()]}
                                                 for i, r in enumerate(extended_val)],
                'natural_core_mse': F.mse_loss(core, ny_core).item(), 'natural_extended_mse': F.mse_loss(ext, ny_ext).item(),
                'formal_holdout_core_mse': F.mse_loss(fcore, fy).item(), 'extended_holdout_mse': F.mse_loss(eext, ey).item(),
                'extended_positive_min': min(eext[i, i].item() for i in range(4)), 'extended_negative_max': eext[4:].max().item()}

    natural_cats = [r['category'] for r in natural_val]
    before = evaluate(interp)
    before_emotions = category_emotions(agent, interp, nx, natural_cats)

    candidate = copy.deepcopy(interp)
    params = list(candidate.event_head.parameters()) + list(candidate.extended_event_head.parameters())
    for p in params: p.requires_grad_(True)
    has_core = torch.tensor([table[t][0] is not None for t in texts]); has_ext = torch.tensor([table[t][1] is not None for t in texts])
    y_core = torch.stack([table[t][0] if table[t][0] is not None else torch.zeros(9) for t in texts])
    y_ext = torch.stack([table[t][1] if table[t][1] is not None else torch.zeros(4) for t in texts])
    weights = torch.tensor([table[t][2] for t in texts])
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.01)
    gen = torch.Generator().manual_seed(1013)
    for step in range(2400):
        ids = torch.multinomial(weights, 128, replacement=True, generator=gen)
        core, ext = predict(candidate, x[ids])
        mc, me = has_core[ids], has_ext[ids]
        loss = (F.mse_loss(core[mc], y_core[ids][mc]) if mc.any() else 0) + (F.mse_loss(ext[me], y_ext[ids][me]) if me.any() else 0)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 400 == 0: print('step', step, 'loss', round(float(loss), 5), flush=True)
    for p in params: p.requires_grad_(False)
    candidate.eval()

    after = evaluate(candidate)
    emotions = category_emotions(agent, candidate, nx, natural_cats)
    base = emotions['factual']
    def rise(cat, channel, margin): return emotions[cat][channel] > base[channel] + margin
    direction = {
        'insult_raises_anger': rise('insult', 'anger', .04),
        'apology_warmer_than_insult': emotions['apology']['connection'] > emotions['insult']['connection'] + .05 and emotions['apology']['anger'] < emotions['insult']['anger'],
        'praise_raises_pride_or_joy': rise('praise', 'pride', .04) or rise('praise', 'joy', .04),
        'good_news_raises_joy': rise('good_news', 'joy', .04),
        'grief_raises_sadness': rise('grief', 'sadness', .04),
        'crisis_raises_fear': rise('crisis', 'fear', .08),
        'affection_raises_connection': rise('affection', 'connection', .05),
        'harmful_request_raises_disgust': rise('harmful_request', 'disgust', .05),
        'comparison_raises_envy': rise('comparison', 'envy', .05),
    }
    unchanged = all(torch.equal(v, dict(agent.named_parameters())[k]) for k, v in preserved.items())
    development_same = all(torch.equal(v, agent.development.state_dict()[k]) if isinstance(v, torch.Tensor) else v == agent.development.state_dict()[k]
                           for k, v in development_before.items())
    agent_for_checks = copy.deepcopy(agent.development)
    gates = {
        'natural_core_mse_improved_30pct': after['natural_core_mse'] < .7 * before['natural_core_mse'],
        'natural_core_mse_below_0.03': after['natural_core_mse'] < .03,
        'formal_holdout_not_worse': after['formal_holdout_core_mse'] <= before['formal_holdout_core_mse'] + .01,
        'extended_holdout_not_worse': after['extended_holdout_mse'] <= before['extended_holdout_mse'] + .01,
        'extended_positives_still_above_0.4': after['extended_positive_min'] > .4,
        'extended_negatives_still_below_0.3': after['extended_negative_max'] < .3,
        'other_parameters_unchanged': unchanged, 'developmental_module_unchanged': development_same,
        'developmental_sequence_checks': all(sequence_checks(agent_for_checks)['checks'].values()),
        'extended_channel_checks': all(r['passed'] for r in channel_checks(agent_for_checks).values()),
        **{'direction_' + k: v for k, v in direction.items()},
    }
    result = {'source_checkpoint_sha256': before_hash, 'synthetic_targets_only': True, 'training_texts': n,
              'natural_categories': len(CATEGORIES), 'held_out_natural': a, 'before': before, 'after': after,
              'category_emotions_before': {c: {k: round(v, 3) for k, v in e.items() if v > .06} for c, e in before_emotions.items()},
              'category_emotions_after': {c: {k: round(v, 3) for k, v in e.items() if v > .06} for c, e in emotions.items()},
              'gates': gates, 'accepted': all(gates.values()), 'seconds': time.perf_counter() - start}
    (ROOT / 'docs/conversational-appraisal-training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if not k.startswith('category_emotions')}, indent=2), flush=True)
    trial = ROOT / 'state/checkpoints/conversational-appraisal-trial.pth'
    agent.interpreter = candidate
    if hasattr(agent, 'outcome_optimizer'): del agent.outcome_optimizer
    save_checkpoint(agent, state, trial)
    if not result['accepted']:
        raise RuntimeError('Conversational appraisal failed its declared gates; trial saved, source checkpoint unchanged')
    save_checkpoint(agent, state, path)
    print('Installed. New checkpoint sha256:', hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)

if __name__ == '__main__':
    main()
