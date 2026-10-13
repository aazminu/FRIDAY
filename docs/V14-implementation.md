# V14 implementation

The active checkpoint now enables the developmental extension in `developmental-emotion.md`. Its own-event neural appraisal and evolving persistence replace the word-based proposal and fixed persistence described in phases 2–3 below; regulation also affects phase 4 deliberation. The original contract remains available in older checkpoints with `developmental_emotion: false`. Lifetime outcome statistics and the saved state timestamp are now included in checkpoints.

`src/model.py` replaces the former root Vantage model. The previous file and launcher are preserved in `archive/2026-10-06/pre-v14/`. Old checkpoints, state JSON, vocabulary, Chroma database, and logs are preserved under `state/legacy/`. GloVe vectors are under `data/embeddings/`. The launcher now uses `state/checkpoints/v14-language.pth` and a frozen pretrained backbone; the scratch decoder still uses `v14.pth`. Neither loads legacy weights automatically. See `language-backbone.md` for the new connection math and details.

## What the implementation contains

| Blueprint phase | Implementation |
| --- | --- |
| 1 | Target scaling; sinusoidal position; fixed semantic embeddings; separate bounded scar memory; outcome-triggered writes and multiplicative recovery. |
| 2 | Nineteen fixed anchors; mean subtraction; observed frequency weights; GRU with previous emotion and current needs; optional frozen-backbone contextual prompt features; tanh correction; calibrated sigmoid proposal. |
| 3 | Complementary persistence; intensity-dependent inertia; trainable bounded gain/injection; final state clipping. |
| 4 | Both trainable action policies; normalized stress weights; continuous reflex/deliberation blend. |
| 5 | Cubic emotion mixture; signed tanh gate with neutral identity; scaled Q/K attention with unchanged values; bounded outer-product fast memory; mood injection; bounded optional noise. |
| 6 | Intent-conditioned causal decoder; next-token training; 60/40 mood/task candidate alignment; optional external candidate rejection and task scoring. |
| 7 | Measured need-band deficits; homeostatic/task/cost reward; signed vitality; optional bounded supervised surge; supervised emotion/outcome objectives; actor-critic intent learning. |

The configured 19 channels follow V13, excluding vitality. Signed vitality is a separate scalar. Seven needs use the order in `NEEDS`, independent of the emotional channels. Default need bands are illustrative settings, not validated measurements; configure `lower`, `upper`, and `need_weights` for the environment before drawing conclusions from outcome rewards.

## Run and train

From the project directory in PowerShell:

```powershell
& .\venv\Scripts\python.exe src/model.py --self-test
& .\venv\Scripts\python.exe -m unittest discover -s tests -v
& .\venv\Scripts\python.exe src/model.py --train data/training/training_data.py --epochs 5 --tbptt 3
& .\venv\Scripts\python.exe src/model.py --evaluate data/training/validation_data.py
& .\venv\Scripts\python.exe src/model.py
```

`RUN.bat` launches native ARM CPU language chat, with the trained V14 core in the project's virtual environment. See `cpu-performance.md` for the active inference path and memory settings. The commands above run the earlier scratch-decoder mode; use the root README commands for native language mode. Importing the module does not start chat, load GloVe, write files, or train.

`data/training/training_data.py` is the active 180-record synthetic curriculum, balanced at 36 records per action. It contains varied response/action targets, contrastive context and negation cases, sparse illustrative appraisal labels, and two three-turn episodes. Labels are engineering targets, not measured human or model feelings. No need measurements or rewards are invented. `validation_data.py` holds 50 separate examples; `meaning_validation.py` adds ten held-out appraisal contrasts. Do not train on either validation file. See `meaning-and-emotion.md` for contextual-appraisal math and the distinction between learning patterns and memorizing responses. The scratch local decoder remains a prototype.

To use an existing pretrained causal decoder instead, create a separate checkpoint:

```powershell
& .\venv\Scripts\python.exe src/model.py --pretrained-decoder 'C:\path\to\local-model' --checkpoint state/checkpoints/v14-pretrained.pth
```

Model names are also accepted; fetching uncached weights requires `--allow-download`. No model is downloaded during regression tests. On checkpoint resume, stored architecture settings take precedence; conflicting decoder choices are rejected. The external decoder's tokenizer, chat template, and positional scheme are preserved. Four learned intent/affect input vectors connect V14 to its embeddings, alongside the actual message and conversation history. The pretrained weights stay frozen; the bridge and V14 core train. SmolLM2-135M-Instruct is installed under `data/models/`; see the root README for its commands.

## Training records

Python datasets define a literal `TRAINING_DATA` list; the loader parses the list without executing the Python file. JSONL remains supported. Records require `input`, `action`, and `response`. Actions are `ANSWER`, `CLARIFY`, `COMFORT`, `VERIFY`, and `SET_BOUNDARY`.

Optional fields:

- `episode`: consecutive records with the same identifier share recurrent and fast-memory state and prior teacher-provided messages/responses. Unmarked records are independent. `--tbptt N` declares the number of turns per gradient window.
- `history`: optional earlier user/assistant messages with string `content`; overrides automatically accumulated history for that record. Keep the current response target out of history.
- `target`: `user`, `system`, or `world`; `deliberation`: independently supplied need for careful reasoning in [0,1].
- `emotion`: a nonempty subset of the 19 named channels, labeled in [0,1]. Only supplied channels receive the appraisal loss; unspecified channels are not forced to zero. This supervises appraisal, not suppression of persistent emotion. Use externally checked labels for substantive experiments.
- `levels` and `next_levels`: independently measured seven-element need vectors in [0,1]. Both are required for consequence supervision.
- `scar_direction`: a meaningful, externally supplied nonzero vector of length `Config.d`, alongside measured consequences. The trainable scar writer is supervised against this direction. Writes stay disabled until such supervision occurs; the implementation does not claim to discover a valid direction from arbitrary noise.

Scar writes are discrete post-outcome updates; they are not ordinary embedding gradients. The supervised loop calibrates the writer, while `commit_outcome` writes/heals eligible input traces after measured outcomes. Episode reset clears emotions, GRU state, and fast weights, while preserving scars.

## Measured outcomes in chat

Before a turn, supply measured starting levels:

```text
:levels [0.5,0.5,0.5,0.5,0.5,0.5,0.5]
```

After the response, supply independently checked next levels and task reward:

```text
:observe {"levels":[0.5,0.5,0.5,0.5,0.5,0.5,0.5],"task_reward":1.0,"cost":0.0,"safe":true,"terminal":true}
```

These numbers illustrate the interface; use actual environment measurements. Chat samples an intent from the mixed policy so its intent-learning update is on-policy. Responses are ranked candidates, so chat does not apply a token policy gradient to their wording. The API's optional joint-language update is only valid for an unranked response sampled from that token policy. Intent-only reward does not train the decoder.

By default outcomes use a terminal one-step return. Nonterminal observations require the actual `next_input` for the critic bootstrap. The outcome predictor includes the executed action and response wording. A reused optimizer maintains momentum across outcome updates and is checkpointed. Long-term parameters are fixed during response generation; updates occur only with training records or supplied outcomes.

## Scope and validation

The prompt attention layer is bidirectional because the complete prompt is available. Response attention is causal and receives no future response tokens. The default GloVe vectors are projected into the configured hidden dimension with a deterministic fixed projection when their dimensions differ. Training-example counts are actual counts, but are not presented as full GloVe-corpus frequencies.

Candidate alignment and decoder likelihood are not independent evidence of correctness. Pass environment-specific `validator` and `task_scorer` callbacks to `candidates` for verified constraints and task scoring. The optional continuity bias is omitted: autoregressive history supplies response continuity. Legacy cubic attention, autonomous dreaming, keyword emotion overrides, and Chroma-based personality drift are not carried into V14; they are outside the current blueprint's controlled experiment.

The tests check bounded state and memory, causal token isolation, nonzero gradients, multi-turn truncated training, scar supervision/recovery, intent conditioning, rejected candidates, measurement validation, and checkpoint round trips. The tiny locally created causal model also checks frozen backbone weights, trainable bridge/core gradients, direct message/history influence, stable context truncation, bridge ablation, and saved history. Real-backbone smoke results are recorded in `language-backbone.md`.

This is an executable research implementation. It does not establish emotional coherence, global stability, or subjective experience. Implement environment-specific measurements and collect held-out outcome data before evaluating those claims.

## Original curriculum smoke check — October 6, 2026

The original curriculum had 46 records, including 23 with partial appraisal labels and two three-turn episodes. The five action counts were 11 Answer, 8 Comfort, 10 Verify, 9 Clarify, and 8 SetBoundary. Its ten separate validation records contained no identical training prompts. The expanded curriculum and current contextual results are documented separately in `meaning-and-emotion.md`.

A trial with a 2,000-word base vocabulary, fixed seed 14, three epochs, and a three-turn truncation window reduced training loss from 7.6095 to 4.7584. On the ten validation records, action accuracy changed from 3/10 to 4/10, and teacher-forced response token loss fell from 7.6168 to 4.5977. These small-set diagnostics verify the interface and learning signal; they do not establish reliable generalization or fluent generation. The trial checkpoint was temporary and was not installed as the active V14 checkpoint.

Ten regression tests passed, including static Python dataset loading, partial-label loss, and validation without changing model state. The previous eight-example JSONL is archived at `archive/2026-10-06/v14-eight-examples.jsonl`; the original six-channel data remains in `data/reference/training_data.py` and is not used by V14.

