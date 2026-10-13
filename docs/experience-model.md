# V14 shared experience and emotional development

The active checkpoint now includes a persistent self/user/relationship model, a trained conversational-event interpreter, and an experience loop that connects ordinary corrections to its own appraisal and lifelong emotional statistics. Restart `RUN.bat` to load it. These additions preserve the existing V14 core and language weights. The pre-change checkpoint and sources are in `archive/2026-10-06/before-experience-model/`.

## Conversation, rather than manual events

You can introduce yourself, describe the relationship, correct a misunderstanding, and give feedback in ordinary words:

```text
My name is Quinn.
We are collaborators building an understanding together.
When I correct your explanations, I am helping you understand, not rejecting you.
Your last answer missed my point. I meant your emotional depth, not your vocabulary.
Yes, that is what I meant.
Who am I?
What are we?
What are you feeling now, and why?
```

Friday starts without an assumed user name or invented shared history. Identity and relationship information retain their source. They can change with later declarations. The initial goals are understanding, usefulness, and continuity; corrections gradually increase the importance of understanding. This is a deliberately small goal model, not autonomous human motivation.

The current-message interpreter uses frozen SmolLM2 contextual features and a new 56,459-parameter classifier. It predicts ordinary conversation, corrections, feedback, introductions, relationship statements, preferences, or self-questions, with separate correction scopes. Scope confidence and overall event confidence are distinct. Uncertain correction scopes prompt clarification and remain tentative. Literal source spans are required before learning, but a source span does not guarantee that the interpretation is correct.

The original Qwen JSON extraction approach made semantic errors even with constrained output; the active launcher uses the trained classifier instead. Its synthetic curriculum contains 229 unique examples and 33 separate validation examples. Neural interpreter weights stay fixed during chat. Confirmed scoped lessons can support later classification through similarity in its learned feature space, and retrieved lessons alter certainty, novelty, and support in appraisal. This combines trained classification with revisable memory; it does not retrain Qwen on each message.

## How words change the engine

The turn loop is interpretation → retrieve relevant lessons → associate explicit feedback with the preceding episode → update expectations/baseline/recovery → appraise the current event → regulate and select an action → produce a reply → record the actual exchange.

An intent correction supplies information about understanding; an answer correction can indicate a setback in usefulness. Neither implies personal rejection. Explicit positive or negative feedback affects developmental outcome statistics and vitality. Ambiguous scopes and external factual assertions do not automatically reward or punish the previous answer. An episode accepts this conversational feedback once. Generated statements such as “I succeeded” are not feedback evidence.

Vitality from verbal evaluation is a signed conversational proxy:

```text
quality = feedback_value * interpretation_confidence
vitality_next = 0.8*vitality_previous + 0.2*quality
```

The manual homeostatic vitality path now normalizes deficits by the maximum attainable deficit before converting them to [-1, 1]. The earlier scale could not become negative under the declared need bounds. Neither proxy is a measurement of biological energy or consciousness.

Conversational need levels are estimates, clearly labeled by `:state`. `:levels` supplies external measurements and takes precedence during that session. `:event` and `:observe` remain available for controlled environment/task observations. Ordinary conversational learning does not require JSON commands.

## Emotional depth and continuity

Language knowledge is present from the start. The new developmental index begins at zero and grows with evidence of learning rather than message count:

```text
evidence = 0.5*distinct_confirmed_corrections + feedback + 2*confirmed_transfers
depth = 1 - exp(-evidence/20)
proposal = (1-depth)*coarse_proposal + depth*differentiated_proposal
```

The coarse proposal averages fear/stress/pain/vulnerability, joy/calm/connection/pride, and anger/sadness/shame/guilt/envy within their groups. Other channels keep their existing proposals. Depth progressively permits finer distinctions from the trained own-event appraiser. It is continuous and has no fixed age stages. Repeated feedback about feedback does not increase the index; uncertain corrections count only after confirmation. A confirmed transfer means feedback endorsed a turn that used earlier scoped learning, not proof of causal or human-like generalization.

This depth formula is an engineering heuristic, not a validated model of childhood or psychological maturity. Nineteen emotion channels remain fixed; new emotions do not emerge. Existing learned baselines, recovery rates, expectations, and regulation values continue adapting throughout use. The index itself increases with accumulated learning evidence while current emotions and recovery remain fluid.

Up to 64 episodes and 64 lessons are retained. Each episode records actual text/reply, interpretation, before/after state, applied lessons, and developmental appraisal. Older entries are evicted; lifetime statistics remain. There is no unlimited autobiographical database or background dreaming/consolidation process.

## Awareness and replies

The generator receives compact identity, user information, relationship, goals, scoped lessons, current interpretation, and actual affect. Narrow identity and own-state questions use a grounded composition path: actual state and recorded exchanges determine the content. Short name/relationship and correction/feedback acknowledgments also use that path. An explicit “I meant …, not …” correction can supply the two contrasted phrases directly. These rules bind speakers and recorded evidence; they do not provide unrestricted semantic paraphrasing. Questions and requests for explanation stay generative. The remaining conversation is generated by Qwen with the same context.

Own-state explanations describe patterns in words and cite the recorded exchange rather than listing emotion scores. A reported decrease in uncertainty requires a corresponding recorded state decrease. Unknown causes remain unknown. Narrow questions about how the engine uses corrections also receive an explanation of the implemented loop. These explanations use designed composition rules, not a learned introspection decoder, and are not evidence of subjective feeling. The self-model supports access to stored identity and state; it does not establish complete situational awareness.

External factual corrections remain unverified user assertions and are excluded from ordinary lesson retrieval. There is no automatic fact-checking service. Profiles and relationship descriptions are user reports, not independently verified identities. The current interpreter classifies the newest message alone; profile extraction and speaker attribution include limited text rules. Rich contextual meaning, subtle irony, long-distance references, novel emotional situations, and reliable perspective-taking remain open limitations. Native replies can still repeat or confuse speakers. The small synthetic scores below must not be presented as human-indistinguishable conversation.

## Persistence and inspection

`:self` shows the compact self/user/relationship context. `:state` shows emotion, vitality, and need provenance. `:reflect` shows the most recent appraisal and regulation. `:growth` shows lifetime learning and depth. `:save` and `:quit` save experience alongside the neural checkpoint. `:reset` clears recent chat and temporary emotion while retaining identity, relationship, lessons, and lifelong learning. Bare or incomplete colon commands produce an input error rather than a fabricated language response.

The installed live checkpoint has no fabricated test biography. Tests use fresh experience models and copied checkpoints. The earlier saved emotional state was preserved during installation; use `:reset` if you intentionally want to clear its temporary activation.

## Tests and reproduction

All 31 regression tests pass, covering bounded state and memory, signed vitality, corrections, ambiguous scope handling, identity provenance, differentiation, checkpoint persistence, and previous V14 behavior. The trained interpreter classified 32/33 held-out synthetic examples correctly; the missed example asked how Friday would describe the relationship. Training and feature extraction took approximately 34 seconds locally. These figures describe this small test set, not unrestricted conversation.

The live integration test includes ten event-classification turns plus two additional language probes (learning-process explanation and generated arithmetic). Its JSON includes full replies so language failures remain visible. The ten event kinds matched, and identity, relationship, verbal-vitality, lesson-retrieval, unverified-fact, and save/reload checks passed. Integration checks do not establish that the generated wording is emotionally coherent. Earlier unsuccessful interpreter runs remain in the results files alongside the final run. In particular, prompt-only attempts repeated relationship acknowledgments, confused who supplied a correction, and copied an illustrative placeholder; this motivated the grounded short acknowledgment path.

Run these commands from the project folder, after closing the chat to avoid loading two copies:

The final twelve-turn run sampled a maximum combined working set of 3.16 GiB. Its median complete turn was 1.08 seconds, but eleven replies used grounded composition rather than native generation. The one generated arithmetic reply took 8.88 seconds with accumulated conversation/context; regression tests were also running during that evaluation. These mixed measurements are not a general generation-speed benchmark. Memory remains bounded, while longer prompts and ordinary generated replies can be appreciably slower than direct state explanations.

```powershell
& .\venv\Scripts\python.exe -m unittest discover -s tests -v
& .\venv\Scripts\python.exe src/evaluate_experience_model.py
# Optional classifier retraining; unnecessary for ordinary chat:
& .\venv\Scripts\python.exe src/train_semantic_interpreter.py
```

The evaluator exercises the experience path and writes `docs/experience-model-results.json`; it does not overwrite the active biography. `docs/semantic-interpreter-results.json` records classifier training and all validation predictions. The older `benchmark_cpu.py` measures the underlying V14/native path without this conversational interpreter, so its older timings should not be substituted for the new experience-loop timings.

These changes implement the agreed experience architecture as a prototype. They do not yet fulfill the broader goal of a fully coherent agent whose emotional conversation is indistinguishable from a human.
