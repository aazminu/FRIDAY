# Learning meaning and emotion

The active launcher now uses native CPU Qwen replies as described in `cpu-performance.md`. The training and generation results below concern the earlier SmolLM2 soft-bridge experiment. The V14 core remains trained on that contextual representation; native Qwen response quality is measured separately.

The target behavior is to respond appropriately to unfamiliar wording and context, rather than reproduce a finite list of replies. V14 generates responses token by token with a pretrained language model. There is no lookup table that selects one of the training sentences. However, supervised next-token learning can still overfit, so generated wording alone is not evidence of understanding.

## What changed

The original emotional proposal used weighted GloVe similarities and a recurrent correction. Its recurrent input averaged the prompt representations. At a fixed prior state, reordering the same words could leave that appraisal input unchanged. Conversation history reached the language decoder but did not reach the emotional core. This was a significant limitation for interpreting negation, perspective, and implied feelings.

The new optional contextual-appraisal adapter reads the actual current message and prior conversation through the frozen backbone. A neutral instruction is used for this encoding; the desired action and target reply are not supplied. The final prompt-position representation incorporates preceding words and their order. A trainable projection maps it into V14's dimensions:

```text
z_t = LayerNorm(F_frozen(prompt_only).last_hidden_state[-1])
s_t = tanh(W_context z_t + b_context)
context_i = existing_V14_context_i + s_t
gut_t = GRU(concat(mean(context_i), previous_emotion, needs), previous_gut)
```

The contextual residual therefore reaches both recurrent appraisal and the attention/action pathway. The existing anchor proposal, bounded correction, persistence, mood gate, fast memory, and output bridge remain. This extends the representation entering V14; it does not replace the V14 state equations with a language-model emotion label or add a keyword override.

The backbone stays frozen and its prompt encoder runs without gradients. The context projection, recurrent appraisal, action policy, and output bridge can learn from supervised targets. The output bridge still receives gradients through the frozen decoder when it trains. No response target is fed into the prompt encoding for its own turn.

These contextual features make distinctions available to the emotional core. They do not guarantee that a 135-million-parameter model represents each distinction correctly or that the learned adapter will use it well. Evaluation is needed.

## Curriculum

`data/training/training_data.py` now contains 180 synthetic records, 36 for each action. It retains the initial 46 examples and adds 134 authored examples. Coverage includes basic explanations, arithmetic, planning, writing, programming, evidence checking, ambiguous references, emotional support, honest limits, and matched meaning cases.

Examples distinguish:

- The user's feelings from a friend's feelings or a fictional character's dialogue.
- Current emotion from emotion that has passed.
- A literal threat from figurative or analytical use of an emotional word.
- Negative wording that asks for a practical answer from a request to be listened to.
- Emotional reassurance from evidence checking; anxiety does not change arithmetic.
- The same vague sentence used after different conversation histories.
- A dishonest request made calmly from the same request made under stress: emotional appraisal may differ while the boundary remains.

Independent examples and complete episode blocks are interleaved with a fixed seed. Episode turns remain in order. This avoids training every action in one long consecutive block. Repeated short prompts are intentional only when history or episode distinguishes them; they are not conflicting labels for identical inputs.

Appraisal labels are sparse illustrative engineering targets, not observations of experienced emotion. Only explicitly named channels receive the appraisal loss. No invented need measurements, rewards, or scar directions are added. Responses use varied wording, but remain reference examples; they are not a complete definition of all good replies.

## Held-out checks

`validation_data.py` now contains 50 separate records, 10 per action. Its original 10 remain first so historical results can be interpreted correctly. Its expanded-set score must not be compared directly with an old ten-record score.

`meaning_validation.py` contains ten new prompts forming five high/low appraisal contrasts: negation, speaker perspective, time, fictional attribution, and conversation history. Some contrast pairs contain the same words in different orders, or the same current sentence with different history. They are not training records.

Evaluation reports action accuracy and response-token loss. Where appraisal labels are present, it also reports label mean squared error and the fraction of contrast pairs with the intended high/low order. A difference smaller than 0.000001 does not count as a correct ordering. The proposal is measured, rather than the persistent emotion, because retained emotion from earlier turns is a separate mechanism.

These ten meaning cases are a small diagnostic. Passing them does not establish general semantic or emotional understanding. Extend evaluation with independently authored paraphrases, new situations, longer dialogues, and blinded judgments of task usefulness and emotional appropriateness. Monitor repeated wording and context-insensitive answers; do not judge success from training loss alone.

## Training and checkpoints

The root README contains commands to train with `--contextual-appraisal` and evaluate both held-out files. The expanded curriculum increases training work compared with 46 examples. Earlier timing estimates for that smaller dataset do not apply unchanged.

An older language checkpoint loads with its original appraisal mode. Adding `--contextual-appraisal` during training initializes the new projection while retaining existing V14 and language weights. The checkpoint then records the adapter and automatically enables it on later chat runs. A newly added adapter resets any stored outcome optimizer because its parameter list changed. Context-aware checkpoints use language contract 3; the earlier output-only contract 2 remains loadable.

The pre-expansion datasets, model code, and original language checkpoint are preserved under `archive/2026-10-06/before-expanded-curriculum/`. Experimental results for the initial contextual run are saved as `docs/contextual-training-results.json`. They should be read as a before/after check on these specific small datasets, not as proof of understanding.

## Initial contextual training results

The original installed checkpoint was continued for two epochs on the 180-record curriculum with the contextual adapter enabled, seed 14, and a three-turn gradient window. Training took 307.48 seconds, about five minutes, on this CPU setup. Combined training loss was 3.5909 in the first epoch and 3.3214 in the second. The resulting checkpoint is installed as `state/checkpoints/v14-language.pth`; the launcher reads its contextual settings automatically.

| Held-out diagnostic | Original checkpoint | Contextual training run |
| --- | ---: | ---: |
| Action accuracy, 50 general examples | 10/50 | 43/50 |
| Response-token NLL, general examples | 3.1043 | 2.9012 |
| Action accuracy, 10 meaning examples | 0/10 | 7/10 |
| Appraisal-label MSE, meaning examples | 0.1120 | 0.0955 |
| Correct high/low appraisal ordering | 2/5 | 4/5 |

The improvement is consistent with learning useful distinctions on unseen prompts. It does not prove general understanding. The curriculum, ordering, and contextual adapter changed together, so this run does not isolate the adapter's contribution. The meaning set is only five pairs, one pair is still misordered, and three meaning-focused action choices remain incorrect.

All eleven regression tests passed. They check, among other things, that word order and conversation history can change appraisal, gradients reach the new adapter, the backbone stays frozen, and tied appraisal outputs are not counted as a correct contrast. Passing those wiring checks is separate from learning the right interpretation.

## Fresh reply failures

Three additional prompts, absent from both training and validation, were generated with greedy intent and greedy decoding. None of the replies exactly matched a training response, but all three exposed limitations:

- A request to subtract eight paper clips from twenty-nine produced a division explanation instead of the correct answer, twenty-one.
- A request to explain different emotional reactions repeated the user's first sentence instead of answering the question.
- An indirect request for listening after a colleague's departure chose COMFORT but generated an irrelevant refusal to advise on work matters and referred the user to HR.

Therefore the improved action and appraisal scores do not establish useful free-form emotional understanding. Avoiding exact memorization is necessary but insufficient. The remaining failing appraisal contrast was negation: the fear proposal was 0.2787 for the frightened case and 0.2835 for the explicitly unfrightened case, the wrong order. Other correct contrast margins were sometimes small, so ordering should not be treated as well-calibrated intensity.

The complete prompts, replies, and per-pair scores are preserved in `contextual-training-results.json`. A comparison against the original pretrained weights confirmed every backbone tensor remained unchanged. Future improvement needs evaluation of generated replies, broader semantic training, and a language backbone whose reasoning and instruction-following capabilities meet the task. Simply increasing the number of repeated passes is not a demonstrated solution.

## Decoder diagnosis and next experiment

`decoder-diagnostic.json` compares the three fresh prompts with the intended action held fixed: the original language model with no soft prefix, four zero soft vectors, and the learned V14 bridge. All runs use greedy decoding and the same 64-token response cap. These are qualitative checks, not a benchmark.

The plain backbone still refused a simple subtraction request and gave an unsupported explanation of different reactions. The learned bridge repeated the emotional-perspective prompt and turned a request for listening into an irrelevant HR referral. Holding the intended action fixed did not resolve those failures. Both backbone limitations and harmful conditioning are therefore plausible contributors. The zero-vector condition also changes the attention input and positions; it is not the same as the unmodified backbone.

Next steps should be:

1. Benchmark a stronger instruction model on task answers and contextual emotion interpretation before connecting V14. [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) is one candidate, not a verified replacement on this machine. Measure memory and latency first; current float32 loading and full-weight checkpoints need adjustment before larger models can be assumed to fit.
2. Distinguish the user's expressed emotion and request from V14's own evolving emotional state. Negation, speaker attribution, time, and preferred kind of help should inform appraisal; a user expressing fear should not automatically force the engine into fear.
3. Bound the output bridge's strength and compare every candidate checkpoint against the unmodified backbone on generated replies. Reject changes that damage task accuracy or instruction following even when reference-token loss improves.
4. Train and evaluate scenario families with paraphrases, changed speakers, negation, indirect feelings, and novel histories. Keep entire held-out scenarios separate. Evaluate listening, relevance, task correctness, repetition, and calibration, rather than only action labels and response likelihood.
5. Use independently checked calculations and other task tools where appropriate. Tool accuracy complements emotional interpretation; it does not establish that interpretation.

The aim is an engine whose emotional state guides appropriate reactions while competent language and task reasoning remain intact. The current small experiment has not established that behavior.
