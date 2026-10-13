# V14 language backbone — October 6, 2026

V14 now connects to a locally downloaded SmolLM2-135M-Instruct language model. This supplies learned vocabulary, grammar, and conversational patterns while V14 continues to calculate persistent emotion, fast memory, and action intent. `RUN.bat` uses this mode. Commands are in the root README. The subsequent contextual-appraisal extension and 180-record curriculum are described in `meaning-and-emotion.md`; the initial-run results below remain historical results for 46 records.

## What reaches the decoder

Each turn has two inputs to language generation:

1. The exact user text and recent user/assistant history, formatted using the backbone's chat template. A system instruction describes the selected action: answer, clarify, comfort, verify, or set a boundary.
2. Four learned input vectors computed from V14's affect representation and action embedding, prepended before the textual prompt.

The raw message is retained separately from the GloVe token IDs. Words outside V14's GloVe vocabulary can reach the language model through its own tokenizer. With contextual appraisal enabled, its frozen prompt representation also reaches V14's recurrent and attention inputs. The fixed anchor-similarity component still uses the GloVe vocabulary.

For affect representation `h_t` of dimension `d`, action embedding `e(a_t)`, language embedding width `D`, and four soft tokens:

```text
c_t = h_t + e(a_t)
P_t = reshape(W_bridge c_t + b_bridge, [4, D])
X_t = concat(P_t, Embedding(chat_prompt), Embedding(previous_response_tokens))
p(y_j | y_<j, prompt, P_t) = softmax(F_frozen(X_t)_j)
```

This is learned conditioning at the model input. It does not add learned prefixes at every attention layer. The chosen action also appears in the text instruction, so changes in wording cannot automatically be attributed to emotional conditioning alone.

## What training changes

The pretrained decoder's parameters have `requires_grad=False` and remain in evaluation mode, including during V14 training. They are excluded from optimizers. Autograd still passes through its computation to the bridge and connected V14 components. Freezing parameters does not prevent that input gradient.

The supervised objective retains response-token cross entropy, action loss, optional partial appraisal labels, and the existing regularization and measured-outcome objectives. Only response tokens, including the final EOS token, receive the language loss. The last actual prompt position predicts the first response token; a bookkeeping sentinel is never inserted into the language prompt.

The bridge initially uses small weights. Training learns how V14's representation should influence the frozen model. This is not retraining the language model from scratch. The existing scratch-decoder checkpoint cannot supply this new bridge; a separate `v14-language.pth` checkpoint is used. Future training resumes that checkpoint's V14 parameters; the supervised optimizer starts fresh on each training invocation.

No new emotion categories or subjective-experience claim is introduced. The 19 channels, seven needs, bounded memories, and V14 state equations remain the emotional core. The backbone improves the available language machinery; the project still needs evidence that its emotional mechanisms improve behavior.

## Context and saved state

Text context uses at most 384 backbone tokens, further reduced to fit the backbone's position limit after reserving four soft tokens and the configured response budget. The newest tokens are retained when the prompt is longer. This simple truncation can discard earlier system instructions or part of an older message; it is not semantic memory retrieval.

The prompt budget is fixed for an entire response, preventing context from shifting as tokens are generated. Response attention stays causal. The GloVe appraisal input separately retains at most 128 local tokens. Interactive responses currently generate up to 64 new tokens, with two candidates scored using the existing V14 selector. Generation caches the backbone's attention keys and values within each candidate, avoiding a full prompt pass at every token. CPU replies can still take several seconds and can end mid-sentence at the token cap.

Consecutive examples sharing an `episode` now share teacher-provided conversation history as well as recurrent emotional state. An optional `history` field accepts a list of `{"role":"user"|"assistant", "content":"..."}` messages before the current input. Do not include the current target response in that history. Independent examples start with empty history. Validation uses the same rule and makes no weight updates.

Chat retains the latest 40 messages and saves them in the checkpoint. `:reset` clears history, emotions, recurrent state, and fast memory; scars remain. The checkpoint includes the complete frozen language weights plus V14 weights and state. Keep `data/models/smollm2-135m-instruct` available because loading also needs its configuration and tokenizer. The downloaded model occupies roughly 270 MB; full float32 checkpoints are larger. The implementation loads float32 for consistent CPU behavior.

Old scratch V14 checkpoints remain supported. Checkpoints from the earlier one-vector pretrained adapter are rejected with an explanatory message because their bridge and context contract differ. The four-vector output-only contract 2 remains supported; contextual appraisal uses contract 3. No legacy weights are overwritten. The code and documentation before this change are preserved in `archive/2026-10-06/before-language-backbone/`.

## Comparing the bridge

`--evaluate data/training/validation_data.py --compare-bridge` evaluates the same checkpoint twice: learned soft vectors enabled, then four zero vectors at the same positions. Both retain the same text, history, reference action, and response targets. Action accuracy stays identical because this comparison changes only decoder conditioning.

This measures the effect of the complete learned bridge, which contains task context and action as well as emotion. It does not isolate emotion. A stronger study would also clamp or shuffle emotion while holding prompt, history, and action fixed; compare multiple seeds; use unfamiliar tasks and longer conversations; and independently judge task correctness and appropriateness. Ten validation records are too few for a strong generalization claim.

## Broader training data

Keep validation separate. Add diverse, independently written examples for explanation, everyday questions, planning, uncertainty, disagreement, and emotionally sensitive dialogue. Include matched scenarios where the task stays the same but history or emotional appraisal changes the appropriate response. Vary response wording instead of repeating a handful of templates. Avoid labeling every negative sentence as needing comfort; action choice should follow context and the user's actual request. Use measured need/outcome fields only when an environment can supply them.

These examples train V14's use of the language model. They do not add new world knowledge to the frozen backbone. A larger local instruction model can be connected through `--pretrained-decoder` with a new checkpoint, but its memory use and CPU latency should be measured first.

## Initial run and checks

The installed `state/checkpoints/v14-language.pth` was trained for two epochs on the existing 46 demonstration records, with seed 14, the default 10,000-word base vocabulary, and a three-turn gradient window. Average combined training loss fell from 5.0420 to 4.4330. No validation examples were used for training.

| Ten held-out records | Before training | After two epochs |
| --- | ---: | ---: |
| Response-token NLL, bridge enabled | 4.8368 | 3.6872 |
| Response-token NLL, four zero soft vectors | 4.9773 | 4.9773 |
| Action accuracy | 4/10 | 2/10 |

The bridge improves fit to the reference wording in this small run, while intent classification gets worse. That is a material limitation: these weights are an initial demonstration, not a calibrated action selector. More diverse action examples, loss balancing, and larger held-out evaluations are needed before treating intent choice as reliable. The fixed textual action instruction uses the reference action during these language diagnostics; interactive chat samples its action from V14's policy instead.

All ten regression tests pass, including the chat-template path, frozen-weight updates, bridge and recurrent gradients, direct text/history influence, context and response causality, equivalent cached/uncached greedy generation, and checkpoint/history round trips. A separate comparison against freshly loaded original weights confirmed that every tensor in the real backbone remained identical after training.

Real-model generation produced an understandable explanation of atmospheric light scattering and practical study suggestions after an exam setback. Both reached the 64-token cap, so completion length remains a limitation. These two examples test execution, not factual reliability or broad language quality. The uncached initial check took about 17–19 seconds per greedy candidate; the active code now uses attention caching, so that timing is not a benchmark of the final implementation.

## Sources

The [official SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) describes the 135-million-parameter instruction model and its Apache 2.0 license. The [Hugging Face guide to soft prompting](https://huggingface.co/docs/peft/main/en/conceptual_guides/prompting) explains the principle of learning input conditioning while retaining pretrained weights. This project implements its own V14-conditioned input bridge rather than importing a PEFT adapter.
