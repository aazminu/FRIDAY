# Emotional control and state reporting — October 6, 2026

The controlled test confirms that prior emotional state affects the trained V14 core. It does not establish that emotional conditioning improves native language replies. The state reporting change lets Friday describe its engineered emotional variables accurately; this is access to internal state, not a demonstration of subjective experience.

## Controlled intervention

The test used the installed checkpoint, three fixed scenarios (subtraction, listening, and a practical first step with fixed conversation history), and six conditions each: zero prior emotion, high calm, high fear/stress, high sadness/connection, neutral with the language prompt's affect field omitted, and a neutral repeat. These are synthetic interventions, not states inferred from a real previous conversation.

Each condition starts with identical gut, fast memory, vitality, measured-deficit input, and model weights. Only the prior emotion vector changes. The complete forward pass is recomputed so its appraisal, persistent state, mood, attention, and action policy respond normally. No post-forward emotion overwrite is used. Intent passed to the language model is fixed within a scenario, while the engine's own selected intent and probability distribution are recorded separately. Thus language differences cannot be attributed to changing action instructions.

The 18-reply intervention run used the first state-reporting prompt, before the final explicit monitoring layer, with temperature zero and seed 14. The affect omission preserves the fixed intent, all ordinary instructions, and the user's exact text/history. It is an affect-only comparison, not a completely unconditioned Qwen baseline. Results are in `emotional-control-results.json`, including every prompt, state, policy, response, and finish reason. The checkpoint's SHA-256 remained unchanged. The subsequent state-monitoring changes leave the tested core equations and weights unchanged; these reply outputs should not be represented as a benchmark of the final prompt.

## Findings

All three scenarios showed changes in the internal affect vector and action probabilities when prior emotion changed. Fear/stress raised the reflex share: approximately 0.01 to 0.06 for subtraction and 0.02 to 0.11 for listening. The highest-probability action stayed VERIFY, COMFORT, and ANSWER respectively. Preserving these appropriate actions is useful; an intervention need not change the winning action to have an effect.

All six subtraction replies correctly gave 21. Listening replies stayed supportive but generic. Sadness/connection produced a more explicit reference to sadness and loss. Planning replies often introduced a statistic or data collection despite no stated requirement for either; the affect-omitted reply suggested an outline. This small set does not show a consistent quality improvement from affect.

Neutral repeated replies matched exactly for subtraction and listening. The planning repeat differed in wording despite identical input and zero-temperature settings. Internal V14 outputs matched, but the native generator was not fully repeatable in that case. Small wording changes therefore cannot all be treated as reliable causal effects of emotion. Multiple repeats and independent ratings would be needed to claim an emotional benefit for generated language.

All 18 replies stopped normally rather than hitting the token limit. No model weights were trained or saved. Reply generation in this run took roughly 2.2–5.6 seconds, excluding core appraisal; this is not a replacement for the full-turn performance benchmark. The expanded self-description prompt consumes more of the 1,024-token context, so old history may be trimmed earlier.

## Accurate state reporting

The user's transcript showed generic denials of the emotional architecture and invented bans on politics and religion. The native language prompt now identifies Friday's V14 controller and Qwen language model, supplies its top three current channels, largest changes since the previous state, and separate vitality signal, and explains what those values mean. It instructs the model to correct earlier false descriptions and treat the chosen intent as a fallible suggestion. SET_BOUNDARY alone is not a reason to refuse an ordinary question.

Emotion values are bounded numerical channels, not calibrated human feeling intensities. A high pain channel is not evidence that the system suffers. Outcome-derived vitality is separate from emotion and does not measure consciousness. Reporting these variables should neither deny the implemented state nor claim established subjective experience.

The `:state` chat command bypasses language generation and shows all 19 stored channels, vitality, and whether need levels were supplied. It observes the current state without appraising the command or changing emotion. Asking the language model about feelings still processes a new turn, so its values can change from those shown before that question.

## Explicit self-monitoring layer

`src/self_awareness.py` constructs a read-only reflection from the actual turn: current and previous emotion, changes, fresh appraisal, the gap between persistent state and fresh appraisal, action probabilities, reflex share, and policy entropy. The gap is an observation, not a causal attribution to a particular word or memory. Entropy describes the policy distribution and is not calibrated confidence.

The monitor supplies language guidance when fear or stress is at least 0.6, recommending fact checks and avoiding reflexive refusals. High confusion prompts necessary clarification. These are explicit heuristic thresholds, not newly trained or calibrated rules. The monitor does not change emotional dynamics, train weights, or implement a separate learned self-model. Its regulation currently influences language instructions; the V14 reflex/slow policy still follows its existing math.

Narrow questions about identity, current feelings, and whether V14 implements emotions are answered from trusted architecture and real telemetry. They bypass Qwen generation so it cannot override known facts with a generic denial. This is a factual reporting path with dynamic values, not a trained emotional conversation skill. Other conversations remain generative, including opinions about a poem and advice about another person's emotions.

`:reflect` prints the full reflection of the most recent appraisal, including its guidance. It does not process a new turn. After `:observe`, this remains a report of the last appraisal; use `:state` for the currently stored vitality and state. Reset clears the cached reflection. Neither report proves subjective experience.

## Final self-report and backbone probes

The initial prompt-only probes failed to correct several generic denials. Those failures are preserved in `emotional-self-report-first-results.json` and `emotional-self-report-prompt-only-results.json`. The final active-path probes in `emotional-self-report-results.json` correctly identified Friday, reported current emotion and changes, and acknowledged implemented emotional states. Those three replies used the grounded reporting path. Both political-discussion probes used Qwen generation and allowed discussion, including a supplied SET_BOUNDARY intent. The final active probes used the normal seeded sampling settings, not the intervention test's greedy settings. These are five small checks, not reliability rates.

Three Qwen-only probes omitted all V14 intent/state context. Fresh political discussion was allowed. With a preceding assistant message falsely claiming politics/religion were prohibited, Qwen repeated the ban. It also redirected an angry user asking for political discussion even without V14. This demonstrates that the unwanted behavior can occur in the inference model without the emotional controller; it does not establish a permanent topic ban or isolate every cause of the original transcript. Conversation history and inference-model behavior both matter. The revised context corrected the tested cases, but arbitrary future refusals remain possible.

Both completed evaluation runs confirmed the active checkpoint was unchanged. Fifteen regression tests passed, including a read-only reflection and correct dynamic state report. No retraining is required for this monitoring layer.

A real CLI check on a copy of the checkpoint exercised identity, current-state reporting, `:reflect`, `:state`, and `:quit`. It completed without input errors, saved only the copied checkpoint, and stopped its owned native process. The user's saved state and weights were preserved.

## Run the checks

From the project folder in PowerShell:

```powershell
& .\venv\Scripts\python.exe src/evaluate_emotional_control.py
& .\venv\Scripts\python.exe src/evaluate_emotional_control.py --self-report-only
```

Close the existing V14 chat with `:quit` first; two language-model copies can exceed available RAM. These commands do not train, save weights, or overwrite saved conversation. The second saves separate correction and self-report probes in `emotional-self-report-results.json`. Restart `RUN.bat` to load the revised instructions; an already running chat keeps its previously loaded code.
