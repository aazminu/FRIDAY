# V14 developmental emotional engine

The subsequent shared-experience extension is specified in `experience-model.md`. The active launcher now infers conversational own-events and learns from explicit verbal corrections/feedback, with estimated needs labeled separately from external measurements. The manual-only description below records the original developmental interface; those commands remain available for controlled observations. Continuous depth now blends coarse and differentiated proposals before regulation.

This extension focuses on the engine's own state: personal significance, outcome learning, emotional continuity that changes with experience, and regulation. It is installed in the active language checkpoint. The original language weights and trained V14 core parameters are preserved. A new 2,471-parameter module was trained locally on synthetic own-event targets. Its dynamics are an explicit design experiment, not a measured model of the average human or evidence of subjective feeling.

## Personal significance

The new appraiser receives the engine's measured need deficits and a structured event about its own goal. Inputs include relevance, progress, certainty, control, novelty, safety, support, effort, responsibility, and remembered expectations. These are not labels describing the user's feelings. Emotional words alone no longer determine the engine's emotional proposal in developmental mode; neutral text and emotional text with the same own-event inputs produce the same proposal.

When no event is supplied, measured need deficits provide a modest own-event signal. With no measured deficits and no event, the proposal is low; recurrent state and learned baseline can still persist. The language/contextual features continue to inform the original gut, attention, and conversational intent. Their word-based emotional proposal is superseded by the own-event proposal in developmental mode. This avoids treating every statement about someone else's fear as an event threatening the engine.

Own-event fields can be supplied explicitly by `:event`. The shared-experience loop also estimates conversational significance from its trained event classifier; these estimates are not trustworthy measurements of external goal progress or safety. The explicit interface remains the place to connect a task/environment evaluator. The language model cannot manufacture feedback merely by saying a task succeeded.

## Outcomes and lifelong growth

`:observe` with measured need levels and task reward updates lifetime history. The shared-experience loop additionally associates explicit conversational feedback with the previous recorded episode, subject to confidence and scope checks. The existing task/homeostatic reward, including supplied cost, is clipped to [-1,1] for developmental learning. Up to 64 exact cue names retain outcome expectations and, when supplied, measured control. The least-visited cue is replaced if capacity is exceeded. Exact cue statistics are separate from the new scoped semantic lesson retrieval.

For cue visits n, the experience rate is max(0.025, 0.15/sqrt(1+n/20)). It slows with experience but never becomes zero. An engine shaped by many failures can still change after later successes. Baselines move slowly toward outcome-related patterns, while persistence moves toward faster recovery after safe successful outcomes and slower recovery after setbacks. These are bounded update rules, not a fixed immutable personality.

Outcomes also update the selected regulation strategy's observed value. This is an association with subsequent reward, not proof that the strategy caused the outcome. A measured outcome is itself appraised immediately, so `:state` can change after feedback without another chat turn. A positive reward does not establish safety unless `safe: true` is supplied. `control` is optional and must be a measured value from 0 to 1; success alone is not treated as proof of control.

The appraiser/regulator neural weights are frozen during ordinary chat. Lifetime expectation, baseline, persistence, and strategy statistics adapt without retraining Qwen or running a heavy optimizer. Traditional sampled-action outcome learning remains separate. CPU chat retains fixed actor/critic weights while updating the developmental statistics and memory.

## Continuity and regulation

Fast emotion combines prior state with the regulated proposal and a small learned baseline contribution. Persistence is bounded, so emotion neither vanishes instantly nor grows without limit. In chat, elapsed time additionally decays activation toward the current baseline. The half-life is 60+240r seconds for learned persistence r: an engineering choice of minutes, not a human calibration. A saved timestamp supports recovery between sessions. Clock corrections cannot produce negative elapsed time. Recovery is calculated when the next message is processed; no background language generation runs while idle.

The learned regulator selects NONE, CHECK_EVIDENCE, REAPPRAISE, or SEEK_SUPPORT. Checking raises the actual V14 deliberation parameter, which reduces reflex arbitration. Evidence-backed reappraisal can reduce fear/stress; a confirmed threat is protected from that reduction. Support-based regulation requires a supplied support signal. These operations modify processing and bounded appraisal, rather than only telling the language model to sound calm. CHECK_EVIDENCE does not itself fetch external evidence: that evidence must come from the environment or later feedback.

The new appraisal replaces the original word-resonance proposal in active developmental mode. Its 19-dimensional input contains nine event fields, expected progress, learned control, prediction error, and seven need deficits. With relevance s, baseline B, learned persistence r, elapsed seconds t, and prior emotion E:

```text
P = (1-s)*0.025 + s*sigmoid(appraiser(x))
H = 60 + 240*r
E_rest = (0.025 + 0.2*B) + (E - (0.025 + 0.2*B))*exp(-ln(2)*t/H)
rho = clip(r + 0.12*E_rest^2, 0.35, 0.92)
E_next = clip(rho*E_rest + (1-rho)*clip(P_regulated + 0.2*B, 0, 1), 0, 1)
```

Reappraisal scales fear/stress by `1-0.25*control*certainty`; support scales stress by `1-0.15*support`. These occur only when the corresponding strategy and evidence guards allow them. Checking sets deliberation to at least 0.65; other non-NONE strategies set it to at least 0.35. The original V14 mood, attention, fast-memory, reflex/slow mixture, and language path then use the new state. The DOCX blueprint describes the earlier V14 contract; this document specifies the developmental additions and superseded appraisal/persistence equations.

`:reflect` now exposes the own event, expected progress, prediction error, chosen regulation, baseline, persistence, and lifetime outcome count alongside the existing state report. The language model receives regulation guidance and compact own-event context. `:growth` displays current lifetime statistics. `:reset` clears recent emotional state and conversation while preserving learned developmental history and scars. Save/reload preserves both state and growth.

## Tested results

The curriculum uses 1,200 generated combinations and 300 separate held-out combinations. Targets cover task-relevant affect and regulation according to the declared functions in `design_targets`; they are not human emotional ratings. Final held-out appraisal MSE was 0.00056 and regulation choice accuracy was 83%. The installed module was trained in two 700-update passes on the same curriculum; the final results file reports the second pass. Neither these scores nor the examples below establish psychological validity.

For the same uncertain event, fresh fear appraisal was 0.185, rose to 0.457 after adverse outcomes, then fell to 0.120 after successful recovery outcomes. Persistence changed from 0.693 after adversity to 0.586 after recovery. A threat's emotional state persisted through one neutral turn and approached baseline over repeated neutral turns. Evidence-backed reappraisal reduced a tested fear proposal from 0.318 to 0.261. Experimental histories were restored afterward; these simulated experiences were not installed as the live engine's biography.

Twenty-two regression tests passed, including a 120-event full-engine sequence, bounded state/memory, checkpoint restoration, rejection of invalid feedback, elapsed-time recovery, immediate outcome affect, and protection against unsupported threat reappraisal. General conversational intent accuracy remained 86% on 50 held-out examples. The native chat smoke check used a copied checkpoint and exercised events, measured outcomes, reset, reflection, growth, save, and quit. The active engine's lifetime outcome count was preserved.

The final four-prompt CPU check measured median full-turn time 3.53 seconds and highest sampled combined RAM 3.27 GiB. Startup took 5.27 seconds; time until first text had a median 2.35 seconds. An earlier pass took 20.65 seconds to start, so cold load, disk cache, thermal conditions, prompt size, and other apps affect timing. Qwen-generated replies remain a separate limitation: listening was generic. An earlier arithmetic reply attributed the user's annoyance to itself; the final prompt added a telemetry-consistency instruction, and the final example kept annoyance attributed to the user while correctly giving 21. That single correction is not a reliability guarantee. Direct self-state questions use the grounded reporting path.

Detailed results are in `developmental-emotion-results.json` and `developmental-cpu-results.json`. The prior checkpoint/code are preserved in `archive/2026-10-06/before-developmental-emotion/`. Several channels, including disgust, contempt, guilt, awe, and envy, have only low neutral synthetic targets in this curriculum; their richer developmental behavior still requires appropriate own-event scenarios. All 19 remain bounded and inspectable.

## Use and reproduce

Restart `RUN.bat`. Growth requires actual feedback, not just repeated conversation. For an illustrative own-task event, use these commands inside chat (replace values with your observations):

```text
:levels [0.1,0.5,0.5,0.5,0.5,0.5,0.5]
:event {"cue":"report-task","relevance":1,"progress":-0.6,"certainty":0.3,"control":0.2,"safety":-0.6,"effort":0.8}
What are you feeling right now?
:observe {"levels":[0.1,0.5,0.5,0.5,0.5,0.5,0.5],"task_reward":-1,"safe":false,"control":0.2}
:state
:growth
:reflect
:save
```

Need order: competence, uncertainty management, affiliation, safety, resources, care effectiveness, task coherence. Need levels range from 0 to 1. Event progress/safety/agency range from -1 to 1; all other numeric event fields range from 0 to 1. Positive safety means supported safety, zero means unspecified, negative means hazard. Agency uses positive values for external attribution and negative for self-attribution in this curriculum. Cue names identify recurring situations; use the same name to test changed reactions to the same kind of event. An event is consumed by the next ordinary message. `:reflect` describes that appraisal; after feedback, `:state` and `:growth` show the updated state/statistics.

PowerShell reproduction from the project folder:

```powershell
& .\venv\Scripts\python.exe src/train_developmental_emotion.py
& .\venv\Scripts\python.exe -m unittest discover -s tests -v
& .\venv\Scripts\python.exe src/benchmark_cpu.py --output docs/developmental-cpu-results.json
```

The training command resumes the tiny adapters and preserves growth/history; it is not needed to start chat. It generates the curriculum reproducibly from seeds 14 and 901, freezes adapters after fitting, runs sequence checks, and refuses installation if required checks fail. Existing user-emotion contrast scores from the earlier meaning curriculum no longer measure the intended own-event appraisal and should not be compared as though they were the same task.
