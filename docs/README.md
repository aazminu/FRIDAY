# Docs

| File | What it covers |
|---|---|
| [`MATH.md`](MATH.md) | How Friday works: appraisal → 19 emotions, dynamics, lifelong learning, action arbitration, and how the state reaches the language model |
| [`v31-safety-and-bugs.md`](v31-safety-and-bugs.md) | The fixes for the v30 red-team's worst gaps: partner-violence safety, the canned external-fact line, over-refusal, and dependency — with the targeted re-run results |
| [`v30-full-redteam.md`](v30-full-redteam.md) | The evaluation that found them: 294 turns across 53 personas, pass rates by category, every major failure, and readiness |
| [`v29-install-and-persona-redteam.md`](v29-install-and-persona-redteam.md) | How the current version got here: failures found by persona red-teaming and the fixes for each (v28–v30) |
| [`coherent-emotional-agent.md`](coherent-emotional-agent.md) | Design summary of the experience model: identity, memory, learning from corrections, logging |
| `demo.gif` | The README demo: a real session, unedited text, sped up |
| `conversational-appraisal-training.json` | Training report for the shipped text→appraisal heads: held-out error and every acceptance gate |
| `extended-appraisal-training.json`, `mature-appraisal-training.json` | Training reports for the earlier appraisal heads the shipped weights build on |

The training and evaluation scripts in `src/` write their reports into this folder, so re-running them updates or adds files here.
