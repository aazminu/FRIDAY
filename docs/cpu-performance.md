# CPU operation on the 16 GB laptop

The active launcher now includes the small trained conversational interpreter and shared experience described in `experience-model.md`. Full-loop timings and sampled RAM are stored with each trial in `experience-model-results.json`. Grounded identity/state replies bypass native generation, so a median across those trials is not a general Qwen generation-speed claim. `benchmark_cpu.py` exercises the older underlying core/native path without the conversational interpreter. The new interpreter adds 56,459 parameters and uses the existing frozen language features.

The latest developmental-extension run is recorded in `developmental-cpu-results.json` and explained in `developmental-emotion.md`: median full turn 3.53 seconds, highest sampled combined RAM 3.27 GiB, and startup 5.27 seconds in that run (20.65 seconds in the earlier pass). The older figures below describe the earlier runtime/prompt. The new own-event appraiser/regulator adds only 2,471 neural parameters; it does not require a larger language model.

The machine has a Snapdragon X1P64100 processor with ten CPU cores and about 16 GB RAM. The existing Python 3.11 environment is AMD64, so Windows runs it through x64 emulation. Generation now uses a native Windows ARM64 llama.cpp process, with CPU execution explicitly selected. No GPU or NPU is required.

`RUN.bat` starts the native language mode. The trained V14 emotional core continues to run in the existing Python environment, including its contextual appraisal, persistent emotion, fast memory, and action choice. Its small frozen SmolLM2 model is retained for the contextual features on which the current V14 weights were trained.

## Language runtime and memory settings

The response model is Qwen3-1.7B in Q4_K_M GGUF format, about 1.28 GB on disk. It runs in the project's `runtime/llama-arm64/` directory. Four generation threads performed better than six or eight in the isolated decoder benchmark, so four is the default. The process uses a single conversation slot, a 1,024-token context window, and a maximum of 128 response tokens. Internal reasoning output is disabled. Replies stream as they are generated.

Only one native response is generated per turn. The original mode generated and rescored two candidates. The smaller context and 128-token compute batches constrain memory overhead. Old conversation pairs are dropped when necessary to fit the rendered prompt; the system instruction and latest user message are preserved. An overlong current message produces a request to shorten it instead of silently cutting the request.

The runtime does not block startup on available RAM. A 16 GB machine with many other applications open can still run short of memory, which makes replies slow or makes the language process exit. Logs are in `state/logs/native-language.log`.

The model process listens only on localhost on an automatically chosen port. The launcher owns that process and stops it when chat exits. It does not connect to a hosted model or transmit conversations off the machine.

## Important architecture distinction

Native language mode uses V14's chosen intent and a compact representation of its emotional state as textual context for Qwen. It does not pass the learned SmolLM2 soft embedding prefix into Qwen: the two models have different tokenizers and embedding spaces, and the native runtime is not a PyTorch gradient path.

Thus this mode keeps the trained emotional/state calculations while changing how language is produced. The original soft-bridge mode remains available by omitting `--cpu-chat`. Supervised V14 training and the existing teacher-forced evaluation commands continue to use the small PyTorch decoder; their response-loss numbers are not evaluations of the native Qwen replies. Use the full-engine benchmark and independent reply evaluation for the active chat mode.

CPU chat chooses the highest-probability intent rather than sampling it. `:observe` still updates memory and vitality from supplied outcomes, but does not apply an on-policy gradient to that deterministic action. Policy weights remain fixed during CPU chat. The original mode retains its sampled-action outcome-learning path. Neither native Qwen weights nor a new Qwen soft adapter were trained in this change.

The subsequent self-monitoring layer supplies real state changes and heuristic regulation guidance, and directly answers narrow questions about the engine's own state. `:state` and `:reflect` expose the stored state and last appraisal. These reporting paths bypass Qwen generation. See `emotional-control.md` for the controlled intervention and self-report probes. The speed figures below were measured before the expanded self-description instructions; a larger prompt consumes more context and can increase processing time.

## Checkpoints

The active V14 checkpoint now stores its trained adapters, emotional core, fixed semantic vectors, memory, and saved conversation, with a fingerprint of the separate frozen language source. Frozen SmolLM2 weights are not copied into every checkpoint. The file shrank from 548,210,476 bytes to about 10,029,983 bytes without retraining or changing learned V14 weights.

Loading is memory mapped, with copies of the semantic/state/optimizer tensors that must survive loading. This also prevents Windows file mapping from blocking a later save to the same filename. The source fingerprint rejects mismatched frozen language files. Keep both language model directories; the compact checkpoint is not self-contained. Old full V14 checkpoints remain readable. The previous full checkpoint and launcher are preserved in `archive/2026-10-06/before-cpu-runtime/`.

## Commands

From the project folder:

```powershell
& .\venv\Scripts\python.exe src/model.py --checkpoint state/checkpoints/v14-language.pth --cpu-chat --threads 4 --native-threads 4
& .\venv\Scripts\python.exe src/benchmark_cpu.py
& .\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The benchmark measures the actual V14-plus-native path on four fresh prompts, records the responses, time until first reply text, total turn time, and both processes' memory use. It does not train or overwrite weights. Detailed results are saved in `docs/cpu-performance-results.json`. `native-cpu-benchmark.json` records the isolated decoder thread comparison. Throughput from that short synthetic benchmark is not the same as end-to-end reply speed.

Longer messages and conversation histories take longer to process. Other applications, power mode, temperature, and CPU load affect timing. The output limit can still truncate a long reply; the prompt requests a complete answer under 80 words to reduce that risk. These settings target responsive operation and bounded resource use, not guaranteed accuracy or complete emotional understanding.

## Measured results — October 6, 2026

The final four-turn run used the settings above, including the 80-word prompt. It measured:

| Measure | Result |
| --- | ---: |
| Median time until first reply text, including V14 appraisal | 0.925 seconds |
| Median complete turn | 3.21 seconds |
| Complete-turn range for these prompts | 2.34–4.27 seconds |
| Highest sampled combined working set, Python plus native process | 3.18 GiB |
| Sum of each process's peak working set, conservative upper bound | 3.42 GiB |
| V14 load plus native startup | 7.96 seconds |
| Isolated decoder generation, four threads | 25.71 tokens/second |

The tests included planning, subtraction, differing emotional perspectives, and a request for listening. The subtraction produced 21 correctly; the perspective prompt received an explanation rather than repetition. The listening reply remained simplistic and made an unnecessary inference about working with the person. These examples are execution and performance checks, not an accuracy or empathy benchmark. All four final replies completed their sentences within the response cap.

Fourteen regression tests passed. A real launcher-path check also exercised streamed chat, saving after load, supplied outcome updates, reset, and exit. The owned language process exited with the chat. Previous PyTorch generated-response failures are preserved in the earlier diagnostic files and are not presented as failures of this native model.

## Provenance

The runtime is [llama.cpp](https://github.com/ggml-org/llama.cpp), pinned to the official Windows CPU ARM64 build b11429. The [Qwen3-1.7B model](https://huggingface.co/Qwen/Qwen3-1.7B) is served using [bartowski's Q4_K_M conversion](https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF). That quantization is a separately published conversion, not an official Qwen Q4 release. Native streaming and template handling follow the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
