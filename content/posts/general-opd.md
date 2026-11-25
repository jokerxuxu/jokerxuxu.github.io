---
title: "Datawhale Agentic RL Task04｜general OPD"
date: 2026-11-25T21:30:00+08:00
summary: "Ten steps of on-policy distillation on DeepMath-103K, with every token accounted for: what reverse KL actually measures, and why the loss curve in SwanLab is just the KL curve divided by a factor."
tags: ["reinforcement-learning", "knowledge-distillation", "on-policy-distillation", "PyTRIO", "SwanLab", "study-notes"]
categories: ["Study Notes"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Notes from Datawhale's group study of *Agentic RL*, Task04: general on-policy distillation. Tutorial: [Agentic RL Lab](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/). Code: [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab). Written 2026-11-25.

OPD and GRPO have almost nothing in common: GRPO's supervision comes from a reward function judging the **final answer**, while OPD's supervision comes from the teacher expressing a preference over **every single token**. This time I reconciled every token of the run, and the conclusion was more interesting than "the curve is going down": **the `trainer/loss_mean` in SwanLab is not new information — it is `opd/reverse_kl_mean` divided by a factor.**

- [This SwanLab run](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-opd/runs/b9myg8ou)

## What exactly was run

| Item | Value |
|---|---|
| Data | DeepMath-103K, first parquet shard, shuffled with seed=42, first 100 problems, **`question` field only** |
| student | `Qwen3.5-4B` + rank-32 LoRA, lr $4\times10^{-5}$; teacher `Qwen3.8-27B` (frozen) |
| Sampling | `temperature=1.0`, `max_tokens=512`, `enable_thinking=false`, $\beta=1.0$ |
| Scale | 10 steps x 4 problems x 4 answers = 160 answers |

## One OPD iteration

![One OPD iteration: the student writes, the teacher scores the same tokens, the student is pulled over](/images/agentic-rl/opd-loop-en.png)

The student gets corrected token by token, on the paths it actually walks. The core is two lines:

```python
reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
advantages = -args.kl_penalty_coef * reverse_kl
```

$$r_t^{\mathrm{KL}} = \log \pi_\theta(y_t \mid x, y_{\lt t}) - \log \pi_T(y_t \mid x, y_{\lt t}),\qquad A_t = -\beta\, r_t^{\mathrm{KL}}$$

By hand: if the student puts 0.2 on a token and the teacher puts 0.4, then $r_t=\ln 0.2-\ln 0.4\approx-0.693$ and $A_t=+0.693$, a positive advantage that pushes the token's probability up. **The signal is per token, not one number per answer.**

The teacher generates nothing new; it just re-scores the tokens the student wrote:

```python
all_logprobs = teacher_client.compute_logprobs(trio.ModelInput.from_ints(prompt_ids + completion_ids)).result()
teacher_lps = all_logprobs[len(prompt_ids):]
```

The whole sequence goes in at once, and a causal model only sees the text before it, so after slicing off the prompt the teacher's first number corresponds to the student's first generated token — **the teacher judges the position the student actually reached, even when the student already went wrong earlier**. The student and the teacher must share a compatible tokenizer, since the same token IDs go to both.

## Autoregressive training shifting

Autoregressive training shifts by one, and the prompt stretch is padded with zeros:

```python
prompt_loss_len = len(prompt_ids) - 1
input_ids       = prompt_ids + completion_ids[:-1]      # length P + 511
padded_advantages = [0.0] * prompt_loss_len + advantages
```

**Only 511 positions per answer carry a gradient** (the final token is a target, never an input position). With 16 datums per step:

$$16 \times 511 = 8{,}176 \quad\text{positions with a gradient}$$

## Results over 10 steps

| step | reverse KL | token-level std | loss_mean | training tokens | prompt tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1508 | 0.7179 | 0.1336 | 9,208 | 1,032 |
| 2 | 0.1503 | 0.7017 | 0.1193 | 10,280 | 2,104 |
| 3 | 0.1086 | 0.5953 | 0.0943 | 9,488 | 1,312 |
| 4 | 0.1036 | 0.5917 | 0.0888 | 9,576 | 1,400 |
| 5 | 0.1311 | 0.6196 | 0.1024 | 10,456 | 2,280 |
| 6 | 0.1388 | 0.6633 | 0.1194 | 9,520 | 1,344 |
| 7 | 0.1218 | 0.6253 | 0.1014 | 9,784 | 1,608 |
| 8 | 0.1080 | 0.5680 | 0.0923 | 9,532 | 1,356 |
| 9 | 0.1276 | 0.6512 | 0.1021 | 10,136 | 1,960 |
| 10 | 0.1233 | 0.6453 | 0.1066 | 9,512 | 1,336 |

"Training tokens" is PyTRIO's `trainer/token_count`; "prompt tokens" is `token_count − 8,176`. Every step builds 16 datums and generates 8,192 tokens, so those columns are not repeated.

![Reverse KL swings between steps while token-level spread stays at 5x the mean](/images/agentic-rl/opd-reverse-kl-en.png)

Three things worth remembering:

- **Do not read a "drop" as a conclusion.** The 10-step mean is 0.1264, the step-to-step relative variation is 12.6%, and the drop from the first five steps to the last five is 3.9% — a factor of three apart.
- **The mean hides the individual differences.** The token-level standard deviation is **4.67 to 5.71 times** the mean at every step; a rough Gaussian estimate puts about 42% of tokens at a negative reverse KL.
- **The swing tracks neither difficulty nor prompt length** (r = −0.03 / +0.22, using DeepMath's own `difficulty` field). The main cause is that every step swaps in fresh problems and fresh trajectories.

## `loss_mean` is a diluted `reverse_kl_mean`

![trainer/loss_mean is not a new signal: it is reverse KL divided by a dilution factor](/images/agentic-rl/opd-loss-dilution-en.png)

The two curves in the left panel coincide on all 10 steps, with a residual of at most 0.78%:

$$\text{trainer/loss\_sum} \approx 8{,}176 \times \text{opd/reverse\_kl\_mean}$$

because `importance_sampling` at $\rho\approx1$ collapses to a sum of $-A_t$, and $A_t=-\beta r_t$. PyTRIO reports `loss_mean` as that sum divided by `token_count` — **and `token_count` also contains the prompt placeholders**:

$$\text{trainer/loss\_mean} = \text{opd/reverse\_kl\_mean} \times \frac{8{,}176}{\text{token\_count}}$$

That denominator is the dilution factor in the right panel, swinging between **1.126 and 1.279**. The correlation between `reverse_kl_mean` and `loss_sum` is **0.99948**, while with `token_count` it is only 0.22.

**So watch `opd/reverse_kl_mean` and ignore `trainer/loss_mean` for OPD.** 

Why is the residual not exactly zero? Because the **last token of each answer is a target only, never an input position**, while the logged `reverse_kl_mean` includes it. Accounting for that, the residual equals the deviation of those 16 "last tokens"; the implied z-scores stay within ±0.8σ across all 10 steps with a mean of +0.03, and no systematic direction — so the residual does not come from $\rho\neq1$.

The dilution factor can be checked independently: `token_count = 4 × Σ prompt tokens + 8,176`. Recomputing the real prompt lengths (57 to 212 tokens) gives step 5 as $4\times(91+173+94+212)+8{,}176=10{,}456$, exactly what the log says, and **all 10 steps match exactly**.

## Token accounting: all 160 answers hit the cap

![Where the training tokens go: 80% in the answers, 20% in prompt placeholders](/images/agentic-rl/opd-token-budget-en.png)

Ten steps generated 81,920 tokens and pushed 97,492 positions through training:

| destination | positions | share |
|---|---:|---:|
| completion positions (carry a gradient) | 81,760 | 83.9% |
| prompt placeholders (advantage = 0) | 15,732 | 16.1% |

The missing 160 positions are the last token of each answer: **generated, scored by the teacher, included in the logged mean — and producing no gradient.**

More important is the right panel: every step's `completion_tokens_mean` is exactly 512.0, which is `max_tokens`. **Not one of the 160 answers stopped on its own.** The consequences chain: no answer ever finished, so not a single `\boxed{}` appeared; what the teacher scored was a chain of thought **truncated to 512 tokens**; and the student only ever learned to "look like the teacher over the first 512 tokens", never rewarded for finishing.

**If this configuration was meant to improve mathematical accuracy, it cannot do so by design.** This needs no evaluation to establish: not a single answer was completed, so even the precondition "the answer can be extracted" does not hold.

## What these 10 steps actually trained

1. **It ran.** 160 answers, 81,920 generated tokens, 8,176 positions with a gradient, nothing missing; the data flow matches the documentation.
2. **There is no evidence the student got stronger.** Reverse KL fell 3.9%, inside the step-to-step noise, and it measures how much the student looks like the teacher on paths the student itself walks, not accuracy.
3. **This run structurally cannot measure accuracy.** All 160 answers hit the 512 cap, none finished.
4. **The most useful thing I learned was how to read the metrics.** `loss_mean` is `reverse_kl_mean` divided by a coefficient between 1.13 and 1.28, and `token_count` can be reconstructed position by position from the prompt lengths.

If I continued: raise `max_tokens` until answers actually finish (that number has to be measured first), set `enable_thinking` consistently with that, and hold out a fixed set of problems so the pre- and post-training students answer them under identical decoding settings, comparing accuracy and answer-extractability together.

## Reproducing it

Reading the raw metrics beats scraping the SwanLab web page (the `.swanlab` file is a record stream, and swanlab's own reader gives back every scalar of every step):

```python
from swanlab.sdk.internal.core_python.store import DataStoreReader
from swanlab.proto.swanlab.record.v1 import record_pb2

reader = DataStoreReader()
reader.open("swanlog/run-20261125_174153-b9myg8ou/run-b9myg8ou.swanlab")
for blob in reader:
    rec = record_pb2.Record(); rec.ParseFromString(blob)
    if rec.HasField("scalar"):
        print(rec.scalar.key, rec.scalar.step, rec.scalar.value.number)
```

```bash
uv run python experiments/general-opd/01-demo-sync.py \
  --teacher-base-model Qwen/Qwen3.8-27B \
  --steps 10 --batch-size 4 --group-size 4 \
  --max-tokens 512 --sample-size 100 --swanlab-mode online
```

The snag I hit: the first run already printed `Loaded 20 DeepMath prompts` and then failed with `model_not_found` while creating the teacher client. I had written `Qwen3.6-27B` from older docs, but the account only exposes `Qwen3.8-27B`. This one line lists the models available to the current account:

```bash
uv run python -c 'import pytrio; print(*pytrio.ServiceClient().get_supported_models(), sep="\n")'
```

## References

- [Datawhale · General OPD tutorial](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/)
- [Sync training script](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/01-demo-sync.py) · [async training script](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/02-demo-async.py)
- [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [DeepMath-103K dataset](https://modelscope.cn/datasets/AI-ModelScope/DeepMath-103K)
