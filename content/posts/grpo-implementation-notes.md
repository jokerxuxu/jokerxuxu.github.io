---
title: "Datawhale Agentic RL Task2: reading GRPO from the formula down to the code"
date: 2026-11-20
summary: "A section-by-section read of the agentic-rl-lab GRPO implementation, then running importance_sampling, ppo and cispo myself and comparing with another record from the group: only 35% of sampled completions reached the update, the same configuration differs by 30% between two days, and the probability ratio is identically 1 in this demo — so the clip never fires and the three losses cannot be separated structurally."
tags: ["reinforcement-learning", "GRPO", "PyTRIO", "implementation", "study-notes"]
categories: ["Study Notes"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Task 02 of the Datawhale *Agentic RL* study group: the GRPO chapter. Tutorial: [GRPO starts with group comparison](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/), quick start: [run guide](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html), code: [datawhalechina/agentic-rl-lab/experiments/grpo](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo). Written 2026-11-20.

The previous note (Task 01) was about the "why": why the baseline can become a group mean, and how the IS / PPO / CISPO weight functions fall out. This one is about the "how" — reading `experiments/grpo/01-demo-sync.py` end to end, and then going back to what my actual run on this machine left behind.

Section 5 compares my runs with the two records a study partner left (a 10-step post and a later async 200-step run), and section 4 is the part where I ran things myself: `importance_sampling`, `ppo` and `cispo` on the same machine, plus **a repeat of one unchanged configuration**. I read the results side by side with an independent record from a study partner ([Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432)).

Up front: **the GRPO implementation is much shorter than the formulas.** What is hard is never typing the equation into code; it is the places where a mistake raises no error and simply makes the model learn something subtly wrong: the policy snapshot used for sampling, the right-shift alignment, degenerate groups, and the question of whether a step with no data should update at all.

## 1. What one step actually does

The whole training loop compresses into six moves. The diagram below comes from the tutorial repo (Apache-2.0):

![GRPO flow: sample one group of answers per question, subtract the group mean to get advantages, update the policy](/images/agentic-rl/grpo-pipeline.png)

Mapped onto `01-demo-sync.py`:

| Stage | Function | Output |
| --- | --- | --- |
| pick questions | `pick_batch` | `batch_size` GSM8K questions |
| build prompt | `build_prompt` | token sequence (few-shot + question + `\boxed{}` request) |
| sample | `run_rollout_group` | `group_size` × `(tokens, logprobs, text)` per question |
| grade | `grade_answer` | one 0/1 reward per answer |
| advantage | `reward - mean_reward` | one group-relative advantage per answer |
| update | `forward_backward` + `optim_step` | one step of LoRA parameters |

Let me go stage by stage.

## 2. Reading the code

### 2.1 Prompt: put the format requirement in the instruction

```python
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
```

Preceded by one few-shot example ("how many r's are in strawberry"), then rendered through the chat template:

```python
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

`enable_thinking=False` keeps sampling short and answer extraction simple. **None of this is algorithmic — it just makes "the answer can be scored by a rule" true.** RLVR needs verifiable answers before anything else can happen.

### 2.2 Reward: only the last `\boxed{}` counts

```python
def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()

def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0
```

The **last** match rather than the first, because a model may compute wrong and then correct itself; `normalize_answer` only strips commas and a trailing dot so that `1,000` and `1000` compare equal.

A missing `\boxed{}` scores 0 — which silently merges "did not follow the format" with "computed the wrong number". That is an obvious place where this demo could be made finer.

### 2.3 Sampling: the old policy must be a snapshot taken at that moment

The single most important line of the whole implementation:

```python
sampling_client = training_client.save_weights_and_get_sampling_client()
```

It sits at the top of every step, which means: **each step saves an anonymous snapshot of the current LoRA weights, and that snapshot is the old policy $\pi_{\text{old}}$ used for this round of sampling.**

Sampling asks for a whole group at once:

```python
result = sampling_client.sample(
    prompt=trio.ModelInput.from_ints(prompt_tokens),
    num_samples=group_size,
    sampling_params=sampling_params,
    return_text=True,
).result()
```

`num_samples=group_size` *is* the definition of GRPO: the same prompt needs multiple completions, otherwise there is nothing to compare within a group. The three things that come back are all the raw material training will ever need:

```python
tokens   = list(sequence.tokens)                  # generated tokens
logprobs = [float(v) for v in sequence.logprobs]  # old-policy logprobs of those tokens
reward   = grade_answer(text, ground_truth)
```

**The `logprobs` must come from the policy that was live while sampling.** They are the denominator of $\rho=\exp(\log\pi_\theta-\log\pi_{\text{old}})$ in every loss below. Recomputing them after the parameter update — the lazy version — forces $\rho\equiv 1$, which disables the correction in all three losses. And it raises no error: it just quietly degrades into plain policy gradient.

### 2.4 Advantage: subtract the mean, drop the degenerate groups

```python
mean_reward = sum(rewards) / len(rewards)
advantage = reward - mean_reward
```

Example: group rewards `[1, 0, 1, 0]`, mean 0.5, advantages `[+0.5, -0.5, +0.5, -0.5]`.

If every reward in the group is identical, all advantages are 0 and the group carries no relative information, so it is skipped:

```python
if all(sample.advantage == 0.0 for sample in rollout_samples):
    n_degenerate += 1
    continue
```

`frac_degenerate` records how many questions of this step were in that state. **It says more about whether this run is learning than the reward does** — which my own data will demonstrate.

### 2.5 Datum: right-shift, and the prompt segment is all padding

PyTRIO's built-in `importance_sampling` / `ppo` losses need three equal-length arrays: `target_tokens`, `logprobs`, `advantages`. `build_grpo_datum` aligns them:

```python
observation_len = len(prompt_tokens) - 1
input_tokens  = prompt_tokens + sample.tokens[:-1]
target_tokens = [0] * observation_len + sample.tokens
padded_logprobs = [0.0] * observation_len + sample.logprobs
padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)
```

The alignment looks like this (`p` is prompt, `c` is completion):

```text
input   = p0 p1 ... p_{n-1} | c0 c1 ... c_{T-2}
target  =  0  0  ...   0   | c0 c1 ... c_{T-1}
adv     =  0  0  ...   0   |  A   A  ...   A
```

Three things worth remembering:

1. **Shifted right by one.** Token $i$ predicts token $i+1$, so the input drops the last completion token and the target is the full completion.
2. **`observation_len` is `len(prompt) - 1`, not `len(prompt)`.** The last prompt token stays in the input as context and predicts the first completion token, so the prompt padding is $n-1$ long and the target starts at the first completion token.
3. **The prompt segment is filled with zeros.** The prompt is context, not the object being optimized; zero `advantages` keeps the gradient from flowing into it.

And one more: **the advantage is broadcast onto every completion token of that answer** (`[sample.advantage] * len(sample.tokens)`). "This answer beat the group average, so all of its tokens should be encouraged" — **token-level credit assignment is simply absent here.** That is what DAPO, GSPO and Dr.GRPO later address; worth keeping in mind.

### 2.6 The update: on an empty batch, the optimizer does not even step

```python
if datums:
    if config.loss_fn in BUILTIN_LOSS_FNS:
        fwd_bwd_future = training_client.forward_backward(datums, loss_fn=config.loss_fn)
    elif config.loss_fn == "cispo":
        ...
    optim_future = training_client.optim_step(adam_params)
    fwd_bwd_result = fwd_bwd_future.result()
    optim_future.result()
    loss_metrics = dict(fwd_bwd_result.metrics)
else:
    loss_metrics = {}
```

Note that `optim_step` sits inside `if datums:`. If every question of that step degenerated, `datums` is empty, and **that step runs no forward/backward and no optimizer update at all** — Adam's moment counters and step count stay put.

That is not necessarily bad: updating on an empty batch would pollute the Adam moments with zero gradients and could advance an lr schedule by one. But it is an *implicit* convention, easy to miss while reading. The log makes it visible: **a step with no `loss_mean` is a step that learned nothing.**

### 2.7 Custom loss: CISPO defines the gradient shape locally

`importance_sampling` and `ppo` are built into PyTRIO; `cispo` is not, and goes through `forward_backward_custom`:

```python
custom_datums = [build_custom_forward_datum(datum) for datum in datums]
fwd_bwd_future = training_client.forward_backward_custom(
    custom_datums,
    make_cispo_loss_fn(sampling_logprobs_list=..., advantages_list=..., ...),
)
```

The division of labour:

1. the remote model runs a forward pass and returns the per-token logprob of the **current** policy on `target_tokens`, differentiable;
2. that logprob comes back to the local process and is handed to `loss_fn`;
3. you write any loss you like in torch and return `(loss, metrics)`;
4. PyTRIO uses that loss to backpropagate and update the parameters.

In other words, **you are not training a model locally; you are only defining locally how those logprobs should become a loss.** This also explains Task 01's warning that the local torch build is CPU-only: the local torch does real work exactly when a custom loss is used.

The core of CISPO is where `detach` sits:

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
clipped_ratio = torch.clamp(prob_ratio, min=clip_low_threshold, max=clip_high_threshold)
cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
```

The clipped ratio acts as a fixed weight; only `target_logprobs` carries the gradient. Same conclusion as in Task 01: **PPO clips the objective itself (out of bounds, it stops pushing); CISPO caps the force (out of bounds, it keeps pushing, just not harder).**

## 3. How this run actually went

### 3.1 Environment and command

In `~/Workspace/agentic-rl-lab` (the tutorial repo, at commit `f66c6fe`):

```bash
uv sync
trio login
swanlab login

uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --swanlab-mode online
```

`--loss-fn` was omitted and took the default `importance_sampling`. Machine: MacBook Air (M5, 16 GB), Python 3.13.18, torch 2.13.0, pytrio 0.2.9, swanlab 0.10.4, datasets 5.0.0.

### 3.2 Configuration and the local archive

Every run leaves a directory under `experiments/grpo/swanlog/` that contains the full record of that experiment:

```text
experiments/grpo/swanlog/run-20261117_210611-2gmiydii/
├── files/config.yaml              # every hyperparameter
├── files/swanlab-metadata.json    # machine, python version, git commit, command line
├── files/requirements.txt         # dependency versions at the time
├── debug/debug.log                # SwanLab warnings
└── run-2gmiydii.swanlab           # binary log
```

`config.yaml` holds the hyperparameters I used:

| Parameter | Value | | Parameter | Value |
| --- | --- | --- | --- | --- |
| base_model | `Qwen/Qwen3.5-4B` | | learning_rate | 4e-5 |
| lora_rank | 32 | | batch_size | 4 |
| steps | 10 | | group_size | 8 |
| max_tokens | 512 | | temperature / top_p | 1.0 / 1.0 |
| loss_fn | `importance_sampling` | | seed | 42 |

Online record: [SwanLab · agentic-rl-lab-grpo](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo/runs/2gmiydii). The final weight name follows the run name: `grpo-sync-qwen35-4b-gsm8k-importance-sampling-steps10`.

### 3.3 What the ten steps say

| step | reward | degenerate | avg_gen_len | loss_mean | completions entering the update |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.000 | 100% | 186.7 | — | 0 |
| 1 | 0.875 | 75% | 253.9 | 0.0540 | 8 |
| 2 | 0.750 | 0% | 380.7 | 0.0445 | 32 |
| 3 | 0.750 | 100% | 266.9 | — | 0 |
| 4 | 0.781 | 75% | 294.5 | −0.0271 | 8 |
| 5 | 0.781 | 25% | 275.4 | 0.0627 | 24 |
| 6 | 1.000 | 100% | 250.8 | — | 0 |
| 7 | 0.938 | 50% | 258.4 | 0.0286 | 16 |
| 8 | 0.969 | 75% | 224.3 | 0.0259 | 8 |
| 9 | 0.875 | 50% | 237.1 | 0.0173 | 16 |

The two figures below are redrawn from the exported SwanLab CSVs (not SwanLab screenshots), because the quantities I wanted are not on its default panels.

![Which groups produced the gradient each step, and which steps produced none](/images/agentic-rl/grpo-signal-budget.png)

**First observation: the signal budget of this run was 35%.**

Ten steps sampled $10\times4\times8=320$ answers, and because degenerate groups are skipped, only **112 of them (35%)** reached the update. **Three of the ten steps (0, 3, 6) produced no datum at all**, and their `loss_mean` is empty in the log — exactly matching §2.6's "on an empty batch even the optimizer does not step". The green bars in the left panel are where a step has usable groups; the three bars labelled "no update" are the empty ones.

**Second observation: the reward curve cannot tell you whether learning happened.**

On the surface the reward wobbles between 0.75 and 1.0 with no upward trend. But the deeper reason is that **the questions change every step.** `pick_batch` takes consecutive slices of the dataset — rows 0–3 at step 0, rows 4–7 at step 1 — so the forty questions of the ten steps are all distinct and differ in difficulty. Step-to-step reward differences therefore mix in question difficulty instead of measuring "the policy got better". The run guide also notes that this repository has **no separate `eval.py`**, so between "training reward" and "the model improved" there is a missing link: a fixed evaluation set.

**Third observation: the two slow steps were not slow because of training.**

![Time between consecutive logged steps: two spikes caused by SwanLab upload retries](/images/agentic-rl/grpo-step-latency.png)

The run went from 21:06:35 to 21:19:40 — 13 minutes, with a median of 35 s per step. But step 7 took 174 s and step 8 took 389 s. In `debug/debug.log` there are seven lines of

```text
Upload failed due to network or server issues, retrying automatically.
```

and all of them fall inside those two steps: **it was SwanLab uploads hanging, not the model computing.** By the median, actual training time was about 350 s, and roughly 435 s were spent waiting for log retries (about 55%). The practical lesson: when a remote training run feels slow, first separate "the compute is slow" from "the logging is slow", otherwise you misjudge both cost and iteration rhythm.

### 3.4 Cost

The tutorial author's own numbers: about 5 CNY for ten steps, 55 CNY to get the whole script working. Mine was a single ten-step demo on the quick-start's smallest configuration; the actual consumption is whatever the PyTRIO console bills, and locally the only thing I can pin down is wall clock — 13 minutes, of which about 7 went to log uploads.

## 4. Running it myself: three losses, same machine

A study partner's write-up reminded me of something: with only one loss, you never see what distinguishes the three objectives. So on 2026-11-20 I ran `ppo` and `cispo` as well, in the same repo on the same machine, and **re-ran `importance_sampling` unchanged** — to find out how much two runs of one configuration can differ. With the 11-17 run that makes four curves.

The hyperparameters never moved (Qwen3.5-4B, LoRA rank 32, lr 4e-5, 10 steps, bs 4, gs 8, 512 tokens, temperature 1.0, seed 42); only `--loss-fn` changed:

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --loss-fn {importance_sampling|ppo|cispo} \
    --swanlab-mode online
```

All four runs live in one SwanLab project: [agentic-rl-lab-grpo](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo) — IS (11-17) `runs/2gmiydii`, PPO (11-20) `runs/wq9npu15`, CISPO (11-20) `runs/ulo34fae`, IS repeat (11-20) `runs/fscv1nz3`.

| Metric | IS 11-17 | IS 11-20 (repeat) | PPO 11-20 | CISPO 11-20 |
| --- | --- | --- | --- | --- |
| starting reward | 1.000 | 1.000 | 0.969 | 1.000 |
| final reward | 0.875 | 0.938 | 0.938 | 0.938 |
| lowest reward | 0.750 (s2) | 0.719 (s3) | 0.750 (s3) | 0.688 (s3) |
| mean reward (steps 1–9) | 0.858 | 0.920 | 0.878 | 0.892 |
| mean frac_degenerate | 0.65 | 0.75 | 0.60 | 0.65 |
| steps with no update | 0, 3, 6 | 0, 7 | 3 | 0, 6 |
| completions used (datums) | 112 | 80 | 128 | 112 |
| train_tokens | 37,906 | 25,920 | 41,194 | 35,412 |
| times loss_mean went negative | 1/7 | 1/8 | 3/9 | 7/8 |
| avg_gen_len peak → end | 380.7→237.1 | 365.0→292.6 | 373.3→225.0 | 370.2→253.0 |
| wall clock per run (incl. logging) | 785 s | 352 s | 360 s | 320 s |

![Four curves: reward does not separate, degenerate groups dominate, only CISPO sits reliably below zero, and the same configuration differs by 30% between days](/images/agentic-rl/grpo-three-losses.png)

### 4.1 First, the noise floor: one configuration, two runs

Look at panel 4: IS 11-17 and IS 11-20 are **the same command**, one day apart.

- completions used: 112 vs 80 — 30% apart (train_tokens 37.9k vs 25.9k, 46% apart)
- mean reward: 0.858 vs 0.920 — 0.062 apart
- lowest reward: 0.750 vs 0.719

**So PPO looking like it produced the most signal (128 datums) says nothing about it being stronger — IS alone differs by 30% between two runs.** No between-group difference smaller than that can be called a result. This is the most useful thing I got out of this batch.

It also explains an oddity: **why do two runs with the same seed 42 differ so much?** Because `np.random.seed` only governs local randomness; the remote sampler is not deterministic.

### 4.2 The deeper finding: the clip never fired

CISPO also records `cispo/ratio_mean` and `cispo/clip_fraction`. I added them as reference metrics, and they pointed at the root of the problem:

![The probability ratio stays at 1 and clip_fraction is zero on every step](/images/agentic-rl/grpo-cispo-ratio.png)

- `ratio_mean` sits within **1 ± 5.3×10⁻⁴** for all ten steps
- `clip_fraction` is **zero on every step**

$\rho=\exp(\log\pi_\theta-\log\pi_{\text{old}})$ is essentially 1, which means **the policy barely moved between sampling and the update.** And if $\rho$ never leaves 1:

$$
w_{\text{IS}}(\rho)=\rho\approx 1,\qquad
w_{\text{PPO}}(\rho)=\operatorname{clip}(\rho,1\pm\varepsilon)\approx \rho,\qquad
w_{\text{CISPO}}(\rho)=\operatorname{sg}(\operatorname{clip}(\rho))\approx \operatorname{sg}(\rho)
$$

then all three losses **took the same branch** during this batch. Task 01's conclusion — that the difference between the three losses is compressed into a one-dimensional force function $w(\rho)$ — is right, but it presumes the different regions of $w$ are actually visited. Here, none of them was.

**So the conclusion "ten steps cannot separate them" has to be rewritten: it is not that there were too few steps, it is that $\rho$ never left 1, so the differences between the three losses never got a chance to act.**

### 4.2.1 Worse: in this demo $\rho$ is identically 1

Push that one step further and the previous paragraph turns out to be wrong in its explanation: the clip does not fail to fire because the learning rate is small — it is **structurally impossible to fire**. Look at the order of operations:

```python
sampling_client = training_client.save_weights_and_get_sampling_client()  # snapshot W_k
...                                                                      # sample with W_k → sampling_logprobs = log π_{W_k}(o)
fwd_bwd_future = training_client.forward_backward(datums, loss_fn=...)    # forward with W_k → target_logprobs = log π_{W_k}(o)
optim_future = training_client.optim_step(adam_params)                    # W_k → W_{k+1}
```

**When the loss is evaluated, the learner's weights are exactly the weights the data came from**, so

$$
\rho_t=\exp(\log\pi_{W_k}(o_t)-\log\pi_{W_k}(o_t))\equiv 1 .
$$

![Weight flow inside one step: sampling and the forward pass share the same $W_k$, so the ratio is identically 1](/images/agentic-rl/grpo-ratio-one.png)

The measured $1\pm5.3\times10^{-4}$ is numerical noise from the forward pass (batched inference versus decoding during sampling), not policy drift — over ten steps it shows no trend, it just random-walks around 1.

Substituting into the three formulas in the [PyTRIO docs](https://docs.pytrio.com/docs/guide/loss_fn):

| loss | formula (from the docs) | at $\rho=1$ |
| --- | --- | --- |
| `importance_sampling` | `loss = -(prob_ratio * advantages).sum()` | $-(1/N)\sum_t A_t$ |
| `ppo` | $-\mathbb{E}\big[\min(\rho A,\ \operatorname{clip}(\rho,1\pm\varepsilon)A)\big]$ | $\operatorname{clip}(1)=1\Rightarrow -(1/N)\sum_t A_t$ |
| `cispo` | $-\sum_t \operatorname{sg}(\operatorname{clip}(\rho))\log\pi_\theta(o_t)A_t$ | $+(1/N)\sum_t \mathrm{NLL}_t A_t$ |

**IS and PPO are literally the same expression at $\rho=1$ — and so are their gradients** (all three give $-\mathbb{E}[A\,\nabla\log\pi]$). Only CISPO is a different function. The three losses can therefore differ only where $\rho\neq1$: several inner epochs over the same batch (from the second epoch on, $\rho$ departs from 1), or a sampler that lags the trainer (genuinely off-policy).

Both `01-demo-sync.py` and `02-demo-async.py` **re-snapshot every step**, so $\rho$ is 1 in both. **Which means: no number of extra steps, no learning rate, no larger model will separate the three losses with this demo.** My earlier suggestion to "raise the learning rate so $\rho$ starts moving" was wrong; corrected here.

### 4.2.2 So what decides the sign and the size of the loss? A batch-level lottery

Write $T_i$ for the length of completion $i$ and $A_i$ for its advantage (zero-mean within the group). Then

$$
L_{\text{IS}}=L_{\text{PPO}}=-\frac1N\sum_i T_i A_i=-\frac1N\sum_i (T_i-\bar T)A_i ,
$$

so **the sign only asks whether the correct answers in this step's batch happened to be longer** — nothing in the algorithm constrains it. And for CISPO:

$$
L_{\text{CISPO}}=\frac1N\sum_i\Big(\sum_t \mathrm{NLL}_{i,t}\Big)A_i\;\approx\;\frac{h}{N}\sum_i T_i A_i=-h\cdot L_{\text{IS}},
$$

where $h$ is the policy's mean per-token negative log-likelihood (entropy, in nats). Two testable predictions follow: **CISPO has the opposite sign to IS/PPO**, and **its magnitude is about $h$ times theirs**.

Measured on the same 40 questions across the four runs:

| Check | Result |
| --- | --- |
| IS 11-17 vs IS 11-20 (two independent samples) | same sign on 6/6 shared steps |
| IS (mean of the two) vs CISPO | **opposite sign on 6/6 steps** |
| PPO vs the IS consensus | same sign on 5/6 steps (the exception, step 1, has PPO = −0.0011, essentially a tie) |
| fit through the origin, $L_{\text{CISPO}}=-h\,L_{\text{IS}}$ | $h=0.78$ nats/token |

![Left: the CISPO loss is the mirror image of the IS loss, and the slope is the policy entropy; right: a per-step sign matrix in which the CISPO row mirrors the two IS rows](/images/agentic-rl/grpo-loss-sign.png)

The right panel shows the mirror most clearly: **the two IS rows are nearly identical (two independent samples), while the CISPO row is flipped overall.** Even *which* steps go negative is the same set of steps (step 4), with the opposite sign.

So **"CISPO's loss is systematically negative" is not a coincidence — it is what the formula says** (it multiplies the logprob rather than the ratio).

**"PPO's loss is positive throughout" has no theoretical basis.** The sharpest counter-evidence is inside the study partner's own numbers: his IS and PPO runs are **the same function** at $\rho=1$, yet his IS went negative 3/9 times. One function producing different signs in two samples is a sign that the sign is a lottery outcome, not an algorithm signature.

### 4.2.3 "The custom loss is slower": real mechanism, unmeasurable at this scale

The mechanism is genuinely more expensive: CISPO ships the logprobs back to the local process, computes the loss with CPU torch, and sends the gradient back, while a built-in loss runs entirely on the remote GPU. At ten steps that overhead is drowned by sampling time and log uploads, so "CISPO 678 s vs IS 638 s" on his side and "CISPO 320 s, actually the fastest" on mine are not in conflict — **the direction is right and the magnitude is unmeasurable**. Measuring it needs logging disabled, many more steps, and repeats.

### 4.3 What reproduced and what did not

| Claim | Study partner's post | Mine |
| --- | --- | --- |
| CISPO's loss is systematically negative | negative 6/8 times | negative 7/8 times ✓ |
| IS produces negative losses | negative 3/9 times | negative 1/7 and 1/8 times |
| PPO's loss is positive throughout | yes | no, negative 3/9 times ✗ |
| "explore longer, then converge" | gen_len peaks around step 2 | all four peak at step 2 (365–381) then fall back to 225–293 ✓ |
| custom loss is slower | CISPO 678 s vs IS 638 s | CISPO 320 s, IS 352 s, PPO 360 s ✗ |

What reproduced is **structural**: CISPO's loss sits below zero almost always (it multiplies the logprob rather than the ratio, see §4.2.2), and all four runs lengthen before they converge. What did not reproduce is the **numeric** kind of claim: how many times a sign flipped, which loss was slower. The former says single runs are noisier than that post suggests; the latter is most likely masked by logging time — in my 11-17 run, 435 s of the ten steps went to waiting for SwanLab to re-upload, an order of magnitude more than the gap between the three losses.

**But "longer first, then converges" needs a discount too.** All four runs (including the study partner's) walk through **the same 40 questions in the same order** — `pick_batch` takes consecutive slices, and `01-demo-sync.py` and `03-grpo.py` are identical at that step. So the four `avg_gen_len` curves correlate at **r = +0.87 to +0.98**, while reward correlates only +0.49 to +0.83 and `frac_degenerate` only +0.20 to +0.80.

**In other words, that length curve is mostly the difficulty profile of these 40 questions, not a dynamic of GRPO** — any algorithm, or even a random baseline, would draw the same shape. Which is the real reason "structural claims reproduce and numeric ones do not": **what reproduces is the shared experimental setup (the same questions, the same script, the same formula structure), not a property of the algorithm.**

![Left: the generation-length curves of the four runs almost coincide; right: the pairwise correlation matrix — length correlates strongly, degenerate groups do not](/images/agentic-rl/grpo-shared-questions.png)

The correlation matrix on the right spells it out: the length block is deep red at **r = 0.87–0.98**, the `frac_degenerate` block only **0.20–0.80**, reward in between. **The higher the correlation, the more the curve is given by the questions rather than by training.**

### 4.4 Timing: the custom loss is not more expensive

The three 11-20 runs took 352 s (IS), 360 s (PPO) and 320 s (CISPO) — within 40 s of each other, and that is the same order as the SwanLab upload stalls. **At ten steps you cannot measure the extra cost of a custom loss**; you would need many more steps and logging disabled for a fair comparison.

## 5. Comparing notes with my study partner

There is another study partner in the group, Kevin, with two Task 02 records:

1. **The Zhihu post** — [Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432): the sync script, 10 steps × 3 losses, exactly the same setup as my first run, so it compares item by item.
2. **A later async long run** — a SwanLab panel with three `grpo-async-qwen35-4b-gsm8k-*-steps200` runs (screenshot taken while running, at ~115 steps). The async script submits the four rollouts of a batch concurrently (`gather`), which drops a step from ~33 s to ~8 s and makes 200 steps affordable.

### 5.1 Item by item (sync, 10 steps)

| Metric | my IS 11-20 | my PPO | my CISPO | his IS | his PPO | his CISPO |
| --- | --- | --- | --- | --- | --- | --- |
| starting reward | 1.000 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| final reward | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 | 0.969 |
| lowest reward | 0.719 (s3) | 0.750 (s3) | 0.688 (s3) | 0.656 (s3) | 0.688 (s3) | 0.750 (s3) |
| mean reward (1–9) | 0.920 | 0.878 | 0.892 | ≈0.882 | ≈0.900 | ≈0.868 |
| frac_degenerate range | 50–100% | 25–100% | 25–100% | 50–100% | 25–100% | 0–100% |
| completions used | 80 | 128 | 112 | 96 | 96 | 112 |
| train_tokens | 25.9k | 41.2k | 35.4k | 33.1k | 29.4k | 39.4k |
| loss_mean negative | 1/8 | 3/9 | **7/8** | 3/9 | 0/9 | 6/8 |
| gen_len peak → end | 365→293 | 373→225 | 370→253 | 344→279 | 366→204 | 414→242 |
| wall clock per run | 352 s | 360 s | 320 s | 638 s | 539 s | 678 s |

**What agrees (the structural kind):**

- **Starting reward is 1.000 everywhere**: the model is too strong, step 0 is already all correct, and degenerate is high from the start.
- **Reward bottoms out around step 3 and recovers**: my minima are all at step 3, his are too. **That is not a coincidence** — both runs walk the same 40 questions in the same order, so the difficulty profile is shared (the correlation matrix in §4.3).
- **"Longer first, then converges"**: my gen_len peaks at step 2 (365–373), his at step 2 as well (344–414).
- **CISPO's loss sits systematically below zero**: 6/8 for him, 7/8 for me (the $-hL_{\text{IS}}$ relation of §4.2.2).

**What does not agree (the numeric kind):**

- **"PPO's loss is positive throughout"**: 0/9 for him, 3/9 for me. And PPO and IS are the same function at $\rho=1$, so this is not an algorithmic difference but a lottery difference (§4.2.2).
- **"The custom loss is slower"**: his CISPO 678 s vs IS 638 s (+6%), while mine has CISPO 320 s, faster than IS at 352 s. The mechanism is genuinely more expensive, but unmeasurable at this scale (§4.2.3).
- **The ranking of effective sample sizes**: his 96/96/112, mine 80/128/112. **The orderings disagree entirely** — because all of those columns sit inside the ±30% noise floor anyway.

### 5.2 The async 200-step runs did not change the conclusion

Three things can be read off his panel:

1. **Reward shows no upward trend over ~115 steps** (oscillating between 0.5 and 1.0, dipping to ~0.1) → "ten steps cannot show learning" upgrades to "**a hundred steps cannot show learning either**".
2. **The sign structure of loss_mean matches my derivation**: the CISPO curve is almost entirely below zero, while PPO and IS are mostly above it.
3. **gen_len oscillates between 100 and 400 with no trend**, and `frac_degenerate` swings between ~0.1 and 1.0 → same conclusion as my ten steps: these quantities are mostly decided by the questions.

**But the crucial point is that switching to async and running 200 steps still does not separate the three losses.** `02-demo-async.py`, like the sync version, **re-snapshots the sampler every step** (`save_weights_and_get_sampling_client_async()`, `:650`), and only runs `forward_backward` + `optim_step` after every rollout has come back (`:708`, `:737`) — so $\rho\equiv1$ holds in the async version too, and the clip still can never fire.

**That is harder than the "not enough statistical power" story I told earlier: it is not the number of steps, and not the sync/async difference — it is structural.** Apart from the value offset (exactly what $-hL_{\text{IS}}$ predicts), every difference between the three curves is sampling noise.

(His SwanLab project is private, so I cannot pull his raw metrics; this paragraph is a qualitative comparison only. The table in §5.1 uses the numbers printed in his post.)

## 6. Implementation details worth keeping

These are the things that only show up when you read the code *and* look at your own run; the tutorial text does not mention them:

1. **`swanlog/` is git-ignored.** The repo's `experiments/grpo/swanlog/.gitignore` contains `*`. So this run's config, metadata and logs exist **only on this machine** and disappear with it. Copy them out if you want the record.
2. **`run_name` is both the experiment name and the weight name.** `build_run_name` produces `grpo-{sync|async}-{model}-{task}-{loss}-steps{N}`; it changes automatically when you switch losses, so a batch of experiments sorts by name in SwanLab, and `save_weights_for_sampler(name=run_name)` can find the matching weights.
3. **Questions wrap around unless `--all-data`.** `pick_batch` indexes `(start + offset) % len(dataset)`, so once steps × batch exceeds the dataset size, questions repeat. Here 40 < 7473, so nothing repeated.
4. **`--all-data` overrides `--steps`.** `effective_steps = ceil(7473/4) = 1869` steps, one pass over the data, no wrapping, each sample used at most once.
5. **Every step saves an anonymous weight snapshot.** `save_weights_and_get_sampling_client()` is called once per step, so the framework handles "one old policy per step" for you — which is why this demo needs no hand-written weight synchronisation.
6. **The local torch is a CPU build, but custom losses really use it.** The CPU torch only starts working when `forward_backward_custom` sends logprobs back for a local loss computation.
7. **The log contains two loss fields.** `loss_mean` and `loss` hold the same value; other trainer keys are prefixed `trainer/`, and CISPO's own metrics keep a `cispo/` prefix — so filter by name when exporting CSVs instead of staring only at `reward`.
8. **You can pull the metrics back afterwards instead of watching the panel.** The SwanLab Python SDK fetches a run's full per-step history locally:

   ```python
   import swanlab
   api = swanlab.Api()
   exp = api.run("jokerxuxu/agentic-rl-lab-grpo/2gmiydii")
   m = exp.metrics(keys=["reward", "frac_degenerate", "loss_mean", "train_tokens"])
   # m["list"] -> [{"key": ..., "metrics": [{"index": step, "data": v, "timestamp": ms}]}]
   print(exp.summary()["reward"])   # also gives min / max / avg / median / stdDev
   ```

   Every figure in this note was redrawn from data pulled this way (cleaner than exporting CSVs by hand, and it reaches metrics like `cispo/clip_fraction`). **Advice: run first, pull everything once, analyse afterwards.**
9. **It still runs when `huggingface.co` is unreachable.** During the second batch of runs HF was completely unreachable, and `load_dataset` hung on HEAD retries for several minutes. GSM8K was already in the local cache, so `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` starts it straight from disk. **If you have already downloaded your data once, do not let the training script die on an unrelated connectivity check.**

## 7. Where this demo differs from the paper's GRPO

Three simplifications were made to keep the pipeline legible (the tutorial's own readme lists them too):

| Paper version | This demo | How to change it |
| --- | --- | --- |
| group normalization $A_i=\frac{r_i-\text{mean}}{\text{std}+\delta}$ | mean subtraction only | add `std_reward = np.std(rewards)` |
| explicit reference KL term | no KL anchor | add $\beta \log\pi_\theta/\pi_{\text{ref}}$ in a custom loss |
| reward model / process reward | 0/1 rule-based reward only | swap in a PRM or finer reward shaping |

The paper's GRPO also separates $\pi_{\text{old}}$ from $\pi_{\text{ref}}$: the former is the per-round sampling snapshot (the denominator of the probability ratio), the latter is the initial SFT model (the KL anchor). This demo has no $\pi_{\text{ref}}$ at all, so do not expect to read a KL signal off its loss curve.

One demo-specific simplification deserves its own line: **the advantage is broadcast across the whole answer.** Paper GRPO does this too (the missing token-level credit assignment is exactly what DAPO / GSPO / Dr.GRPO address), but "the code does it this way" and "the algorithm is defined this way" are two different statements.

## 8. Next

- **Make $\rho\neq1$ if you want to compare the three losses.** Right now they are the same function (§4.2.1), so more steps, a bigger learning rate, or switching between sync and async will not help — the study partner's async 200-step runs did not separate them either (§5.2). Two workable routes: several inner epochs over the same datums (from the second epoch on, $\rho$ departs from 1), or a deliberately stale sampler (stale weights, genuinely off-policy). This is also where the clip actually fires in the DAPO/GSPO-style setups.
- **Run a small "inner epoch" experiment to verify this.** Five lines of change: call `forward_backward` + `optim_step` twice on the same `datums`, then check whether `cispo/ratio_mean` visibly departs from 1, whether `clip_fraction` becomes non-zero, and whether the three loss curves finally fork. That is the shortest path from "the clip never fires" as a derivation to a measurement.
- **Raise the group size and the task difficulty.** The current bottleneck is that 60–75% of sampling is thrown away: GSM8K is too easy for Qwen3.5-4B, and step 0 is already all correct. Try harder questions (MATH), or raise `group_size` from 8 to 16, and see whether `frac_degenerate` drops.
- **Pair every comparison with a repeat of the same configuration.** That 30% noise floor from §4.1 was the most expensive lesson of this batch: without a repeat you cannot tell an algorithmic difference from luck.
- **Fix the questions and add a fixed evaluation set.** If the length curve correlates at r≈0.9 across runs, the curve is decided by the questions; to see whether the policy improved you need the same batch every step plus a held-out set that never trains. This is the requirement that came out of the observation in §3.3.
- **Restore the simplifications**: group normalization plus a KL term — good practice for `forward_backward_custom`.

> Series index: [Agentic RL Lab notes](/series/agentic-rl-lab/). Previous: [Task 01 · a roboticist's first pass at LLM RL](/posts/from-controller-to-policy-gradient/).

## References

- Course repo: [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task 02 tutorial: [GRPO starts with group comparison](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) ｜ [quick start](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html)
- Code for this run: [experiments/grpo/01-demo-sync.py](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)
- Independent record from a study partner: [Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432) (part of §4 comes from comparing against it)
- DeepSeekMath (origin of GRPO): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PyTRIO docs: [pytrio.cn](https://pytrio.cn/) ｜ SwanLab: [swanlab.cn](https://swanlab.cn/)
