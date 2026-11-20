---
title: "Datawhale Agentic RL Task2: reading GRPO from the formula down to the code"
date: 2026-11-20
summary: "A section-by-section read of the agentic-rl-lab GRPO implementation — what rollout, reward, advantage, Datum and the update actually do — followed by the real records left by my ten-step run: only 112 of 320 sampled completions reached the update, and three steps produced no gradient at all."
tags: ["reinforcement-learning", "GRPO", "PyTRIO", "implementation", "study-notes"]
categories: ["Study Notes"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Task 02 of the Datawhale *Agentic RL* study group: the GRPO chapter. Tutorial: [GRPO starts with group comparison](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/), quick start: [run guide](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html), code: [datawhalechina/agentic-rl-lab/experiments/grpo](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo). Written 2026-11-20.

The previous note (Task 01) was about the "why": why the baseline can become a group mean, and how the IS / PPO / CISPO weight functions fall out. This one is about the "how" — reading `experiments/grpo/01-demo-sync.py` end to end, and then going back to what my actual run on this machine left behind.

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

## 4. Implementation details worth keeping

These are the things that only show up when you read the code *and* look at your own run; the tutorial text does not mention them:

1. **`swanlog/` is git-ignored.** The repo's `experiments/grpo/swanlog/.gitignore` contains `*`. So this run's config, metadata and logs exist **only on this machine** and disappear with it. Copy them out if you want the record.
2. **`run_name` is both the experiment name and the weight name.** `build_run_name` produces `grpo-{sync|async}-{model}-{task}-{loss}-steps{N}`; it changes automatically when you switch losses, so a batch of experiments sorts by name in SwanLab, and `save_weights_for_sampler(name=run_name)` can find the matching weights.
3. **Questions wrap around unless `--all-data`.** `pick_batch` indexes `(start + offset) % len(dataset)`, so once steps × batch exceeds the dataset size, questions repeat. Here 40 < 7473, so nothing repeated.
4. **`--all-data` overrides `--steps`.** `effective_steps = ceil(7473/4) = 1869` steps, one pass over the data, no wrapping, each sample used at most once.
5. **Every step saves an anonymous weight snapshot.** `save_weights_and_get_sampling_client()` is called once per step, so the framework handles "one old policy per step" for you — which is why this demo needs no hand-written weight synchronisation.
6. **The local torch is a CPU build, but custom losses really use it.** The CPU torch only starts working when `forward_backward_custom` sends logprobs back for a local loss computation.
7. **The log contains two loss fields.** `loss_mean` and `loss` hold the same value; other trainer keys are prefixed `trainer/`, and CISPO's own metrics keep a `cispo/` prefix — so filter by name when exporting CSVs instead of staring only at `reward`.

## 5. Where this demo differs from the paper's GRPO

Three simplifications were made to keep the pipeline legible (the tutorial's own readme lists them too):

| Paper version | This demo | How to change it |
| --- | --- | --- |
| group normalization $A_i=\frac{r_i-\text{mean}}{\text{std}+\delta}$ | mean subtraction only | add `std_reward = np.std(rewards)` |
| explicit reference KL term | no KL anchor | add $\beta \log\pi_\theta/\pi_{\text{ref}}$ in a custom loss |
| reward model / process reward | 0/1 rule-based reward only | swap in a PRM or finer reward shaping |

The paper's GRPO also separates $\pi_{\text{old}}$ from $\pi_{\text{ref}}$: the former is the per-round sampling snapshot (the denominator of the probability ratio), the latter is the initial SFT model (the KL anchor). This demo has no $\pi_{\text{ref}}$ at all, so do not expect to read a KL signal off its loss curve.

One demo-specific simplification deserves its own line: **the advantage is broadcast across the whole answer.** Paper GRPO does this too (the missing token-level credit assignment is exactly what DAPO / GSPO / Dr.GRPO address), but "the code does it this way" and "the algorithm is defined this way" are two different statements.

## 6. Next

- **Run the three losses and compare.** Same hyperparameters, only `--loss-fn` changes — this is what the repo most wants you to do. It needs cloud quota, so I will check the balance before launching.
- **Raise the group size and the task difficulty.** The current bottleneck is that 65% of sampling is thrown away: GSM8K is too easy for Qwen3.5-4B, and step 0 is already all correct. Try harder questions (MATH), or raise `group_size` from 8 to 16, and see whether `frac_degenerate` drops.
- **Restore the simplifications**: group normalization plus a KL term — good practice for `forward_backward_custom`.
- **Add a fixed evaluation set** to separate "training reward" from "generalization reward" — a requirement that came directly out of the observation in §3.3.

> Series index: [Agentic RL Lab notes](/series/agentic-rl-lab/). Previous: [Task 01 · a roboticist's first pass at LLM RL](/posts/from-controller-to-policy-gradient/).

## References

- Course repo: [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task 02 tutorial: [GRPO starts with group comparison](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) ｜ [quick start](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html)
- Code for this run: [experiments/grpo/01-demo-sync.py](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)
- DeepSeekMath (origin of GRPO): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PyTRIO docs: [pytrio.cn](https://pytrio.cn/) ｜ SwanLab: [swanlab.cn](https://swanlab.cn/)
