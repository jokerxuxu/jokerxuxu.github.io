---
title: "Datawhale Agentic RL Task02｜Which answers actually reach the update?"
date: 2026-11-20
summary: "From the GSM8K scoring rule down to the per-token policy gradient, and back to four PyTRIO runs: degenerate groups, the probability ratio, and how many updates actually happened."
tags: ["reinforcement-learning", "GRPO", "PyTRIO", "implementation", "study-notes"]
categories: ["Study Notes"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

In the previous note I tried to use the robot RL I know to understand LLM training: how several answers to the same question provide each other with a baseline, and how IS / PPO / CISPO change what one token contributes to the policy gradient. I also ran ten steps on PyTRIO and found that the three rounds in which the whole batch degenerated happened to have no loss record at all. [Previous · A roboticist's first pass at LLM RL](/posts/llm-rl-begin/)

This time I want to read down through the source: how one question becomes a group of training samples, how one answer's advantage is handed out to its tokens, and where the probability recorded at sampling time enters the update? Once that is clear, I go back to the differences between the four runs.

## 1. From GSM8K to the group-relative advantage

I run GSM8K with Qwen3.5-4B. The training set holds 7,473 word problems, each with a question and a worked solution. The script takes whatever follows `####` in the reference answer as the grading target. The prompt the model sees is built from a few question-answer examples plus the question itself, and asks for the final answer written inside `\boxed{...}`. After generation, `grade_answer` extracts the last boxed match and compares it after stripping commas, leading and trailing whitespace, and a trailing period. A reward of 1 therefore means **the final answer passed this grading rule**. The reasoning chain earns nothing on its own, and a missing box also scores 0.

In this example the **policy** is the model's conditional distribution over the next token. The state is "the question plus the answer written so far", the action is choosing the next token. Letting the model generate one complete answer along its own choices is one **rollout**. The reward only arrives when the whole answer is done.

![One prompt produces several answers; the group is compared before the policy is updated](/images/agentic-rl/grpo-pipeline.png)

*Figure 1　Reading the course's flow diagram: the grouping happens inside a single question. One of my batches holds 4 questions, so there are 4 independent within-question comparisons.*

A batch contains several questions. The script calls `run_rollout_group` once per question, sampling a group of completions from the same prompt. Let the reward of answer $i$ be $r_i$, and let each group hold $G$ answers. What the source implements is

$$
A_i=r_i-\frac{1}{G}\sum_{j=1}^{G}r_j.
$$

```python
mean_reward = sum(rewards) / len(rewards)
# when building each RolloutSample
advantage = reward - mean_reward
```

That mean is the baseline for this question. It gives me a way to read the advantage: how much better was this answer than the model's own average on the same question? This implementation subtracts the mean only — there is no division by the standard deviation, and no critic that has to be fitted.

## 2. Turning one answer into data the model can train on

The line I used to misread most is `observation_len = len(prompt_tokens) - 1`. Laying out a short sequence is easier than staring at that minus one. Suppose the question has three tokens, the answer has two, and the whole answer carries advantage $A$:

| token fed to the model | p₁ | p₂ | p₃ | y₁ |
| --- | --- | --- | --- | --- |
| target this position predicts | placeholder | placeholder | y₁ | y₂ |
| logprob saved at sampling | 0 | 0 | ℓ₁-old | ℓ₂-old |
| advantage used in training | 0 | 0 | A | A |

When the model reaches p₃ it has already seen the whole question, so it can predict the first answer token y₁. The prompt therefore holds 3 tokens but only the first 2 prediction positions need to be zeroed. The prefix is still context the model needs in order to predict; the zero advantage makes those placeholder targets contribute nothing directly to the policy loss.

`build_grpo_datum` wraps those four rows into one `trio.Datum`, and one answer corresponds to one datum. At this point, **the advantage is answer-level while the old logprob is token-level**. The tokens inside one answer share a single $A$, and each of them keeps the probability it was sampled with, inside its own context.

## 3. When does PyTRIO sample, and when does it change the weights?

I sorted the PyTRIO calls by execution order. While reading this part I watch two things: the trainable LoRA parameters of the moment, and the probability record left behind by this round's sampling.

| call in the script | what it does in this training run |
| --- | --- |
| `create_lora_training_client(..., rank=32)` | Sets up the training side: low-rank adapter parameters trained on top of the base model. My configuration uses Adam with a learning rate of 4e-5. |
| `save_weights_and_get_sampling_client()` | At the start of each round, saves a snapshot of the current weights and returns a sampler dedicated to this round. The old policy here is the policy that produced this batch of answers. |
| `sample(..., num_samples=8).result()` | Requests 8 answers for one question and waits for the result. Besides the text, it hands back the generated tokens and their logprobs. |
| `forward_backward(datums, loss_fn=...)` | The training side recomputes the current logprob for the already-sampled tokens and computes the gradient with the built-in IS or PPO objective. |
| `forward_backward_custom(...)` | CISPO takes this branch. The callback receives the differentiable current logprob, while the old logprob from sampling and the advantages are passed in through the closure. |
| `optim_step(adam_params)` | Called once per non-empty batch. The script waits for the forward/backward and the update requests to finish before starting the next round. |
| `save_weights_for_sampler(...)` | Saves the final weights for sampling at the end of training; the terminal prints the corresponding path. |

A future can be read simply as "the request has been submitted, the result can be fetched later", and `.result()` is what waits for it. All 4 questions are generated and graded before a single update happens. If all 4 groups of a round degenerate, `datums` is empty and both the forward/backward and the optimizer update are skipped. So `--steps 10` in the command means ten rounds of the outer loop.

## 4. The probability ratio sits at 1 — how can the model still learn?

The training side looks at the same prefix $s_t$ and the same target token $y_t$ again. Let the current logprob be $\ell_t$ and the one recorded at sampling time be $\ell_{t,\mathrm{old}}$; the probability ratio is

$$
\rho_t=\exp(\ell_t-\ell_{t,\mathrm{old}}).
$$

Following the call order, when this ratio is computed **this round's optimizer has not run yet**. Sampling and training use the same version of the weights, so when both ends agree the result is naturally 1. The denominator is the stored number while the numerator is still differentiable with respect to the current parameters, so even when the ratio is exactly 1 we still have $\nabla_\theta\rho_t=\rho_t\nabla_\theta\ell_t$. The next round samples again, and the comparison baseline is refreshed along with it.

Reusing the notation from the previous note, write the gradient of a single token's contribution to the maximized objective as $A\,w\,\nabla_\theta\ell_t$. The difference between the three branches can be compressed into one table. The table below ignores the clipping boundary points.

| branch | gradient weight $w$ | the distinction that helped me most when reading the code |
| --- | --- | --- |
| IS | $\rho$ | uses the probability ratio directly. |
| PPO | 0 when the advantage is positive and $\rho>u$, or the advantage is negative and $\rho<l$; $\rho$ otherwise | the standard clipped objective looks at both the sign of the advantage and the direction of the excursion. |
| CISPO | $\operatorname{clip}(\rho,l,u)$ | the source applies `detach()` to the clipped weight, so the gradient flows back through the current logprob. |

Take a numerical example: $A>0$, upper bound 1.2, $\rho=1.5$. Standard PPO takes the constant term on this side, so that token's objective gradient is zero; CISPO, with the same upper bound, still contributes gradient with weight 1.2. **The 1.2 is only a number for contrasting the formulas** — my actual CISPO run used the bounds 0 and 4. The script hands PPO to the built-in implementation and does not set its thresholds explicitly in this file. [Course sync script](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py) · [PyTRIO loss docs](https://docs.pytrio.com/docs/guide/loss_fn)

## 5. The actual run: 32 answers, but only 8 of them were trained on

Here is the command I used. When switching loss I only change the value of the last argument.

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 --max-tokens 512 \
    --swanlab-mode online --loss-fn ppo
```

This is the first training log line of my PPO run:

```text
Step 0 | reward: 0.969 | degenerate: 75% | datums: 8
       | train_tokens: 1702 | avg_gen_len: 189.5 | loss_mean: 0.0936
```

Each round has 4 questions and each question has 8 answers. Look at this line: 0.969 is 31/32 rounded. Since only one answer was wrong, there must be 3 questions that were all correct across their 8 answers, plus 1 question with 7 correct and 1 wrong.

| PPO round 0 | group mean | advantage of each answer | answers kept |
| --- | ---: | --- | ---: |
| three questions, each all correct | 1 | all 0 | 0 |
| one question with 7 correct and 1 wrong | 7/8 | seven +1/8, one −7/8 | 8 |

So the only wrong answer **was kept**. It is what gave this group the largest-magnitude negative advantage. The other three groups have all-zero advantages and are skipped whole by the `continue` in the main loop. That is the 75% degenerate rate in the log. The other way round, a question answered wrongly all 8 times would be skipped by the same line of code.

Now the two length metrics in the log can be explained too. 189.5 is the mean generation length over all 32 answers; 1,702 is the total token count of the 8 answers that were kept, so those 8 average about 213 tokens each. The two numbers count different sets.

## 6. The results of the four runs

The IS first run of 17 November is the experiment analysed in the previous note. On the 20th I added PPO and CISPO, and re-ran IS with the original configuration. Every time it was the same first 40 questions and 320 sampled answers. The "valid question groups" below are the questions where a 0 and a 1 both appear inside the group. [The four runs' SwanLab project](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo) · [The first IS run's log](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo/runs/2gmiydii)

| ten rounds in total | IS first run | IS re-run | PPO | CISPO |
| --- | ---: | ---: | ---: | ---: |
| valid question groups / 40 | 14 | 10 | 16 | 14 |
| answers entering training / 320 | 112 | 80 | 128 | 112 |
| completion tokens used in training | 37,906 | 25,920 | 41,194 | 35,412 |
| actual updates / 10 | 7 | 8 | 9 | 8 |
| correct answers in the last round / 32 | 28 | 30 | 30 | 30 |

![The question groups kept in each round of the first IS run, and the rounds that skipped the update](/images/agentic-rl/grpo-signal-budget.png)

*Figure 2　The first IS run. Compare rounds 2 and 3: both have a 75% accuracy, but only round 2 produced an update.*

**The same accuracy can give completely different learning signal.** Round 2 and round 3 of the IS first run are both 24/32. In round 2 all four groups had both correct and wrong answers, so all 32 answers were kept; in round 3 three questions were all correct and one was all wrong, so no group had any spread and there were zero updates. Looking only at the reward curve, I would have seen those two rounds as the same two points.

**A re-run of the same configuration also changes how much actually gets trained.** Both IS runs sampled 320 answers, but the number entering training went from 112 to 80 — about 29% fewer. Interestingly the second run also updated one more time: it had fewer valid groups, but they were spread over more rounds. Comparing only how many steps the optimizer took still misses part of the picture. One repeat can show that there really is sampling variance in this experiment; it is not enough to estimate the stable range of that variance.

**Whether the clipping fired has to be read from the metrics of the matching branch.** In CISPO's 8 valid steps, the per-step mean probability ratio stayed between 0.99954 and 1.00053, and all 8 `clip_fraction` values were 0. The first answers "where is the mean", the second answers "were there training tokens beyond this round's threshold". The thresholds are [0, 4], so this result cannot tell me how close each individual token is to 1. PPO's log keeps `loss_mean` but leaves behind no probability-ratio distribution and no effective clipping rate, so its actual clipping ratio remains unknown.

**Why didn't I rank the three methods by the sign of the loss?** Look at that question in PPO's round 0 with 7 correct and 1 wrong. Summed by answer, the advantages are zero; expanded per token, each advantage repeats as many times as the length of its answer, so the sum need not be zero. CISPO's objective also explicitly contains the current logprob, so even when the gradient weight is close to IS, the scalar loss it displays can differ. And since every round uses new questions, the 30/32 of the last round is only the sampling result on that round's four questions. To compare the final models I still need a fixed set of independent evaluation questions.

## 7. After the run, how do I want to verify the clipping?

The smallest change I want to make is to update twice on the same batch of answers, always using the old logprob left behind by the first sampling. During the first forward pass both ends have the same weights; during the second forward pass the numerator already comes from the updated model. Recording the quantiles of the probability ratio for these two passes, and then counting — by sign of the advantage — the fraction of tokens where PPO really enters the constant branch, would show "how far does one update actually push the policy". Whether that is enough to trigger the clipping has to be answered by measurement.

When I compare against my study partner's record, I also want to keep one distinction: generating the same batch of questions concurrently shortens the waiting time; only when a parameter update actually happens between the training side and the sampling snapshot is there one extra probability difference caused by the weight change. To judge that, when reading the code I should look for the order of the snapshot and the optimizer step. [A colleague's record from the same group](https://zhuanlan.zhihu.com/p/2106862160429295432)

Having read this much, when I look at one line of log I will first find `degenerate` and `datums`, work out which answers provided the gradient, and only then ask how large the probability ratio of those tokens is. The result of PPO's round 0 now makes sense: 31 answers passed the grading rule, and what actually triggered learning was a disagreement on one of the questions.

[Series index · Agentic RL Lab notes](/series/agentic-rl-lab/) · [Course GRPO experiment directory](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo)

## References

- Course repo: [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task 02 tutorial: [GRPO starts with group comparison](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) ｜ [quick start](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html)
- Code for this run: [experiments/grpo/01-demo-sync.py](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)
- An independent record from a study partner: [Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432)
- DeepSeekMath (origin of GRPO): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PyTRIO docs: [pytrio.cn](https://pytrio.cn/) ｜ SwanLab: [swanlab.cn](https://swanlab.cn/)
