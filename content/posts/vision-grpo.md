---
title: "Datawhale Agentic RL Task03｜One geometry figure, and what 100 GRPO steps really trained"
date: 2026-11-24T00:30:00+08:00
summary: "How a GeoQA figure enters the policy, and what a 100-step run actually updated: 34 steps carried a gradient, 152 of 800 answers never wrote a parsable option."
tags: ["reinforcement-learning", "GRPO", "multimodal", "PyTRIO", "study-notes"]
categories: ["Study Notes"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

In the previous note I followed GRPO on GSM8K. This time the task is a geometry multiple-choice question: a picture, a stem, four options. I expected to have to learn a separate recipe for vision reinforcement learning, but reading the training script made me stop on two very concrete questions instead: how many positions does one image occupy in the sequence, and when the model writes a long geometric derivation, which part of it does the program actually score?

This time I ran the full 100 steps and recomputed every number from the raw SwanLab metrics. All figures below come from two runs: the first one crashed after step 68, and the second one resumed from the step-65 checkpoint and finished at step 100. The three steps that ran twice are counted only once, from the last run. The per-step metrics I decoded from the local `.swanlab` files match the data SwanLab serves on the cloud, key by key: 100 steps × 6 metrics, zero differences. The charts below are screenshots taken directly from the SwanLab dashboard.

- [First run (steps 1–68)](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-vision-grpo/runs/l11udwhb) · [Resumed run (steps 66–100)](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-vision-grpo/runs/rawcelig)

## 1. What one sample looks like

A GeoQA training record holds the figure, the question stem, four options, the correct label, and the worked solution. The model receives the first three. The label is kept for the scorer, and the solution is never concatenated into the prompt. That single decision decides what the model can learn: it may try different derivations, but the training program only knows whether the final option was correct.

![The figure used by step 1 and the prompt handed to the model](/images/agentic-rl/vg-sample-prompt-en.png)

The left half of the figure is the sample actually used by step 1 (GeoQA id=9419); the gold letter is B. The right half shows what that sample looks like on its way into the model.

In PyTRIO, passing an image takes two steps. First the chat template decides where the image goes, and then that position is replaced by an `ImageChunk` that carries the real image bytes. The script also flattens transparent images onto a white background, so the transparent background of a geometry figure does not turn black. The final input can be written as:

```text
EncodedTextChunk  +  ImageChunk  +  EncodedTextChunk
    template head       figure          template tail
```

The part that is easy to miss is the length. An `ImageChunk` is one object in Python, but it can occupy many positions in the model sequence. The script uses the image processor to compute the expected number of visual tokens, and after sampling it checks that the remote `input_tokens` equals the local `len(prompt)`. That length is what later tells the code where the answer starts.

## 2. The image predicts, the reward lands on the answer

Assume the whole image-plus-text prompt occupies $P$ positions and the model generated three tokens, $a,b,c$. Shortening the arrays to just those three outputs in `build_grpo_datum` is what finally made the `prompt_length - 1` in the code make sense.

| position in the training input | what is predicted there | advantage written |
|---|---|---|
| the first $P-1$ prompt positions | the rest of the prompt, targets filled with placeholders | 0 |
| the last prompt position | the first answer token $a$ | $A$ |
| the $a$ in the answer | $b$ | $A$ |
| the $b$ in the answer | $c$ | $A$ |

Give $P$ zeros instead and the training signal for the first answer token shifts by one position. The image sits inside the prompt, yet the information it carries still changes the probability the model assigns to $a,b,c$.

The experiment extracts the last recognisable boxed option from each answer, for example `\boxed{B}`. Matching the gold label scores 1, everything else scores 0. The 8 answers of one question share one mean reward, and each answer gets the advantage $A_i=r_i-\bar r$. That advantage is copied onto every token of the answer and handed, together with the per-token log probabilities saved at sampling time, to the `importance_sampling` loss. The cost of that choice is plain enough: even if a step of the derivation is wrong, as long as the final letter is right, the scorer cannot tell.

## 3. The result of 100 steps: only 34 of them updated the weights

Let the SwanLab dashboard speak first. The two figures below are screenshots taken straight from SwanLab's chart comparison view: red is the first run (steps 1–68), green is the resumed run (steps 66–100), both on the same step axis.

![SwanLab chart comparison: reward and train_datums](/images/agentic-rl/vg-swanlab-reward-datums.png)

The left panel is reward. The curve keeps jumping between 0 and 1: every step is a different question, so the score mostly reflects the difficulty of the question rather than a model that is steadily getting better.

The right panel is `train_datums`, the number of answers that actually entered forward/backward in that step. It is either 0 or 8 — **there is no value in between**. The number of red and green spikes standing up is exactly the number of steps that updated the weights.

![SwanLab chart comparison: loss_mean and mean completion length](/images/agentic-rl/vg-swanlab-loss-length.png)

The left panel is `trainer/loss_mean`. It only has a value on the steps where an update happened, and the line is broken everywhere else. Its values run from 0.0058 to 0.2247, with a mean of 0.0629. The magnitude is small partly because every step performs a single LoRA update, and partly because all 8 answers of a group enter the same batch.

The right panel is the mean completion length per step, and its peaks land exactly on 1,024, the `max_tokens` cap.

Now let me lay the same data out again to add what the dashboard itself cannot show.

![Per-step score, and per-step output length against the 1,024-token cap](/images/agentic-rl/vg-reward-format-en.png)

I used Qwen3.5-4B, LoRA rank 32 and a learning rate of $4\times10^{-5}$. Each step takes one question, samples 8 answers, and each answer may generate at most 1,024 tokens.

The left panel is the mean reward of the 8 answers of each step. Green marks the 34 steps with mixed outcomes — these really went through forward/backward and the optimizer. Pink marks the 66 steps where all 8 answers scored the same: the relative advantage inside the group is identically zero, so the script skips the group.

The right panel is the mean output length per step. Five steps (6, 13, 23, 63 and 94) had all 8 answers truncated at 1,024 tokens, and those five steps score exactly 0. The rest split into two groups: the 34 updated steps average 760 tokens, the 66 skipped steps only 354. **The long answers are the ones the model was not sure about.**

![Where the 800 answers went, and how many were correct per step](/images/agentic-rl/vg-answers-funnel-en.png)

Laying the 800 answers out one by one gives a sharper picture than the average score:

| counted from the logs | result |
|---|---:|
| questions trained / answers sampled | 100 / 800 |
| option successfully extracted | 648 / 800 (81%) |
| option matched the gold label | 640 / 800 (80%) |
| steps that really updated the parameters | 34 / 100 |
| answers that entered forward/backward | 272 / 800 |

What surprised me most: of the 160 zero-score answers, **152 never even produced an option that could be parsed**, and only 8 extracted a wrong letter. The logs alone cannot tell how many of those 152 answered with a broken derivation, how many forgot the box, and how many ran out of tokens before giving an answer. If I want to know where the model actually gets stuck, those raw answers are worth more than an averaged reward curve.

The right panel counts the correct answers per step: 5 steps all wrong, 61 steps all right, 66 steps in total on the two ends — exactly the steps that were skipped. Only the middle 34 steps ever mixed right and wrong answers, so the number of answers that entered the update is $34\times8=272$.

## 4. An all-correct group also reports `loss_mean=n/a`

This looks odd, but the code says it plainly: `if len(set(rewards)) == 1: continue` — identical rewards for the whole group means no relative advantage, so the group is skipped.

Step 64 of the original run is a good control: all 8 answers were well formatted and 7 were correct. The mean reward is 0.875, so the correct answers carry an advantage of +0.125 and the wrong one −0.875, and all 8 answers join the update. Step 65 is the case where all 8 answers are correct: every advantage is zero and the script skips the group.

One caveat matters here: the **80% is the share of correct answers sampled during training**. The same question is answered 8 times while the model keeps changing, and every step uses a different question. The 71% to 87% reported by the course comes from a fixed set of 100 test questions, and the tutorial trains on 8 questions per step. What I can confirm here is that the run works and that the final weights were saved; to claim the model improved, the base model and the trained model still have to be evaluated on the same test set.

## 5. Two crashes told me the difference between "answers per group" and "answers per request"

The first attempt followed the tutorial: 8 questions per step, 8 answers each. Sampling went fine, and the error only appeared at the forward/backward stage: too many images. The reason sits in the structure of the training data — every valid answer is packed into a `Datum` together with the original figure. So 8 different figures can become 64 image-carrying samples in a single training request. I lowered the number of questions per step to 1, kept 8 answers per question, and the updates went through.

Two more crashes happened while waiting for sampled responses. Both disconnected after 163,840 bytes, which suggested asking for less per request, but the logs could not tell which layer closed the connection. The fix I kept was to request 2 answers at a time and issue 4 requests against the same policy version, so that the group average is only computed once 8 answers are in hand. **Where the split happens matters**: averaging every 2 answers would give the same 8 answers different advantages, and the experiment would no longer be the same experiment.

![Rollout time per step, and the wall-clock of the two runs](/images/agentic-rl/vg-step-cost-en.png)

The timings show how stubborn that crash was: for the same question, the fastest step took 4 seconds and the slowest took 29 minutes. Steps 24, 42 and 61 each ran for more than 17 minutes. The 100 rollout steps add up to about 2.9 hours, the first run took 3.5 hours overall, and the final resume only 28.3 minutes.

The first attempt stopped at step 69, and the most recent complete training state was saved at step 65. Step 67 did perform an update, so simply moving the progress number to 68 would skip it. I restored the model and the optimizer from step 65, resampled from step 66, and waited until the log printed the two save addresses (sampler and state) for step 100. Next time a run breaks I will look for those two pieces of information first: which step was saved last, and whether the failure happened during sampling or during the parameter update. [First run](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-vision-grpo/runs/l11udwhb) · [Resumed run](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-vision-grpo/runs/rawcelig)

## References

- [Datawhale · Vision GRPO experiment and tutorial results](https://datawhalechina.github.io/agentic-rl-lab/experiments/vision-grpo/)
- [Training and evaluation source](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/vision-grpo)
- [GeoQA dataset](https://huggingface.co/datasets/hz2475/geoQA)
- [DeepSeekMath · the origin of the GRPO algorithm](https://arxiv.org/abs/2402.03300)
