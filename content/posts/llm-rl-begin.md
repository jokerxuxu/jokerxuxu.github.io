---
title: "Datawhale Agentic RL Task1: a roboticist's first pass at LLM RL"
date: 2026-11-17
summary: "A first LLM RL run on PyTRIO, and GRPO told in robot-RL vocabulary: why a baseline reduces variance, why a group mean is enough, and where my control intuition breaks."
tags: ["reinforcement-learning", "GRPO", "PyTRIO", "LLM", "study-notes"]
categories: ["Study Notes"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Task 01 of the Datawhale *Agentic RL* study group: environment, basic concepts. Tutorial: [Agentic RL Lab](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/). Code: [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab). Written 2026-11-17.

I have some exposure to LLM and agent work, but most of my day-to-day is robot learning. The motive for studying LLM RL is direct: **what happens to the same RL paradigm when the domain changes** — where it forces a different approach, and where nothing changes at all.

So this is not another glossary. It is an answer to my own question: **how far does my existing intuition carry me, and where does it break?** Longer term I also want to learn how people on the LLM side think.

## 1. It trains without a local GPU

**Architecture**

| Component | Runs on | Responsibility |
| --- | --- | --- |
| PyTRIO `0.2.9` | cloud | forward, backward, optimizer step, sampling, weight storage |
| SwanLab `0.10.4` | cloud + local logs | curves, hyperparameters, stdout |
| repo scripts | local | data, reward, training loop |

I know this arrangement in another form: simulator on a server, control law and scoring locally. The difference is what comes back — state trajectories there, gradients and weights here.

Setup is three commands (`uv sync`, `trio login`, `swanlab login`) and one ten-step demo:

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --loss-fn importance_sampling \
    --swanlab-mode online
```

Two things the tutorial keeps warning about, and that a first-timer will almost certainly hit:

1. **The local torch is CPU-only.** `pyproject.toml` pins a `pytorch-cpu` index; do not "helpfully" swap in the CUDA build. It only does real work when you pass a custom loss through `forward_backward_custom`.
2. **Every command needs `uv run`.** `trio` and `swanlab` live inside `.venv`; from a system shell you get command not found.

Before the first run, it is also worth checking these three things:

3. **Check the network in stages.** Dependency installation, the Hugging Face download and the cloud training request are three different stages; when something times out, look at which stage the log stops in first and then check that stage's network and proxy settings, instead of restarting the whole training over and over.
4. **Leave room for the HF cache.** The model, tokenizer and dataset downloads can take a lot of local disk; when space is tight, point `HF_HOME` at a directory with enough capacity and keep the path consistent across runs, so nothing is downloaded twice.
5. **Run small first, then look at the quota.** Use the ten-step demo above to verify the pipeline and check the remaining quota on the cloud account and the actual consumption; only then increase the number of steps, the batch size, the group size or the generation length.

**The cloud turns owning compute into calling compute**, and for anyone without a lab machine that is the precondition for this whole exercise.

## 2. Translating it into robot RL

The notation from robot RL maps almost one-to-one onto this side. The full picture:

| Symbol | Robot RL | LLM RL |
| --- | --- | --- |
| state $s_t$ | joint angles, velocities, IMU, contacts | prompt + tokens generated so far |
| action $a_t$ | torque or joint target, usually continuous and high-dimensional | next token, discrete, the vocabulary |
| policy $\pi_\theta$ | MLP emitting a Gaussian mean and variance | the same transformer emitting a distribution over the vocab |
| reward | usually dense, sometimes sparse | usually one terminal score from an external grader |
| exploration | Gaussian noise, entropy bonus, max-entropy | temperature, top-p |
| algorithms | PPO, SAC, TD3 | PPO, GRPO, DAPO, GSPO |

My reading: **LLM RL is an MDP whose action space is the vocabulary, whose episodes run thousands of steps, which pays out only at the end, and which sets $\gamma=1$.** Almost every difference from robot RL follows from that sentence.

**One: the reward moves from every step to the end.** Robot PPO computes advantages with GAE:

$$
A_t=\sum_{k\ge 0}(\gamma\lambda)^k\,\delta_{t+k},
\qquad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

because every step has an $r_t$. Here the intermediate steps all have $r_t=0$ and $\gamma=1$, so the return along a trajectory is a single constant and the per-token advantage can only be broadcast from it. Credit assignment is a secondary problem in robot RL — per-step rewards plus GAE handles it — and becomes the main problem here. That is also why people build process reward models (PRMs): the point is to manufacture the per-step reward that is missing.

**Two: the baseline changes source but not purpose.** Robot PPO ships a value function; GRPO replaces it with the mean over $G$ answers to the same prompt. Robot RL has the same trick: **run N envs in parallel from the same initial state and use the group's mean return as the baseline.**

**Three: data gets exactly one use.** Robot RL has a simulator, so SAC can keep drawing from a replay buffer. Here the "environment" is the policy itself: the moment the weights move, the old data is no longer on-policy, so the loop is collect once, train once. This is part of why off-policy methods like SAC and TD3 stay rare here — a critic would have to value a text prefix, which is both hard to train and not backed by cheap replayed data.

Two things that do not transfer:

- **Exploration means something else.** Robots explore with action noise and entropy regularization; an LLM's temperature flattens the token distribution, and raising it is not the same as exploring a new solving strategy.
- **The locality of actions is gone.** In continuous control, nearby actions produce nearby outcomes, and much of the machinery leans on that; tokens have no such geometry, so "one digit wrong" and "a whole paragraph wrong" sit the same distance apart.

## 3. Baseline: from GAE to a group mean

The second point deserves its own section, because it is exactly where PPO felt obvious to me and GRPO did not.

In robot PPO the advantage leans on a learned value function. Switch to $\gamma=1$ with a terminal-only reward and every token's return is identical: the problem collapses to "I only need a baseline", and that baseline has a cheaper estimator.

Policy gradient is

$$
\nabla_\theta J = \mathbb{E}\left[R\,\nabla_\theta \log \pi_\theta\right].
$$

Replace $R$ with $R-b$: as long as $b$ does not depend on the sampled action, the expectation is unchanged while the variance can shrink. The ideal $b$ is the expected return in that state, $\mathbb{E}[R\mid s]$ — PPO learns a value network $V_\psi$ to estimate it, its only extra model.

GRPO's change is one sentence: **sample a group of answers to the same question and use the group mean as the baseline.** One question is one state, so the group mean is an unbiased estimate of $\mathbb{E}[R\mid s]$, and the critic disappears entirely.

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)+\delta}.
$$

Why does it have to be the same question? Because average reward across different questions is not comparable to begin with. It is the same habit as a controlled comparison: freeze the operating point, then look at relative performance. A group of answers is a set of repeats under one condition.

Two things worth internalizing: **rewards can all be positive while advantages still take both signs**, and **a zero-variance group yields all-zero advantages** and no gradient.

**When the group has no spread, the baseline collapses.** If every answer is right or every one is wrong, all $r_i$ are equal, every advantage is zero, and the group is wasted; the log calls it `frac_degenerate`. Curriculum learning in robotics hits the same wall: when task difficulty and current policy are mismatched, success and failure stop separating, the gradient is zero, and the only fix is to change the task. The difference is that there you can edit the task directly, while here you can only shift the question distribution or `group_size`.

## 4. IS / PPO / CISPO: three losses, one derivation

The `loss-functions` chapter is where I got stuck, and the sticking point was specific: **the three formulas look almost identical, so why are they three algorithms?** IS and PPO differ by a `min` and a `clip`; CISPO adds a `detach`. If those are implementation details, why give them a chapter.

It clicked once I differentiated with respect to $\theta$: all three end up in the same shape, and the only thing that differs is **how hard each token gets pushed**, $w(\rho)$. Let me peel that apart.

### 4.1 Notation and one premise

Let the policy be $\pi_\theta$, and write $o_{i,t}$ for token $t$ of sample $i$ of one answer. The sampling policy $\pi_{\text{old}}$ is frozen; the probability ratio is

$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid s_{i,t})}{\pi_{\text{old}}(o_{i,t}\mid s_{i,t})}.
$$

The advantage $A_i$ is the group-normalized relative advantage in GRPO and the GAE advantage in PPO. **The key premise here is that $A_i$ is a constant with respect to $\theta$** ($\nabla_\theta A_i=0$): it is computed during rollout and frozen for training. Only because of that can the three losses collapse into one shape; if the advantage depended on $\theta$ (an online critic trained jointly with the policy), the derivation gains a term $\mathbb{E}[w(\rho)\nabla_\theta A_i]$ and the unified form fails.

Write $r_{i,t}=\pi_\theta(o_{i,t}\mid s_{i,t})$. From $\nabla\log\pi=\nabla\pi/\pi$:

$$
\nabla_\theta\rho_{i,t}=\frac{\nabla_\theta r_{i,t}}{\pi_{\text{old}}(o_{i,t}\mid s_{i,t})}=\rho_{i,t}\,\nabla_\theta\log\pi_\theta(o_{i,t}\mid s_{i,t}).
$$

**The gradient of $\rho$ is $\rho$ times the score function.** That is the engine of the whole section.

### 4.2 IS: $w(\rho)=\rho$

IS corrects the off-policy expectation directly with the ratio:

$$
\mathcal{L}_{\text{IS}}(\theta)=-\,\mathbb{E}\left[\rho_{i,t}\,A_i\right].
$$

The minus sign comes from "minimizing the loss = maximizing the objective". Differentiating:

$$
\nabla_\theta\mathcal{L}_{\text{IS}}=-\,\mathbb{E}\left[A_i\,\nabla_\theta\rho_{i,t}\right],
\qquad
w_{\text{IS}}(\rho)=\rho .
$$

Nothing caps $\rho$. At $\rho=1.8$ the gradient for that token is multiplied by 1.8 — in control terms, a proportional term with no limiter: the larger the deviation, the harder the push, and one step can send a probability flying.

### 4.3 PPO: the flat region flips with the sign of $A$

Let $l=1-\epsilon$ and $u=1+\epsilon$. The single-sample PPO surrogate is

$$
L(\theta)=\min\left(\rho A,\ \operatorname{clip}(\rho,l,u)A\right).
$$

Once $A$ is fixed, this reduces to two lines:

$$
L(\theta)=\begin{cases} A\min(\rho,u), & A\gt 0,\\ A\max(\rho,l), & A\lt 0.\end{cases}
$$

The `max` on the second line comes from multiplying by a negative number, which flips the inequality. This one step checks every branch. Writing $\nabla_\theta L=A\,w(\rho)\nabla_\theta\log\pi_\theta$ and using $\nabla_\theta\rho=\rho\nabla_\theta\log\pi_\theta$:

| advantage | $\rho\lt l$ | $l\lt\rho\lt u$ | $\rho\gt u$ |
| --- | --- | --- | --- |
| $A\gt 0$ | $w=\rho$ | $w=\rho$ | $w=0$ |
| $A\lt 0$ | $w=0$ | $w=\rho$ | $w=\rho$ |

In compact form:

$$
w_{\text{PPO}}(\rho)=\begin{cases} 0, & A\gt 0,\ \rho\gt u,\\ 0, & A\lt 0,\ \rho\lt l,\\ \rho, & \text{otherwise (including overshoot the other way)}.\end{cases}
$$

**Reading it:**

- **The side is decided by the sign of $A$.** With $A\gt 0$ the objective rewards raising the probability of the sampled action, so $\rho\gt u$ means "raised too far" and the flat region sits on the **upper** side; with $A\lt 0$ it rewards lowering that probability, so $\rho\lt l$ means "pushed down too far" and the flat region sits on the **lower** side.
- Overshooting in the **opposite** direction ($A\gt 0$ with $\rho\lt l$, or $A\lt 0$ with $\rho\gt u$) keeps the unclipped term, so the gradient survives to correct the change. That is the other half of "one-sided conservatism".
- Both sides use the same $\epsilon$, so the structure is symmetric about the *favourable versus unfavourable* direction but asymmetric on the numerical $\rho$ axis — the flat region flips with the sign of $A$.
- "Stopping" only means **that sample's direct gradient on the PPO surrogate is zero**. Other samples still move the shared parameters, and clipping does not lock the actual ratio inside the band.

### 4.4 CISPO: $w(\rho)=\operatorname{clip}(\rho)$, detached

CISPO keeps the clipped ratio as a coefficient only, with no gradient through it:

$$
\mathcal{L}_{\text{CISPO}}(\theta)=-\,\mathbb{E}\left[\operatorname{sg}\big(\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\big)\,A_i\,\log\pi_\theta(o_{i,t})\right].
$$

$\operatorname{sg}(\cdot)$ is stop-gradient (`detach`). It is treated as a constant when differentiating, so only $\log\pi_\theta$ carries gradient:

$$
\nabla_\theta\mathcal{L}_{\text{CISPO}}=-\,\mathbb{E}\left[\operatorname{sg}\big(\operatorname{clip}(\rho_{i,t})\big)\,A_i\,\nabla_\theta\log\pi_\theta(o_{i,t})\right],
$$

hence

$$
w_{\text{CISPO}}(\rho)=\operatorname{clip}(\rho,1-\epsilon,1+\epsilon).
$$

**The key comparison**: without detaching, $\nabla_\theta\operatorname{clip}(\rho)$ is $\rho\nabla\log\pi$ inside the band and $0$ outside, so it degenerates back to PPO's "zero past the edge". Detach turns the coefficient into a read-only magnitude:

- capped: at $\rho\gt u$ the force tops out at $u$ instead of growing without bound as in IS;
- **never zero**: past the edge the coefficient is still $l$ or $u$ (both positive), so the gradient keeps existing, it just stops growing.

That is "force capped, not switched off", which patches exactly the "cut off" defect of PPO's flat region.

### 4.5 All three together

$$
\nabla_\theta\mathcal{L}=-\,\mathbb{E}\big[w(\rho_{i,t})\,A_i\,\nabla_\theta\log\pi_\theta(o_{i,t})\big]
$$

| loss | $w(\rho)$ | cap | overshoot, favourable side | overshoot, opposite side |
| --- | --- | --- | --- | --- |
| IS | $\rho$ | none | keeps growing | keeps growing |
| PPO | $\rho$ inside the band, $0$ past the favourable edge | yes (by zeroing) | stops (gradient is 0) | keeps $\rho$, still correcting |
| CISPO | $\operatorname{clip}(\rho)$ (detached) | yes ($u$ / $l$) | capped, still pushing | capped, still pushing |

The three differ **only in this scalar function $w(\rho)$**; the loss form, the `min`, and the `detach` are merely ways of implementing that $w$. That is the answer to "three formulas that look identical but count as three algorithms": the algorithmic difference is compressed into a one-dimensional force function.

### 4.6 A clarification: GRPO is not on this axis

- **GRPO** decides where $A_i$ comes from: sample a group of answers to the same prompt and use the group mean / standard deviation as the relative advantage ($A_i=(R_i-\operatorname{mean}(R_{1:G}))/\operatorname{std}(R_{1:G})$). It says **nothing** about how $A_i$ becomes a gradient.
- **PPO / CISPO / IS** decide how $A_i$ becomes a gradient, that is, $w(\rho)$.

So "GRPO + PPO's clip" and "GRPO + CISPO's detach-clip" are both legitimate combinations: the first is GRPO supplying the advantage, PPO supplying the force function; the second keeps the same advantage and swaps the force function. Listing GRPO alongside PPO/CISPO as "three algorithms" conflates two orthogonal axes — **advantage construction** versus **force function**.

## 5. The GRPO objective

The previous section took the force function apart; this one lays out the GRPO objective. Training maximizes it (code minimizes $-J$):

$$
J(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\left(\min\left(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)
-\beta\, D_{\mathrm{KL}}^{(i,t)}\right)\right]
$$

Its policy part — the `min` term inside the bracket — is exactly the PPO form from 4.3, applied as is: **group-relative advantage plus PPO's weight**. Only two things are added: the KL term, and two levels of averaging.

Five readings:

1. The old policy samples $G$ answers; samples and old logprobs are frozen;
2. each answer gets one advantage, broadcast to all of its tokens;
3. `min` plus `clip` bounds how hard each update pushes — the shape of $w$ was derived in 4.3 and is simply reused;
4. the KL term holds the policy near the reference;
5. average over tokens, then average equally over answers — long answers should not get more say.

**One distinction that must not be blurred: $\pi_{\text{old}}$ and $\pi_{\text{ref}}$ are different objects.** The first is this round's sampling snapshot, the denominator of the ratio. The second is usually the initial SFT model, the anchor of the KL term. They may start out identical and then diverge in update cadence entirely. Confuse them and the KL curve will read strangely.

At this point all of GRPO's novelty sits in the advantage: the weight function is reused wholesale from PPO, and only the estimate of the advantage changes. It is not free either — dropping the critic costs $G-1$ extra samples per question.

## 6. Ten steps, measured: three readings and one coincidence

This is the repo's default demo: `--steps 10 --batch-size 4 --group-size 8 --max-tokens 512 --loss-fn importance_sampling`, Qwen3.5-4B on GSM8K, logged to SwanLab.

Training record: [SwanLab · agentic-rl-lab-grpo](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo/v1/liwxpw/runs/2gmiydii/chart). The charts and the table below all come from this single run.

| step | reward | degenerate | avg_gen_len | loss_mean |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 100% | 186.7 | — |
| 1 | 0.875 | 75% | 253.9 | 0.0540 |
| 2 | 0.750 | 0% | 380.7 | 0.0445 |
| 3 | 0.750 | 100% | 266.9 | — |
| 4 | 0.781 | 75% | 294.5 | −0.0271 |
| 5 | 0.781 | 25% | 275.4 | 0.0627 |
| 6 | 1.000 | 100% | 250.8 | — |
| 7 | 0.938 | 50% | 258.4 | 0.0286 |
| 8 | 0.969 | 75% | 224.3 | 0.0259 |
| 9 | 0.875 | 50% | 237.1 | 0.0173 |

![Group-mean reward over ten steps, with no trend](/images/agentic-rl/grpo-demo-reward.png)

**One: reward barely moves.** It bounces between 0.75 and 1.0 with no trend, and the reason is easy to guess: GSM8K is too easy for Qwen3.5-4B, which was already perfect at step 0. **Ten steps cannot show learning** — that is expected, not a failure.

![More than half the steps waste entire groups](/images/agentic-rl/grpo-demo-degenerate.png)

**Two: a coincidence you can only find by looking at the data — the three steps with 100% degenerate (0, 3, 6) have no loss record at all.** The whole batch was wasted, so the code skipped it and produced zero datums. I did not spot this in the curve; I found it by lining up the missing CSV values with `frac_degenerate = 1`. It turns "the group is wasted" from section 3 into something checkable: **a missing loss line is not a logging bug, it is a step where nothing was learned.**

![Loss jumps across sign; its mean has no absolute meaning](/images/agentic-rl/grpo-demo-loss.png)

**Three: length swings a lot and proves nothing.** 187 → 381 → 237, with step 2 the longest — and step 2 is the only one with zero degenerate groups. Ten steps are too few to claim "longer means smarter" or "longer means length hacking".

![Mean generation length: step-to-step noise beats any trend](/images/agentic-rl/grpo-demo-genlen.png)

So this run proves the **pipeline**, not the algorithm: environment wired up, credits spent, curves logged, weights saved. Which of the three losses is steadier needs a seed-fixed comparison with everything else held constant; at ten steps the noise is larger than the signal.

## 7. Still unresolved

- That reward design matters more than the algorithm is something I currently believe rather than have evidence for.
- I will keep going on GRPO next.

> Series index: [Agentic RL Lab](/series/agentic-rl-lab/).

## References

- Course repo: [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task 01: [environment & concepts](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/)
- DeepSeekMath, where GRPO appears (§4.1): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PPO, clipped surrogate objective: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- DeepSeek-R1, rule-based reward and GRPO on verifiable tasks: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
