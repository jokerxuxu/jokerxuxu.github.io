---
title: "Datawhale Agentic RL Task02｜一次 GRPO 更新，究竟用上了哪些回答？"
date: 2026-11-20
summary: "从 GSM8K 的评分规则读到逐 token 的策略梯度，再回到四次 PyTRIO 运行，分析退化组、概率比和实际更新次数。"
tags: ["强化学习", "GRPO", "PyTRIO", "实现", "组队学习"]
categories: ["学习笔记"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

上一篇，我试着用熟悉的机器人强化学习来理解语言模型训练：同一道题的几个回答怎样给彼此提供 baseline，IS、PPO 和 CISPO 又怎样改变一个 token 对策略梯度的贡献。我还用 PyTRIO 跑了十轮，发现整批题都退化的那三轮恰好没有 loss 记录。[上一篇 · 一个机器人方向读者的 LLM 强化学习入门](/zh/posts/llm-rl-begin/)

这篇我想顺着源码往下读：一道题怎样变成一组训练样本，一个回答的优势怎样分给各个 token，采样时存下的概率又在哪一步参与更新？弄清这些，再回头看四次运行的差别。

## 一、从 GSM8K 到组内优势

我用 Qwen3.5-4B 跑 GSM8K。训练集有 7,473 道数学应用题，每道题有题面和带解题过程的标准答案。脚本取标准答案里 `####` 后面的内容作判分依据。模型看到的 prompt 则由少量问答示例和本题题面组成，并要求用 `\boxed{...}` 写最终答案。模型生成后，`grade_answer` 提取最后一个匹配到的方框答案，比对前去掉逗号、首尾空白和末尾句点。因此 1 分表示**最终答案通过了这条判分规则**。推理过程没有单独得分，漏写方框也会得 0 分。

在这个例子里，**策略**就是模型给下一个 token 的条件概率分布。状态是“题目加已经写出的回答”，动作是接下来选哪个 token。让模型沿着自己的选择生成一次完整回答，就是一次 **rollout**。整条回答结束时才有奖励。

![同一个 prompt 产生多条回答，组内比较之后再更新策略](/images/agentic-rl/grpo-pipeline.png)

*图 1　沿着课程的流程图看，分组发生在同一道题内部。我的一次 batch 有 4 道题，因此会做 4 次独立的组内比较。*

一次 batch 包含多道题。脚本对每道题分别调用 `run_rollout_group`，用同一个 prompt 采样一组 completion（模型生成的回答）。设第 $i$ 个回答的奖励为 $r_i$，每组有 $G$ 个回答，源码实现的是

$$
A_i=r_i-\frac{1}{G}\sum_{j=1}^{G}r_j.
$$

```python
mean_reward = sum(rewards) / len(rewards)
# 构造每条 RolloutSample 时
advantage = reward - mean_reward
```

这个均值就是本题的 baseline。它给了我一个理解 advantage 的办法：模型这一次回答，比它在同一道题上的平均表现好多少？这份实现只减均值，没有再除以标准差，也不需要拟合一个估计价值的 critic。

## 二、把一个回答拆成模型能训练的数据

我原来最容易读错的是 `observation_len = len(prompt_tokens) - 1`。直接拿一个短序列排开，比盯着这个减一好理解。假设题目有三个 token，回答有两个 token，整条回答的优势为 $A$：

| 送入模型的 token | p₁ | p₂ | p₃ | y₁ |
| --- | --- | --- | --- | --- |
| 该位置要预测的目标 | 占位 | 占位 | y₁ | y₂ |
| 采样时保存的 logprob | 0 | 0 | ℓ₁-old | ℓ₂-old |
| 训练用的优势 | 0 | 0 | A | A |

模型读到 p₃ 时已经看完题目，正好可以预测第一个回答 token y₁。因此 prompt 有 3 个 token，却只有最前面 2 个预测位置需要置零。前缀仍然是模型预测所需的上下文，零优势使这些占位目标对策略损失的直接贡献为零。

`build_grpo_datum` 把上面四行封装为一个 `trio.Datum`，一条回答对应一个 datum。到这里，**优势是回答级的，旧 logprob 是 token 级的**。同一回答里的 token 共享一个 $A$，各自保留在其上下文中被采到的概率。

## 三、PyTRIO 在什么时候采样，什么时候改权重？

我把 PyTRIO 的调用按执行顺序整理了一遍。读这部分时，我会盯住两样东西：当前可训练的 LoRA 参数，以及本轮采样时留下的那份概率记录。

| 脚本中的调用 | 在这次训练里做什么 |
| --- | --- |
| `create_lora_training_client(..., rank=32)` | 建立训练端，在基础模型上训练低秩适配参数。我的配置使用 Adam，学习率为 4e-5。 |
| `save_weights_and_get_sampling_client()` | 每轮开始保存当前权重快照，得到本轮专用的采样器。这里的 old policy 就是生成本批回答的策略。 |
| `sample(..., num_samples=8).result()` | 对一道题请求 8 个回答，并等待结果。除了文本，还取出生成的 token 和各自的 logprob。 |
| `forward_backward(datums, loss_fn=...)` | 训练端对已采到的 token 重新计算当前 logprob，按内置 IS 或 PPO 目标计算梯度。 |
| `forward_backward_custom(...)` | CISPO 走这个分支。回调拿到可求导的当前 logprob，采样时的旧 logprob 和优势由闭包传入。 |
| `optim_step(adam_params)` | 每个非空 batch 调一次优化器。脚本等待前后向和更新请求完成，再开始下一轮。 |
| `save_weights_for_sampler(...)` | 训练结束时保存可供采样使用的最终权重，终端打印对应路径。 |

这里的 future 可以简单理解成“已提交请求，结果稍后可取”，`.result()` 用来等待结果。4 道题都生成完、判完分才更新一次。如果这一轮 4 个组全退化，`datums` 为空，前后向和优化器更新都会跳过。因此命令里的 `--steps 10` 表示走十轮外层循环。

## 四、概率比接近 1，模型怎么还能学？

训练端再次看同一个前缀 $s_t$ 和同一个目标 token $y_t$。记当前 logprob 为 $\ell_t$，采样时记录的为 $\ell_{t,\mathrm{old}}$，概率比就是

$$
\rho_t=\exp(\ell_t-\ell_{t,\mathrm{old}}).
$$

顺着调用顺序看，计算这个比值时，**本轮的优化器还没走**。采样和训练使用同一版本权重，两端计算一致时自然得到 1。分母是保存下来的数，分子仍然对当前参数可导，因此即使比值恰好等于 1，也有 $\nabla_\theta\rho_t=\rho_t\nabla_\theta\ell_t$。下一轮重新采样，比较基准也随之刷新。

沿用上一篇的记号，把单个 token 对最大化目标的梯度写成 $A\,w\,\nabla_\theta\ell_t$，三个分支的区别可以压成一张表。下表忽略截断边界点。

| 分支 | 梯度权重 $w$ | 对我读代码最有用的区别 |
| --- | --- | --- |
| IS | $\rho$ | 直接使用概率比。 |
| PPO | 正优势且 $\rho>u$，或负优势且 $\rho<l$ 时为 0，其余为 $\rho$ | 标准 clipped objective 同时看优势的符号和越界方向。 |
| CISPO | $\operatorname{clip}(\rho,l,u)$ | 源码对截断权重执行 `detach()`，让梯度从当前 logprob 回传。 |

举个数值例子，设 $A>0$、上界为 1.2、$\rho=1.5$。标准 PPO 在这一侧取常数项，这个 token 的目标梯度为零；CISPO 若采用同一上界，仍以 1.2 为权重贡献梯度。**1.2 只是用来对照公式的示例**，我的 CISPO 实际运行使用的上下界是 0 和 4。脚本把 PPO 交给内置实现，没有在这个文件里显式指定其阈值。[课程同步脚本](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py) · [PyTRIO loss 文档](https://docs.pytrio.com/docs/guide/loss_fn)

## 五、实际运行：32 个回答，为什么只训练了 8 个？

我用的命令如下。换 loss 时只改最后一个参数的值。

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 --max-tokens 512 \
    --swanlab-mode online --loss-fn ppo
```

这是我跑 PPO 时的第一行训练日志：

```text
Step 0 | reward: 0.969 | degenerate: 75% | datums: 8
       | train_tokens: 1702 | avg_gen_len: 189.5 | loss_mean: 0.0936
```

每轮 4 道题，每题 8 个回答。看这一行，0.969 是 31/32 四舍五入的结果。既然只有一个回答得 0，那么一定有 3 道题各自 8 次全对，剩下 1 道题 7 对 1 错。

| PPO 第 0 轮 | 组均分 | 每个回答的优势 | 留下的回答 |
| --- | ---: | --- | ---: |
| 三道各自全对的题 | 1 | 全是 0 | 0 |
| 一道 7 对 1 错的题 | 7/8 | 七个 +1/8，一个 −7/8 | 8 |

所以唯一的错误回答**留下了**。它给这一组提供了绝对值最大的负优势。其余三组的优势全为零，被主循环里的 `continue` 整组跳过。这就是日志中的 75% 退化率。反过来，即便一道题 8 次全错，也会被同一行代码跳过。

现在也能解释日志里的两个长度指标了。189.5 是全部 32 个回答的平均生成长度；1,702 是留下的 8 个回答的 token 总数，因此这 8 条平均约 213 个 token。两个数统计的是不同集合。

## 六、四次运行的结果

11 月 17 日的 IS 首跑就是上一篇分析的那次实验。20 日补跑 PPO、CISPO，又用原配置重复了一次 IS。每次都是同样的前 40 道题、320 个采样回答。下面的有效题组指组内同时出现 0 分和 1 分的题。[四次运行的 SwanLab 项目](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo) · [第一次 IS 的日志](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo/runs/2gmiydii)

| 十轮合计 | IS 首跑 | IS 重跑 | PPO | CISPO |
| --- | ---: | ---: | ---: | ---: |
| 有效题组 / 40 | 14 | 10 | 16 | 14 |
| 进入训练的回答 / 320 | 112 | 80 | 128 | 112 |
| 训练用 completion token | 37,906 | 25,920 | 41,194 | 35,412 |
| 实际更新次数 / 10 | 7 | 8 | 9 | 8 |
| 最后一轮答对数 / 32 | 28 | 30 | 30 | 30 |

![第一次 IS 运行中每轮保留的题组与跳过更新的轮次](/images/agentic-rl/grpo-signal-budget.png)

*图 2　第一次 IS 运行。可以对照第 2、3 轮：两轮正确率同为 75%，但只有第 2 轮发生了更新。*

**同一个正确率，能给出完全不同的学习信号。** IS 首跑的第 2 轮和第 3 轮都是 24/32。第 2 轮四组都有对有错，留下全部 32 个回答；第 3 轮则是三题全对、一题全错，每道题内部都没有分差，更新次数为零。只看 reward 曲线，我会把这两轮看成相同的两个点。

**同配置重跑，也会改变有效训练量。** 两次 IS 都采了 320 个回答，进入训练的从 112 个变成 80 个，后者少约 29%。有意思的是，第二次反而多更新了一次：它的有效题组更少，却分布在更多轮次里。只比较优化器走了几步，同样会漏掉一部分信息。一次重复能说明本实验里确实有采样波动，还不足以估计这种波动的稳定范围。

**截断有没有生效，要看对应分支记录的指标。** CISPO 的 8 个有效步里，逐步平均概率比在 0.99954–1.00053 之间，8 次 `clip_fraction` 均为 0。前者回答“均值在哪”，后者回答“有没有训练 token 超出这次的阈值”。阈值是 [0, 4]，这个结果无法告诉我每个 token 距离 1 有多近。PPO 的日志保留了 `loss_mean`，没有留下相应的概率比分布或有效截断率，因此它的实际截断比例仍然未知。

**为什么我没按 loss 的正负给三种方法排名？** 就看PPO 第 0 轮那道 7 对 1 错的题。按回答求和，优势是零；按 token 展开，每个优势会重复相应回答的长度次，求和就未必为零。CISPO 的目标还显式含有当前 logprob，即使梯度权重接近 IS，两者显示的标量 loss 也可以不同。再加上每轮都换题，最后一轮的 30/32 只是对当轮四道题的采样结果。要比较最终模型，我还缺一组固定的独立评测题。

## 七、跑完之后，我想怎样验证截断？

我想做的最小改动，是同一批回答连续更新两次，并一直使用第一次采样留下的 old logprob。第一次前向时两端权重相同；第二次前向时，分子已经来自更新后的模型。分别记录这两次的概率比分位数，再按优势符号统计 PPO 真正进入常数分支的 token 比例，就能观察“一次更新究竟把策略推开了多少”。是否足以触发截断，要由测量回答。

和学习搭子的记录对照时，我还想保留一个区别：同一批题并发生成可以缩短等待时间；只有训练端和采样快照之间确实发生了参数更新，才多出了一项由权重变化造成的概率差异。要判断这一点，读代码时应找出保存快照和执行优化器的先后顺序。[同组学习记录](https://zhuanlan.zhihu.com/p/2106862160429295432)

这次读完，我再看一行日志，会先找 `degenerate` 和 `datums`，弄清楚哪些回答提供了梯度，再问这些 token 的概率比多大。PPO 第 0 轮的结果也就说得通了：31 个回答通过判分，真正触发学习的，是其中一道题上出现了分歧。

[系列索引 · Agentic RL Lab 实战笔记](/zh/series/agentic-rl-lab/) · [课程 GRPO 实验目录](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo)

## 参考

- 课程仓库：[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task02 教程：[GRPO 从组内比较开始](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) ｜ [快速启动](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html)
- 本次 run 的代码：[experiments/grpo/01-demo-sync.py](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)
- 同组学习搭子的独立记录：[Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432)
- DeepSeekMath（GRPO 出处）：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PyTRIO 文档：[pytrio.cn](https://pytrio.cn/) ｜ SwanLab：[swanlab.cn](https://swanlab.cn/)
