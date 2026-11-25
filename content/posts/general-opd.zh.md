---
title: "Datawhale Agentic RL Task04｜算到 token 级的 reverse KL：160 条回答、97,492 个训练 token 和一个被稀释的 loss"
date: 2026-11-25T21:30:00+08:00
summary: "在 DeepMath-103K 上跑通 10 步 on-policy 蒸馏，把每个 token 的账都对上：reverse KL 到底在量什么，以及 SwanLab 上那条 loss 曲线为什么其实就是 KL 曲线除以了一个系数。"
tags: ["强化学习", "知识蒸馏", "on-policy distillation", "PyTRIO", "SwanLab", "学习笔记"]
categories: ["学习笔记"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Datawhale 组队学习《Agentic RL》Task04 的记录：通用 On-policy Distillation。教程见 [Agentic RL Lab](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/)，代码见 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)。记录于 2026-11-25。

OPD（on-policy distillation）和 GRPO 几乎没有共同点：GRPO 的监督信号来自奖励函数对**最终答案**的判断，OPD 的监督信号来自老师对**每一个 token** 的偏好。**SwanLab 上那条 `trainer/loss_mean`，不是新信息，它就是 `opd/reverse_kl_mean` 除以了一个系数。**

- [本次 SwanLab 运行](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-opd/runs/b9myg8ou)

## 这次跑的到底是什么

| 项 | 值 |
|---|---|
| 数据 | DeepMath-103K 第 1 个 parquet 分片，seed=42 打乱取前 100 题，**只读 `question`** |
| student | `Qwen3.5-4B` + rank 32 LoRA，lr $4\times10^{-5}$；teacher `Qwen3.8-27B`（冻结） |
| 采样 | `temperature=1.0`，`max_tokens=512`，`enable_thinking=false`，$\beta=1.0$ |
| 规模 | 10 步 × 4 题 × 4 条回答 = 160 条回答 |

## 一轮 OPD 就是这五个盒子

![一次 OPD 迭代：学生采样，老师在同一批 token 上打分，学生被拉过去](/images/agentic-rl/opd-loop.png)

学生在自己会走的路上，被老师逐 token 纠正。核心就两行：

```python
reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
advantages = -args.kl_penalty_coef * reverse_kl
```

$$r_t^{\mathrm{KL}} = \log \pi_\theta(y_t \mid x, y_{\lt t}) - \log \pi_T(y_t \mid x, y_{\lt t}),\qquad A_t = -\beta\, r_t^{\mathrm{KL}}$$

手算：学生给某 token 0.2、老师给 0.4，则 $r_t=\ln 0.2-\ln 0.4\approx-0.693$，$A_t=+0.693$，正优势把这个 token 的概率往上推。**信号是逐 token 的，不是整条回答一个数。**

老师这一步不生成新答案，只是把学生写出来的 token 重新打一遍分：

```python
all_logprobs = teacher_client.compute_logprobs(trio.ModelInput.from_ints(prompt_ids + completion_ids)).result()
teacher_lps = all_logprobs[len(prompt_ids):]
```

一次传入整段序列，因果模型只能看到前文，所以切掉题面之后第一个数恰好对应学生生成的第一个 token——**老师评价的是学生实际走到的位置，哪怕学生前面已经算错了**。前提是 student 和 teacher 共享兼容的 tokenizer，因为同一组 token ID 要同时喂给两者。

## 自回归训练

自回归训练要右移一位，prompt 区间用 0 占位：

```python
prompt_loss_len = len(prompt_ids) - 1
input_ids       = prompt_ids + completion_ids[:-1]      # 长度 P + 511
padded_advantages = [0.0] * prompt_loss_len + advantages
```

**每条回答只有 511 个位置带梯度**（最后一个 token 只作 target，不作输入位置）。一步 16 条 datum，于是

$$16 \times 511 = 8{,}176 \quad\text{个「有梯度的位置」}$$

## 10 步的结果

| 步 | reverse KL | token 级 std | loss_mean | 训练 token | 题面 token |
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

"训练 token"是 PyTRIO 的 `trainer/token_count`，"题面 token"是 `token_count − 8,176`。每步都是 16 条 datum、8,192 个生成 token，不再单列。

![reverse KL 在步之间来回摆动，token 级离散度始终是均值的 5 倍](/images/agentic-rl/opd-reverse-kl.png)

三件事值得记住：

- **别把"下降"当结论。** 10 步均值 0.1264，步间相对波动 12.6%，而前 5 步到后 5 步只降 3.9%——差 3 倍以上。
- **均值掩盖了个体差异。** 每步的 token 级标准差都是均值的 **4.67 ~ 5.71** 倍，粗略按正态估计约 42% 的 token 上 reverse KL 是负的。
- **波动跟题目难度、题面长度都不相关**（r = −0.03 / +0.22，用 DeepMath 自带的 `difficulty` 字段算的）。主因是每步都换新题、新轨迹。

## loss_mean 是被稀释的 reverse KL

![trainer/loss_mean 不是新信号：它是 reverse KL 除以一个稀释系数](/images/agentic-rl/opd-loss-dilution.png)

左图两条线在 10 步上完全重合，残差 ≤ 0.78%：

$$\text{trainer/loss\_sum} \approx 8{,}176 \times \text{opd/reverse\_kl\_mean}$$

因为 `importance_sampling` 在 $\rho\approx1$ 时退化成 $-A_t$ 求和，而 $A_t=-\beta r_t$。PyTRIO 报的 `loss_mean` 是这个和除以 `token_count`——**而 `token_count` 里还含 prompt 占位**：

$$\text{trainer/loss\_mean} = \text{opd/reverse\_kl\_mean} \times \frac{8{,}176}{\text{token\_count}}$$

那个分母就是右图的稀释系数，10 步在 **1.126 ~ 1.279** 之间摆动。`reverse_kl_mean` 与 `loss_sum` 的相关系数是 **0.99948**，与 `token_count` 只有 0.22。

**所以看 OPD 只看 `opd/reverse_kl_mean`，`trainer/loss_mean` 不会给你新信息。**

残差为什么不是 0？因为每条回答的**最后一个 token 只作 target、不作输入位置**，而日志里的 `reverse_kl_mean` 把它算进平均了。把这一点算进去，残差等于那 16 个"最后一个 token"的偏离，隐含 z 分数 10 步全在 ±0.8σ 内、均值 +0.03——没有系统性方向，说明残差不是来自 $\rho\neq1$。

稀释系数还能被独立验证：`token_count = 4 × Σ题面token + 8,176`。用脚本复刻出每步 4 道题的真实题面长度（57 ~ 212 token），第 5 步是 $4\times(91+173+94+212)+8{,}176=10{,}456$，与日志完全一致，**10 步全部精确吻合**。

## token 账本：160 条回答全部撞上上限

![训练 token 花在哪：八成在回答上，两成是 prompt 占位](/images/agentic-rl/opd-token-budget.png)

10 步生成 81,920 个 token，送进训练的是 97,492 个位置：

| 去向 | 位置数 | 占比 |
|---|---:|---:|
| completion 位置（带梯度） | 81,760 | 83.9% |
| prompt 占位（advantage = 0） | 15,732 | 16.1% |

差出来的 160 个位置，是每条回答的最后一个 token：**生成了、老师也打了分、还进了日志均值，但没有产生梯度。**

更要紧的是右图：每一步的 `completion_tokens_mean` 都恰好等于 512.0，也就是 `max_tokens`。**160 条回答没有一条自己停下来。** 后果是连锁的——没有一条写完，`\boxed{}` 一个都没出现；老师打分的对象是一条**被截断到 512 token 的推理**；学生只学会了"在前 512 个 token 上像老师"，从没被奖励"把话说完"。

**如果这次配置是为了提升数学正确率，那它在设计上就做不到。** 这条不用等评测就能确定：所有回答都没写完，连"答案能不能被提取出来"这个前置条件都不成立。

## 这 10 步到底训练出了什么

1. **跑通了。** 160 条回答、81,920 个生成 token、8,176 个带梯度的位置一步不少，数据流与文档一致。
2. **没有证据表明学生变强。** reverse KL 只降 3.9%，落在步间噪声里；而且它衡量的是"学生在学生自己会走的路上有多像老师"，不等于正确率。
3. **这次运行结构上无法衡量正确率。** 全部撞上 512 上限，没有一条写完。
4. **最有用的收获反而是指标的读法。** `loss_mean` 是 `reverse_kl_mean` 除以一个 1.13–1.28 的系数；`token_count` 能被题面长度逐位还原。

继续做的话我会改三件事：`max_tokens` 提到足够让回答写完（这个数得先测出来）、`enable_thinking` 配套设置、留一批固定题让训练前后的学生用同一套解码参数各答一遍，比较正确率和答案可提取率。

## 复现

读原始指标比看 SwanLab 网页方便（`.swanlab` 是记录流，用 swanlab 自带的 reader 就能解出每一步的每个标量）：

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

踩到的坑：第一次运行已打印 `Loaded 20 DeepMath prompts`，却在创建 teacher client 时报 `model_not_found`——我按旧文档写了 `Qwen3.6-27B`，账号里可见的是 `Qwen3.8-27B`。这一行能列出当前账号可用的模型：

```bash
uv run python -c 'import pytrio; print(*pytrio.ServiceClient().get_supported_models(), sep="\n")'
```

## 参考资料

- [Datawhale · General OPD 实验教程](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/)
- [同步训练脚本](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/01-demo-sync.py) · [异步训练脚本](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/02-demo-async.py)
- [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [DeepMath-103K 数据集](https://modelscope.cn/datasets/AI-ModelScope/DeepMath-103K)
