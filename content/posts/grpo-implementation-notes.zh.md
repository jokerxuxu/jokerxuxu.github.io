---
title: "Datawhale Agentic RL Task2：把 GRPO 从公式读到代码"
date: 2026-11-20
summary: "逐段读 agentic-rl-lab 的 GRPO 实现，再自己把 importance_sampling / ppo / cispo 三种 loss 各跑一遍：320 条采样只有 112 条进入更新，同一份配置两次跑就差 30%，而概率比始终贴着 1，clip 一次都没触发。"
tags: ["强化学习", "GRPO", "PyTRIO", "实现", "组队学习"]
categories: ["学习笔记"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Task02 记录：GRPO 章。教程见 [GRPO 从组内比较开始](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/)，快速启动见 [运行指南](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html)，代码见 [datawhalechina/agentic-rl-lab/experiments/grpo](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo)。记录于 2026-11-20。

上一篇（Task01）讲的是"为什么"：baseline 为什么能换成组内均值、IS / PPO / CISPO 三个力度函数怎么推。这一篇讲"怎么写"——把 `experiments/grpo/01-demo-sync.py` 从头到尾读一遍，看清每一步在干什么；然后回头看这台机器上真的跑过的那次 run 留下了什么。

第四节是我自己动手的部分：把 `importance_sampling` / `ppo` / `cispo` 三种 loss 在同一台机器上各跑一遍，另外加一次**同配置的重复 run**。结果和一位学习搭子的独立记录对照着读（他的那篇：[Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432)）。

先给结论：**GRPO 的实现比公式短得多。** 难的从来不是把公式敲进代码，而是那些"写错了也不报错，只是悄悄学歪"的地方：采样时的策略快照、右移对齐、退化组、以及"没数据的那一步到底该不该更新"。

## 一、一次 step 到底发生了什么

整个训练循环可以压成六步。下图是教程仓库里的示意图（Apache-2.0）：

![GRPO 流程：同一道题采一组回答，减去组内均值得到 advantage，再去更新策略](/images/agentic-rl/grpo-pipeline.png)

对应到 `01-demo-sync.py`：

| 阶段 | 函数 | 产出 |
| --- | --- | --- |
| 取题 | `pick_batch` | `batch_size` 道 GSM8K 题 |
| 造 prompt | `build_prompt` | token 序列（few-shot + 题目 + `\boxed{}` 要求） |
| 采样 | `run_rollout_group` | 每题 `group_size` 条 `(tokens, logprobs, text)` |
| 打分 | `grade_answer` | 每条一个 0/1 reward |
| 算优势 | `reward - mean_reward` | 每条一个组内 advantage |
| 更新 | `forward_backward` + `optim_step` | LoRA 参数走一步 |

下面逐段看。

## 二、逐段读代码

### 2.1 prompt：把格式要求写进指令

```python
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
```

前面再拼一个 few-shot 示例（"strawberry 里有几个 r"），让模型稳定地按格式回答，最后走 chat template：

```python
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

`enable_thinking=False` 是因为带思维链的模板会让采样变长、也会让 reward 提取变复杂。**这里不是算法，只是让"答案能被规则打分"这件事成立**——RLVR 的前提是答案可验证。

### 2.2 reward：只认最后一个 `\boxed{}`

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

取**最后一个**而不是第一个，是因为模型可能中途算错又改口；`normalize_answer` 只做去逗号和去尾点的轻量归一化，防止 `1,000` 和 `1000` 被判成不同。

没有 `\boxed{}` 就直接 0 分——注意这会把"格式没按要求"和"算错了"混成同一个信号，是这份 demo 明显可以再拆细的地方。

### 2.3 采样：old policy 必须是"当场的快照"

这是整份实现里最关键的一行：

```python
sampling_client = training_client.save_weights_and_get_sampling_client()
```

它放在每个 step 的开头，意思是：**每个 step 都从当前 LoRA 权重存一份匿名快照，这个快照就是这一轮采样用的 old policy $\pi_{\text{old}}$。**

采样的时候一次要一整组：

```python
result = sampling_client.sample(
    prompt=trio.ModelInput.from_ints(prompt_tokens),
    num_samples=group_size,
    sampling_params=sampling_params,
    return_text=True,
).result()
```

`num_samples=group_size` 就是 GRPO 的定义：同一个 prompt 要采多个 completion，否则组内没法比较。采样回来的三样东西是后面训练的全部原材料：

```python
tokens   = list(sequence.tokens)          # 生成的 token
logprobs = [float(v) for v in sequence.logprobs]  # 旧策略对这些 token 的 logprob
reward   = grade_answer(text, ground_truth)
```

**`logprobs` 必须来自采样那一刻的策略。** 后面 IS / PPO / CISPO 里的概率比 $\rho=\exp(\log\pi_\theta-\log\pi_{\text{old}})$ 的分母就是它。如果偷懒在参数更新之后再重新算一遍 logprob，$\rho$ 会恒等于 1，三种 loss 的修正项全部失效——而且不会报错，只会安静地变成普通 policy gradient。

### 2.4 advantage：减均值，退化的组直接丢掉

```python
mean_reward = sum(rewards) / len(rewards)
advantage = reward - mean_reward
```

例：组内 reward 是 `[1, 0, 1, 0]`，均值 0.5，advantage 就是 `[+0.5, -0.5, +0.5, -0.5]`。

如果组内 reward 全相同，`advantage` 全是 0，这一组没有任何相对优劣信息，代码直接跳过：

```python
if all(sample.advantage == 0.0 for sample in rollout_samples):
    n_degenerate += 1
    continue
```

`frac_degenerate` 记的就是"这一步里有多少道题属于这种情况"。**它比 reward 更能说明这次训练有没有在学**——这一点后面用我的真实数据会再遇到。

### 2.5 Datum：右移对齐，prompt 段全部占位

PyTRIO 的内置 `importance_sampling` / `ppo` loss 需要三样等长的东西：`target_tokens`、`logprobs`、`advantages`。代码在 `build_grpo_datum` 里对齐：

```python
observation_len = len(prompt_tokens) - 1
input_tokens  = prompt_tokens + sample.tokens[:-1]
target_tokens = [0] * observation_len + sample.tokens
padded_logprobs = [0.0] * observation_len + sample.logprobs
padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)
```

对齐关系是这样（`p` 是 prompt，`c` 是 completion）：

```text
input   = p0 p1 ... p_{n-1} | c0 c1 ... c_{T-2}
target  =  0  0  ...   0   | c0 c1 ... c_{T-1}
adv     =  0  0  ...   0   |  A   A  ...   A
```

三点值得记住：

1. **右移一位。** 用第 $i$ 个 token 预测第 $i+1$ 个，所以 input 丢掉 completion 的最后一个，target 是完整的 completion。
2. **`observation_len` 是 `len(prompt) - 1` 而不是 `len(prompt)`。** 最后一个 prompt token 留在 input 里当上文，用来预测第一个 completion token；所以 prompt 占位是 $n-1$ 个，target 从第一个 completion token 开始。
3. **prompt 段全部填 0。** prompt 是上下文，不是被优化的对象；`advantages` 为 0 保证梯度不会回传到 prompt 上。

还有一条：**advantage 被广播到了这条回答的每一个 completion token 上**（`[sample.advantage] * len(sample.tokens)`）。也就是说"这条回答整体比同组平均好，所以它每个 token 都该被鼓励"——**token 级的信用分配在这里是缺失的**。这一条是后面 DAPO、GSPO 要修的东西，先记在这里。

### 2.6 更新：空 batch 的那一步，连优化器都不走

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

注意 `optim_step` 也在 `if datums:` 里面。如果这一步所有题都退化，`datums` 是空的，**这一步不会有 forward/backward，也不会有一步 optimizer 更新**（Adam 的动量计数和 step 数都不动）。

这未必是坏事——空 batch 去更新，Adam 的动量会被零梯度污染，甚至可能把 lr schedule 推进一格。但这是"隐式约定"，读代码时很容易看漏。对照日志就更好认：**`loss_mean` 缺失的步，就是这一步什么都没学。**

### 2.7 custom loss：CISPO 是在本地定义的梯度形状

`importance_sampling` 和 `ppo` 是 PyTRIO 内置的，`cispo` 不是，走 `forward_backward_custom`：

```python
custom_datums = [build_custom_forward_datum(datum) for datum in datums]
fwd_bwd_future = training_client.forward_backward_custom(
    custom_datums,
    make_cispo_loss_fn(sampling_logprobs_list=..., advantages_list=..., ...),
)
```

分工是这样的：

1. 远端模型 forward，算出**当前**策略对 `target_tokens` 的逐 token logprob（可求导）；
2. 这个 logprob 回到本地，交给 `loss_fn`；
3. 本地用 torch 写任意 loss，返回 `(loss, metrics)`；
4. PyTRIO 拿这个 loss 去做 backward 和参数更新。

也就是说，**你不是在本地训练模型，只是在本地定义"这些 logprob 应该怎么变成 loss"**。这也解释了 Task01 里那条"本地 torch 是 CPU 版"的注意事项：只有走 custom loss 时，本地 torch 才真的参与计算。

CISPO 的核心是 `detach` 的位置：

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
clipped_ratio = torch.clamp(prob_ratio, min=clip_low_threshold, max=clip_high_threshold)
cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
```

裁剪后的 ratio 只当固定砝码，梯度只从 `target_logprobs` 回传。与 Task01 推出来的结论一致：**PPO 是把目标本身裁掉（越界就停手），CISPO 是把力度封顶（越界也继续推，只是不再加大）。**

## 三、我这次是怎么跑完的

### 3.1 环境和命令

在 `~/Workspace/agentic-rl-lab`（跟着教程 clone 下来的仓库，commit `f66c6fe`）里：

```bash
uv sync
trio login
swanlab login

uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --swanlab-mode online
```

`--loss-fn` 没写，走的默认值 `importance_sampling`。环境是 MacBook Air（M5，16 GB），Python 3.13.18，torch 2.13.0，pytrio 0.2.9，swanlab 0.10.4，datasets 5.0.0。

### 3.2 配置与存档

每一次 run 都会在 `experiments/grpo/swanlog/` 下留一个目录，里面是这次实验的完整存档：

```text
experiments/grpo/swanlog/run-20261117_210611-2gmiydii/
├── files/config.yaml              # 全部超参
├── files/swanlab-metadata.json    # 机器、python 版本、git commit、命令行
├── files/requirements.txt         # 当时的依赖版本
├── debug/debug.log                # SwanLab 的告警
└── run-2gmiydii.swanlab           # 二进制日志
```

`config.yaml` 里记着我用的超参：

| 参数 | 值 | | 参数 | 值 |
| --- | --- | --- | --- | --- |
| base_model | `Qwen/Qwen3.5-4B` | | learning_rate | 4e-5 |
| lora_rank | 32 | | batch_size | 4 |
| steps | 10 | | group_size | 8 |
| max_tokens | 512 | | temperature / top_p | 1.0 / 1.0 |
| loss_fn | `importance_sampling` | | seed | 42 |

线上记录：[SwanLab · agentic-rl-lab-grpo](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo/runs/2gmiydii)。最终权重名和 run 名同规则生成：`grpo-sync-qwen35-4b-gsm8k-importance-sampling-steps10`。

### 3.3 十个 step 的读数

| step | reward | degenerate | avg_gen_len | loss_mean | 进入更新的 completion |
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

下面两张图是从 SwanLab 导出的 CSV 重画的（不是 SwanLab 的截图），因为想看的量不在它的默认面板里。

![每一步的梯度来自哪几组，以及哪些步根本没有更新](/images/agentic-rl/grpo-signal-budget.png)

**第一个观察：这次 run 的"信号预算"只有 35%。**

十步一共采了 $10\times4\times8=320$ 条回答，因为退化组被跳过，真正进入更新的只有 **112 条**（35%）；**10 步里有 3 步（0、3、6）一个 datum 都没产生**，那三步的 `loss_mean` 在日志里是空的——和 2.6 节说的"空 batch 连优化器都不走"完全对上。左图里绿色柱子就是这三步竖着写"no update"的地方。

**第二个观察：reward 曲线不能用来判断学没学。**

表面上 reward 在 0.75–1.0 之间抖，没有上升趋势。但更根本的原因是：**每一步的题目都在换。** `pick_batch` 取的是数据集的连续切片，第 0 步是第 0–3 行，第 1 步是第 4–7 行，十步四十道题互不重复、难度也不一样。所以 step 之间的 reward 差异里，混着"题目难度"这个噪声，而不是纯粹的"策略变好"。而教程的 run 指南里也写明了这个仓库**没有独立的 `eval.py`**——训练 reward 和"模型变好了"之间，缺一个固定评测集这个环节。

**第三个观察：慢的两个步，跟训练没关系。**

![相邻两步之间的时间：有两个尖刺，来自 SwanLab 上传重试](/images/agentic-rl/grpo-step-latency.png)

十步从 21:06:35 跑到 21:19:40，一共 13 分钟，中位数 35 秒一步。但第 7 步花了 174 秒、第 8 步花了 389 秒。对照 `debug/debug.log`，里面有 7 条

```text
Upload failed due to network or server issues, retrying automatically.
```

全部落在这两步里——**是 SwanLab 上传卡住，不是模型在算。** 按中位数估，十步真正的训练时间大约 350 秒，剩下 435 秒在等日志重试（约 55%）。这个提醒很实际：远程训练跑得慢时，先分清是"算得慢"还是"日志传得慢"，否则会误判成本和迭代节奏。

### 3.4 成本

教程作者自己给的经验是：十步大约 5 元，调通整个脚本花了 55 元。我这次是跟着快速启动里的最小配置跑的一次 ten-step demo，实际消耗以 PyTRIO 控制台账单为准；本地能确定的只有 wall clock——13 分钟，其中约 7 分钟在等日志上传。

## 四、自己跑一遍：三种 loss 的同机对照

搭子那篇提醒了我一件事：只跑一种 loss，看不出三种目标的差别。于是 2026-11-20 我在同一个仓库、同一台机器上把 `ppo` 和 `cispo` 也各跑了一遍，另外**把 `importance_sampling` 原样重跑了一次**——为了知道"同一份配置两次跑能差多少"。加上 11-17 那次，一共四条曲线。

超参完全没动（Qwen3.5-4B、LoRA rank 32、lr 4e-5、10 步、bs 4、gs 8、512 tokens、temperature 1.0、seed 42），只改 `--loss-fn`：

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --loss-fn {importance_sampling|ppo|cispo} \
    --swanlab-mode online
```

四次 run 都在同一个 SwanLab project：[agentic-rl-lab-grpo](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo) —— IS（11-17）`runs/2gmiydii`、PPO（11-20）`runs/wq9npu15`、CISPO（11-20）`runs/ulo34fae`、IS 重跑（11-20）`runs/fscv1nz3`。

| 指标 | IS 11-17 | IS 11-20（重跑） | PPO 11-20 | CISPO 11-20 |
| --- | --- | --- | --- | --- |
| 起点 reward | 1.000 | 1.000 | 0.969 | 1.000 |
| 最终 reward | 0.875 | 0.938 | 0.938 | 0.938 |
| reward 最低点 | 0.750（s2） | 0.719（s3） | 0.750（s3） | 0.688（s3） |
| 平均 reward（1–9） | 0.858 | 0.920 | 0.878 | 0.892 |
| frac_degenerate 均值 | 0.65 | 0.75 | 0.60 | 0.65 |
| 零更新的步 | 0, 3, 6 | 0, 7 | 3 | 0, 6 |
| 有效 completion（datums） | 112 | 80 | 128 | 112 |
| train_tokens | 37,906 | 25,920 | 41,194 | 35,412 |
| loss_mean 为负的次数 | 1/7 | 1/8 | 3/9 | 7/8 |
| avg_gen_len 峰→末 | 380.7→237.1 | 365.0→292.6 | 373.3→225.0 | 370.2→253.0 |
| 单组耗时（含日志等待） | 785 s | 352 s | 360 s | 320 s |

![四条曲线：reward 分不开、退化组占主导、只有 CISPO 稳定在零以下、同一配置两次差了 30%](/images/agentic-rl/grpo-three-losses.png)

### 4.1 先定噪声地板：同一份配置跑两次

看图 4：IS 11-17 和 IS 11-20 是**同一条命令**，只差了一天。

- 有效样本 112 vs 80，差 30%（train_tokens 37.9k vs 25.9k，差 46%）
- 平均 reward 0.858 vs 0.920，差 0.062
- reward 最低点 0.750 vs 0.719

**所以 PPO 看起来"信号最多"（128 个 datums）并不能说明它更强——同一个 IS 自己两次就差了 30%。** 任何小于这个量级的组间差异都不能当结论。这是我这批实验里最有用的一条。

顺带解释了一个反直觉现象：**为什么 seed 都是 42，两次结果还差这么多？** 因为 `np.random.seed` 只管本地随机数，管不到远端采样服务——采样本身不是确定性的。

### 4.2 更根本的发现：clip 一次都没触发

CISPO 额外记了 `cispo/ratio_mean` 和 `cispo/clip_fraction`，本来只是参考指标，结果它们把问题指到了根上：

![概率比始终贴着 1，clip_fraction 每一步都是 0](/images/agentic-rl/grpo-cispo-ratio.png)

- `ratio_mean` 十步里始终在 **1 ± 5.3×10⁻⁴**
- `clip_fraction` **每一步都是 0**

$\rho=\exp(\log\pi_\theta-\log\pi_{\text{old}})$ 几乎恒等于 1，意味着**在采样和更新之间，策略几乎没动**。既然 $\rho$ 没离开 1：

$$
w_{\text{IS}}(\rho)=\rho\approx 1,\qquad
w_{\text{PPO}}(\rho)=\operatorname{clip}(\rho,1\pm\varepsilon)\approx \rho,\qquad
w_{\text{CISPO}}(\rho)=\operatorname{sg}(\operatorname{clip}(\rho))\approx \operatorname{sg}(\rho)
$$

三者在这批实验里**走的是同一个分支**。Task01 推出来的"三种 loss 的差别被压缩进一维力度函数 $w(\rho)$"这句话没错，但前提是 $w$ 的不同区域会被走到——这次一个都没走到。

**所以"10 步拉不开差距"这个结论要改写：不是步数不够，而是 $\rho$ 从来没远离 1，三种 loss 的差异连生效的机会都没有。**

### 4.2.1 更要命的是：这个 demo 里 $\rho$ 恒等于 1

再往里想一层，会发现上一段那句"clip 没触发"不是学习率太小造成的，而是**结构上不可能触发**。看代码的顺序：

```python
sampling_client = training_client.save_weights_and_get_sampling_client()  # 快照当前权重 W_k
...                                                                      # 用 W_k 采样 → sampling_logprobs = log π_{W_k}(o)
fwd_bwd_future = training_client.forward_backward(datums, loss_fn=...)    # 用 W_k 前向 → target_logprobs = log π_{W_k}(o)
optim_future = training_client.optim_step(adam_params)                    # W_k → W_{k+1}
```

**算 loss 的时候，learner 的权重就是采样时那份权重**，所以

$$
\rho_t=\exp(\log\pi_{W_k}(o_t)-\log\pi_{W_k}(o_t))\equiv 1 .
$$

实测的 $1\pm5.3\times10^{-4}$ 是前向的数值差异（批量前向 vs 采样时解码），不是策略漂移——十步下来它没有趋势，只是随机游走。

代入 [PyTRIO 官方文档](https://docs.pytrio.com/docs/guide/loss_fn) 给的三个公式：

| loss | 公式（文档原文） | 在 $\rho=1$ 处 |
| --- | --- | --- |
| `importance_sampling` | `loss = -(prob_ratio * advantages).sum()` | $-(1/N)\sum_t A_t$ |
| `ppo` | $-\mathbb{E}\big[\min(\rho A,\ \operatorname{clip}(\rho,1\pm\varepsilon)A)\big]$ | $\operatorname{clip}(1)=1\Rightarrow -(1/N)\sum_t A_t$ |
| `cispo` | $-\sum_t \operatorname{sg}(\operatorname{clip}(\rho))\log\pi_\theta(o_t)A_t$ | $+(1/N)\sum_t \mathrm{NLL}_t A_t$ |

**IS 和 PPO 在 $\rho=1$ 处逐字相同；连梯度也一样**（三者的梯度都是 $-\mathbb{E}[A\,\nabla\log\pi]$）。只有 CISPO 是另一个函数。所以这三种 loss 的差别只在 $\rho\neq1$ 的地方才会出现——同一批数据做多个 inner epoch（第二个 epoch 起 $\rho$ 才偏离 1），或者采样器落后于 trainer（真正的 off-policy）。

而 `01-demo-sync.py` 和 `02-demo-async.py` 都是**每步重新快照**，所以两个脚本里 $\rho$ 永远等于 1。**这意味着：无论加步数、提学习率、还是换模型，这个 demo 都分不开三种 loss。** 我原来写的"把学习率提上去让 $\rho$ 跑起来"是错的，这里更正。

### 4.2.2 那 loss 的符号和数值由什么决定？一次 batch 级抽签

把第 $i$ 条回答的长度记作 $T_i$、优势记作 $A_i$（组内零均值），则

$$
L_{\text{IS}}=L_{\text{PPO}}=-\frac1N\sum_i T_i A_i=-\frac1N\sum_i (T_i-\bar T)A_i ,
$$

**符号只取决于"这一步里答对的回答是不是更长"**，算法对它没有任何约束——这是一次抽签。而 CISPO：

$$
L_{\text{CISPO}}=\frac1N\sum_i\Big(\sum_t \mathrm{NLL}_{i,t}\Big)A_i\;\approx\;\frac{h}{N}\sum_i T_i A_i=-h\cdot L_{\text{IS}},
$$

其中 $h$ 是策略的平均每 token 负对数似然（熵，nats）。于是有两条可检验的预测：**CISPO 与 IS/PPO 反号**，且**量级约为 $h$ 倍**。

同一批 40 道题、四个 run 的实测：

| 检验 | 结果 |
| --- | --- |
| IS 11-17 vs IS 11-20（两次独立抽样） | 6/6 个共有步同号 |
| IS/PPO 的共识 vs CISPO | 5/6 步反号（唯一例外 step 1，PPO = −0.0011，几乎打平） |
| $\lvert L_{\text{CISPO}}\rvert/\lvert L_{\text{IS}}\rvert$ | 0.20–1.47，均值 ≈ 0.73 → 对应 $h\approx0.2$–$1.5$ nats/token |

所以 **"CISPO 的 loss 系统性偏负"不是巧合，是公式本身决定的**（它乘的是 logprob 而不是 ratio）。

而 **"PPO 的 loss 全程为正"没有理论依据**。最有力的反证在搭子自己的数据里：他跑的 IS 和 PPO 在 $\rho=1$ 处是**同一个函数**，可他的 IS 有 3/9 次为负。同一个函数在两次抽样里符号不同，就说明符号是抽签结果，不是算法签名。

### 4.2.3 "custom loss 更慢"：机制成立，但这个规模测不出来

机制上确实更贵：CISPO 要把 logprob 传回本地进程、在 CPU 上用 torch 算 loss、再把梯度传回远端；内置 loss 全在远端 GPU 上算。10 步的量级下这点开销被采样时间和日志上传淹没，所以"他那边 CISPO 678 s vs IS 638 s"和"我这边 CISPO 320 s 反而最快"都不矛盾——**方向对、量级测不出来**。要测得关掉日志同步、拉长步数、再多跑几次取平均。

### 4.3 能复现的和不能复现的

| 结论 | 搭子那篇 | 我这边 |
| --- | --- | --- |
| CISPO 的 loss 系统性偏负 | 6/8 次为负 | 7/8 次为负 ✓ |
| IS 会出现负 loss | 3/9 次为负 | 1/7、1/8 次为负（两次 IS 都不多） |
| PPO 的 loss 全程为正 | 是 | 否，3/9 次为负 ✗ |
| "先变长探索、再收敛" | gen_len 在 step 2 冲高 | 四组都在 step 2 冲高（365–381）后回落到 225–293 ✓ |
| custom loss 更慢 | CISPO 678 s vs IS 638 s | CISPO 320 s、IS 352 s、PPO 360 s ✗ |

能复现的是**结构性**的：CISPO 的 loss 几乎总在零以下（它乘的是 logprob 而不是 ratio，见 4.2.2），四组都先变长再收敛。不能复现的是那些**数值性**结论：负号出现几次、哪个 loss 更慢。前者说明单次 run 的噪声比那篇大，后者更可能是被日志上传耗时掩盖了——我 11-17 那次跑，10 步里有 435 秒花在等 SwanLab 重传，比三个 loss 之间的差距大一个量级。

**但"先变长再收敛"这条也要打个折扣。** 四次 run（含搭子的）跑的是**同样的前 40 道题、同样的顺序**——`pick_batch` 取的是数据集的连续切片，`01-demo-sync.py` 和 `03-grpo.py` 在这一步完全一样。所以四次 run 的 `avg_gen_len` 曲线相关系数高达 **r = +0.87 ~ +0.98**，而 reward 只有 +0.49 ~ +0.83、`frac_degenerate` 只有 +0.20 ~ +0.80。

**换句话说，那条长度曲线主要是这 40 道题的难度剖面，而不是 GRPO 的动力学**——换成任何算法、甚至一条随机基线，也会画出同样的形状。反过来说，这正是"结构性结论能复现、数值性结论不能"的真正原因：**复现的是共享的实验条件（同一批题、同一段脚本、同一套公式结构），不是算法的性质。**

### 4.4 耗时：custom loss 没有更贵

三次 11-20 的 run 分别是 352 s（IS）、360 s（PPO）、320 s（CISPO），差距在 40 秒以内，而且这个量级和 SwanLab 上传等待同阶。**在 10 步这个规模上测不出 custom loss 的额外开销**；要测得把步数拉长，并关掉日志同步再对比。

## 五、跑完这一遍之后，记下来的实现细节

这些是读代码 + 看自己的 run 才注意到的东西，教程正文里不会写：

1. **`swanlog/` 目录被 `.gitignore` 掉了。** 仓库里 `experiments/grpo/swanlog/.gitignore` 写的是 `*`。所以这次 run 的配置、metadata、日志**只存在本地**，换机器就丢。想留档要自己拷走。

2. **`run_name` 同时是实验名和权重名。** 规则是 `grpo-{sync|async}-{模型}-{任务}-{loss}-steps{N}`，由 `build_run_name` 生成；改 loss 时它会自动变，所以同一批实验在 SwanLab 里能按名字排序，也方便 `save_weights_for_sampler(name=run_name)` 找到对应权重。

3. **非 `--all-data` 模式下题目会回绕。** `pick_batch` 用 `(start + offset) % len(dataset)` 取索引，步数乘 batch 超过数据规模后就会重复出题。我这次 40 < 7473，没有重复。

4. **`--all-data` 会忽略 `--steps`。** `effective_steps = ceil(7473/4) = 1869` 步，全量跑一遍；这个模式下不再回绕，每条样本最多用一次。

5. **每一步都会存一次匿名权重快照。** `save_weights_and_get_sampling_client()` 每步调用一次，所以"每步一个 old policy"这件事是框架替你做掉的；也是为什么这个 demo 的"权重同步"不需要自己写。

6. **本地 torch 是 CPU 版，但 custom loss 真的会用到它。** 只有 `forward_backward_custom` 会把 logprob 送回本地算 loss，这时 CPU torch 才开始工作。

7. **日志里有两个 loss 字段。** `loss_mean` 和 `loss` 内容相同，另外 trainer 返回的其他键会被加上 `trainer/` 前缀，CISPO 自己的指标保留 `cispo/` 前缀——所以导出 CSV 时按名字筛，别只盯着 `reward`。
8. **`huggingface.co` 断了也能跑。** 第二次跑三组实验时 HF 完全连不上（`load_dataset` 卡在 HEAD 请求重试里好几分钟）。GSM8K 已经在本地缓存里，所以加上 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 就直接读缓存启动，不用等网络。**如果你的数据已经下过一次，就别让训练脚本死在一次无关的联网检查上。**

## 六、和论文版 GRPO 的差距

这份 demo 为了讲清楚链路，做了三处简化（教程自己在 readme 里也列了）：

| 论文版 | 这份 demo | 怎么改 |
| --- | --- | --- |
| 组内标准化 $A_i=\frac{r_i-\text{mean}}{\text{std}+\delta}$ | 只减均值 | 加一行 `std_reward = np.std(rewards)` |
| 显式 reference KL 项 | 没有 KL 锚点 | 在 custom loss 里加 $\beta \log\pi_\theta/\pi_{\text{ref}}$ |
| reward model / 过程奖励 | 只有 0/1 规则 reward | 换成 PRM 或更细的 reward shaping |

另外论文里的 GRPO 是 $\pi_{\text{old}}$ 与 $\pi_{\text{ref}}$ 分开的：前者是每轮采样的快照（概率比的分母），后者是初始 SFT 模型（KL 的锚点）。这个 demo 里根本没有 $\pi_{\text{ref}}$，所以也别指望从它的 loss 曲线里读出 KL 项。

还有一个 demo 特有的简化值得单独点名：**advantage 广播到整条回答**。论文版 GRPO 也是这么做的（token 级信用分配的缺失正是 DAPO / GSPO / Dr.GRPO 要处理的问题），但知道"这份代码就这么写"和"算法本来就这样"是两件事。

## 七、下一步

- **想让三种 loss 真的分开，必须让 $\rho\neq1$。** 现在它们在 $\rho=1$ 处是同一个函数（4.2.1），加步数、提学习率都没用。可行的做法有两条：同一批 datums 做多个 inner epoch（第二个 epoch 起 $\rho$ 才偏离 1），或者故意让采样器落后于 trainer（stale weights，真正的 off-policy）。这也是 DAPO/GSPO 那些方法里 clip 真正会触发的地方。
- **加大 group size、提高任务难度。** 现在的瓶颈是 60–75% 的采样被丢掉：GSM8K 对 Qwen3.5-4B 太简单，第 0 步就全对。换成更难的题（比如 MATH），或者把 `group_size` 从 8 加到 16，看看 `frac_degenerate` 会不会掉下来。
- **每次比较都配一个同配置的重复 run。** 4.1 节那 30% 的噪声地板是这次最贵的教训：不做重复，就分不清"算法差异"和"这次运气"。
- **固定题目 + 固定评测集。** 既然四条 run 的长度曲线相关到 r≈0.9 说明曲线由题目决定，那要评估"策略有没有变好"，就得每步用同一批题、并且另有一个不参与训练的固定评测集。这是 3.3 节那个观察逼出来的需求。
- **把简化补回去**：组内标准化 + KL 项，正好用 `forward_backward_custom` 练手。

> 系列索引：[Agentic RL Lab 实战笔记](/zh/series/agentic-rl-lab/)。上一篇：[Task01 · 一个机器人方向读者的 LLM 强化学习入门](/zh/posts/from-controller-to-policy-gradient/)。

## 参考

- 课程仓库：[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task02 教程：[GRPO 从组内比较开始](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) ｜ [快速启动](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/run.html)
- 本次 run 的代码：[experiments/grpo/01-demo-sync.py](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)
- 同组学习搭子的独立记录：[Task02 打卡：用 PyTRIO 在 GSM8K 上跑通 GRPO](https://zhuanlan.zhihu.com/p/2106862160429295432)（第 4 节的部分结论来自与它的对照）
- DeepSeekMath（GRPO 出处）：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PyTRIO 文档：[pytrio.cn](https://pytrio.cn/) ｜ SwanLab：[swanlab.cn](https://swanlab.cn/)
