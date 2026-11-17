---
title: "Datawhale Agentic RL Task1：一个机器人方向读者的 LLM 强化学习入门"
date: 2026-11-17
summary: "在 PyTRIO 上跑通第一次 LLM 强化学习，并用控制系统的语言重讲一遍 GRPO：baseline 为什么能降方差、组内均值为什么够用，以及我的控制直觉在哪一步失效。"
tags: ["强化学习", "GRPO", "PyTRIO", "LLM", "组队学习"]
categories: ["学习笔记"]
series: ["Agentic RL Lab"]
math: true
ShowToc: true
---

> Datawhale 组队学习《Agentic RL》Task01 的记录：环境、基础概念。教程见 [Agentic RL Lab](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/)，代码见 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)。记录于 2026-11-17。

虽然我有部分涉猎做LLM和Agent相关知识，但平时更多接触的还是机器人学习。来学习 LLM 的 RL，动机也很直接：**同一个 RL 范式，换一个领域会变成什么样**——哪里被逼着改了做法，哪里其实没变。

所以这篇不打算再抄一遍词汇表。我想回答自己的问题：**已经有的控制直觉能带我走多远，又会在哪一步失效。** 后续也希望了解做LLM人的思维方式。

## 一、本地不用GPU也可以训练

**架构**

| 组件 | 跑在哪 | 负责什么 |
| --- | --- | --- |
| PyTRIO `0.2.9` | 云端 | 前向、反向、优化器更新、采样、权重保存 |
| SwanLab `0.10.4` | 云端 + 本地日志 | 曲线、超参数、终端输出 |
| 仓库脚本 | 本地 | 数据、reward、训练循环 |

我熟悉的是同一个分工的另一个版本：仿真器放在服务器，本地只跑控制律和打分。区别在于返回的东西——那边返回状态轨迹，这边返回梯度和权重。

装起来只有三步，`uv sync`、`trio login`、`swanlab login`，然后一次十步的 demo：

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --loss-fn importance_sampling \
    --swanlab-mode online
```

教程里反复强调容易踩坑的点：

1. **本地的 torch 是 CPU 版**。它在 `pyproject.toml` 里指定了 `pytorch-cpu` 源，不要"贴心"地换成 CUDA 版。只有走 `forward_backward_custom`（自定义 loss）时它才真正参与计算。
2. **每条命令都要 `uv run`**。`trio` 和 `swanlab` 装在 `.venv` 里，系统终端直接敲是 command not found。


另外，首次运行前也可以顺手排查这三点：

3. **网络要分段检查**。依赖安装、Hugging Face 下载和云端训练请求是不同环节；遇到超时，先看日志卡在哪一步，再检查对应的网络和代理配置，避免反复重启整个训练。
4. **HF 缓存要留空间**。模型、tokenizer 和数据集下载可能占用本地磁盘；空间紧张时，可以用 `HF_HOME` 指向容量充足的目录，并在后续运行中保持路径一致，减少重复下载。
5. **先小跑，再看额度**。先用上面的十步 demo 验证流程，并检查云端账户的剩余额度和实际消耗，再增加训练步数、batch size、group size 或生成长度。

**云端服务把"拥有算力"变成了"调用算力"**，对不守着机房的读者来说，这是这次能跑起来的前提。

## 二、用机器人 RL 的语言翻译一遍

机器人 RL 那套记号在这边几乎能逐项对上：

| 记号 | 机器人 RL | LLM RL |
| --- | --- | --- |
| 状态 $s_t$ | 关节角、角速度、IMU、接触 | prompt + 已经生成的 token |
| 动作 $a_t$ | 力矩或目标关节角等，通常连续，高维 | 下一个 token，离散，词表 |
| 策略 $\pi_\theta$ | MLP 输出高斯均值和方差 | 同一个 transformer 输出词表上的分布 |
| 奖励 | 一般是密集，也有稀疏奖励 | 通常只有末尾终局由外部打分器给 |
| 探索 | 高斯噪声、熵正则、最大熵 | 温度采样、top-p |
| 算法 | PPO、SAC、TD3 | PPO、GRPO、DAPO、GSPO |

个人理解：**LLM 的 RL 是一个动作空间为词表、episode 有几千步、只有终局给分、$\gamma=1$ 的 MDP。** 它和机器人 RL 的差别，几乎都能从这句话推出来。

**第一，奖励从"每步都有"变成"只在结尾"。** 机器人 PPO 里 advantage 用 GAE 逐步算：

$$
A_t=\sum_{k\ge 0}(\gamma\lambda)^k\,\delta_{t+k},
\qquad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

因为每一步都有 $r_t$。LLM 这边中间步骤全是 $r_t=0$、$\gamma=1$，整条轨迹的回报是个常数，逐 token 的 advantage 只能整条广播。信用分配在机器人 RL 里是次要问题——有逐步奖励，GAE 就够；在这里它变成主要矛盾。这也是为什么有人去做过程奖励模型（PRM），本质上是想把"逐步奖励"补回来。

**第二，baseline 换了来源，但目的没变。** 机器人 PPO 标配一个 value function；GRPO 把它换成"同一 prompt 的 $G$ 条回答的均值"。这在机器人里也有对应：**同一初始状态并行跑 N 个 env，用组内平均 return 当 baseline**。

**第三，数据基本只能用一次。** 机器人有模拟器，SAC 可以从 replay buffer 里反复抽；LLM 的"环境"就是策略自己，权重一更新，旧数据立刻不再 on-policy，所以基本是采一遍、训一遍。这也是 SAC、TD3 这类 off-policy 方法在这边少见的原因之一：critic 要估"一段文本前缀值多少分"，既难训，也没有可重放的廉价数据。

还有两处不能照搬：

- **探索的含义不同。** 机器人靠动作噪声和熵正则探索；LLM 的温度改的是 token 分布的平坦程度，调高温度不等于探索到新的解题策略。
- **动作的局部性没了。** 连续动作空间里相邻动作结果相近，这是这类方法能 work 的前提之一；token 之间没有这种几何结构，"答错一个数字"和"答错一整段"在动作空间里的距离是一样的。

## 三、baseline：从 GAE 到组内均值

上一节的第二条值得单独展开，因为我第一遍看 PPO 觉得"都懂"、看到 GRPO 却卡住，卡的就是这里。

在机器人 PPO 里，advantage 依赖一个学出来的价值函数；把设定换成 $\gamma=1$、只有终局奖励，所有 token 的 return 相同，问题就退化成"只需要一个 baseline"。而这个 baseline 有更便宜的估法。


策略梯度是

$$
\nabla_\theta J = \mathbb{E}\left[R\,\nabla_\theta \log \pi_\theta\right].
$$

把 $R$ 换成 $R-b$，只要 $b$ 不依赖采样到的动作，期望就不变，方差却可以变小。理想的 $b$ 是当前状态下的期望回报 $\mathbb{E}[R\mid s]$——PPO 学一个价值网络 $V_\psi$ 去估它，这也是 PPO 里唯一的额外模型。

GRPO 的改动只有一句话：**对同一道题采一组回答，用组内均值当 baseline。** 同一道题就是同一个状态，组内均值恰好是 $\mathbb{E}[R\mid s]$ 的无偏估计，于是 critic 整个消失了。

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)+\delta}.
$$

为什么必须是"同一道题"？因为跨题的平均奖励本来就不可比。一组回答互为参照，实际上就是在同工况下做重复实验。

两个容易想的点：**reward 全为正，advantage 仍然有正负**（比平均差就是负的）；**组内方差为零时 advantage 全零**，这一组不产生梯度。


**组内没有区分度时，这个 baseline 就失效了。** 全对或全错时 $r_i$ 全相同，advantage 全零，这一组白采，日志里记作 `frac_degenerate`。机器人在课程学习里撞的是同一堵墙：任务难度和当前策略不匹配时，成功和失败没有分化，梯度也就是零，只能去调难度。区别是那边能直接改任务，这边只能改题目分布或者 `group_size`。


## 四、IS / PPO / CISPO：三个 loss 的推导

`loss-functions` 那一章卡的地方很具体：**三个公式长得几乎一样，凭什么算三个算法？** IS 和 PPO 只差一个 `min` 加 `clip`，CISPO 只是多一个 `detach`。如果只是实现细节，为什么要单独写一章。

后来发现，把 loss 对 $\theta$ 求一次梯度，差别就全露出来了：三者最后是同一个形状，不同的只有**每个 token 被推动的力度** $w(\rho)$。下面把这层剥开。

### 4.1 记号与前提

设策略为 $\pi_\theta$，一条回答的第 $i$ 个样本的第 $t$ 个 token 记作 $o_{i,t}$。采样用的旧策略记作 $\pi_{\text{old}}$，冻结不动；概率比定义为

$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid s_{i,t})}{\pi_{\text{old}}(o_{i,t}\mid s_{i,t})}.
$$

优势 $A_i$ 在 GRPO 里是组内标准化后的相对优势，在 PPO 里是 GAE 优势。**这里的关键前提是 $A_i$ 对 $\theta$ 是常数**（$\nabla_\theta A_i=0$）：它在 rollout 阶段算好、训练时冻结。正因如此，三种 loss 才可能塌缩成同一个形状；若 advantage 自己也依赖 $\theta$（在线 critic 与策略联合训练），推导要多出一项 $\mathbb{E}[w(\rho)\nabla_\theta A_i]$，统一形式就不成立了。

再记 $r_{i,t}=\pi_\theta(o_{i,t}\mid s_{i,t})$，由 $\nabla\log\pi=\nabla\pi/\pi$：

$$
\nabla_\theta\rho_{i,t}=\frac{\nabla_\theta r_{i,t}}{\pi_{\text{old}}(o_{i,t}\mid s_{i,t})}=\rho_{i,t}\,\nabla_\theta\log\pi_\theta(o_{i,t}\mid s_{i,t}).
$$

**对 $\rho$ 求梯度，等于 $\rho$ 乘以 score function。** 这是整节推导的发动机。

### 4.2 IS：$w(\rho)=\rho$

IS 直接用概率比修正 off-policy 的期望：

$$
\mathcal{L}_{\text{IS}}(\theta)=-\,\mathbb{E}\left[\rho_{i,t}\,A_i\right].
$$

负号来自"最小化 loss = 最大化目标"。求梯度：

$$
\nabla_\theta\mathcal{L}_{\text{IS}}=-\,\mathbb{E}\left[A_i\,\nabla_\theta\rho_{i,t}\right],
\qquad
w_{\text{IS}}(\rho)=\rho .
$$

$\rho$ 无上限。$\rho=1.8$ 时，该 token 的梯度被放大 1.8 倍——控制里这就是只有比例项、没有限幅，偏差大时推力也大，容易一步把概率推飞。

### 4.3 PPO：平坦区的位置由 $A$ 的符号决定

记 $l=1-\epsilon$、$u=1+\epsilon$。单样本的 PPO 代理目标是

$$
L(\theta)=\min\left(\rho A,\ \operatorname{clip}(\rho,l,u)A\right).
$$

固定 $A$ 之后可以直接化成两行：

$$L(\theta)= \begin{cases} A\min(\rho,u), & A\gt 0,\\ A\max(\rho,l), & A\lt 0. \end{cases}$$

第二行的 `max` 来自乘负数会翻转不等号。这一步就能把所有分支检验完。沿用记号 $\nabla_\theta L=A\,w(\rho)\nabla_\theta\log\pi_\theta$，配合 $\nabla_\theta\rho=\rho\nabla_\theta\log\pi_\theta$：

| 优势 | $\rho\lt l$ | $l\lt \rho\lt u$ | $\rho\gt u$ |
| --- | --- | --- | --- |
| $A\gt 0$ | $w=\rho$ | $w=\rho$ | $w=0$ |
| $A\lt 0$ | $w=0$ | $w=\rho$ | $w=\rho$ |

写成紧凑形式：

$$w_{\text{PPO}}(\rho)= \begin{cases} 0, & A\gt 0,\ \rho\gt u,\\ 0, & A\lt 0,\ \rho\lt l,\\ \rho, & \text{其余（含相反方向越界）}. \end{cases}$$

**解读：**

- **方向由 $A$ 的符号决定。** $A\gt 0$ 时目标鼓励提高已采样动作的概率，$\rho\gt u$ 表示提得过头，平坦区在**上侧**；$A\lt 0$ 时目标鼓励降低该动作的概率，$\rho\lt l$ 表示压得过头，平坦区在**下侧**。
- 向**相反方向**越界（$A\gt 0$ 且 $\rho\lt l$，或 $A\lt 0$ 且 $\rho\gt u$）时，目标取未裁剪项，梯度保留，用来纠正这个变化。这是"单向保守"的另一半。
- 两侧用同一个 $\epsilon$，所以结构关于"有利 / 不利方向"对称，但关于 $\rho$ 的数值轴不对称——平坦区位置随 $A$ 的符号翻转。
- "停手"只指**该样本对 PPO 代理目标的直接梯度为零**。其他样本仍会通过共享参数继续改变这个动作的概率，clipping 并没有把实际概率比锁在区间里。

### 4.4 CISPO：$w(\rho)=\operatorname{clip}(\rho)$，且 detach

CISPO 把裁剪后的概率比只当系数、不回传梯度：

$$
\mathcal{L}_{\text{CISPO}}(\theta)=-\,\mathbb{E}\left[\operatorname{sg}\big(\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\big)\,A_i\,\log\pi_\theta(o_{i,t})\right].
$$

$\operatorname{sg}(\cdot)$ 是 stop-gradient（即 `detach`）。求梯度时它视为常数，只有 $\log\pi_\theta$ 回传：

$$
\nabla_\theta\mathcal{L}_{\text{CISPO}}=-\,\mathbb{E}\left[\operatorname{sg}\big(\operatorname{clip}(\rho_{i,t})\big)\,A_i\,\nabla_\theta\log\pi_\theta(o_{i,t})\right],
$$

于是

$$
w_{\text{CISPO}}(\rho)=\operatorname{clip}(\rho,1-\epsilon,1+\epsilon).
$$

**关键对比**：如果**不** detach，$\nabla_\theta\operatorname{clip}(\rho)$ 在区间内是 $\rho\nabla\log\pi$、区间外是 0，结果退化成 PPO 的"越界归零"。detach 把系数变成只读的力度标量：

- 有上限：$\rho\gt u$ 时力度封顶在 $u$，不会像 IS 那样无限放大；
- **永不为零**：$\rho$ 越界时系数仍是 $l$ 或 $u$（都大于 0），梯度照常存在，只是不再增大。

这就是"推力封顶，而不是停手"，正好补上 PPO 平坦区的"剪没"缺陷。

### 4.5 三者统一

$$
\nabla_\theta\mathcal{L}=-\,\mathbb{E}\big[w(\rho_{i,t})\,A_i\,\nabla_\theta\log\pi_\theta(o_{i,t})\big]
$$

| loss | $w(\rho)$ | 上限 | 有利侧越界 | 相反侧越界 |
| --- | --- | --- | --- | --- |
| IS | $\rho$ | 无 | 继续放大 | 继续放大 |
| PPO | 区间内 $\rho$，有利侧越界取 0 | 有（靠归零实现） | 停手（梯度为 0） | 保留 $\rho$，继续纠正 |
| CISPO | $\operatorname{clip}(\rho)$（detach） | 有（$u$／$l$） | 封顶但继续推 | 封顶但继续推 |

三者**唯一的实质差别就是 $w(\rho)$ 这个标量函数**；loss 写法、`min`、`detach` 都只是实现这个 $w$ 的手段。这就是"三个公式长得几乎一样、却算三个算法"的答案：算法差异被压缩进了一个一维的力度函数。

### 4.6 顺带澄清：GRPO 不在这个维度上

- **GRPO** 规定的是 $A_i$ 怎么来：同一 prompt 采一组回答，用组内均值 / 标准差做相对优势（$A_i=\frac{R_i-\operatorname{mean}(R_{1:G})}{\operatorname{std}(R_{1:G})}$）。它**不规定** $A_i$ 怎么变成梯度。
- **PPO / CISPO / IS** 规定的是 $A_i$ 怎么变成梯度，即 $w(\rho)$。

所以"GRPO + PPO 的 clip"和"GRPO + CISPO 的 detach-clip"都是合法组合：前者是 GRPO 提供优势、PPO 提供力度函数；后者是同一优势、换一个力度函数。把 GRPO 和 PPO/CISPO 并列成"三个算法"会混淆两个正交的维度——**优势构造** vs **力度函数**。

## 五、GRPO 的目标函数

上一节把力度函数拆完了，这一节把 GRPO 的目标函数摊开。训练时最大化它（代码里最小化 $-J$）：

$$
J(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\left(\min\left(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)
-\beta\, D_{\mathrm{KL}}^{(i,t)}\right)\right]
$$

它的策略部分（括号里 `min` 那一项）就是 4.3 的 PPO 写法，直接套用：**组内相对优势 + PPO 的权重**。多出来的只有两样东西：KL 项，和两层平均。

读法五句：

1. 旧策略采 $G$ 条，采样结果和旧 logprobs 冻结；
2. 每条拿一个组内优势，广播到它所有 token；
3. `min` 与 `clip` 限制更新力度——$w$ 的形状 4.3 已经推完，这里只是照用；
4. KL 项拉住当前策略，别离参考策略太远；
5. 先按 token 平均，再按回答等权平均——长短回答不该因为长度获得不同话语权。

**一个必须分清的地方：$\pi_{\text{old}}$ 和 $\pi_{\text{ref}}$ 不是同一个东西。** 前者是本轮采样的快照，做概率比的分母；后者通常是初始 SFT 模型，做 KL 的锚点。训练开始时两者可能一模一样，之后更新节奏完全不同。搞混了，KL 曲线会给出很奇怪的读数。

到这里可以看到，GRPO 的新意全部集中在 advantage 上：权重函数整段沿用 PPO，换的只是优势怎么估。它也不是免费的——省掉 critic，代价是每道题多采 $G-1$ 条。

## 六、十步实测：三个读数和一个巧合

跑的是仓库默认 demo：`--steps 10 --batch-size 4 --group-size 8 --max-tokens 512 --loss-fn importance_sampling`，Qwen3.5-4B + GSM8K，记录在 SwanLab 上。

训练记录：[SwanLab · agentic-rl-lab-grpo](https://swanlab.cn/@jokerxuxu/agentic-rl-lab-grpo/v1/liwxpw/runs/2gmiydii/chart)。截图里的曲线和下面这张表都来自同一次 run。

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

![十步的组内平均 reward，没有趋势](/images/agentic-rl/grpo-demo-reward.png)

**第一，reward 基本没动。** 十步在 0.75–1.0 之间来回，没有趋势。原因不难猜：GSM8K 对 Qwen3.5-4B 太简单，第 0 步就已经全对。**十步看不出学习**，这是预期内的，不是训练失败。

![一半以上的步里，整组题都白采](/images/agentic-rl/grpo-demo-degenerate.png)

**第二，一个只有对着数据才能发现的巧合：degenerate 100% 的三步（0、3、6）没有 loss 记录。** 因为整批题都白采，代码直接跳过，那一步连一个 datum 都没产生。这不是我从曲线看出来的，是把 CSV 的缺失值和 `frac_degenerate = 1` 对出来的。它把第三节那句"这一组就白采"变成了可验证的事实：**日志里少一行 loss，不是漏记，是那一步真的什么都没学。**

![loss 在正负之间跳，均值本身没有绝对意义](/images/agentic-rl/grpo-demo-loss.png)

**第三，长度波动大，且不能当结论。** 187 → 381 → 237，第 2 步最长，而它恰好是唯一 degenerate 为 0 的一步。十步的样本量太小，既不能说"变长=变聪明"，也不能说"变长=刷长度"。

![平均生成长度，单步波动比趋势更明显](/images/agentic-rl/grpo-demo-genlen.png)



## 七、还没想清楚的

- **reward 设计比算法更影响结果**，这句话我现在只是相信，还没有自己的证据。
- 有了解到在自动驾驶领域也有人用 GRPO 做后训练，可能在完成task后做一个概念上的调研。
- 接下来会进一步学习 GRPO。

> 系列索引：[Agentic RL Lab 实战笔记](/zh/series/agentic-rl-lab/)。

## 参考

- 课程仓库：[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ Task01 教程：[environment & concepts](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/)
- DeepSeekMath（GRPO 出处，§4.1）：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PPO（Clipped Surrogate Objective）：[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- DeepSeek-R1（规则奖励与可验证任务上的 GRPO）：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
