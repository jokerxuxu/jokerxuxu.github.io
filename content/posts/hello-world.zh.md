---
title: "你好，世界"
date: 2026-10-29
summary: "为什么开这个站、准备写什么，以及它怎么搭的。"
tags: ["随笔", "hugo"]
math: true
ShowToc: true
---

## 为什么又开一个个人站

大部分时间我都在读论文、把仿真搞崩、然后第二次修同一类 bug。这些大多留在了笔记软件里。这个站想把其中有用的部分公开出来：那些我六个月前希望能搜到的短技术笔记。

它也是一个比简历上的一行字更诚实的项目记录——试过什么、什么失败了、下次会怎么做。

## 你会在这里看到

- **绳驱动的手与软体机器人**——走线、传动、摩擦与控制。
- **仿真笔记**——MuJoCo 里容易搞错的地方。
- **论文笔记**——短，但会写清我认为重要的假设。
- **偶尔的力学**——不想重推一遍的推导。

我不会写组里未公开的结果，也不会把它写成日记。某个月没有更新，也没关系。

## 一个排版检查

这篇也顺便当模板检查用，所以放一小段我最近反复看的力学。对跨过某个关节的绳索，肌腱长度和关节力矩是

$$
L(q) = L_0 + \sum_i r_i(q_i)\, q_i,
\qquad
\tau_i = r_i(q_i)\, T,
$$

其中 $T$ 是肌腱张力，$r_i$ 是力臂。麻烦的地方在于 $r_i$ 不是常数：它随包角几何变化，导槽上的摩擦损失也跟着变。

再来一段代码，确认高亮和复制按钮：

```python
def tendon_length(q, moment_arms, rest_length=0.0):
    """Tendon length for a serial chain, ignoring wrapping dynamics."""
    return rest_length + sum(r * qi for r, qi in zip(moment_arms, q))
```

如果上面的式子排版成了公式、而不是原样的 `$$ ... $$`，说明 KaTeX 工作正常。

## 这个站怎么搭的

[Hugo](https://gohugo.io/) + [PaperMod](https://github.com/adityatelange/hugo-PaperMod) 主题，用 GitHub Actions 部署到 GitHub Pages。内容是纯 Markdown，公式用 KaTeX，源码在 [GitHub](https://github.com/jokerxuxu)。

## 接下来

把肌腱走线的模型认真写一遍——包括那些没跑通的部分。
