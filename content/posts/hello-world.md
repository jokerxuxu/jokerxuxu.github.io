---
title: "Hello, world"
date: 2026-10-29
summary: "Why this site exists, what I plan to write about, and how it is built."
tags: ["meta", "hugo"]
math: true
ShowToc: true
---

## Why another personal site

I spend most of my time reading papers, breaking simulations, and fixing the same class of bug twice. Most of that never leaves my notes app. This site is an attempt to make the useful part public: short technical notes that I would have wanted to find six months ago.

It is also a place to keep a project record that is more honest than a CV line — what I tried, what failed, and what I would do differently.

## What you will find here

- **Tendon-driven hands and soft robotics** — routing, transmission, friction, and control.
- **Simulation notes** — MuJoCo modeling details that are easy to get wrong.
- **Paper summaries** — short, with the assumptions I think matter.
- **Occasional mechanics** — derivations I do not want to redo.

I will not write about unpublished results from my group, and I will not turn this into a diary. If a month has no post, that is fine.

## A formatting check

This post also doubles as a template check, so here is a small piece of the mechanics I keep coming back to. For a cable crossing a joint, the tendon length and joint torque are

$$
L(q) = L_0 + \sum_i r_i(q_i)\, q_i,
\qquad
\tau_i = r_i(q_i)\, T,
$$

where $T$ is the tendon tension and $r_i$ is the moment arm. The annoying part is that $r_i$ is not a constant: it changes with the wrapping geometry, and so does the friction loss across the guides.

And a code block, to confirm syntax highlighting and the copy button:

```python
def tendon_length(q, moment_arms, rest_length=0.0):
    """Tendon length for a serial chain, ignoring wrapping dynamics."""
    return rest_length + sum(r * qi for r, qi in zip(moment_arms, q))
```

If the equation above is typeset rather than shown as raw `$$ ... $$`, KaTeX is working.

## How the site is built

[Hugo](https://gohugo.io/) with the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, deployed to GitHub Pages by a GitHub Actions workflow. Content is plain Markdown, math is KaTeX, and the source is on [GitHub](https://github.com/jokerxuxu).

## What's next

A proper write-up of the tendon routing model — including the parts that did not work.
