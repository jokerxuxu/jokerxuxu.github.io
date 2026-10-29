---
title: "Tendon Routing: Statics, Friction, and Moment Arms"
date: 2026-08-10
summary: "Working notes on how a cable's path through a hand changes the force a joint actually sees."
tags: ["tendon-driven", "mechanics", "modeling"]
hidemeta: true
ShowToc: false
ShowPostNavLinks: false
---

## Why this note exists

A tendon is not a rigid link. It wraps around pulleys and guides, its tension drops across every contact, and the moment arm it produces is a function of joint angle — not a constant. Most quick models quietly assume otherwise, and then the controller spends its life compensating for the assumption.

These are my working notes on the static side of that problem.

## The pieces I keep coming back to

- **Moment arm as a function of configuration.** For a cable crossing joint $i$, the torque is $r_i(q)\,T$, where $r_i$ changes as the cable wraps.
- **Capstan friction.** Tension across a wrap follows $T_{\text{out}} = T_{\text{in}}\,e^{-\mu\theta}$; a hand with several guides can lose a large fraction of the commanded force before it reaches the fingertip.
- **Routing as a design variable.** Where the cable enters and how many guides it crosses changes both the moment arm profile and the friction loss — the two cannot be tuned independently.

<!-- TODO: turn this into a full derivation with diagrams once the notes are cleaned up. -->

## Open questions

- When is a constant-moment-arm approximation acceptable for grasp force control?
- How much hysteresis is needed before a rate-independent friction model stops being useful?

## Status

Notes in progress; no polished write-up yet.
