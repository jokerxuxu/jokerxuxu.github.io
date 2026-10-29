---
title: "A Tendon-Driven Biomimetic Hand in MuJoCo"
date: 2026-09-01
summary: "Simulation-first study of a tendon-driven anthropomorphic hand: routing, transmission, and joint-space control."
tags: ["MuJoCo", "tendon-driven", "dexterous manipulation"]
hidemeta: true
ShowToc: false
ShowPostNavLinks: false
---

## Why

Tendon-driven hands move the actuators out of the hand and into the forearm, then route force through cables. That buys compactness and low distal inertia, and it costs you: tendon length depends on every joint the cable crosses, friction and wrapping are hard to model, and the joints are coupled even when the mechanism looks simple.

I wanted a simulation model where I could change the routing, break the transmission, and see what happens — before committing to hardware.

## What I built

- A parametric hand model in **MuJoCo**, with tendon actuators whose length is computed from joint coordinates rather than hard-coded.
- A **joint-space outer loop** that tracks reference angles, plus the low-level tendon force allocation underneath it.
- Scripts to sweep routing parameters (moment arms, pretension, friction) and log tracking error.

<!-- TODO: link the repository and add 1–2 concrete results, e.g. tracking RMSE on a grasp trajectory. -->

## What I'm figuring out

- How much tracking error comes from **friction and wrapping** versus model error in the moment arms.
- Whether a simple Coulomb-plus-viscous friction model is enough for pick-and-place, or whether hysteresis has to be modeled explicitly.
- How to validate the simulated transmission against a real tendon-driven finger, not just against itself.

## Status

In progress. Code and results will be linked here once they are presentable.
