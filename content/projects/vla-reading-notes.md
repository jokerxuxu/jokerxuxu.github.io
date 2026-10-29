---
title: "Vision-Language-Action Models: Reading Notes"
date: 2026-07-15
summary: "Notes from reading VLA work, with an eye on what it assumes about the robot body underneath."
tags: ["embodied AI", "VLA", "manipulation"]
hidemeta: true
ShowToc: false
ShowPostNavLinks: false
---

## Why I'm reading this

VLA models are usually evaluated on arms with parallel grippers. A tendon-driven hand has compliance, coupling, and friction that a position-controlled gripper does not — and I want to know which of those the policy is implicitly relying on.

## What I'm collecting

- **Action spaces.** Joint positions, end-effector poses, or latent actions — each one hides a different assumption about the controller.
- **Data.** How much of the success is the policy, and how much is a well-tuned impedance controller underneath it?
- **Hardware-aware policies.** Work that conditions on hand morphology or tendon state instead of treating the hand as a black box.

<!-- TODO: add the specific papers you have read, with one-line takeaways. -->

## Status

Ongoing reading notes. I will publish short summaries as I finish each paper.
