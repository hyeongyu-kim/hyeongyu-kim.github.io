---
layout: editorial_note
permalink: /notes/test-time-adaptation/
title: What actually changes at test time?
description: Separate the update rule, the state that changes, and the measurements used to judge it.
topic: Test-time adaptation
reading_time: 4
updated: 2026-10-08
research_note: true
note_order: 1
---

A classifier can work well on its training distribution and fail when brightness, noise, acquisition settings, or the input population changes. Test-time adaptation uses information in the incoming inputs to change the model's behavior. The difficult part is deciding which changes are justified without target labels.

It helps to separate three things: **what is updated, what drives the update, and how its consequences are measured**.

## The state that changes

Adaptation does not have to update every weight. It can change normalization statistics, a small set of affine parameters, activation parameters, or additional adaptation modules.

In [AcTTA](https://arxiv.org/abs/2603.26096), the adaptable part is the activation function: its response can change while the pretrained network weights remain fixed. In [Buffer layers](https://arxiv.org/abs/2510.21271), small inserted modules provide adaptation capacity around a frozen backbone. These choices determine both the cost of an update and what the model can change.

The [comparison lab]({{ '/lab/#tta-lab' | relative_url }}) applies Tent, Buffer, and AcTTA to the same inputs and source checkpoint. They share an entropy objective but change different parts of the model:

| Method | What learns                    | What to inspect                          |
| ------ | ------------------------------ | ---------------------------------------- |
| Tent   | BN scale and shift             | Changes to γ and β                       |
| Buffer | Added residual branches        | A signed spatial correction              |
| AcTTA  | Activation center and response | The activation curve and its local slope |

The lab uses recorded PyTorch updates on CIFAR-10-C. Convolutions in the pretrained backbone stay frozen. All three adaptive methods use current-batch BN statistics even before their first learned update, so their step-zero differences from Source should not be attributed to optimization.

## The information used by the update

Training labels were used to learn the source classifier. **Target labels are not passed to the adaptation function.** The lab minimizes mean prediction entropy on the current unlabeled batch. Labels are used afterward to calculate accuracy and calibration error.

This separation matters. Choosing an update, its strength, or an early stopping point using target accuracy would make the experiment a different setting. An unlabeled adaptation rule should be identifiable without reading the evaluation labels.

## When adaptation hurts

An entropy objective encourages confident predictions. A confident mistake can also have low entropy, so a decrease in the objective is not itself evidence that adaptation has helped. Small batches make both normalization statistics and the displayed evaluation estimates noisy.

Try Appearance shift, then Small batch. Change the corruption, severity, or update count, and inspect the same image and channel in all four methods. Compare accuracy, ECE, and entropy. All recorded outcomes are retained; an improvement is not built into the interface.

## A batch, a stream, and a reset

One-batch mode repeatedly updates the same batch. Its scores measure the adapted inputs, not generalization to unseen data. Stream mode moves through clean, noise, blur, and clean inputs again. Select Return to clean, then compare carrying adapted state with resetting before every incoming batch.

The stream history records predictions **before** each batch's update. The table and internal state describe that batch **after** its update. These are different measurements. Reset restores both model parameters and optimizer state.

When describing a TTA result, specify the update state, the target information it receives, the batch or stream protocol, and the reset policy. Those details make the result interpretable.

## Reading

- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c)
- [Revisiting Batch Normalization for Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)
- [AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](https://arxiv.org/abs/2603.26096)
- [Buffer layers for Test-Time Adaptation](https://arxiv.org/abs/2510.21271)
