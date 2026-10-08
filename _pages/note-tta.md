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

The [digit experiment]({{ '/lab/#digit-lab' | relative_url }}) uses a simpler mechanism: feature-statistic alignment. For each feature in the first hidden layer, it estimates the mean and standard deviation of an unlabeled target batch, then maps that feature toward the statistics measured on the source training set:

```text
aligned feature = (target feature − target mean) / target std
                  × source std + source mean
```

The classifier weights stay fixed. This is a small statistical adaptation experiment, with a mechanism related to the idea of matching normalization statistics across domains. It is separate from the methods evaluated in the papers.

## The information used by the update

Training labels were used to learn the source digit classifier. **Target labels are not passed to its adaptation function.** That function receives hidden features and source statistics. The held-out labels are used afterward to calculate accuracy and calibration error.

This separation matters. Choosing an update, its strength, or an early stopping point using target accuracy would make the experiment a different setting. An unlabeled adaptation rule should be identifiable without reading the evaluation labels.

## When adaptation hurts

Matching statistics can help with some appearance shifts. It can also confuse a change in class composition with a change in appearance. A batch containing mostly one digit may have different feature means for a legitimate reason. Small batches add another source of error: their estimates are noisy.

Try a clean batch at severity zero, then a shifted batch. Compare batch sizes, and check whether accuracy and ECE move together. The experiment recalculates its numbers; an improvement is not built into the interface.

## A batch, a stream, and a reset

The lab adapts each batch independently and resets its adapted state when the batch or shift changes. A continuously updated model is a different protocol: earlier inputs can affect later predictions, and a reset changes the history available to the model.

When describing a TTA result, specify the update state, the target information it receives, the batch or stream protocol, and the reset policy. Those details make the result interpretable.

## Reading

- [Revisiting Batch Normalization for Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)
- [AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](https://arxiv.org/abs/2603.26096)
- [Buffer layers for Test-Time Adaptation](https://arxiv.org/abs/2510.21271)
