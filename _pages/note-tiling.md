---
layout: editorial_note
permalink: /notes/matmul-tiling/
title: Reading a matrix multiply through its tiles
description: The arithmetic stays the same. The working set and data reuse depend on the loop structure.
topic: Efficient inference
reading_time: 5
updated: 2026-10-08
research_note: true
note_order: 2
---

For `C = A @ B`, let A have shape M × K and B have shape K × N. The output has shape M × N. Counting a multiply and an addition as two operations gives `2 × M × N × K` arithmetic operations.

Tiling reorganizes that work. It chooses how much of A, B, and C is active at a time, and how often those pieces need to move.

## Keep one output tile resident

Consider an output-stationary loop. Pick a tile of C, initialize its accumulator, and walk through K in chunks. Each chunk reads a tile from A and B. The accumulator stays resident until the reduction is complete, then is written once.

```python
for mo in range(0, M, tile_m):
    for no in range(0, N, tile_n):
        accumulator = zeros_for_this_output_tile()
        for ko in range(0, K, tile_k):
            accumulator += A_tile(mo, ko) @ B_tile(ko, no)
        write_C_tile(mo, no, accumulator)
```

In the [tile experiment]({{ '/lab/#tile-lab' | relative_url }}), the highlighted C tile remains in place while the A and B tiles move along K. That motion is the loop order made visible.

## What fits in local storage?

With FP16 inputs and FP32 accumulation, one full tile operation needs:

| Resident data |                 Bytes |
| ------------- | --------------------: |
| A tile        | `2 × tile_m × tile_k` |
| B tile        | `2 × tile_k × tile_n` |
| C accumulator | `4 × tile_m × tile_n` |

For a 16 × 16 output tile and a K tile of 32, these are 1 KiB, 1 KiB, and 1 KiB: a **3 KiB working set**. Double buffering, alignment, padding, and scratch data would add to this number; this small model excludes them.

## Count the repeated reads

Assume no reuse of input tiles across different output tiles. A is read once for each output-column tile, while B is read once for each output-row tile:

```text
A elements read = M × K × ceil(N / tile_n)
B elements read = K × N × ceil(M / tile_m)
C elements written = M × N
```

For M = N = K = 128 and tiles 16 × 16 × 32, the model counts **576 KiB of traffic**: 256 KiB each for A and B, plus 64 KiB for C. The arithmetic count is 4,194,304 operations.

Increasing tile_m or tile_n can reduce repeated input reads, while increasing the resident footprint. Changing only tile_k changes the footprint and the number of reduction steps; it does not change the total traffic under this particular reuse assumption.

## Edge tiles still matter

A dimension of 96 with tile size 64 has one full tile and one partial tile. Count actual elements for the partial tile. Multiplying the full tile size by the number of tiles would overestimate both storage or traffic in some calculations.

The lab counts edge ranges explicitly. It reports the largest resident tile and sums the modeled traffic across the actual matrix dimensions.

## A memory count is not a latency estimate

These counts describe the loop and its reuse policy. Predicting hardware performance also requires bandwidth, execution resources, instruction scheduling, layout, alignment, and memory-bank behavior. A better traffic count is useful evidence, but does not establish which schedule will run fastest.

## Reading

- [Tensor program abstraction in Apache TVM](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- [Try the interactive tile experiment]({{ '/lab/#tile-lab' | relative_url }})
