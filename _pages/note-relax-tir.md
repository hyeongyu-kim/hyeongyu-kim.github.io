---
layout: editorial_note
permalink: /notes/relax-to-tir/
title: From a tensor graph to a tensor program
description: Follow a matrix multiply from graph-level meaning to loops, storage, and execution.
topic: Compiler notes
reading_time: 4
updated: 2026-10-08
research_note: true
note_order: 3
---

At the graph level, a matrix multiply says what result is needed. At the tensor-program level, its computation and memory accesses are explicit. This distinction helps explain the roles of Relax and TIR in TVM.

## Relax: relationships between tensor operations

Relax represents a model as operations on tensors, with shape and type information and relationships between expressions. At this level, a transformation can reason about a sequence of operations or decide whether operations should be grouped.

A simple example is a matrix multiply followed by an elementwise activation. The graph describes that dependency without committing to one tile size or one hardware instruction sequence.

## Legalization: making an operation concrete

TVM's `LegalizeOps` transformation can rewrite high-level operations into calls to tensor programs. A Relax `call_tir` connects a graph expression to a TIR function. This is a boundary between graph-level composition and an explicit computation.

The exact route depends on the operator, target, and compiler pipeline. Some operations may call an external implementation rather than become a newly generated tensor program.

## TIR: loops and buffers

An explicit matrix multiply has loops over output rows, output columns, and a reduction dimension. Its buffer accesses tell you which A and B elements contribute to each C element.

```python
# Mathematical loop sketch, not TVMScript syntax
for i in range(M):
    for j in range(N):
        C[i, j] = 0
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

The sketch exposes work that was hidden inside a graph operator. It still leaves many execution choices open.

## Scheduling: choosing an execution structure

Splitting loops creates outer tile loops and inner loops. Reordering them changes traversal. Introducing local buffers changes data movement. Tensorization can map a suitable computation pattern to a target-specific implementation.

For example, holding a C tile resident across K chunks gives the output-stationary structure shown in the [tiling note]({{ '/notes/matmul-tiling/' | relative_url }}). The arithmetic meaning remains a matrix multiply, while its working set and reuse are explicit.

## Beyond the tensor program

A backend still has to map the program to the target's available operations and execution model. Instruction selection, register or local-memory constraints, and binary emission belong to later implementation stages. A custom accelerator can make different choices from a CPU or GPU.

Reading a compiler pipeline becomes easier when each representation is tied to a concrete question: What computation is required? How is it organized? Where does the data live? Which target operations execute it?

## Reading

- [Relax in Apache TVM](https://tvm.apache.org/docs/deep_dive/relax/index.html)
- [Relax transformation tutorial](https://tvm.apache.org/docs/deep_dive/relax/tutorials/relax_transformation.html)
- [Tensor program abstraction in Apache TVM](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
