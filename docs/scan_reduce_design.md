# Prepared scan and reduce

Include `<iane/algorithms/scan.hpp>` and `<iane/algorithms/reduce.hpp>`.
The example uses the same API with CPU and CUDA execution contexts.

```cpp
auto exec = iane::make_execution_context(policy);

// Input pixels and both uint64_t outputs have the same 2D extents.
auto rows = exec.prepare(
    iane::inclusive_scan_along<0>(plus_u64, std::uint64_t{0}), pixels, row_sums);
auto columns = exec.prepare(
    iane::inclusive_scan_along<1>(plus_u64, std::uint64_t{0}), row_sums, integral);

exec.run(rows, pixels, row_sums);
exec.run(columns, row_sums, integral).wait();
```

Here `plus_u64` is a host/device callable taking two unsigned 64-bit values.
The result at `(x, y)` sums the input rectangle from `(0, 0)` through
`(x, y)`, inclusive. For input rows `[1, 2, 3]` and `[4, 5, 6]`,
the output rows are `[1, 3, 6]` and `[5, 12, 21]`.

## Operations

| Factory | Result |
| --- | --- |
| `inclusive_scan(op, init)`, `exclusive_scan(op, init)` | One prefix sequence over the whole view, with dimension 0 varying fastest. Output has the input shape. |
| `inclusive_scan_along<axis>(op, init)`, `exclusive_scan_along<axis>(op, init)` | One independent prefix sequence per line along the compile-time axis. Output has the input shape. |
| `reduce(op, init)` | One value in a writable 1D view of extent 1. |
| `reduce_along<axis>(op, init)` | One value per line, with the reduced axis removed. A 1D input uses the same extent-1 output as full reduction. |

Each sequence starts with `init`, applied once. Inclusive scan includes the
current element and exclusive scan stops before it. Empty scans produce no
writes. An empty reduction produces `init` for each existing output coordinate.
For example, reducing axis 0 of a `0 × 5` view produces five copies of
`init`. The type of `init` defines the accumulator type. Convert each input
element to that type before combining and use the same type for the output.

Require a pure associative operation for scan. Preserve operand order while
allowing regrouping. Also require commutativity for reduce, so the backend can
reorder operands. Floating-point results may differ across backends or worker
counts. Support readable expressions and strided inputs and outputs through
logical coordinates. Require distinct writable output locations and disjoint
input and output storage. The accumulator must be trivially copyable with
fundamental alignment.

## Execution and scratch

Reuse `execution_context`. Infer compatibility from the source and destination
view types, including nested expression operands, and check the policy against
that backend during preparation. Reduction writes into a view on that backend.
A CUDA scalar can feed subsequent CUDA work on the same stream or be transferred
explicitly when the host needs it.

Prepare using prototype views to validate shape and determine scratch size.
A prepared operation can reuse its reservation with new views of the same types,
extents and supported layouts. Keep the operation configuration in the prepared
state. Keep temporary storage in the context and suballocate it on each run.
Two sequential operations share a reservation sized to the larger requirement.
The row sums in the example are an intermediate image passed explicitly between
operations. CUDA runs queue on the context stream, with the existing lifetime
and completion rules from [Prepared algorithms](prepared_algorithms.md).

The serial CPU path needs no algorithm scratch. With multiple workers, CPU
axis operations assign whole lines when enough lines are available. Otherwise
they split long lines into chunks using the CPU policy's grain size. Full-view
operations use the same chunking with a single logical line. CUDA submits
independent chunks of up to 128 elements.

When a line has multiple chunks, the first pass computes chunk totals.
For reductions, a second pass combines them into one result per line. For
scans, the second pass computes the carry into each chunk, and a final pass
rereads source elements and writes prefixes starting with those carries.
Only the first carry starts with `init`. Scratch holds one accumulator per
chunk, reused for carries. The carry pass processes each line's chunk totals
in order. Single-chunk lines write their output directly without scratch.
