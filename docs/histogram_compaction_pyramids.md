# Histogram, compaction and pyramids

## Histogram and equalization

Include `<iane/algorithms/histogram.hpp>` and `<iane/algorithms/scan.hpp>`.
Here `to_bin` is the bin mapper and `add` sums two unsigned counts.

```cpp
iane::histogram(pixels, bins, to_bin, policy);
auto exec = iane::make_execution_context(policy);
auto scan = exec.prepare(iane::inclusive_scan(add, 0u), bins, cdf);
exec.run(scan, bins, cdf);
iane::equalization_lut(cdf, lut, policy);
iane::copy(iane::lookup_view(pixels, lut, to_bin), equalized, policy).wait();
```

Bins are unsigned int counters. Each histogram call replaces their contents.
The mapper returns an integral bin index. Negative and out-of-range indices
are ignored. Histogram updates use atomic counters on both backends. The
output must provide distinct, naturally aligned counters, and the source
element count must fit in an unsigned int.

The equalization table contains normalized float values. It maps the first
occupied bin to zero and the last occupied bin to one using the cumulative
counts. An empty or single-bin histogram produces a normalized identity ramp.
The CDF must be an inclusive, nondecreasing cumulative count. A lookup view
borrows both inputs. Every index used to evaluate it must be in range.
Use one stream for the entire pipeline and keep borrowed buffers alive
through completion.

## Stable coordinate compaction

Include `<iane/algorithms/compact.hpp>`. Here `is_selected` is the pixel
predicate.

```cpp
auto output = iane::compact_output{coordinates, count};
auto exec = iane::make_execution_context(policy);
auto select = exec.prepare(iane::compact(is_selected), mask, output);
exec.run(select, mask, output).wait();
```

The coordinate buffer is a writable 1D view whose element type is the source
coordinate type. The count is a writable one-element 1D view of
`std::size_t` on the same backend. Coordinates appear in logical order with
axis zero varying fastest. If capacity is short, the output contains the first
matches that fit and the count still reports all matches. Unused entries stay
untouched. The pure predicate may be evaluated twice. The context reserves and
reuses one temporary count per chunk. The source and both outputs must be
disjoint and writable coordinates must refer to distinct storage.

## Resolution levels

Include `<iane/image/pyramid.hpp>`. In this example the first step
reduces both axes and the second reduces only axis zero. The caller's
`downsample` operation determines how each step samples and combines
pixels. For example, the same level extents can be used with a Gaussian
filter or a neighborhood maximum before sampling.

```cpp
using image_t = iane::host_image<float, 2>;
using extent_t = iane::math::vec2i;
const extent_t extents[]{extent_t{9, 7}, extent_t{5, 4}, extent_t{3, 4}};
iane::image_pyramid<image_t> pyramid(extents);
auto levels = pyramid.view();
iane::copy(input, levels.level(0));
for (std::size_t i = 1; i < levels.level_count(); ++i) {
	downsample(levels.level(i - 1), levels.level(i));
}
```

Level zero conventionally denotes the finest resolution. Extents are explicit
and the same storage can be used with different construction rules. The
descriptor table is on the host. Pass individual level views to algorithms,
including CUDA algorithms. An owning pyramid allocates each level through
its image type. A `pyramid_view` can also borrow a caller-provided table of
views, including strided or externally owned levels. Keep the table and its
images alive while borrowing the pyramid. Keep any level storage alive through
completion of operations on it.
