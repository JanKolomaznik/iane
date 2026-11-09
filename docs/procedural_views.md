# Procedural views

Include `<iane/view/procedural.hpp>` and `<iane/execution/run_on.hpp>`.

A procedural view computes a value when indexed. Use a view as the domain
for a coordinate generator or a constant, then compose equal-sized views
with pointwise callables.

```cpp
iane::run_on(options, input.const_view(), output.view(),
	[](auto aSource, auto aDestination, auto aExecution) {
		auto doubled = iane::map_view(aSource, double_pixel{});
		auto bias = iane::generate_view(aSource, column_index{});
		auto baseline = iane::constant_view(aSource, 1);
		auto offset = iane::zip_view(bias, baseline, add_pixels{});
		return iane::copy(iane::zip_view(doubled, offset, add_pixels{}),
			aDestination, aExecution);
	});
```

Here `double_pixel` returns twice its input, `column_index` returns the first
coordinate, and `add_pixels` adds two values. With those callables, the output
at each coordinate is twice the input plus its column index plus one.
For CUDA, make the callable operators device-callable with
`IANE_DECL_HYBRID`. The backend follows the domain or input views, and zip
requires both operands to use the same backend and coordinate system.
A shape mismatch throws when creating the zip.

The factories copy view handles and callables by value. The coordinate and
constant generators only retain the domain's dimensions. Map and zip still
borrow the storage of their operands, which must stay valid through evaluation
and CUDA completion. Index only within the view's extents. Materializing with
`copy` evaluates the expression in a single traversal without intermediate
image buffers. The aliasing rules of `copy` apply to each operand.

## Arithmetic

Include `<iane/view/algebra.hpp>` for pointwise `+`, `-`, `*` and `/`.
Each operator accepts two readable views, or a readable view and an
arithmetic scalar in either order. Unary `-` negates each pixel.

```cpp
auto expression = source * 2 + 1;
iane::copy(expression, destination).wait();
```

An input of `{2, 3, 5}` produces `{5, 7, 11}`. Two view operands must have
matching extents and the same backend. Scalars are copied into the expression.
Pixel arithmetic uses the corresponding C++ operator, including its
promotions, division rules and valid operand ranges. The expression borrows
its input storage and follows the same lifetime and aliasing rules as zip.

The same header provides `iane::log`, `log1p`, `log2`, `log10`, `exp`,
`exp2`, `sqrt`, `sin`, `cos` and `pow` as lazy pointwise functions.
`pow` accepts two views or broadcasts a scalar on either side. The
corresponding standard-library scalar functions determine result types and
domain behavior. For example, `iane::log1p(iane::pow(source, 2.0f))`
evaluates each pixel as `log(1 + pixel²)`.
