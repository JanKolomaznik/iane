#pragma once

#include <iane/algorithms/convolution.hpp>
#include <iane/execution/run_on.hpp>
#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>

#include <boost/test/unit_test.hpp>

namespace meta_tests {

// A non-scalar element with componentwise arithmetic and no numeric conversions.
struct pixel {
	int first{};
	int second{};

	IANE_DECL_HYBRID pixel operator+(pixel aOther) const
	{
		return {first + aOther.first, second + aOther.second};
	}

	IANE_DECL_HYBRID pixel operator*(int aFactor) const
	{
		return {first * aFactor, second * aFactor};
	}
};

static_assert(iane::concepts::ConvolutionElements<pixel, int, pixel>);
static_assert(!iane::concepts::ConvolutionElements<pixel, pixel, pixel>);

// Generate a different value for every coordinate in a 3-wide test image.
template<class TPosition>
IANE_DECL_HYBRID pixel pattern(const TPosition& aPosition)
{
	int value = 1;
	int stride = 1;
	for (int i = 0; i < TPosition::dim; ++i) {
		value += aPosition[i] * stride;
		stride *= 3;
	}
	return {value, -value};
}

// A read-only generated view with the same backend as the view under test.
template<class TView>
struct generated_view {
	using value_type = pixel;
	using coord_t = typename TView::coord_t;
	using extents_t = typename TView::extents_t;
	static constexpr int dimension = TView::dimension;

	extents_t size;

	IANE_DECL_HYBRID extents_t dimensions() const { return size; }
	IANE_DECL_HYBRID pixel operator[](const coord_t& aPosition) const { return pattern(aPosition); }
};

template<class TView>
struct host_pattern : generated_view<TView> {
	using is_host_view_tag = void;
};

template<class TView>
struct device_pattern : generated_view<TView> {
	using is_device_view_tag = void;
};

template<class TView>
auto make_pattern(TView aView)
{
	using view_t = std::conditional_t<iane::concepts::HostView<TView>,
		host_pattern<TView>, device_pattern<TView>>;
	return view_t{{aView.dimensions()}};
}

// Initialize, mutate and transform the two-channel element through the value visitors.
struct initialize_pixel {
	template<class TPosition>
	IANE_DECL_HYBRID void operator()(pixel& aPixel, const TPosition& aPosition) const
	{
		aPixel = pattern(aPosition);
	}
};

struct double_pixel {
	IANE_DECL_HYBRID void operator()(pixel& aPixel) const { aPixel = aPixel * 2; }
};

struct bias_pixel {
	IANE_DECL_HYBRID pixel operator()(pixel aPixel) const { return aPixel + pixel{1, -1}; }
};

// Combine the source with the previous destination value at the same coordinate.
template<class TView>
struct combine_at_position {
	TView output;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID pixel operator()(pixel aPixel, const typename TView::coord_t& aPosition) const
	{
		return output[aPosition] + aPixel + pixel{aPosition[TView::dimension - 1], 0};
	}
};

// Exercise a locator callback that writes to a separate destination.
template<class TView>
struct store_previous_sample {
	TView output;

	IANE_CUDA_HD_WARNING_DISABLE
	template<iane::concepts::Locator TLocator>
	IANE_DECL_HYBRID void operator()(TLocator aLocator) const
	{
		typename TLocator::offset_t offset{};
		offset[TView::dimension - 1] = -1;
		output[aLocator.position()] = aLocator[offset];
	}
};

// Sample the neighbor along the outermost axis to exercise higher dimensions.
struct next_sample {
	IANE_CUDA_HD_WARNING_DISABLE
	template<iane::concepts::Locator TLocator>
	IANE_DECL_HYBRID pixel operator()(TLocator aLocator) const
	{
		typename TLocator::offset_t offset{};
		offset[TLocator::dimension - 1] = 1;
		return aLocator[offset];
	}
};

// Verify storage directly so a traversal bug cannot hide an unvisited coordinate.
template<class TImage, class TExpected>
void check_pixels(const TImage& aImage, TExpected aExpected)
{
	int plane_size = 1;
	for (int i = 1; i < TImage::dim; ++i) {
		plane_size *= 3;
	}
	for (int index = 0; index < plane_size * 3; ++index) {
		const auto expected = aExpected(index + 1, index / plane_size, plane_size);
		BOOST_TEST_CONTEXT("linear index " << index) {
			BOOST_TEST(aImage.pointer()[index].first == expected.first);
			BOOST_TEST(aImage.pointer()[index].second == expected.second);
		}
	}
}

template<int tDim>
void check_meta_algorithms(iane::backend aBackend)
{
	using image_t = iane::host_image<pixel, tDim>;
	using coord_t = typename image_t::extents_t;
	const coord_t size(3, std::true_type{});
	const iane::run_options options{.backend = aBackend};

	image_t input(size);
	image_t output(size);
	iane::copy(make_pattern(input.const_view()), input.view()).wait();

	iane::run_on(options, input.const_view(), output.view(), [](auto aSrc, auto aDst, auto& aExec) {
		iane::for_each_position(aDst, initialize_pixel{}, aExec);
		iane::for_each(aDst, double_pixel{}, aExec);
		iane::transform(aDst, aDst, bias_pixel{}, aExec);
		return iane::transform_position(aSrc, aDst, combine_at_position{aDst}, aExec);
	});
	check_pixels(output, [](int aValue, int aLayer, int) {
		return pixel{3 * aValue + 1 + aLayer, -3 * aValue - 1};
	});

	iane::run_on(options, input.const_view(), output.view(), [](auto, auto aDst, auto& aExec) {
		return iane::for_each_locator(make_pattern(aDst), store_previous_sample{aDst},
			iane::border_mode::zero, aExec);
	});
	check_pixels(output, [](int aValue, int aLayer, int aPlaneSize) {
		const int previous = aLayer == 0 ? 0 : aValue - aPlaneSize;
		return pixel{previous, -previous};
	});

	iane::run_on(options, input.const_view(), output.view(), [](auto aSrc, auto aDst, auto& aExec) {
		return iane::transform_locator(aSrc, aDst, next_sample{}, iane::border_mode::clamp, aExec);
	});
	check_pixels(output, [](int aValue, int aLayer, int aPlaneSize) {
		const int next = aLayer == 2 ? aValue : aValue + aPlaneSize;
		return pixel{next, -next};
	});

	coord_t kernel_size(1, std::true_type{});
	kernel_size[tDim - 1] = 3;
	iane::host_image<int, tDim> kernel(kernel_size);
	kernel.pointer()[0] = 2;
	kernel.pointer()[1] = -1;
	kernel.pointer()[2] = 0;
	iane::run_on(options, std::tuple{input.const_view(), kernel.const_view()}, output.view(),
		[](auto aSrc, auto aKernel, auto aDst, auto& aExec) {
			return iane::convolve(aSrc, aKernel, aDst, {iane::border_mode::zero}, aExec);
		});
	check_pixels(output, [](int aValue, int aLayer, int aPlaneSize) {
		const int next = aLayer == 2 ? 0 : aValue + aPlaneSize;
		return pixel{2 * next - aValue, aValue - 2 * next};
	});
}

} // namespace meta_tests
