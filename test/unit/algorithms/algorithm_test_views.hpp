#pragma once

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

// Build packed input directly to keep test setup independent of the tested algorithms.
template<int tDim>
auto make_input()
{
	using image_t = iane::host_image<pixel, tDim>;
	image_t image(typename image_t::extents_t(3, std::true_type{}));
	int count = 1;
	for (int i = 0; i < tDim; ++i) {
		count *= 3;
	}
	for (int i = 0; i < count; ++i) {
		image.pointer()[i] = {i + 1, -i - 1};
	}
	return image;
}

} // namespace meta_tests
