#pragma once

#include <iane/algorithms/resample.hpp>
#include <iane/execution/run_on.hpp>
#include <iane/image/image.hpp>
#include <iane/locator/sampling_image_locator.hpp>
#include <iane/resource/host/resource_host.hpp>
#include <iane/view/interpolated_view.hpp>

#include <boost/test/unit_test.hpp>

namespace resample_tests {

// Map destination centers to edge-aligned source positions at twice the scale.
struct scale_x {
	IANE_DECL_HYBRID iane::math::vec2f operator()(iane::math::vec2f aPosition) const
	{
		aPosition[0] *= 2;
		return aPosition;
	}
};

// Select the plane midway between the two source slices.
struct middle_plane {
	IANE_DECL_HYBRID iane::math::vec3f operator()(iane::math::vec2f aPosition) const
	{
		return iane::math::vec3f{aPosition[0], aPosition[1], 1.0f};
	}
};

// Sample a subpixel offset on the backend executing the traversal.
template<iane::concepts::InterpolatedView TView>
struct sample_neighborhood {
	TView source;

	IANE_DECL_HYBRID void operator()(float& aOutput, iane::math::vec2i aIndex) const
	{
		const iane::sampling_image_locator locator(source,
			iane::math::vec2f{0.5f + aIndex[0], 0.5f + aIndex[1]});
		static_assert(iane::concepts::SamplingLocator<decltype(locator)>);
		aOutput = locator.sample(iane::math::vec2f{-0.5f, -0.5f});
	}
};

inline void check_resampling(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<int, 2> input(6, 2);
	iane::host_image<float, 2> output(2, 2);
	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < 6; ++x) {
			input.view()[coord_t{x, y}] = 10 * (x + 4 * y);
		}
	}
	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto crop = aSource.subview(coord_t{1, 0}, coord_t{4, 2});
			const auto linear = iane::make_interpolated_view(crop, iane::linear);
			static_assert(iane::concepts::InterpolatedView<decltype(linear)>);
			static_assert(std::same_as<typename decltype(linear)::value_type, int>);
			static_assert(std::same_as<decltype(linear.sample(iane::math::vec2f{})), float>);
			return iane::resample(linear, aDestination, scale_x{}, aExecution);
		});
	const float expected[]{15, 35, 55, 75};
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(output.pointer()[i] == expected[i]);
	}

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto crop = aSource.subview(coord_t{1, 0}, coord_t{4, 2});
			const auto linear = iane::make_interpolated_view(crop, iane::linear, iane::border_mode::zero);
			return iane::for_each_position(aDestination, sample_neighborhood{linear}, aExecution);
		});
	const float neighborhood[]{2.5f, 7.5f, 15.0f, 35.0f};
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(output.pointer()[i] == neighborhood[i]);
	}
}

inline void check_plane(iane::backend aBackend)
{
	iane::host_image<float, 3> volume(2, 2, 2);
	iane::host_image<float, 2> output(2, 2);
	for (int i = 0; i < 8; ++i) {
		volume.pointer()[i] = static_cast<float>(i);
	}
	iane::run_on(iane::run_options{.backend = aBackend}, volume.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			return iane::resample(iane::make_interpolated_view(aSource, iane::linear),
				aDestination, middle_plane{}, aExecution);
		});
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(output.pointer()[i] == i + 2.0f);
	}
}

} // namespace resample_tests
