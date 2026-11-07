#pragma once
#include <type_traits>

namespace iane::concepts {

/* ---------------------------------------------------------
   Basic markers implemented by view types:
   struct is_host_view_tag {};
   struct is_device_view_tag {};
   --------------------------------------------------------- */

/**
 * @brief Concept satisfied by any host-side image view.
 */
template<typename TElement>
concept HostView = requires { typename TElement::is_host_view_tag; };

/**
 * @brief Concept satisfied by any device-side image view.
 */
template<typename TElement>
concept DeviceView = requires { typename TElement::is_device_view_tag; };

/**
 * @brief Any image view, host or device.
 */
template<typename TElement>
concept AnyView = HostView<TElement> || DeviceView<TElement>;

} // namespace iane::concepts
