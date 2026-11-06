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
template<typename T>
concept HostView = requires { typename T::is_host_view_tag; };

/**
 * @brief Concept satisfied by any device-side image view.
 */
template<typename T>
concept DeviceView = requires { typename T::is_device_view_tag; };

/**
 * @brief Any image view, host or device.
 */
template<typename T>
concept AnyView = HostView<T> || DeviceView<T>;

} // namespace iane::concepts
