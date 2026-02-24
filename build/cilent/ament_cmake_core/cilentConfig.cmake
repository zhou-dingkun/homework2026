# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_cilent_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED cilent_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(cilent_FOUND FALSE)
  elseif(NOT cilent_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(cilent_FOUND FALSE)
  endif()
  return()
endif()
set(_cilent_CONFIG_INCLUDED TRUE)

# output package information
if(NOT cilent_FIND_QUIETLY)
  message(STATUS "Found cilent: 0.0.0 (${cilent_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'cilent' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${cilent_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(cilent_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_targets-extras.cmake;ament_cmake_export_dependencies-extras.cmake")
foreach(_extra ${_extras})
  include("${cilent_DIR}/${_extra}")
endforeach()
