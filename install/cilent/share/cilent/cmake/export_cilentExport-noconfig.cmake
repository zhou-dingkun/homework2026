#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cilent::kalman_filter" for configuration ""
set_property(TARGET cilent::kalman_filter APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cilent::kalman_filter PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libkalman_filter.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cilent::kalman_filter )
list(APPEND _IMPORT_CHECK_FILES_FOR_cilent::kalman_filter "${_IMPORT_PREFIX}/lib/libkalman_filter.a" )

# Import target "cilent::image_processor" for configuration ""
set_property(TARGET cilent::image_processor APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cilent::image_processor PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libimage_processor.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cilent::image_processor )
list(APPEND _IMPORT_CHECK_FILES_FOR_cilent::image_processor "${_IMPORT_PREFIX}/lib/libimage_processor.a" )

# Import target "cilent::serial_sender" for configuration ""
set_property(TARGET cilent::serial_sender APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cilent::serial_sender PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libserial_sender.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cilent::serial_sender )
list(APPEND _IMPORT_CHECK_FILES_FOR_cilent::serial_sender "${_IMPORT_PREFIX}/lib/libserial_sender.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
