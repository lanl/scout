if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  # ===== Scout ==============================
  # don't build debugserver use the xcode one
  # that is already installed and signed
  #add_subdirectory(debugserver)
  # =========================================
endif()
  add_subdirectory(driver)
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
  add_subdirectory(lldb-gdbserver)
endif()
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_subdirectory(lldb-platform)
endif()
