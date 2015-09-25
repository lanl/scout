project(scout_test_scc_display_sgh_sgh2d)

  file(GLOB sgh2d_SOURCES "*.sc")
  file(GLOB sgh2d_HEADERS "*.sch")
  #ignore warnings about i/o in forall
  set(CMAKE_SCC_FLAGS "${CMAKE_SCC_FLAGS} -Wno-calling-io-function")

  set (sgh2d_INCLUDE_DIRS "")
  foreach (_headerFile ${sgh2d_HEADERS})
      get_filename_component(_dir ${_headerFile} PATH)
      list (APPEND sgh2d_INCLUDE_DIRS ${_dir})
  endforeach()
  list(REMOVE_DUPLICATES sgh2d_INCLUDE_DIRS)

  include_directories(${sgh2d_INCLUDE_DIRS})
  set(fname "scc_display_sgh_sgh2d")
  add_executable (${fname} ${sgh2d_SOURCES})
  add_test("${fname}" "${TEST_BINARY_DIR}/${fname}")
  message(" + added runtime regression test entry: ${fname}")
