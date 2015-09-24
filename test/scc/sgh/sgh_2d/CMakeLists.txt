project(scout_test_scc_sgh_sgh2d)

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
  set(fname "scc_sgh_sgh2d")
  add_executable (${fname} ${sgh2d_SOURCES})
  add_test("${fname}" "${TEST_BINARY_DIR}/${fname}")
#add_test(scout_test_scc_sgh_sgh2d_check_verts_L diff -q verts_L.txt ${CMAKE_SOURCE_DIR}/scc/sgh/sgh_2d/verts_L.txt.gold)
#add_test(scout_test_scc_sgh_sgh2d_check_verts_R diff -q verts_R.txt ${CMAKE_SOURCE_DIR}/scc/sgh/sgh_2d/verts_R.txt.gold)
# add_test(scout_test_scc_sgh_sgh2d_check_cells_L diff -q cells_L.txt ${CMAKE_SOURCE_DIR}/scc/sgh/sgh_2d/cells_L.txt.gold)
#add_test(scout_test_scc_sgh_sgh2d_check_cells_R diff -q cells_R.txt ${CMAKE_SOURCE_DIR}/scc/sgh/sgh_2d/cells_R.txt.gold)
  message(" + added runtime regression test entry: ${fname}")
