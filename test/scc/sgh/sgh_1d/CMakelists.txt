project(scout_test_scc_sgh_sgh1d)

  file(GLOB sgh1d_SOURCES "sgh_solver_1d.sc")
  #ignore warnings about i/o in forall
  set(CMAKE_SCC_FLAGS "${CMAKE_SCC_FLAGS} -Wno-calling-io-function -Wno-mesh-variable-scope")

  set(fname "scc_sgh_sgh1d")
  add_executable (${fname} ${sgh1d_SOURCES})
  add_test("${fname}" "${TEST_BINARY_DIR}/${fname}")
  message(" + added runtime regression test entry: ${fname}")
