#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(mac_rt_tests)

include_directories(
  ${OPENGL_INCLUDE_DIR}
  ${SCOUT_INCLUDE_DIR}
)

set(LIBS
  ${OPENGL_LIBRARY}
  scRuntime
)

file(GLOB SRCS *.cpp)

# Add an executable and test entry for each source file...
foreach (test ${SRCS})
  get_filename_component(tname ${test} NAME_WE) 
  if (NOT tname) 
    message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
  else (NOT tname)
    add_executable(${tname} ${test})
    target_link_libraries(${tname} ${LIBS})
    set(CMAKE_EXE_LINKER_FLAGS 
      "${CMAKE_EXE_LINKER_FLAGS} -L${SCOUT_LIB_DIR} -L${CUDA_LIBRARY_DIR} -framework Cocoa -framework CoreGraphics") 
    add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
    message(" + added runtime regression test entry: ${tname}")
  endif(NOT tname)
endforeach(test)

