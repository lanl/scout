#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(scout_test_volren)

# only run these tests if the $DISPLAY variable is set

set(CMAKE_SCC_FLAGS "-DSC_ENABLE_MPI -I${CMAKE_CURRENT_SOURCE_DIR}")

set(LIBS
  ${MPI_LIBRARIES}
)

include_directories(
  ${MPI_INCLUDE_PATH}
)

# Note: By default we pull in all the .sc source files in this
# directory for inclusion into the regression checks.  
#
file(GLOB SC_SRCS *.sc) 
foreach (test ${SC_SRCS})
  get_filename_component(tname ${test} NAME_WE) 
  if (NOT tname)
    message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
  else (NOT tname)
    add_executable(${tname} ${tname}.sc )
    target_link_libraries(${tname} ${LIBS})
    if(DEFINED ENV{DISPLAY})
      if(tname STREQUAL "volrenmp")
        add_test(${tname} ${MPIEXEC} -n 8 ${CMAKE_CURRENT_BINARY_DIR}/${tname} 2 2 2)
      elseif(tname STREQUAL "combustion-volren")
        if(DEFINED ENV{SC_LANL_ONLY})
          add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
        endif()
      else()
        add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
      endif()
      message(" + added runtime regression test entry: ${tname}")
   endif()
  endif(NOT tname)
endforeach(test)

