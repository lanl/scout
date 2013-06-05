#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(scout_test_mthread)

# run tests w/ the -mt threading option to scc 
set(CMAKE_SCC_FLAGS "-mt")

# Note: By default we pull in all the .sc source files in this
# directory for inclusion into the regression checks.  
file(GLOB SC_SRCS *.sc) 

# Add an executable and test entry for each source file...
foreach (test ${SC_SRCS})
  get_filename_component(tname ${test} NAME_WE) 
  if (NOT tname) 
    message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
  else (NOT tname)
    add_executable(${tname} ${test})
    add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
    message(" + added runtime regression test entry: ${tname}")
  endif(NOT tname)
endforeach(test)

