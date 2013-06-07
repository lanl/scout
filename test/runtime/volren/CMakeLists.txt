#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(scout_rt_test_display)

# Build w/ clang as volren uses blocks
set(CMAKE_CXX_COMPILER ${SC_BUILD_DIR}/../llvm/bin/clang++)
set(CMAKE_CXX_FLAGS "-fblocks")


include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${SCOUT_INCLUDE_DIR}
  ${SCOUT_INCLUDE_DIR}/scout #find our unistd.h wrapper to deal with __block in linux unistd.h
  ${OPENGL_INCLUDE_DIR}
  ${SDL_INCLUDE_DIR}
  ${HWLOC_INCLUDE_DIR}
  ${MPI_INCLUDE_PATH}
)

if(APPLE)
  FIND_LIBRARY(APPKIT_LIBRARY AppKit)
else()
  SET(APPKIT_LIBRARY "")
endif()

set(LIBS
  scRuntime
  BlocksRuntime
  ${CUDA_LIBRARIES}
  ${OPENGL_LIBRARY}
  ${APPKIT_LIBRARY}
  ${SDL_LIBRARIES}
  ${HWLOC_LIBRARIES}
  ${MPI_LIBRARIES}
)

file(GLOB RT_VOLREN_SRCS *.cpp)

# Add an executable and test entry for each source file...
foreach (test ${RT_VOLREN_SRCS})
  get_filename_component(tname ${test} NAME_WE) 
  if (NOT tname) 
    message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
  else (NOT tname)
    add_executable(${tname} ${test})
    target_link_libraries(${tname} ${LIBS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SCOUT_LIB_DIR} -L${CUDA_LIBRARY_DIR} -L${SDL_LIBRARY_DIR}") 
    if(DEFINED ENV{DISPLAY}) 
      if(tname STREQUAL "combustion-volren_rt")
        if(DEFINED ENV{SC_LANL_ONLY})
          add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
          message(" + added runtime regression test entry: ${tname}")
        endif()
      else()
        add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
        message(" + added runtime regression test entry: ${tname}")
      endif()
    endif()
  endif(NOT tname)
endforeach(test)

