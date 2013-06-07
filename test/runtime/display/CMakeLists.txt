#
###########################################################################
# Copyright (c) 2010, Los Alamos National Security, LLC.
# All rights reserved.
# 
#  Copyright 2010. Los Alamos National Security, LLC. This software was
#  produced under U.S. Government contract DE-AC52-06NA25396 for Los
#  Alamos National Laboratory (LANL), which is operated by Los Alamos
#  National Security, LLC for the U.S. Department of Energy. The
#  U.S. Government has rights to use, reproduce, and distribute this
#  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
#  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
#  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
#  derivative works, such modified software should be clearly marked,
#  so as not to confuse it with the version available from LANL.
#
#  Additionally, redistribution and use in source and binary forms,
#  with or without modification, are permitted provided that the
#  following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided 
#      with the distribution.  
#
#    * Neither the name of Los Alamos National Security, LLC, Los
#      Alamos National Laboratory, LANL, the U.S. Government, nor the
#      names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
#
#  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
#  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGE.
#
###########################################################################
#
  
project(scout_rt_test_display)

FIND_LIBRARY(COCOA_LIBRARY Cocoa)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${SCOUT_INCLUDE_DIR}
  ${SCOUT_INCLUDE_DIR}/scout #find our unistd.h wrapper to deal with __block in linux unistd.h
  ${OPENGL_INCLUDE_DIR}
  ${SDL_INCLUDE_DIR}
  ${HWLOC_INCLUDE_DIR}
  ${MPI_INCLUDE_DIR}
)

set(LIBS
  scRuntime
  ${CUDA_LIBRARIES}
  ${OPENGL_LIBRARY}
  ${SDL_LIBRARIES}
  ${HWLOC_LIBRARIES}
  ${MPI_LIBRARIES}
)

if (COCOA_LIBRARY)
    set(LIBS ${LIBS} ${COCOA_LIBRARY})
endif()

file(GLOB RT_DISPLAY_SRCS *.cpp)

# Add an executable and test entry for each source file...
foreach (test ${RT_DISPLAY_SRCS})
  get_filename_component(tname ${test} NAME_WE) 
  if (NOT tname) 
    message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
  else (NOT tname)
    add_executable(${tname} ${test})
    target_link_libraries(${tname} ${LIBS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SCOUT_LIB_DIR} -L${CUDA_LIBRARY_DIR} -L${SDL_LIBRARY_DIR}") 
    if(DEFINED ENV{DISPLAY}) 
      add_test(${tname} ${CMAKE_CURRENT_BINARY_DIR}/${tname})
      message(" + added runtime regression test entry: ${tname}")
    endif()
  endif(NOT tname)
endforeach(test)

