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
  
project(scout_rt_test_glfw)

message(STATUS "scout install include directory: ${SCOUT_INSTALL_INCLUDE_DIR}")
message(STATUS "scout install lib directory: ${SCOUT_INSTALL_LIB_DIR}")
message(STATUS "glfw library dir: ${GLFW_LIBRARY_DIR}")
message(STATUS "glfw libs : ${GLFW_LIBS}")
message(STATUS "opengl libs : ${OPENGL_LIBRARY}")

if(NOT APPLE)
  set(CMAKE_CXX_FLAGS "") #don't use c++11
endif()

include_directories(
  ${SCOUT_INSTALL_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${GLFW_INCLUDE_DIR}
)

set(LIBS
  scRuntime
  ${GLFW_LIBS}
  ${OPENGL_LIBRARY}
  ${COCOA_LIBRARY}
  ${IOKIT_LIBRARY}
  ${COREVIDEO_LIBRARY}
  ${COREFOUNDATION_FRAMEWORK}
)

file(GLOB RT_GLFW_SRCS simple.cpp)

set(CMAKE_CXX_COMPILER ${SCOUT_BUILD_DIR}/../bin/clang++)

if (SCOUT_ENABLE_GLFW)
  # Add an executable and test entry for each source file...
  foreach (test ${RT_GLFW_SRCS})
    get_filename_component(tname ${test} NAME_WE) 
    if (NOT tname) 
      message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
    else (NOT tname)
      add_executable(runtime_glfw_${tname} ${test})
      if(DEFINED ENV{DISPLAY}) 
        add_test(runtime_glfw_${tname} ${TEST_BINARY_DIR}/runtime_glfw_${tname})
        message(" + added runtime regression GLFW test entry: runtime_glfw_${tname}")
      endif()
      if ("${CMAKE_SYSTEM}" MATCHES "Linux")
        #GL and GLU are added here as findOpenGL is broken on Ubuntu
        target_link_libraries(runtime_glfw_${tname} ${LIBS} GL GLU X11 Xxf86vm Xrandr pthread Xi Xcursor rt)
      else()
        target_link_libraries(runtime_glfw_${tname} ${LIBS})
      endif()
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SCOUT_INSTALL_LIB_DIR} -L${GLFW_LIBRARY_DIR}") 
    endif(NOT tname)
  endforeach(test)
else()
endif()
