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
  
project(glyph_interop)

find_package(SDL)

include_directories(
  ${PNG_INCLUDE_DIR} 
  ${SCOUT_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${SDL_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

SET(CUDA_SDK_LIBS cudpp_x86_64)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

SET(GLYPH_INTEROP_CUDA_SRCS compute.cu )
SET(GLYPH_INTEROP_CUDA_FILES compute.o )

#try and figure out if the machine is 64bit
if( CMAKE_SIZEOF_VOID_P MATCHES "8" )
  SET(NVCC_FLAGS "-m64")
else()
  SET(NVCC_FLAGS "")
endif()

add_custom_command(
  OUTPUT compute.o
  COMMAND ${CUDA_NVCC} ${NVCC_FLAGS} -c -I ${SCOUT_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/runtime/interop/compute.cu 
  )

SET(GLYPH_INTEROP_SRCS glyph_interop.cpp)

if(APPLE) 
  FIND_LIBRARY(APPKIT_LIBRARY AppKit)
else()
  SET(APPKIT_LIBRARY "")  
endif()

set(GLYPH_INTEROP_LIBS
  scRuntime
  ${PNG_LIBRARIES}
  ${OPENGL_LIBRARY}
  ${CUDA_LIBRARIES}
  ${HWLOC_LIBRARIES}
  ${MPI_LIBRARIES}
  ${APPKIT_LIBRARY}
  ${SDL_LIBRARIES}
)

ADD_EXECUTABLE(glyph_interop
  ${GLYPH_INTEROP_CUDA_FILES}
  ${GLYPH_INTEROP_SRCS}
  ) 

TARGET_LINK_LIBRARIES(glyph_interop ${GLYPH_INTEROP_LIBS} )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SCOUT_LIB_DIR} -L${CUDA_LIBRARY_DIR} -L${SDL_LIBRARY_DIR}") 

if(DEFINED ENV{DISPLAY} AND DEFINED ENV{SC_NVIDIA}) 
  add_test(glyph_interop ${CMAKE_CURRENT_BINARY_DIR}/glyph_interop)
  message(" + added runtime regression test entry: glyph_interop")  
endif()

