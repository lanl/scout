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
# Notes
#
#   This is a simplified find module for CUDA. 
#
#   This module sets the following variables:
#
#     CUDA_FOUND       - Set to TRUE if CUDA is found.
#     CUDA_INCLUDE_DIR - Path for reaching CUDA header files. 
#     CUDA_LIBRARY_DIR - Path to CUDA library file(s).
#     CUDA_LIBRARIES   - Set of libraries required for linking.
#     CUDA_NVCC        - location of CUDA_NVCC
#     CUDA_VERSION_MAJOR - major version number
#     CUDA_VERSION_MINOR - minor version number 
# 
#####

  # CUDA 5 introduced a framework bundle on Mac OS X.  Unfortunately, 
  # the header files associated with the the framework are incomplete; 
  # therefore we disable looking for frameworks...
  if (APPLE)
    set(_SAVE_FIND_FRAMEWORK ${CMAKE_FIND_FRAMEWORK})
    set(CMAKE_FIND_FRAMEWORK NEVER)
  endif()

  ##### DETERMINE CUDA VERSION INFORMATION.
  #
  find_program(CUDA_NVCC
    NAMES nvcc
    PATH_SUFFIXES bin cuda/bin
    HINTS $ENV{CUDA_DIR}/bin
    PATHS 
    /usr
    /usr/local
    /opt
    NO_DEFAULT_PATH
    DOC "Path to the CUDA compiler (nvcc)."
  )

  # Dig out the version information from nvcc if it exists... This is based on the CMake
  # provided FindCUDA module. 
  if (CUDA_NVCC MATCHES "NOTFOUND")
    set(CUDA_VERSION_MAJOR "0")
    set(CUDA_VERSION_MINOR "0")
  else() 
    execute_process(COMMAND ${CUDA_NVCC} "--version" OUTPUT_VARIABLE _NVCC_VERSION)
    string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${_NVCC_VERSION})
    string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${_NVCC_VERSION})
    unset(_NVCC_VERSION)
  endif()
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" 
    CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)


  ##### INCLUDE DIRECTORY AND FILES
  #
    find_path(CUDA_INCLUDE_DIRS
      cuda.h
      PATH_SUFFIXES include include/cuda cuda/include
      HINTS $ENV{CUDA_DIR}
      PATHS
      /usr
      /usr/local
      /opt
      /opt/local
      DOC "Path for CUDA's include files."
      )
  #
  #####


  #####
  #
    if (APPLE) 
      set(DYNLIB_EXT dylib)    
    else()
      set(DYNLIB_EXT so)
    endif()

    message(STATUS "find-cuda: looking for libcudart.${DYNLIB_EXT}")
    find_path(CUDA_LIBRARY_DIR
      libcudart.${DYNLIB_EXT}
      PATH_SUFFIXES lib64 cuda/lib64 lib cuda/lib 
      HINTS $ENV{CUDA_DIR}
      /usr
      /usr/local
      /opt
      /opt/local
      DOC "Path to CUDA libraries."
      )
  #
  #####

  if (CUDA_LIBRARY_DIR) 
   if (APPLE) 
      set(CUDA_LIBRARIES "-lcuda -lcudart")
    else()
      set(CUDA_LIBRARIES "-ldl -lcuda -lcudart")
    endif()
  endif()

  find_package_handle_standard_args(CUDA
    REQUIRED_VARS 
    CUDA_LIBRARY_DIR 
    CUDA_INCLUDE_DIRS
    )


  # Clean up... 
  if (APPLE)
    set(CMAKE_FIND_FRAMEWORK _SAVE_FIND_FRAMEWORK)
    unset(_SAVE_FIND_FRAMEWORK)
  endif()
