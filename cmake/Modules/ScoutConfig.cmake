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
#####

if (NOT __SC_CONFIG_CMAKE)
  set(__SC_CONFIG_CMAKE)
else()
  return()
endif()

##### SCOUT SOURCE & BUILD PATHS
# These variables are used throughout the configuration.
#
  set(SC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(SC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
  set(SC_INCLUDE_DIR ${SC_SRC_DIR}/include)
  set(SC_CONFIG_DIR ${SC_BUILD_DIR}/scout-config)
  set(SC_INC_PATH ${SC_INCLUDE_DIR} ${SC_CONFIG_DIR}/include)

  set(SC_LLVM_SRC_DIR ${SC_SRC_DIR}/llvm)
  set(SC_LLVM_BUILD_DIR ${SC_BUILD_DIR}/llvm)
  set(SC_LLVM_INC_PATH ${SC_LLVM_SRC_DIR}/include ${SC_LLVM_BUILD_DIR}/include)

  set(SC_CLANG_SRC_DIR ${SC_LLVM_SRC_DIR}/tools/clang)
  set(SC_CLANG_BUILD_DIR ${SC_LLVM_BUILD_DIR}/tools/clang)
  set(SC_CLANG_INC_PATH ${SC_CLANG_SRC_DIR}/include 
    ${SC_CLANG_BUILD_DIR}/include)
#
#####

##### SCOUT BUILD SETTINGS
# Check to see if we have a build-type set -- if not, default to a
# debugging build...
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE DEBUG)
    message(STATUS "scout: Defaulting to debug build configuration...")
  endif()
#
#####


##### SCOUT VERSION INFORMATION 
# The following variables are used to help us track Scout 
# version information. 
#
  set(SC_VERSION_MAJOR 0)
  set(SC_VERSION_MINOR 1)
  set(SC_VERSION_PATCH 0)

  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Scout Programming Language")
  set(CPACK_PACKAGE_VENDOR scout)
  set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
  set(CPACK_PACKAGE_VERSION_MAJOR "${SC_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${SC_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${SC_VERSION_PATCH}")
#
#####  


##### CMAKE MODULE PATHS
# Overhaul the path to CMake moudles so we can find some of our 
# own (and LLVM's) configuration details. 
#
  set(CMAKE_MODULE_PATH
    "${SC_SRC_DIR}/cmake/Modules"
    "${SC_BUILD_DIR}/cmake/Modules"
    "${SC_LLVM_SRC_DIR}/cmake/modules"
     ${CMAKE_MODULE_PATH}
    )

  message(STATUS "scout: CMake modules search path: ${CMAKE_MODULE_PATH}")
#
#####


##### PACKAGE DEPENDENCIES & CONFIGURATION 
# We search for the various external software packages we require, or
# can leverage, in this section. The results of these searches also 
# drives Scout's feature set.

  # --- OpenGL support. 
  find_package(OpenGL)
  if (OPENGL_FOUND)
    message(STATUS "scout: OpenGL found, enabling support.")
    set(SC_ENABLE_OPENGL ON  CACHE BOOL "Enable OpenGL support.")
    set(SC_HAVE_OPENGL 1)
  else()
    set(SC_HAVE_OPENGL 0)
  endif()

  # --- CUDA support. 
  find_package(CUDA)
  if (CUDA_FOUND) 
    message(STATUS "scout: CUDA found, enabling PTX codegen support.")
    message(STATUS "scout: CUDA include path: ${CUDA_INCLUDE_DIRS}")
    set(SC_ENABLE_CUDA ON CACHE BOOL 
      "Enable CUDA/PTX code generation and runtime support.")

    set(SC_ENABLE_LIB_NVVM OFF CACHE BOOL 
      "Enable NVIDIA's compiler SDK vs. LLVM's PTX backend.")

    if (SC_ENABLE_LIB_NVVM)
      message(STATUS "scout: Enabling NVIDIA libnvvm support.")
    endif()

  else()
    message(STATUS "scout: CUDA not found disabling support.")
    set(SC_ENABLE_CUDA OFF CACHE BOOL 
      "Enable CUDA/PTX code generation and runtime support.")
  endif()  

  # --- OpenCL support.  
  #only look for OpenCL if we can't find Cuda
  if(NOT CUDA_FOUND) 
    #OpenCL support not currently working on Apple
    if(NOT APPLE)
      find_package(OpenCL)
    endif()
  endif()
  if (OPENCL_FOUND)
    # TODO - should these separated or wrapped into one?  
    message(STATUS "scout: OpenCL found, enabling AMDIL codegen support.")    
    set(SC_ENABLE_OPENCL ON CACHE BOOL "Enable OpenCL code generation and runtime support.")
    set(SC_ENABLE_AMDIL ON CACHE BOOL "Enable AMD IL code generation and runtime support.")
    #add_definitions(-DSC_ENABLE_OPENCL -DSC_ENABLE_AMDIL)
  else()
    message(STATUS "scout: OpenCL not found, disabling support")    
    set(SC_ENABLE_OPENCL OFF CACHE BOOL "Enable OpenCL code generation and runtime support.")
    set(SC_ENABLE_AMDIL OFF CACHE BOOL "Enable AMD IL code generation and runtime support.")
   endif()

  # --- NUMA support. 
  find_package(HWLOC)
  if (HWLOC_FOUND)
    message(STATUS "scout: Found hwloc -- enabling NUMA support.")
    set(SC_ENABLE_NUMA ON CACHE BOOL "Enable NUMA (via libhwloc) runtime support.")
    if (APPLE) 
      message(STATUS "scout: Note NUMA support under Mac OS X has limited features.")
    endif()
  else()
    message(STATUS "scout: NUMA support not found -- disabling support.")
    set(SC_ENABLE_NUMA OFF CACHE BOOL "Enable NUMA (via libhwloc) runtime support.")
  endif()

  # --- Thread support. 
  find_package(Threads)
  if (Threads_FOUND) 
    message(STATUS "scout: Found Threads -- enabling support.")
    set(SC_ENABLE_THREADS ON CACHE BOOL "Enable Threads support.") 
  else()
    set(SC_ENABLE_THREADS OFF CACHE BOOL "Enable Threads support.") 
  endif()

  # --- MPI support. 
  find_package(MPI)
  if (MPI_FOUND) 
    message(STATUS "scout: Found MPI -- enabling support.")
    set(SC_ENABLE_MPI ON CACHE BOOL "Enable MPI runtime support.") 
  else()
    set(SC_ENABLE_MPI OFF CACHE BOOL "Enable MPI runtime support.") 
  endif()


  # --- GLFW support. 
  find_package(GLFW)
  if (GLFW_FOUND) 
    message(STATUS "scout: GLFW found, enabling runtime support.")
    message(STATUS "scout: GLFW include path: ${GLFW_INCLUDE_DIR}")
    set(SC_ENABLE_GLFW ON CACHE BOOL 
      "Enable GLFW runtime support.")
  else()
    message(STATUS "scout: GLFW not found, disabling support.")
    set(SC_ENABLE_GLFW OFF CACHE BOOL 
      "Enable GLFW runtime support.")
  endif()  


  # --- SDL support. 
  find_package(SDL REQUIRED)
  if (SDL_FOUND) 
    message(STATUS "scout: Found SDL -- enabling support.")
    set(SC_ENABLE_SDL ON CACHE BOOL "Enable SDL support (required).")
  endif()

  # Disable PNG for now -- some Linux systems are having a hard time 
  # with matching the API we've used an we haven't had time to fully
  # investigate.  Putting it on the backburner for now...
  find_package(PNG)

  if (PNG_FOUND)
    message(STATUS "scout: Found PNG -- enabling support.")
    set(SC_ENABLE_PNG ON CACHE BOOL "Enable PNG support in scout's runtime libraries.")
    add_definitions(${PNG_DEFINITIONS})
  else()
    set(SC_ENABLE_PNG OFF CACHE BOOL "Enable PNG support in scout's runtime libraries.")
  endif()

# --- THRUST support.
  find_package(THRUST)

  if (THRUST_FOUND)
    set (THRUST_DIR ${THRUST_INCLUDE_DIR}/thrust  CACHE PATH "Thrust directory")
    message(STATUS "scout: THRUST found")

  else ()
    message(STATUS "scout: THRUST not found -- disabling support.")
    set(SC_ENABLE_CUDA_THRUST OFF CACHE BOOL "Enable THRUST support via CUDA in scout's runtime libraries.")
    set(SC_ENABLE_TBB_THRUST OFF CACHE BOOL "Enable THRUST support via TBB in scout's runtime libraries.")
    set(SC_ENABLE_CPP_THRUST OFF CACHE BOOL "Enable THRUST support via CPP in scout's runtime libraries.")

  endif ()

  if (THRUST_FOUND)

    set(SC_ENABLE_CPP_THRUST ON CACHE BOOL "Enable THRUST support via CPP in scout's runtime libraries.")

    find_package(CUDA)
    if (CUDA_FOUND)
      message(STATUS "scout: CUDA found.")
#set(SC_ENABLE_CUDA_THRUST ON CACHE BOOL "Enable THRUST support via CUDA in scout's runtime libraries.")
    endif ()

    find_package(TBB)
    if (TBB_FOUND)
      message(STATUS "scout: TBB found.")
#      set(SC_ENABLE_TBB_THRUST ON CACHE BOOL "Enable THRUST support via TBB in scout's runtime libraries.")
    endif ()

  endif ()

#
#####
