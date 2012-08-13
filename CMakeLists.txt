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
#   - For details and help with CMake see:
#
#       http://www.cmake.org/cmake/help/documentation.html
#
#   - Scout-centric configuration options are of the form
#     SC_[description].  Please continue to use this form when
#     modifying or adding to the build system.
#
#   - Note that we include the building of LLVM as a subproject.  This
#     means that there can be some tricky ordering issues between
#     project dependencies so use care when chaning the ordering of
#     things within the build configuration(s).
#
#   - Our philosophy for determining the build configuration is based
#     on what packages are available (found) on your system at
#     configuration time.  You can override these settings manually
#     with 'ccmake' or from the command line with 'cmake'. 
#  
# <-- LANL ONLY
#
#   - Lines within our source that match this pattern should be used
#     for code that is intended for use within LANL (e.g. as a way to
#     protect NDA information we have under vendor collaborations.
#
# LANL ONLY -->
# 
#####

project(Scout)

  cmake_minimum_required(VERSION 2.8.8)

  enable_testing()

  set(SC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(SC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
  set(SC_INCLUDE_DIR ${SC_SRC_DIR}/include)
  set(SC_REGRESSION_DIR ${SC_SRC_DIR}/test)

  set(SC_LLVM_SRC_DIR ${SC_SRC_DIR}/llvm)
  set(SC_LLVM_INC_DIR ${SC_LLVM_SRC_DIR}/include)
  set(SC_LLVM_BUILD_INC_DIR ${SC_BUILD_DIR}/llvm/include)

  set(SC_CLANG_SRC_DIR ${SC_SRC_DIR}/llvm/tools/clang)
  set(SC_CLANG_INC_DIR ${SC_CLANG_SRC_DIR}/include) 
  set(SC_CLANG_BUILD_INC_DIR ${SC_BUILD_DIR}/llvm/tools/clang/include)

  set(CMAKE_VERBOSE_MAKEFILE OFF)

  ##### PACKAGE CREATION (via CPack)
  #
  # http://www.cmake.org/Wiki/CMake:Packaging_With_CPack)
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
  include(CPack)
  #
  #
  #####


  ##### CMAKE MODULES
  #
  # Add paths for our custom cmake modules.  Note that we also 
  # include llvm's modules -- this makes life much easier for 
  # building... 
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/llvm/cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/llvm/cmake/modules"
    )
  #
  #
  #####


  ##### BUILD SETTINGS AND FLAGS
  #
  # 

  # Check to see if we have a build-type set -- if not, default to a
  # debugging build...
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE DEBUG)
    message(STATUS "scout: Defaulting to debug build configuration...")
  endif()
  #
  #
  #####


  ##### PACKAGE-DISCOVERY SETTINGS
  #
  # The following section searches the system for various packages and
  # configures the build based on what packages are found. 
  #

  # --- OpenGL support. 
  find_package(OpenGL)
  if (OPENGL_FOUND)
    message(STATUS "scout: OpenGL found, enabling support.")
    set(SC_ENABLE_OPENGL ON  CACHE BOOL "Enable OpenGL support.")
    add_definitions(-DSC_ENABLE_OPENGL)
  endif()


  # --- CUDA (NVIDIA GPU) support. 
  # 
  find_package(CUDA)
  if (CUDA_FOUND) 
    message(STATUS "scout: CUDA found, enabling PTX codegen support.")
    message(STATUS "scout: CUDA include path: ${CUDA_INCLUDE_DIRS}")
    set(SC_ENABLE_CUDA ON CACHE BOOL "Enable CUDA/PTX code generation and runtime support.")
    add_definitions(-DSC_ENABLE_CUDA) 

    #<-- LANL-ONLY 
    #
    set(SC_ENABLE_LIB_NVVM OFF CACHE BOOL "Enable NVIDIA's compiler SDK vs. LLVM's PTX backend.")

    if (SC_ENABLE_LIB_NVVM)
      message(STATUS "scout: Enabling NVIDIA libnvvm support.")
      add_definitions(-DSC_ENABLE_LIB_NVVM)
    endif()
    #
    # LANL ONLY --> 

  endif()  


  #<-- LANL-ONLY 
  #
  # --- OpenCL (AMD GPU) support.  
  # TODO: This needs to be flushed out but we are currently using an
  # NDA version of the LLVM backend (that will hopefully soon be open
  # sourced).  Should we find a way to break the build if you are not
  # within a pre-designated git branch?
  #
  #  !!! This is currently not a FindOpenCL module...
  #
  # find_package(OpenCL)
  # if (OPENCL_FOUND)

    # TODO - should these separated or wrapped into one?  

    # message(STATUS "scout: OpenCL found, enabling AMDIL codegen support.")    
    # set(SC_ENABLE_OPENCL ON CACHE BOOL "Enable OpenCL code generation and runtime support.")
    # set(SC_ENABLE_AMDIL ON CACHE BOOL "Enable AMD IL code generation and runtime support.")
    # add_definitions(-DSC_ENABLE_OPENCL -DSC_ENABLE_AMDIL)

  # endif()
  #
  # LANL ONLY --> 


  # --- NUMA (HWLOC) support. 
  #
  find_package(HWLOC)
  if (HWLOC_FOUND)
    message(STATUS "scout: Found hwloc -- enabling NUMA support.")
    set(SC_ENABLE_NUMA ON CACHE BOOL "Enable NUMA (via libhwloc) runtime support.")
    add_definitions(-DSC_ENABLE_NUMA)
    if (APPLE) 
      message(STATUS "scout: Note NUMA support under Mac OS X has limited features.")
    endif()
  endif()


  # --- MPI support. 
  # 
  find_package(MPI)

  if (MPI_FOUND) 
    message(STATUS "scout: Found MPI -- enabling support.")
    set(SC_ENABLE_MPI ON CACHE BOOL "Enable MPI runtime support.") 
    add_definitions(-DSC_ENABLE_MPI) 

  endif()

  # Disable PNG for now -- some Linux systems are having a hard time 
  # with matching the API we've used an we haven't had time to fully
  # investigate.  Putting it on the backburner for now...
  find_package(PNG)

  if (PNG_FOUND)
    message(STATUS "scout: Found PNG -- enabling support.")
    set(SC_ENABLE_PNG ON  CACHE_BOOL "Enable PNG support in scout's runtime libraries.")
    add_definitions(-DSC_ENABLE_PNG ${PNG_DEFINITIONS})
  endif()
  #
  #
  #####


  ##### LLVM CONFIGURATION DETAILS
  #
  # Note: Here we try to override some of the LLVM default
  # configurations such that we can skip building some of backends
  # that we really don't care about...
  #
  set(LLVM_TARGETS_TO_BUILD X86 NVPTX CppBackend
    CACHE STRING "Semicolon-separated list of LLVM backend targets to build, or \"all\".")
  message(STATUS "scout: Reduced LLVM codegen targets (${LLVM_TARGETS_TO_BUILD}).")
  #
  #
  #####


  ##### INSTALL DETAILS
  #
  # Note: There are additional install configuration details
  # within LLVM and we need to make sure we are compatible 
  # with them as well... 
  #

  # Install details...
  if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(CMAKE_INSTALL_PREFIX ${SC_BUILD_DIR}/scout)
    message(STATUS "scout: Debug build enabled, setting in-source install prefix of '${CMAKE_INSTALL_PREFIX}'.")
  endif()

  # NOTE: There is a bug in CMake 2.8.8 that will not correctly
  # install header files on Mac OS X using the syntax we have below.
  # The patch for this is available here:
  #
  #    http://cmake.org/gitweb?p=cmake.git;a=commitdiff;h=a41557a2
  #
  install(DIRECTORY ${SC_INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING 
      PATTERN  "*.h"
      PATTERN  "*.sch"
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  )

  #
  #####


  ##### SUBPROJECTS
  #
  # Include the next level of projects below.  Please place only
  # one level of projects below vs. paths to individual/single
  # target projects below (e.g. use lib/CMakeLists.txt to add a
  # library subproject vs. adding the library itself below). 
  #
  # Note that we are dependent upon llvm's build configuration 
  # and ordering issues can become problematic/tricky (especially
  # for parallel builds) if you are not careful here. 
  #
  # Summary of subprojects:
  #
  #    - tools  : various tools that support scout development.
  #    - llvm   : our modified/extended versions of llvm, clang.
  #    - lib    : supporting libraries (e.g. the runtime).
  #    - sandbox: proof-of-concept test codes.
  #    - tests  : regression testing
  #
  add_subdirectory(tools)
  add_subdirectory(llvm)
  add_subdirectory(tools/llvm)
  add_subdirectory(tools/clang)
  add_subdirectory(lib)
  #add_subdirectory(sandbox)

  set(SC_ENABLE_REGRESS_TESTS ON CACHE STRING
        "Building the suite of regression tests.")
  if (SC_ENABLE_REGRESSIONS)
    add_subdirectory(test)
  endif()
  #
  #
  #####
