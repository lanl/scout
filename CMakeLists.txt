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
#####

project(Scout)
  ##### CMAKE REQUIREMENTS
  # There are a few gremlins in cmake we've stumbled into -- we force the
  # version here to avoid chasing our tails again...  If your system doesn't
  # have at least the version number below download it from:
  #
  #   http://www.cmake.org/cmake/resources/software.html
  #
    cmake_minimum_required(VERSION 2.8.12)
  #
  #####
  include(cmake/Modules/ScoutConfig.cmake)
  include(cmake/Modules/ScoutLLVMConfig.cmake)
  include(CTest)

  set(CMAKE_VERBOSE_MAKEFILE ON)

  ##### INSTALL DETAILS
  #
  # Note: There are additional install configuration details
  # within LLVM and we need to make sure we are compatible
  # with them as well...
  #
  set(CMAKE_INSTALL_PREFIX ${SCOUT_BUILD_DIR})
  set(SCOUT_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

  # Install details...
  if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS
      "scout: Debug build, install prefix '${CMAKE_INSTALL_PREFIX}'.")
  endif()

  install(DIRECTORY ${SCOUT_INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING
      PATTERN  "*.h"
      PATTERN  "*.sch"
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
      )

  install(DIRECTORY ${SCOUT_CONFIG_DIR}/include/
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
  # Include the next level of projects below.  Please place only one
  # level of projects below vs. paths to individual/single target
  # projects below (e.g. use lib/CMakeLists.txt to add a library
  # subproject vs. adding the library itself below).
  #
  # Note that we are dependent upon llvm's build configuration and
  # ordering issues can become problematic/tricky (especially for
  # parallel builds) if you are not careful here.
  #
  # Summary of subprojects:
  #
  #    - llvm    : modified/extended versions of llvm, clang for scout.
  #    - include : auto-generation of configuration headers.
  #    - config  : library of scout's build-time configuration details.
  #    - tools   : various tools that support scout development.
  #    - runtime : scout's runtime library.
  #
    if(CMAKE_SCC_RUNTIME_ONLY STREQUAL "ON")
      message(STATUS "scout: build runtime only")
      add_subdirectory(lib/Runtime)
      #add_subdirectory(lib/NewRuntime)
    else()
      message(STATUS "scout: build all")
      add_subdirectory(llvm)
      add_subdirectory(include)
      add_subdirectory(lib/Config)
      add_subdirectory(cmake)
      add_subdirectory(tools)
      #build w/ clang
      add_subdirectory(legion)
      add_subdirectory(lib/Runtime)
    endif()

    file(GLOB scout_incs ${SCOUT_INCLUDE_DIR}/scout/*)
    file(MAKE_DIRECTORY ${SCOUT_BUILD_DIR}/include/scout)

    foreach(entry ${scout_incs})
      get_filename_component(name ${entry} NAME)
      set(dest ${SCOUT_BUILD_DIR}/include/scout)
      if (NOT ${name} STREQUAL "Config") 
	message("${entry} --> ${dest}")
        file(COPY ${entry} DESTINATION ${dest})
      endif()
    endforeach()

    #-----
    # Note: we have to build the standard library and test directory
    # separately because the 'scc' driver needs to be bootstrapped.
    # We therefore treat those directories as entirely separate projects.
    # The configuration file generation below helps out with the bootstrap
    # process...
    #-----
    message(STATUS "scout: generating pre-configuration for stdlib build.")
    configure_file(
      lib/Standard/cmake/Modules/Configuration.cmake.in
      ${CMAKE_BINARY_DIR}/lib/Standard/CMake/Modules/Configuration.cmake
      )
    message(STATUS "scout: generating pre-configuration for testing build.")
    configure_file(
      test/cmake/Modules/Configuration.cmake.in
      ${CMAKE_BINARY_DIR}/test/CMake/Modules/Configuration.cmake
      )

    #-----
    # Some optional parts of the build...
    #
    set(sandbox scout-local/sandbox)
    if(EXISTS ${sandbox} AND IS_DIRECTORY ${sandbox})
      message(STATUS "scout: generating configuration file for sandbox.")
      configure_file(
        ${sandbox}/SandboxConfig.cmake.in
        ${CMAKE_SOURCE_DIR}/${sandbox}/SandboxConfig.cmake
        )
    endif()
    #
    #-----
  #
  #
  #####
