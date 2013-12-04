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

# Determine the compiler to use for Scout programs.  Note that a
# generator may set CMAKE_SCC_COMPILER before loading this file to
# force a specific compiler to be used.  You may also use the
# environment variable SCC first (if defined), next use the cmake
# variable CMAKE_GENERATOR_SCC which can be defined by a generator.
#
# Sets the following variables:
#   CMAKE_SC_COMPILER
#   CMAKE_AR
#   CMAKE_RANLIB
#
# If not already set before, it also sets
#   _CMAKE_TOOLCHAIN_PREFIX

IF(NOT CMAKE_SCC_COMPILER)

  SET(CMAKE_SCC_COMPILER_INIT NOTFOUND)

  # Prefer the environment variable SCC.
  IF($ENV{SCC} MATCHES ".+")

    GET_FILENAME_COMPONENT(CMAKE_SCC_COMPILER_INIT $ENV{SCC} 
      PROGRAM PROGRAM_ARGS CMAKE_SCC_FLAGS_ENV_INIT)

    IF(CMAKE_SCC_FLAGS_ENV_INIT)
      SET(CMAKE_SCC_COMPILER_ARG1 "${CMAKE_SCC_FLAGS_ENV_INIT}" 
      CACHE STRING "First argument to SCC compiler")
    ENDIF(CMAKE_SCC_FLAGS_ENV_INIT)

    IF(NOT EXISTS ${CMAKE_SCC_COMPILER_INIT})
      MESSAGE(FATAL_ERROR 
        "Compiler command not set in environment SCC:\n$ENV{SCC}.\n${CMAKE_SCC_COMPILER_INIT}")
    ENDIF(NOT EXISTS ${CMAKE_SCC_COMPILER_INIT})

  ENDIF($ENV{SCC} MATCHES ".+")

  # Next prefer the generator specified compiler.
  IF(CMAKE_GENERATOR_SCC)

    IF(NOT CMAKE_SCC_COMPILER_INIT)
      SET(CMAKE_SCC_COMPILER_INIT ${CMAKE_GENERATOR_SCC})
    ENDIF(NOT CMAKE_SCC_COMPILER_INIT)

  ENDIF(CMAKE_GENERATOR_SCC)

  # finally list compilers to try
  IF(CMAKE_SCC_COMPILER_INIT)
    SET(CMAKE_SCC_COMPILER_LIST ${CMAKE_SCC_COMPILER_INIT})
  ELSE(CMAKE_SCC_COMPILER_INIT)
    SET(CMAKE_SCC_COMPILER_LIST scc)
  ENDIF(CMAKE_SCC_COMPILER_INIT)

  # Find the compiler.
  IF (_CMAKE_USER_SCC_COMPILER_PATH)

    FIND_PROGRAM(CMAKE_SCC_COMPILER 
      NAMES ${CMAKE_SCC_COMPILER_LIST} 
      PATHS ${_CMAKE_USER_SCC_COMPILER_PATH} 
      DOC "Scout C Compiler (SCC)" 
      NO_DEFAULT_PATH)

  ENDIF (_CMAKE_USER_SCC_COMPILER_PATH)

  FIND_PROGRAM(CMAKE_SCC_COMPILER NAMES ${CMAKE_SCC_COMPILER_LIST} DOC "Scout C Compiler (SCC)")

  IF(CMAKE_SCC_COMPILER_INIT AND NOT CMAKE_SCC_COMPILER)

    SET(CMAKE_SCC_COMPILER "${CMAKE_SCC_COMPILER_INIT}" CACHE FILEPATH "SCC compiler" FORCE)

  ENDIF(CMAKE_SCC_COMPILER_INIT AND NOT CMAKE_SCC_COMPILER)

ELSE(NOT CMAKE_SCC_COMPILER)
  # We get here if CMAKE_SCC_COMPILER was on the command line (via -D)
  # or via a pre-made CMakeCache.txt (e.g. via ctest) or set in CMake's
  # CMAKE_TOOLCHAIN_FILE.
  #
  # If CMAKE_SCC_COMPILER is a list of length 2, use the first item as
  # CMAKE_SCC_COMPILER and the 2nd one as CMAKE_SCC_COMPILER_ARG1
  LIST(LENGTH CMAKE_SCC_COMPILER _CMAKE_SCC_COMPILER_LIST_LENGTH)

  IF("${_CMAKE_SCC_COMPILER_LIST_LENGTH}" EQUAL 2)
    LIST(GET CMAKE_SCC_COMPILER 1 CMAKE_SCC_COMPILER_ARG1)
    LIST(GET CMAKE_SCC_COMPILER 0 CMAKE_SCC_COMPILER)
  ENDIF("${_CMAKE_SCC_COMPILER_LIST_LENGTH}" EQUAL 2)

  # If a compiler was specified by the user but did not include a full
  # path, try to find its full path.  If it is found, force it into
  # the cache.  If it isn't found, don't overwrite the setting (which
  # was given by the user) with "NOTFOUND" if the SCC compiler already
  # had a path, reuse it for searching the C compiler

  GET_FILENAME_COMPONENT(_CMAKE_USER_SCC_COMPILER_PATH "${CMAKE_SCC_COMPILER}" PATH)

  IF(NOT _CMAKE_USER_SCC_COMPILER_PATH)

    FIND_PROGRAM(CMAKE_SCC_COMPILER_WITH_PATH NAMES ${CMAKE_SCC_COMPILER})
    MARK_AS_ADVANCED(CMAKE_SCC_COMPILER_WITH_PATH)

    IF(CMAKE_SCC_COMPILER_WITH_PATH)
      SET(CMAKE_SCC_COMPILER ${CMAKE_SCC_COMPILER_WITH_PATH} CACHE STRING "SCC compiler" FORCE)
    ENDIF(CMAKE_SCC_COMPILER_WITH_PATH)

  ENDIF(NOT _CMAKE_USER_SCC_COMPILER_PATH)
ENDIF(NOT CMAKE_SCC_COMPILER)

MARK_AS_ADVANCED(CMAKE_SCC_COMPILER)

IF (NOT _CMAKE_TOOLCHAIN_LOCATION)
  GET_FILENAME_COMPONENT(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_SCC_COMPILER}" PATH)
ENDIF (NOT _CMAKE_TOOLCHAIN_LOCATION)

IF(NOT CMAKE_SCC_COMPILER_ID_RUN)
  SET(CMAKE_SCC_COMPILER_ID_RUN 1)

  # Each entry in this list is a set of extra flags to try
  # adding to the compile line to see if it helps produce
  # a valid identification file.
  SET(CMAKE_SCC_COMPILER_ID_TEST_FLAGS
    # Try compiling to an object file only.
    "-c"
    )

  # Try to identify the compiler.
  SET(CMAKE_SCC_COMPILER_ID)
  FILE(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in
    CMAKE_SCC_COMPILER_ID_PLATFORM_CONTENT)
  INCLUDE(${SCOUT_CMAKE_DIR}/Scout/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(SCC SCCFLAGS CMakeSCCCompilerId.sc)

ENDIF(NOT CMAKE_SCC_COMPILER_ID_RUN)

INCLUDE(${CMAKE_ROOT}/Modules/CMakeClDeps.cmake)
INCLUDE(CMakeFindBinUtils)

# configure all variables set in this file
CONFIGURE_FILE(${SCOUT_CMAKE_DIR}/Scout/CMakeSCCCompiler.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CMakeSCCCompiler.cmake
  @ONLY IMMEDIATE # IMMEDIATE must be here for compatibility mode <= 2.0
  )
#for cmake 2.8.10
CONFIGURE_FILE(${SCOUT_CMAKE_DIR}/Scout/CMakeSCCCompiler.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${CMAKE_VERSION}/CMakeSCCCompiler.cmake
  @ONLY IMMEDIATE # IMMEDIATE must be here for compatibility mode <= 2.0
  )

SET(CMAKE_SCC_COMPILER_ENV_VAR "scc")
