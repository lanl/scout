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
INCLUDE(CMakeTestCompilerCommon)

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that that selected compiler can actually compile and link
# the most basic of programs.  If not, a fatal error is set and cmake
# stops processing commands and will not generate any makefiles or
# projects.
IF(NOT CMAKE_SCXX_COMPILER_WORKS)
  PrintTestCompilerStatus("SCXX" "")
  FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/testSCXXCompiler.scpp 
    "#ifndef __cplusplus\n"
    "# error \"The CMAKE_SCXX_COMPILER is set to a C compiler\"\n"
    "#endif\n"
    "int main(){return 0;}\n")
  TRY_COMPILE(CMAKE_SCXX_COMPILER_WORKS ${CMAKE_CURRENT_BINARY_DIR} 
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/testSCXXCompiler.scpp
    OUTPUT_VARIABLE __CMAKE_SCXX_COMPILER_OUTPUT)
  SET(SCXX_TEST_WAS_RUN 1)
ENDIF(NOT CMAKE_SCXX_COMPILER_WORKS)

IF(NOT CMAKE_SCXX_COMPILER_WORKS)
  PrintTestCompilerStatus("SCXX" " -- broken")
  # if the compiler is broken make sure to remove the platform file
  # since Windows-cl configures scxx files need to be removed
  # when c or c++ fails
  FILE(REMOVE ${CMAKE_PLATFORM_ROOT_BIN}/CMakeSCXXPlatform.cmake)
  FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CMakeError.log
    "Determining if the SCXX compiler works failed with "
    "the following output:\n${__CMAKE_SCXX_COMPILER_OUTPUT}\n\n")
  MESSAGE(FATAL_ERROR "The Scout compiler \"${CMAKE_SCXX_COMPILER}\" "
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n ${__CMAKE_SCXX_COMPILER_OUTPUT}\n\n"
    "CMake will not be able to correctly generate this project.")
ELSE(NOT CMAKE_SCXX_COMPILER_WORKS)
  IF(SCXX_TEST_WAS_RUN)
    PrintTestCompilerStatus("SCXX" " -- works")
    FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CMakeOutput.log
      "Determining if the SCXX compiler works passed with "
      "the following output:\n${__CMAKE_SCXX_COMPILER_OUTPUT}\n\n")
  ENDIF(SCXX_TEST_WAS_RUN)
  SET(CMAKE_SCXX_COMPILER_WORKS 1 CACHE INTERNAL "")

  IF(CMAKE_SCXX_COMPILER_FORCED)
    # The compiler configuration was forced by the user.
    # Assume the user has configured all compiler information.
  ELSE(CMAKE_SCXX_COMPILER_FORCED)
    # Try to identify the ABI and configure it into CMakeSCXXCompiler.cmake
    INCLUDE(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
    CMAKE_DETERMINE_COMPILER_ABI(SCXX ${SCOUT_CMAKE_DIR}/CMakeSCXXCompilerABI.scpp)
    CONFIGURE_FILE(
      ${SCOUT_CMAKE_DIR}/CMakeSCXXCompiler.cmake.in
      ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeSCXXCompiler.cmake
      @ONLY IMMEDIATE # IMMEDIATE must be here for compatibility mode <= 2.0
      )
    #for cmake 2.8.10
    CONFIGURE_FILE(
      ${SCOUT_CMAKE_DIR}/CMakeSCXXCompiler.cmake.in
      ${CMAKE_BINARY_DIR}/CMakeFiles/${CMAKE_VERSION}/CMakeSCXXCompiler.cmake
      @ONLY IMMEDIATE # IMMEDIATE must be here for compatibility mode <= 2.0
      )
    INCLUDE(${CMAKE_BINARY_DIR}/CMakeFiles/CMakeSCXXCompiler.cmake)
  ENDIF(CMAKE_SCXX_COMPILER_FORCED)
  IF(CMAKE_SCXX_SIZEOF_DATA_PTR)
    FOREACH(f ${CMAKE_SCXX_ABI_FILES})
      INCLUDE(${f})
    ENDFOREACH()
    UNSET(CMAKE_SCXX_ABI_FILES)
  ENDIF()
ENDIF(NOT CMAKE_SCXX_COMPILER_WORKS)

UNSET(__CMAKE_SCXX_COMPILER_OUTPUT)
