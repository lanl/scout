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

include(${CMAKE_BINARY_DIR}/CMake/Modules/Configuration.cmake)

set(CMAKE_MODULE_PATH
  ${SCOUT_CMAKE_PATH}
  ${CMAKE_MODULE_PATH}
)

#ignore warnings about i/o in forall
set(CMAKE_SCC_FLAGS "${CMAKE_SCC_FLAGS} -Wno-calling-io-function")

project(scout_test_scc)
    
  cmake_minimum_required(VERSION 2.8.9)

  # Note: By default we pull in all the .sc source files in this
  # directory for inclusion into the regression checks.  If you want do
  # build more complex tests we suggest creating a subdirectory for the
  # source.
  #
  file(GLOB SC_SRCS *.sc) 

  # Add an executable and test entry for each source file...
  foreach (test ${SC_SRCS})
    get_filename_component(tname ${test} NAME_WE) 
    if (NOT tname) 
      message(FATAL_ERROR "runtime regression check ${test} produced an empty name")
    else (NOT tname)
      set(fname "scc_${tname}")
      add_executable(${fname} ${test})
      add_test("${fname}" "${TEST_BINARY_DIR}/${fname}")
      message(" + added runtime regression test entry: ${fname}")
    endif(NOT tname)
  endforeach(test)

  add_subdirectory(display) 
  add_subdirectory(driver) 
  add_subdirectory(error) 
  add_subdirectory(warning) 
