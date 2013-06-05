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

project(scc)

  # Set our visibility settings for compatibility with LLVM's build
  # flags...  Without this you'll get an odd message at link time,
  # "likely caused by different translation units being compiled with
  # different visibility settings"...
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fno-rtti")
  
  include(sc-llvm-link-libs)

  add_llvm_definitions( -D__STDC_CONSTANT_MACROS )
  add_llvm_definitions( -D__STDC_FORMAT_MACROS )
  add_llvm_definitions( -D__STDC_LIMIT_MACROS )

  include_directories(
    ${SC_INC_PATH}
    ${SC_LLVM_INC_PATH}
    ${SC_CLANG_INC_PATH}
    ${CMAKE_INSTALL_PREFIX}/include 
    )

  set(LLVM_LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    asmparser
    bitreader
    bitwriter
    codegen
    instrumentation
    ipo
    linker
    selectiondag
    )

  add_llvm_executable(scc
    driver.cpp
    cc1_main.cpp
    cc1as_main.cpp
    ScoutVisitor.cpp
    )

  add_dependencies(scc
    clang-headers
    scConfig
    clang
    )

  target_link_libraries(scc
    clangFrontendTool
    clangAST
    clangAnalysis
    clangBasic
    clangCodeGen
    clangDriver
    clangEdit
    clangFrontend
    clangLex
    clangParse
    clangEdit
    clangARCMigrate
    clangRewriteCore
    clangRewriteFrontend
    clangSema
    clangSerialization
    clangStaticAnalyzerFrontend
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    ${SC_LLVM_LINK_LIBS}
    scConfig
    )

  # TODO: We need to replace CLANG_EXECUTABLE_VERSION below with 
  # SCC_EXECUTABLE_VERSION... 
  set_target_properties(scc PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
  install(TARGETS scc
    RUNTIME DESTINATION bin)
