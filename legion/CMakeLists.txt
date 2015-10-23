#
###########################################################################
# Copyright (c) 2014, Los Alamos National Security, LLC.
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

project(lsci)

# Build w/ clang ignoring any delete-non-virtual-dtor warnings
find_package(Clang)
set(CMAKE_C_COMPILER ${CLANGXX_COMMAND})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSHARED_LOWLEVEL -Wno-delete-non-virtual-dtor -Wno-format")
set(CMAKE_CXX_COMPILER ${CLANGXX_COMMAND})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHARED_LOWLEVEL -Wno-delete-non-virtual-dtor -Wno-format")

set(LG_RT_DIR ${SCOUT_SRC_DIR}/legion/runtime)

include_directories(${SCOUT_SRC_DIR}/lib/Runtime/mesh)

list(APPEND low_level_src 
  ${LG_RT_DIR}/shared_lowlevel.cc
  ${LG_RT_DIR}/realm/logging.cc 
  ${LG_RT_DIR}/realm/cmdline.cc 
  ${LG_RT_DIR}/realm/profiling.cc 
  ${LG_RT_DIR}/realm/timers.cc
)

list(APPEND mapper_src 
  ${LG_RT_DIR}/mappers/default_mapper.cc 
  ${LG_RT_DIR}/mappers/shim_mapper.cc 
  ${LG_RT_DIR}/mappers/mapping_utilities.cc
)

list(APPEND high_level_src 
  ${LG_RT_DIR}/legion/legion.cc 
  ${LG_RT_DIR}/legion/legion_c.cc 
  ${LG_RT_DIR}/legion/legion_ops.cc 
  ${LG_RT_DIR}/legion/legion_tasks.cc 
  ${LG_RT_DIR}/legion/legion_trace.cc 
  ${LG_RT_DIR}/legion/legion_spy.cc 
  ${LG_RT_DIR}/legion/legion_profiling.cc 
  ${LG_RT_DIR}/legion/legion_instances.cc 
  ${LG_RT_DIR}/legion/legion_views.cc 
  ${LG_RT_DIR}/legion/legion_analysis.cc 
  ${LG_RT_DIR}/legion/region_tree.cc 
  ${LG_RT_DIR}/legion/runtime.cc
  ${LG_RT_DIR}/legion/garbage_collection.cc
)

list(APPEND sclegion_src ${SCOUT_SRC_DIR}/legion/sclegion/sclegion.cc)

include_directories(
  sclegion
  runtime
  runtime/legion
  runtime/mappers
  runtime/realm
)

add_library(sclegion STATIC ${low_level_src} ${mapper_src} ${high_level_src} ${sclegion_src})
add_dependencies(sclegion scc)
install(TARGETS sclegion DESTINATION lib)
