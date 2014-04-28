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

project(scRuntime)

find_library(GLFW glfw)

# Build w/ clang as volren uses blocks
set(CMAKE_CXX_COMPILER ${SCOUT_BUILD_DIR}/bin/clang++)
if(NOT APPLE)
  set(CMAKE_CXX_FLAGS "") #don't use c++11 for runtime
endif()

include_directories(
  ${SCOUT_INCLUDE_DIR}
  ${SCOUT_BUILD_DIR}/config/include
  ${SDL_INCLUDE_DIR}
  ${GLFW_INCLUDE_DIR}
  ${PNG_INCLUDE_DIR}
  ${SCOUT_INCLUDE_DIR}/scout/sys/linux #find our unistd.h wrapper to deal with __block in linux unistd.h
)

file(GLOB rt_hdrs
  ${SCOUT_INCLUDE_DIR}/runtime/*.h
  ${SCOUT_BUILD_DIR}/config/include/*.h
  )

if (APPLE)
  file(GLOB rt_srcs
    *.cpp
    *.mm
    )
else()
  file(GLOB rt_srcs
    *.cpp
    )
endif()

#cpu
file(GLOB rt_srcs_cpu cpu/*.cpp)
list(APPEND rt_srcs ${rt_srcs_cpu})

#gpu
#file(GLOB rt_srcs_gpu gpu/*.cpp)
#list(APPEND rt_srcs ${rt_srcs_gpu})

#renderall
file(GLOB rt_srcs_renderall renderall/*.cpp)
list(APPEND rt_srcs ${rt_srcs_renderall})
file(GLOB rt_srcs_renderall_cpu renderall/cpu/*.cpp)
list(APPEND rt_srcs ${rt_srcs_renderall_cpu})

if (SCOUT_ENABLE_OPENGL)

  # Add OpenGL source files to the build...
  file(GLOB gl_hdrs ${SCOUT_INCLUDE_DIR}/Runtime/opengl/*.h)
  file(GLOB gl_srcs opengl/*.cpp)

  if (APPLE)
    #find CoreGraphics headers if they are inside of ApplicationServices
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -F/System/Library/Frameworks/ApplicationServices.framework/Frameworks/")
    file(GLOB apple_gl_hdrs ${SCOUT_INCLUDE_DIR}/Runtime/opengl/macosx/*.h)
    file(GLOB apple_gl_srcs opengl/macosx/*.cpp opengl/macosx/*.mm)
    list(APPEND gl_hdrs ${apple_gl_hdrs})
    list(APPEND gl_srcs ${apple_gl_srcs})
  else()
    #disable glx stuff on linux for now
    #file(GLOB glx_hdrs ${SCOUT_INCLUDE_DIR}/Runtime/opengl/glx/*.h)
    #file(GLOB glx_srcs opengl/macosx/*.cpp opengl/glx/*.cpp)
    list(APPEND gl_hdrs ${glx_hdrs})
    list(APPEND gl_srcs ${glx_srcs})
  endif()

  # Add supporting opengl shaders to the build dependencies.  In this
  # case we actually package up the shaders into the library source by
  # converting them into C/C++ constant char strings that are included
  # into the build process...  See the toplevel tools/glsl-to-str for
  # details.

  # We need a directory to output the generated/wrapped shaders.
  if (NOT EXISTS ${SCOUT_BUILD_DIR}/include)
    file(MAKE_DIRECTORY ${SCOUT_BUILD_DIR}/include)
  endif()

  # Make sure the shaders actually compile before we pack
  # them into the runtime library for use.  We do this
  # using the 'glsl-cc' utility (in the toplevel level tools
  # directory).
  file(GLOB gl_vert_shaders opengl/glsl/*_vs.glsl)
  set(GLSLCC    ${SCOUT_BUILD_DIR}/bin/glsl-cc)
  foreach(v_shader ${gl_vert_shaders})

    # file name w/out extension
    get_filename_component(name ${v_shader} NAME_WE)

    # only compile shaders if $DISPLAY is set
    if(DEFINED ENV{DISPLAY})
      add_custom_command(
        OUTPUT ${scRuntime_BINARY_DIR}/${name}_pass.dep
        COMMAND ${GLSLCC} --vertex ${v_shader}
        COMMAND touch ${scRuntime_BINARY_DIR}/${name}_pass.dep
        DEPENDS ${GLSLCC} ${v_shader}
        )
    else()
      add_custom_command(
        OUTPUT ${scRuntime_BINARY_DIR}/${name}_pass.dep
        COMMAND touch ${scRuntime_BINARY_DIR}/${name}_pass.dep
        )
    endif()

    add_custom_target(compile_${name}
      ALL
      DEPENDS ${GLSLCC} ${v_shader} ${scRuntime_BINARY_DIR}/${name}_pass.dep
      )
  endforeach()

  file(GLOB gl_frag_shaders opengl/glsl/*_fs.glsl)
  foreach(f_shader ${gl_frag_shaders})

    # file name w/out extension
    get_filename_component(name ${f_shader} NAME_WE)

    # only compile shaders if $DISPLAY is set
    if(DEFINED ENV{DISPLAY})
      add_custom_command(
        OUTPUT ${scRuntime_BINARY_DIR}/${name}_pass.dep
        COMMAND ${GLSLCC} --fragment ${f_shader}
        COMMAND touch ${scRuntime_BINARY_DIR}/${name}_pass.dep
        DEPENDS ${GLSLCC} ${f_shader}
        )
    else()
      add_custom_command(
        OUTPUT ${scRuntime_BINARY_DIR}/${name}_pass.dep
        COMMAND touch ${scRuntime_BINARY_DIR}/${name}_pass.dep
        )
    endif()

    add_custom_target(compile_${name}
      ALL
      DEPENDS ${GLSLCC} ${f_shader} ${scRuntime_BINARY_DIR}/${name}_pass.dep
      )
  endforeach()

  # If all the shaders compiled we then add a custom step in which we
  # convert them into C character string format so they may be
  # embedded within the runtime library.
  file(GLOB gl_shaders opengl/glsl/*.glsl)
  set(GLSL2STR ${SCOUT_BUILD_DIR}/bin/glsl-to-str)
  foreach(shader ${gl_shaders})

    # file name w/out extension
    get_filename_component(name ${shader} NAME_WE)

    add_custom_command(
      OUTPUT ${SCOUT_BUILD_DIR}/include/${name}.h
      COMMAND ${GLSL2STR}
        --in      ${shader}
        --out     ${SCOUT_BUILD_DIR}/include/${name}.h
        --varname ${name}
      DEPENDS ${GLSL2STR} ${shader}
      )

    add_custom_target(generate_${name}
      ALL
      DEPENDS ${SCOUT_BUILD_DIR}/include/${name}.h ${GLSL2STR}
      )
  endforeach()

  # Update the include path to include the generated include files that
  # hold the individual shaders...
  include_directories(${SCOUT_BUILD_DIR}/include)

  if (SCOUT_ENABLE_GLFW)
    file(GLOB glfw_hdrs ${SCOUT_INCLUDE_DIR}/runtime/opengl/glfw/*.h)
    file(GLOB glfw_srcs opengl/glfw/*.cpp)
  endif()

endif()

if (SCOUT_ENABLE_MPI)
  file(GLOB gl_srcs_mpi opengl/mpi/*.cpp)
  list(APPEND gl_srcs ${gl_srcs_mpi})
  file(GLOB volren_srcs volren/hpgv/*.cpp)
endif()

# Finally add the library and dependencies...  Note that if we
# are running w/out PNG/OpenGL support the associated variables
# are empty....  This is the best way (to date) that I've found
# to deal with the cruft of handling multiple configurations of
# the runtime...
add_library(scRuntime STATIC ${rt_srcs} ${gl_srcs} ${glfw_srcs})
add_dependencies(scRuntime
  scc
  glsl2cstr
  )

install(TARGETS scRuntime
  DESTINATION lib)
