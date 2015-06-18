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

project(scRuntimeLibrary)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Build w/ clang
find_package(Clang)
set(CMAKE_CXX_COMPILER ${CLANGXX_COMMAND})

include_directories(
  ${SCOUT_INCLUDE_DIR}
  ${SCOUT_INCLUDE_DIR}/scout/Runtime/volren
  ${SCOUT_BUILD_DIR}/config/include
  ${PNG_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${R_INCLUDE_DIRS}
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

find_package(QT5 REQUIRED)
if (QT5_FOUND)
  find_package(Qt5Widgets)
  find_package(Qt5Gui)
endif()

#gpu
if (SCOUT_ENABLE_CUDA)
  file(GLOB rt_srcs_gpu gpu/*.cpp)

  #cuda_include_directories(${SCOUT_INCLUDE_DIR}/scout/Runtime/volren)
  #set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  #set(CMAKE_CXX_FLAGS -g)
  #cuda_compile(RENDERER_O volren/volumeRender_kernel.cu)
  #set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
  
  file(GLOB rt_srcs_volren volren/*.cpp)

  qt5_wrap_cpp(moc_srcs_volren volren/glwidget.h ${SCOUT_INCLUDE_DIR}/scout/Runtime/volren/VolumeRendererWindow.h)

  list(APPEND rt_srcs ${rt_srcs_gpu})
  list(APPEND rt_srcs ${rt_srcs_volren})
  list(APPEND rt_srcs ${moc_srcs_volren})
  list(APPEND rt_srcs ${RENDERER_O})

endif()

#plot
if (SCOUT_ENABLE_PLOT)
  file(GLOB rt_srcs_plot plot/*.cpp)
  list(APPEND rt_srcs ${rt_srcs_plot})
endif()

#plot2
if (QT5_FOUND)  
  file(GLOB qt_hdrs
   ${SCOUT_INCLUDE_DIR}/scout/Runtime/opengl/qt/*.h)
  file(GLOB rt_srcs_plot2 plot2/*.cpp)
  list(APPEND rt_srcs ${rt_srcs_plot2})

  file(GLOB qt_srcs opengl/qt/*.cpp)

  qt5_wrap_cpp(moc_srcs ${qt_hdrs})
endif()  

if (SCOUT_ENABLE_OPENGL)

  # Add OpenGL source files to the build...
  file(GLOB gl_hdrs ${SCOUT_INCLUDE_DIR}/Runtime/opengl/*.h)
  file(GLOB gl_srcs opengl/*.cpp)

  if (APPLE)
    #find CoreGraphics headers if they are inside of ApplicationServices
    if(NOT EXISTS /System/Library/Frameworks/CoreGraphics.framework)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -F/System/Library/Frameworks/ApplicationServices.framework/Frameworks/")
    endif()
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
        OUTPUT ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        COMMAND ${GLSLCC} --vertex ${v_shader}
        COMMAND touch ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        DEPENDS ${GLSLCC} ${v_shader}
        )
    else()
      add_custom_command(
        OUTPUT ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        COMMAND touch ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        )
    endif()

    add_custom_target(compile_${name}
      ALL
      DEPENDS ${GLSLCC} ${v_shader} ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
      )
  endforeach()

  file(GLOB gl_frag_shaders opengl/glsl/*_fs.glsl)
  foreach(f_shader ${gl_frag_shaders})

    # file name w/out extension
    get_filename_component(name ${f_shader} NAME_WE)

    # only compile shaders if $DISPLAY is set
    if(DEFINED ENV{DISPLAY})
      add_custom_command(
        OUTPUT ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        COMMAND ${GLSLCC} --fragment ${f_shader}
        COMMAND touch ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        DEPENDS ${GLSLCC} ${f_shader}
        )
    else()
      add_custom_command(
        OUTPUT ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        COMMAND touch ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
        )
    endif()

    add_custom_target(compile_${name}
      ALL
      DEPENDS ${GLSLCC} ${f_shader} ${scRuntimeLibrary_BINARY_DIR}/${name}_pass.dep
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
link_directories(${CUDA_LIBRARY_DIR})

add_library(scRuntime SHARED ${rt_srcs} ${gl_srcs} ${qt_srcs} ${moc_srcs})

set_target_properties(scRuntime 
  PROPERTIES 
  POSITION_INDEPENDENT_CODE 1
  COMPILE_FLAGS "-fPIC"
  MACOSX_RPATH 1) 

message(STATUS "this is it 1: ${CUDA_LIBRARIES}")
message(STATUS "this is it 2: ${CUDA_TOOLKIT_ROOT_DIR}")

#set(_link_libs ${OPENGL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${R_LI#BRARIES} Qt5::Widgets Qt5::Gui)

set(_link_libs ${OPENGL_LIBRARIES} ${CUDA_LIBRARIES} ${R_LIBRARIES} Qt5::Widgets Qt5::Gui)

if (APPLE) 
  set(_link_libs ${_link_libs} "-framework Cocoa -framework Foundation -framework IOKit -framework CoreVideo")
endif()
target_link_libraries(scRuntime ${_link_libs})

add_dependencies(scRuntime
  scc
  glsl2cstr
  )

install(TARGETS scRuntime
  DESTINATION lib)

unset(_link_libs)
