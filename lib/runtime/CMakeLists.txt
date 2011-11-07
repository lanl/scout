#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(scRuntime)
enable_testing()

set(CMAKE_C_COMPILER ${LLVM_BINARY_DIR}/bin/scc)
set(CMAKE_CXX_COMPILER ${LLVM_BINARY_DIR}/bin/scc)

include_directories(
  ${SCOUT_INCLUDE_DIR}
  ${PNG_INCLUDE_DIR}
  ${HWLOC_INCLUDE_DIR}
  ${SCOUT_SRC_DIR}/packages/hwloc/include
)

file(GLOB rt_hdrs ${SCOUT_INCLUDE_DIR}/runtime/*.h)

if(${APPLE})
  file(GLOB rt_srcs *.cpp *.mm)
else()
  file(GLOB rt_srcs *.cpp)  
endif()

if(SCOUT_ENABLE_RT_OPENGL) 
  # Add OpenGL source files to the build...
  file(GLOB gl_hdrs ${SCOUT_INCLUDE_DIR}/runtime/opengl/*.h)
  file(GLOB gl_srcs opengl/*.cpp)

  # Add supporting opengl shaders to the build dependencies.  In this
  # case we actually package up the shaders into the library source by
  # converting them into C/C++ constant char strings that are included
  # into the build process...  See the toplevel utils/glsl2cstr for
  # details...


  # Make sure the shaders actually compile before we pack
  # them into the runtime library for use.  We do this 
  # using the 'glslcc' utility (in the toplevel level utils
  # directory). 
  file(GLOB gl_vert_shaders opengl/glsl/*_vs.glsl) 
  set(GLSLCC    ${SCOUT_BINARY_DIR}/bin/glslcc)
  foreach(v_shader ${gl_vert_shaders}) 

    get_filename_component(name ${v_shader} NAME_WE) # file name w/out extension

    add_custom_command(
      OUTPUT ${scRuntime_BINARY_DIR}/${name}_pass.dep
      COMMAND ${GLSLCC} --vertex ${v_shader} 
      COMMAND touch ${scRuntime_BINARY_DIR}/${name}_pass.dep
      DEPENDS ${GLSLCC} ${v_shader}
      )

    add_custom_target(compile_${name}
      ALL
      DEPENDS ${GLSLCC} ${v_shader} ${scRuntime_BINARY_DIR}/${name}_pass.dep
      )
  endforeach()

  file(GLOB gl_frag_shaders opengl/glsl/*_fs.glsl) 
  foreach(f_shader ${gl_frag_shaders}) 

    get_filename_component(name ${f_shader} NAME_WE) # file name w/out extension

    add_custom_command(
      OUTPUT ${scRuntime_BINARY_DIR}/${name}_pass.dep
      COMMAND ${GLSLCC} --fragment ${f_shader} 
      COMMAND touch ${scRuntime_BINARY_DIR}/${name}_pass.dep
      DEPENDS ${GLSLCC} ${f_shader}
      )

    add_custom_target(compile_${name}
      ALL
      DEPENDS ${GLSLCC} ${f_shader} ${scRuntime_BINARY_DIR}/${name}_pass.dep
      )
  endforeach()


  # If all the shaders compiled we then add a custom step in which we
  # convert them into C character string format so they may be
  # embedded within the runtime library.
  file(GLOB gl_shaders opengl/glsl/*.glsl)
  set(GLSL2CSTR ${SCOUT_BINARY_DIR}/bin/glsl2cstr)
  foreach(shader ${gl_shaders}) 

    get_filename_component(name ${shader} NAME_WE) # file name w/out extension

    add_custom_command(
      OUTPUT ${SCOUT_BINARY_DIR}/include/${name}.h
      COMMAND ${GLSL2CSTR} 
        --in ${shader} 
	--out ${SCOUT_BINARY_DIR}/include/${name}.h 
        --name ${name}
      DEPENDS ${GLSL2CSTR} ${shader}
      )

    add_custom_target(generate_${name}
      ALL
      DEPENDS ${SCOUT_BINARY_DIR}/include/${name}.h ${GLSL2CSTR} 
      )
  endforeach()

  # Update the include path to include the generated include files that
  # hold the individual shaders... 
  include_directories(${SCOUT_BINARY_DIR}/include)

endif()

# Check to see if we should add PNG support... 
if(SCOUT_ENABLE_PNG_SUPPORT) 
  file(GLOB png_srcs png/*.cpp)
endif()


if (SCOUT_ENABLE_RT_THREADS) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()


# Finally add the library and dependencies...  Note that if we
# are running w/out PNG/OpenGL support the associated variables
# are empty....  This is the best way (to date) that I've found
# to deal with the cruft of handling multiple configurations of
# the runtime...
add_library(scRuntime STATIC ${rt_srcs} ${gl_srcs} ${png_srcs})
add_dependencies(scRuntime 
  scc
  glsl2cstr
  )
