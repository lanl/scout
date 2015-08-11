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
#  This file should only be included at the top-level of the build
#  configuration, we guard against multiple inclusions from the wrong
#  spot in the hierarchy.
#
#####

if (NOT __SCOUT_CONFIG_CMAKE)
  set(__SCOUT_CONFIG_CMAKE)
else()
  return()
endif()

##### SCOUT SOURCE & BUILD PATHS
#
  set(SCOUT_SRC_DIR            ${PROJECT_SOURCE_DIR})
  set(SCOUT_INCLUDE_DIR        ${PROJECT_SOURCE_DIR}/include)
  set(SCOUT_BUILD_DIR          ${PROJECT_BINARY_DIR})
  set(SCOUT_CONFIG_DIR         ${SCOUT_BUILD_DIR}/config)
  set(SCOUT_LLVM_SRC_DIR       ${SCOUT_SRC_DIR}/llvm)
  set(SCOUT_LLVM_INCLUDE_DIR   ${SCOUT_LLVM_SRC_DIR}/include)
  set(SCOUT_LLVM_BUILD_DIR     ${SCOUT_BUILD_DIR}/llvm)
  set(SCOUT_CLANG_SRC_DIR      ${SCOUT_LLVM_SRC_DIR}/tools/clang)
  set(SCOUT_CLANG_INCLUDE_DIR  ${SCOUT_CLANG_SRC_DIR}/include)
  set(SCOUT_CLANG_BUILD_DIR    ${SCOUT_LLVM_BUILD_DIR}/tools/clang)
  set(SCOUT_CMAKE_DIR          ${PROJECT_SOURCE_DIR}/cmake)

  set(SCOUT_INCLUDE_PATH  
    ${SCOUT_INCLUDE_DIR}
    ${SCOUT_BUILD_DIR}/include
    ${SCOUT_CONFIG_DIR}/include 
    )

  set(SCOUT_LLVM_INCLUDE_PATH
    ${SCOUT_LLVM_INCLUDE_DIR}
    ${SCOUT_LLVM_BUILD_DIR}/include
    )

  set(SCOUT_CLANG_INCLUDE_PATH
    ${SCOUT_CLANG_INCLUDE_DIR}
    ${SCOUT_CLANG_BUILD_DIR}/include
    )

  # Try and coalesce all of our output (built) files into single
  # location -- this gives us a set of tools at build time that we can
  # use w/out the need to do a full-blown install (which can be a bane
  # during active development).
  set(EXECUTABLE_OUTPUT_PATH ${SCOUT_BUILD_DIR}/bin)
  set(LIBRARY_OUTPUT_PATH    ${SCOUT_BUILD_DIR}/lib)
#
#####

##### SCOUT BUILD SETTINGS
# Check to see if we have a build-type set -- if not, default to a
# debugging build...
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
    message(STATUS "scout: Defaulting to release w/ debug configuration...")
  endif()
#
#####

##### SCOUT VERSION INFORMATION
# The following variables are used to help us track Scout
# version information.
#
  set(SCOUT_VERSION_MAJOR      1)
  set(SCOUT_VERSION_MINOR      0)
  set(SCOUT_PATCHLEVEL         0)
  set(SCOUT_VERSION
       "${SCOUT_VERSION_MAJOR}.${SCOUT_VERSION_MINOR}.${SCOUT_PATCHLEVEL}")

  if (${SCOUT_VERSION} MATCHES "[0-9]+\\.[0-9]+\\.[1-9]+")
    set(SCOUT_HAS_VERSION_PATCHLEVEL 1)
    string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([1-9]+)" "\\1"
           SCOUT_PATCHLEVEL ${SCOUT_VERSION})
  else()
    set(SCOUT_HAS_VERSION_PATCHLEVEL 0)
  endif()


  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Scout Programming Language")
  set(CPACK_PACKAGE_VENDOR LANL)
  set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
  set(CPACK_PACKAGE_VERSION_MAJOR "${SCOUT_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${SCOUT_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${SCOUT_PATCHLEVEL}")
#
#####


##### CMAKE MODULE PATHS
# Overhaul the path to CMake moudles so we can find some of our
# own (and LLVM's) configuration details.
#
  set(CMAKE_MODULE_PATH
    "${SCOUT_SRC_DIR}/cmake/Modules"
    "${SCOUT_BUILD_DIR}/cmake/Modules"
    "${SCOUT_LLVM_SRC_DIR}/cmake/modules"
     ${CMAKE_MODULE_PATH}
    )

  message(STATUS "scout: CMake modules search path: ${CMAKE_MODULE_PATH}")
#
#####


##### PACKAGE DEPENDENCIES & CONFIGURATION
# We search for the various external software packages we require, or
# can leverage, in this section. The results of these searches also
# drives Scout's feature set.

  # --- OpenGL support.
  find_package(OpenGL)
  if (OPENGL_FOUND)
    message(STATUS "scout: OpenGL found, enabling support.")
    set(SCOUT_ENABLE_OPENGL ON  CACHE BOOL "Enable OpenGL support.")
    set(SCOUT_HAVE_OPENGL 1)
  else()
    set(SCOUT_HAVE_OPENGL 0)
  endif()

  # --- CUDA support.
  find_package(CUDA)
  if (CUDA_FOUND)
    get_filename_component(_CUDA_LIBRARY_DIR ${CUDA_CUDA_LIBRARY} DIRECTORY)
    set(CUDA_LIBRARY_DIR "${_CUDA_LIBRARY_DIR}" CACHE STRING "Cuda library dir")
    get_filename_component(_CUDA_RTLIBRARY_DIR ${CUDA_CUDART_LIBRARY} DIRECTORY)
    set(CUDA_RTLIBRARY_DIR "${_CUDA_RTLIBRARY_DIR}" CACHE STRING "Cuda rt library dir")
    set(CUDA_LIBRARIES "-L${CUDA_LIBS_PATH} -L${CUDA_RTLIB_PATH} -lcudart -lcuda")

    if (APPLE)
      set(CUDA_LIBRARIES "${CUDA_LIBRARIES} -F/Library/Frameworks")
    endif()

    if (CUDA_VERSION_MAJOR VERSION_GREATER 6 OR CUDA_VERSION_MAJOR VERSION_EQUAL 6)
      message(STATUS "scout: CUDA found, enabling PTX codegen support.")
      message(STATUS "scout: CUDA include path: ${CUDA_INCLUDE_DIRS}")
      set(SCOUT_ENABLE_CUDA ON CACHE BOOL
        "Enable CUDA/PTX code generation and runtime support.")
  
      set(SCOUT_ENABLE_LIB_NVVM OFF CACHE BOOL
          "Enable NVIDIA's compiler SDK vs. LLVM's PTX backend.")
  
      if (SCOUT_ENABLE_LIB_NVVM)
       message(STATUS "scout: Enabling NVIDIA libnvvm support.")
      endif()
   else() #CUDA VERSION
     message(STATUS "scout: CUDA >=6.0 required, disabling support .")
     set(SCOUT_ENABLE_CUDA OFF CACHE BOOL
      "Enable CUDA/PTX code generation and runtime support.")
     set(CUDA_VERSION_MAJOR 0)
     set(CUDA_VERSION_MINOR 0)
     set(CUDA_INCLUDE_DIRS "")
     set(CUDA_LIBRARY_DIR "")
     set(CUDA_RTLIBRARY_DIR "")
     set(CUDA_LIBRARIES "")
   endif() # CUDA_VERSION
  else() # CUDA_FOUND
    message(STATUS "scout: CUDA not found disabling support.")
    set(SCOUT_ENABLE_CUDA OFF CACHE BOOL
      "Enable CUDA/PTX code generation and runtime support.")
    set(CUDA_VERSION_MAJOR 0)
    set(CUDA_VERSION_MINOR 0)
    set(CUDA_INCLUDE_DIRS "")
    set(CUDA_LIBRARY_DIR "")
    set(CUDA_RTLIBRARY_DIR "")
    set(CUDA_LIBRARIES "")
  endif() #CUDA_FOUND

  # --- Legion support
    message(STATUS "scout: enabling Legion by default.")
    set(SCOUT_ENABLE_LEGION ON CACHE BOOL
      "Enable Legion support.")

  # --- OpenCL support.
  #only look for OpenCL if we can't find Cuda
  if(NOT CUDA_FOUND)
    #OpenCL support not currently working on Apple
    if(NOT APPLE)
# disable OpenCL for now
#     find_package(OpenCL)
    endif()
  endif()
  if (OPENCL_FOUND)
    # TODO - should these separated or wrapped into one?
    message(STATUS "scout: OpenCL found, enabling AMDIL codegen support.")
    set(SCOUT_ENABLE_OPENCL ON CACHE BOOL "Enable OpenCL code generation and runtime support.")
    set(SCOUT_ENABLE_AMDIL ON CACHE BOOL "Enable AMD IL code generation and runtime support.")
    #add_definitions(-DSCOUT_ENABLE_OPENCL -DSCOUT_ENABLE_AMDIL)
  else()
    message(STATUS "scout: OpenCL not found, disabling support")
    set(SCOUT_ENABLE_OPENCL OFF CACHE BOOL "Enable OpenCL code generation and runtime support.")
    set(SCOUT_ENABLE_AMDIL OFF CACHE BOOL "Enable AMD IL code generation and runtime support.")
   endif()

  # --- NUMA support.
  #find_package(HWLOC)
  #if (HWLOC_FOUND)
  #    message(STATUS "scout: Found hwloc -- enabling NUMA support.")
  #  set(SCOUT_ENABLE_NUMA ON CACHE BOOL "Enable NUMA (via libhwloc) runtime support.")
  #  if (APPLE)
  #    message(STATUS "scout: Note NUMA support under Mac OS X has limited features.")
  #  endif()
  #else()
  #  message(STATUS "scout: NUMA support not found -- disabling support.")
    set(SCOUT_ENABLE_NUMA OFF CACHE BOOL "Enable NUMA (via libhwloc) runtime support.")
  #endif()

  # --- Thread support.
  find_package(Threads)
  if (Threads_FOUND)
    message(STATUS "scout: Found Threads -- enabling support.")
    set(SCOUT_ENABLE_THREADS ON CACHE BOOL "Enable Threads support.")
  else()
    set(SCOUT_ENABLE_THREADS OFF CACHE BOOL "Enable Threads support.")
  endif()

  # --- MPI support.
  #find_package(MPI)
  #if (MPI_FOUND)
  #  message(STATUS "scout: Found MPI -- enabling support.")
  #  set(SCOUT_ENABLE_MPI ON CACHE BOOL "Enable MPI runtime support.")
  #else()
  set(SCOUT_ENABLE_MPI OFF CACHE BOOL "Enable MPI runtime support.")
  #endif()

  # --- QTsupport.
  find_package(QT5 REQUIRED)
  if (QT5_FOUND)
    if(APPLE)
      # QT5 needs COCOA, IOKIT and COREVIDEO on the mac.
      find_library(COCOA_LIBRARY Cocoa)
      find_library(IOKIT_LIBRARY IOKit)
      find_library(COREVIDEO_LIBRARY CoreVideo)
      find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
      if(COCOA_LIBRARY AND IOKIT_LIBRARY AND COREVIDEO_LIBRARY AND COREFOUNDATION_FRAMEWORK)
          set(SCOUT_ENABLE_QT5 ON CACHE BOOL
            "Enable QT runtime support.")
      else()
          message(STATUS "scout: COCOA, IOKIT, COREVIDEO or COREFOUNDATION not found, disabling QT5 support.")
          set(SCOUT_ENABLE_QT5 OFF CACHE BOOL
            "Enable QT5 runtime support.")
      endif()
    else()    
        message(STATUS "scout: QT5 found, enabling runtime support.")
        set(SCOUT_ENABLE_QT5 ON CACHE BOOL
          "Enable QT5 runtime support.")
    endif()
  else()
    message(STATUS "scout: Required QT5 not found.")
    set(SCOUT_ENABLE_QT5 OFF CACHE BOOL
      "Enable QT5 runtime support.")
  endif()


  # Disable PNG for now -- some Linux systems are having a hard time
  # with matching the API we've used an we haven't had time to fully
  # investigate.  Putting it on the backburner for now...
  find_package(PNG)

  if (PNG_FOUND)
    message(STATUS "scout: Found PNG -- enabling support.")
    set(SCOUT_ENABLE_PNG ON CACHE BOOL "Enable PNG support in scout's runtime libraries.")
    add_definitions(${PNG_DEFINITIONS})
  else()
    set(SCOUT_ENABLE_PNG OFF CACHE BOOL "Enable PNG support in scout's runtime libraries.")
  endif()

# --- THRUST support.
  #find_package(THRUST)

  #if (THRUST_FOUND)
  # set (THRUST_DIR ${THRUST_INCLUDE_DIR}/thrust  CACHE PATH "Thrust directory")
  #  message(STATUS "scout: THRUST found")
  #
  #else ()
  #  message(STATUS "scout: THRUST not found -- disabling support.")
  #  set(SCOUT_ENABLE_CUDA_THRUST OFF CACHE BOOL "Enable THRUST support via CUDA in scout's runtime libraries.")
  #  set(SCOUT_ENABLE_TBB_THRUST OFF CACHE BOOL "Enable THRUST support via TBB in scout's runtime libraries.")
  #  set(SCOUT_ENABLE_CPP_THRUST OFF CACHE BOOL "Enable THRUST support via CPP in scout's runtime libraries.")
  #endif ()

  #if (THRUST_FOUND)
  #  set(SCOUT_ENABLE_CPP_THRUST ON CACHE BOOL "Enable THRUST support via CPP in scout's runtime libraries.")
  #  find_package(CUDA)
  #  if (CUDA_FOUND)
  #    message(STATUS "scout: CUDA found.")
  #  #set(SCOUT_ENABLE_CUDA_THRUST ON CACHE BOOL "Enable THRUST support via CUDA in scout's runtime libraries.")
  #  endif ()

  #  find_package(TBB)
  # if (TBB_FOUND)
  #    message(STATUS "scout: TBB found.")
  #      set(SCOUT_ENABLE_TBB_THRUST ON CACHE BOOL "Enable THRUST support via TBB in scout's runtime libraries.")
  #  endif ()
  #endif ()


# --- lldb supprt
  if(DEFINED ENV{SC_BUILD_LLDB}) 
    find_package(PythonLibs REQUIRED)
    find_package(SWIG REQUIRED)
    if (APPLE) 
      find_package(XCODE REQUIRED)
    else()
      find_package(LIBEDIT REQUIRED)
    endif()
    if(APPLE) 
      if(XCODE_VERSION VERSION_GREATER 5.1.1 OR XCODE_VERSION VERSION_EQUAL 5.1.1)
        if(PYTHONLIBS_FOUND AND SWIG_FOUND) 
          set(SCOUT_ENABLE_LLDB ON CACHE BOOL "Enable building of lldb")
        else()
          message(FATAL_ERROR "scout: Pythonlibs and SWIG required for LLDB, not building")
        endif()
      else()
        message(FATAL_ERROR "scout: xcode 5.1.1 or greater required for LLDB, not building")
      endif()
    else() 
      if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.8 OR CMAKE_C_COMPILER_VERSION VERSION_EQUAL 4.8)
        if(PYTHONLIBS_FOUND AND SWIG_FOUND AND LIBEDIT_FOUND)
          set(SCOUT_ENABLE_LLDB ON CACHE BOOL "Enable building of lldb")
        else()
          message(FATAL_ERROR "scout: Pythonlibs, SWIG and libedit required for LLDB, not building")
        endif()
      else()      
        message(FATAL_ERROR "scout: gcc 4.8 or greater required for LLDB, not building")
      endif()
    endif()
  endif()

  #setup RPATH
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH ${GCC_INSTALL_PREFIX}/lib64 ${SCOUT_BUILD_DIR}/lib ${CUDA_LIBRARY_DIR})


#
#####
