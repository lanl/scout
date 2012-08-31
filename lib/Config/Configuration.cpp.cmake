/*
 *	
 *###########################################################################
 * Copyrigh (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */

#include "scout/Config/Configuration.h"

// These macros give us an easy way to deal with the CMake settings
// for 'on' and 'off' within our code.  This of course requires we use
// a consistent naming convention for enable (ON) vs. disabled (OFF)
// build configurations.
#define OFF false
#define ON  true

// Access to all of our configuration-centric settings should live
// within the scout::config namesapce. 
namespace scout {
  
  namespace config {
  
    bool Configuration::OpenGLSupport = ${SC_ENABLE_OPENGL};
    bool Configuration::CUDASupport   = ${SC_ENABLE_CUDA};
    bool Configuration::NUMASupport   = ${SC_ENABLE_NUMA};
    bool Configuration::MPISupport    = ${SC_ENABLE_MPI};

    //bool Configuration::OpenCLSupport = ${SC_ENABLE_OPENCL};


    int Configuration::CudaVersion[2] = {
      ${CUDA_VERSION_MAJOR}, 
      ${CUDA_VERSION_MINOR}
    }; 
  
    const char* Configuration::IncludePaths[] = {

      "-I${CMAKE_INSTALL_PREFIX}/include",

      #ifdef SC_ENABLE_OPENGL
      
      "-I${SDL_INCLUDE_DIR}",
    
      #ifndef APPLE  // We'll use frameworks on Mac OS X.
      "-I${OPENGL_INCLUDE_DIR}",  
      #endif
      
      #endif // OPENGL
    
      #ifdef SC_ENABLE_CUDA
      "-I${CUDA_INCLUDE_DIRS}",
      #endif

      #ifdef SC_ENABLE_NUMA 
      "-I${HWLOC_INCLUDE_DIR}",
      #endif

      #ifdef SC_ENABLE_MPI 
      "-I${MPI_C_INCLUDE_PATH}",
      #endif

      0 // end of include paths. 
    };

    const char* Configuration::LibraryPaths[] = {

      "-L${CMAKE_INSTALL_PREFIX}/lib",
      "-L${SDL_LIBRARY_DIR}",    

      #ifdef SC_ENABLE_OPENGL
    
      #ifdef APPLE
      "-framework OpenGL",
      #else
      // Ignore this on Mac OS X (frameworks handle details).      
      "${OPENGL_gl_LIBRARY}",
      #endif // APPLE 

      #endif  // OpenGL 
    
      #ifdef SC_ENABLE_CUDA    
      "-L${CUDA_LIBRARY_DIR}", 
      #endif

      #ifdef SC_ENABLE_NUMA 
      "-L${HWLOC_LIBRARY_DIR}",
      #endif

      #ifdef SC_ENABLE_MPI
      // Note MPI includes compiler flags in dirs string.       
      "${MPI_C_LINK_DIRS}",
      #endif
    
      0 // mark end of library paths.
    };
  
    const char* Configuration::Libraries[] = {

      "-lscRuntime -lscStandard",
      #ifdef SC_ENABLE_CUDA
      "-lscCudaError",
      #endif

      "${SDL_LIBRARIES}",

      #ifdef APPLE 
      "-framework Cocoa", 
      "-framework Foundation", 
      #endif

      #ifdef SC_ENABLE_OPENGL

      // Frameworks handle MacOS X details for us so we skip details here.
      #ifndef APPLE
      "-lGLU -lGL",
      #endif

      #endif // OPENGL 
    
      #ifdef SC_ENABLE_CUDA
      "${CUDA_LIBRARIES} -lscCudaError",
      #endif

      #ifdef SC_ENABLE_NUMA
      "${HWLOC_LIBRARIES}",
      #endif

      #ifdef SC_ENABLE_MPI
      // Note MPI includes compiler flags in libs string.      
      "${MPI_C_LINK_LIBS}", 
      #endif
    
      0 // mark end of library paths.      
    };
  }
}
