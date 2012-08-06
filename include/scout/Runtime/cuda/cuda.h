/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#ifndef SCOUT_CUDA_H_
#define SCOUT_CUDA_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef SC_ENABLE_OPENGL
#include <cuda_gl_interop.h>
#endif

extern void cuda_error_check(cudaError_t err, const char* file, int line);

// Todo: It would probably be nice to have an option of turning on or
// off these checks in the build configuration. --psm 
#define CUDAErrorCheck(func)  cuda_error_check((func), __FILE__, __LINE__)

#endif


