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

#ifndef _SC_CUDA_ERROR_H_
#define _SC_CUDA_ERROR_H_

#include <stdio.h>
#include <cuda.h>

extern "C" {
  void CheckCudaError(char *name, int err);
}

#endif
