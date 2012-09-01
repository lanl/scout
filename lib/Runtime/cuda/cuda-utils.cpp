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
#include <iostream>

#include "scout/Runtime/cuda/Cuda.h"

using namespace std;

// ----- cuda_error_check
//
void cuda_error_check(cudaError_t err, const char* file, int line)
{
  if (err != cudaSuccess) {
    cerr << "cuda runtime error:\n";
    cerr << "  " << file << "(" << line << "): "
         << cudaGetErrorString(err) << endl;
    abort();
  }
}

    
    
