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

#include "llvm/Transforms/Scout/CudaError/CudaError.h"

void CheckCudaError(char *name, int err) {
  if(err == CUDA_SUCCESS) return;

  printf("CUDA Error in %s: ", name);
  if(err == CUDA_ERROR_INVALID_VALUE)                      printf("Invalid value.");
  else if(err == CUDA_ERROR_OUT_OF_MEMORY)                 printf("Out of memory.");
  else if(err == CUDA_ERROR_NOT_INITIALIZED)               printf("Driver not initialized.");
  else if(err == CUDA_ERROR_DEINITIALIZED)                 printf("Driver deinitialized.");
  else if(err == CUDA_ERROR_NO_DEVICE)                     printf("No CUDA-capable device available.");
  else if(err == CUDA_ERROR_INVALID_DEVICE)                printf("Invalid device.");
  else if(err == CUDA_ERROR_INVALID_IMAGE)                 printf("Invalid kernel image.");
  else if(err == CUDA_ERROR_INVALID_CONTEXT)               printf("Invalid context.");
  else if(err == CUDA_ERROR_CONTEXT_ALREADY_CURRENT)       printf("Context already current.");
  else if(err == CUDA_ERROR_MAP_FAILED)                    printf("Map failed.");
  else if(err == CUDA_ERROR_UNMAP_FAILED)                  printf("Unmap failed.");
  else if(err == CUDA_ERROR_ARRAY_IS_MAPPED)               printf("Array is mapped.");
  else if(err == CUDA_ERROR_ALREADY_MAPPED)                printf("Already mapped.");
  else if(err == CUDA_ERROR_NO_BINARY_FOR_GPU)             printf("No binary for GPU.");
  else if(err == CUDA_ERROR_ALREADY_ACQUIRED)              printf("Already acquired.");
  else if(err == CUDA_ERROR_NOT_MAPPED)                    printf("Not mapped.");
  else if(err == CUDA_ERROR_INVALID_SOURCE)                printf("Invalid source.");
  else if(err == CUDA_ERROR_FILE_NOT_FOUND)                printf("File not found.");
  else if(err == CUDA_ERROR_INVALID_HANDLE)                printf("Invalid handle.");
  else if(err == CUDA_ERROR_NOT_FOUND)                     printf("Not found.");
  else if(err == CUDA_ERROR_NOT_READY)                     printf("CUDA not ready.");
  else if(err == CUDA_ERROR_LAUNCH_FAILED)                 printf("Launch failed.");
  else if(err == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)       printf("Launch exceeded resources.");
  else if(err == CUDA_ERROR_LAUNCH_TIMEOUT)                printf("Launch exceeded timeout.");
  else if(err == CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING) printf("Launch with incompatible texturing.");
  else if(err == CUDA_ERROR_UNKNOWN)                       printf("Unknown error.");
  else                                                     printf("Returned code not identifiable by CUDA.");
  printf("  Exiting...\n");
  exit(1);
}
