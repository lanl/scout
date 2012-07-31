#include "runtime/opencl/scout_opencl.h"

#include <cstring>
#include <cassert>

bool __sc_opencl = false;
cl_context __sc_opencl_context;
cl_program __sc_opencl_program;
cl_command_queue __sc_opencl_command_queue;
cl_device_id __sc_opencl_device_id;

void __sc_init_opencl(){
  __sc_opencl = true;

  cl_uint numPlatforms;

  cl_int ret = clGetPlatformIDs(0, NULL, &numPlatforms);
  assert(ret == CL_SUCCESS);

  assert(numPlatforms > 0 && "No OpenCL platforms available");

  cl_platform_id platform;

  ret = clGetPlatformIDs(1, &platform, NULL);
  assert(ret == CL_SUCCESS && "Error getting OpenCL platform IDs");
  
  cl_uint numDevices;

  ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  assert(ret == CL_SUCCESS && "Error getting # of OpenCL devices");
  assert(numDevices > 0 && "No OpenCL GPU devices");
  
  ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                       1, &__sc_opencl_device_id, NULL);
  assert(ret == CL_SUCCESS && "Error getting first OpenCL device");
  
  __sc_opencl_context = 
    clCreateContext(NULL, 1, &__sc_opencl_device_id, NULL, NULL, &ret);
  assert(ret == CL_SUCCESS && "Error creating OpenCL context");
  
  __sc_opencl_command_queue = 
    clCreateCommandQueue(__sc_opencl_context, __sc_opencl_device_id,
                         0, &ret);
  assert(ret == CL_SUCCESS && "Error creating OpenCL command queue");
}

extern "C"
void __sc_opencl_build_program(const void* bitcode, size_t size){
  // create a stub kernel so we can generate the AMD binary ELF image
  // and fill it in with the real program LLVM IR bitcode
  static const char* stub = "__kernel void stub(__global int* a){}";

  cl_int ret;

  size_t stubSize = strlen(stub);
  cl_program stubProgram = 
    clCreateProgramWithSource(__sc_opencl_context, 1, 
                              (const char**)&stub, &stubSize, &ret);
  assert(ret == CL_SUCCESS && "Error creating OpenCL binary stub");

  clBuildProgram(stubProgram, 1, &__sc_opencl_device_id, 
                 "-fno-bin-amdil -fno-bin-exe", NULL, NULL);

  size_t programSize;
  size_t numDevices = 1;
  ret = clGetProgramInfo(stubProgram, CL_PROGRAM_BINARY_SIZES, 
                         sizeof(size_t), &programSize, &numDevices);
  assert(ret == CL_SUCCESS && "Error reading OpenCL stub size");

  char* programData = (char*)malloc(sizeof(char)*programSize);

  size_t numPrograms = 1;
  ret = 
    clGetProgramInfo(stubProgram, CL_PROGRAM_BINARIES, 
                     programSize, programData, &numPrograms);
  assert(ret == CL_SUCCESS && "Error reading OpenCL stub program");

  // TODO - add the bitcode for our actual program to the ELF image
  // we need to resize the .llvmir section and adjust section
  // offsets

  size_t newProgramSize;
  const unsigned char* newProgramBinary;

  cl_int status;
  
  __sc_opencl_program = 
    clCreateProgramWithBinary(__sc_opencl_context,
                              1,
                              &__sc_opencl_device_id,
                              &newProgramSize,
                              &newProgramBinary,
                              &status,
                              &ret);

  assert(status == CL_SUCCESS && ret == CL_SUCCESS &&
         "Error creating OpenCL program");
}

extern "C"
void* __sc_get_opencl_device_ptr(const char* meshName,
                                 const char* fieldName){
  
}

extern "C"
void __sc_put_opencl_device_ptr(const char* meshName,
                                const char* fieldName,
                                void* memPtr){

}

