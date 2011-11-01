/**
 * @file   PTXBackendTestSuite.cpp
 * @date   08.08.2009
 * @author Helge Rhodin
 *
 *
 * Copyright (C) 2009, 2010 Saarland University
 *
 * This file is part of llvmptxbackend.
 *
 * llvmptxbackend is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * llvmptxbackend is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with llvmptxbackend.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <cuda/cuda.h>
#include <iostream>
#include <cstdio>

#include "PTXTestFunctions.cpp"

#define check(ERROR) check(ERROR, __FILE__, __LINE__)

const char *cuErrorString(CUresult error) {
  switch(error) {
  case CUDA_SUCCESS: return "No errors.";
  case CUDA_ERROR_INVALID_VALUE: return "Invalid value.";
  case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory.";
  case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized.";
  case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized.";
  case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available.";
  case CUDA_ERROR_INVALID_DEVICE: return "Invalid device.";
  case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image.";
  case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context.";
  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current.";
  case CUDA_ERROR_MAP_FAILED: return "Map failed.";
  case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed.";
  case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped.";
  case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped.";
  case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU.";
  case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired.";
  case CUDA_ERROR_NOT_MAPPED: return "Not mapped.";
  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "Mapped resource not available for access as an array.";
  case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "Mapped resource not available for access as a pointer.";
  case CUDA_ERROR_INVALID_SOURCE: return "Invalid source.";
  case CUDA_ERROR_FILE_NOT_FOUND: return "File not found.";
  case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle.";
  case CUDA_ERROR_NOT_FOUND: return "Not found.";
  case CUDA_ERROR_NOT_READY: return "CUDA not ready.";
  case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed.";
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources.";
  case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout.";
  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing.";
  case CUDA_ERROR_POINTER_IS_64BIT: return "Attempted to retrieve 64-bit pointer via 32-bit API function.";
  case CUDA_ERROR_SIZE_IS_64BIT: return "Attempted to retrieve 64-bit size via 32-bit API function.";
  case CUDA_ERROR_UNKNOWN: return "Unknown error.";
  default: return "Bad error.";
  }
}

CUresult initialize(int device, CUcontext *phContext, CUdevice *phDevice, CUmodule *phModule, CUstream *phStream)
{
  // Initialize the device and create the context
  cuInit(0);
  cuDeviceGet(phDevice, device);
  CUresult status = cuCtxCreate(phContext, 0, *phDevice);
  if (status != CUDA_SUCCESS)
    {std::cout << "ERROR: could not create context\n"; exit(0);}

    status = cuModuleLoad(phModule, "PTXTestFunctions.o.ptx");

  if (status != CUDA_SUCCESS)
    {std::cout << "ERROR: could not load .ptx module: " << cuErrorString(status) << "\n"; exit(0);}

  // Create stream
  status = cuStreamCreate(phStream, 0);
  if (status != CUDA_SUCCESS)
    {printf("ERROR: during stream creation\n"); exit(0);}

  return status;
}

CUresult loadAndRunTestFunction(CUmodule *phModule, std::string name, CUdeviceptr &d_data,
                                DataStruct *h_data, unsigned int memSize,
                                int thread_x=1,int thread_y=1,int thread_z=1,
                                int block_x=1, int block_y=1, int block_z=1)
{
  //  std::cout << "  Start Loading" << std::endl;

  // load data the to device
  cuMemcpyHtoD(d_data, h_data, memSize);

  // Locate the kernel entry point
  CUfunction phKernel = 0;
  CUresult status = cuModuleGetFunction(&phKernel, *phModule, name.data());
   if (status != CUDA_SUCCESS)
     {printf("ERROR: could not load function\n");}

  // Set the kernel parameters
  status = cuFuncSetBlockShape(phKernel, thread_x, thread_y, thread_z);
   if (status != CUDA_SUCCESS)
     {printf("ERROR: during setBlockShape\n");}

  int paramOffset = 0;
  status = cuParamSetv(phKernel, paramOffset, &d_data, sizeof(DataStruct*));
  paramOffset += sizeof(DataStruct*);
  status = cuParamSetSize(phKernel, paramOffset);
   if (status != CUDA_SUCCESS)
     {printf("ERROR: during cuParamSetv\n");}

  // Launch the kernel
  status = cuLaunchGrid(phKernel, block_x, block_y);
  if (status != CUDA_SUCCESS)
    {printf("ERROR: during grid launch\n");}

  //  std::cout << "  launched CUDA kernel!!" << std::endl;

  // Copy the result back to the host
  status = cuMemcpyDtoH(h_data, d_data, memSize);
  if (status != CUDA_SUCCESS)
    {printf("ERROR: during MemcpyDtoH\n");}
}

CUresult loadAndRunDualTestFunction(CUmodule *phModule, std::string name, CUdeviceptr &d_data0,
                                CUdeviceptr &d_data1,
                                DataStruct *h_data0,
                                DataStruct *h_data1,
                                unsigned int memSize,
                                int thread_x=1,int thread_y=1,int thread_z=1,
                                int block_x=1, int block_y=1, int block_z=1)
{
  //  std::cout << "  Start Loading" << std::endl;

  // load data the to device
  cuMemcpyHtoD(d_data0, h_data0, memSize);
  cuMemcpyHtoD(d_data1, h_data1, memSize);

  // Locate the kernel entry point
  CUfunction phKernel = 0;
  CUresult status = cuModuleGetFunction(&phKernel, *phModule, name.data());
   if (status != CUDA_SUCCESS)
     {printf("ERROR: could not load function\n");}

  // Set the kernel parameters
  status = cuFuncSetBlockShape(phKernel, thread_x, thread_y, thread_z);
   if (status != CUDA_SUCCESS)
     {printf("ERROR: during setBlockShape\n");}

  int paramOffset = 0, size=0;

  size = sizeof(CUdeviceptr);
  status = cuParamSetv(phKernel, paramOffset, &d_data0, size);
  paramOffset += size;
  status = cuParamSetv(phKernel, paramOffset, &d_data1, size);
  paramOffset += size;



  status = cuParamSetSize(phKernel, paramOffset);
   if (status != CUDA_SUCCESS)
     {printf("ERROR: during cuParamSetv\n");}

  // Launch the kernel
  status = cuLaunchGrid(phKernel, block_x, block_y);
  if (status != CUDA_SUCCESS)
    {printf("ERROR: during grid launch\n");}

  //  std::cout << "  launched CUDA kernel!!" << std::endl;

  // Copy the result back to the host
  status = cuMemcpyDtoH(h_data0, d_data0, memSize);
  status = cuMemcpyDtoH(h_data1, d_data1, memSize);
  if (status != CUDA_SUCCESS)
    {printf("ERROR: during MemcpyDtoH\n");}
}

void setZero(DataStruct *h_data, DataStruct *h_data_reference)
{
  h_data->i = h_data_reference->i = 0;
  h_data->f = h_data_reference->f = 0;
  h_data->d = h_data_reference->d = 0;
  h_data->u = h_data_reference->u = 0;
  for(int i=0; i<ARRAY_N; i++)
    h_data->ia[i] = h_data_reference->ia[i] = 0;
  for(int i=0; i<ARRAY_N; i++)
    h_data->fa[i] = h_data_reference->fa[i] = 0;

}

bool compareData(DataStruct *h_data, DataStruct *h_data_reference)
{
  bool correct = true;
  if(h_data->i != h_data_reference->i) {
    std::cout << "  DIFFERENCE: data.i: DEVICE " << h_data->i << " != " << h_data_reference->i << " REFERENCE\n";
    correct = false;}
  if(abs(h_data->f / h_data_reference->f-1)>0.001) {// || (abs(h_data_reference->f)<0.01 && abs(h_data->f-h_data_reference->f)>0.001)) {
    std::cout << "  DIFFERENCE: data.f: DEVICE " << h_data->f << " != " << h_data_reference->f << " REFERENCE\n";
    correct = false;}
  if(h_data->d != h_data_reference->d) {
    std::cout << "  DIFFERENCE: data.d: DEVICE " << h_data->d << " != " << h_data_reference->d << " REFERENCE\n";
    correct = false;}
  for(int i=0; i<ARRAY_N; i++)
  {
    if(h_data->ia[i] != h_data_reference->ia[i]) {
      std::cout << "  DIFFERENCE: data.ia[" << i << "]: DEVICE " << h_data->ia[i] << " != " << h_data_reference->ia[i] << " REFERENCE\n";
      correct = false;}
  }
  for(int i=0; i<ARRAY_N; i++)
  {
    if(abs(h_data->fa[i] - h_data_reference->fa[i])>0.001 || (abs(h_data->fa[i])<0.01 && abs(h_data->fa[i]-h_data_reference->fa[i])>0.001)) {
      std::cout << "  DIFFERENCE: data.fa[" << i << "]: DEVICE " << h_data->fa[i] << " != " << h_data_reference->fa[i] << " REFERENCE\n";
      correct = false;
    }
  }
  return correct;
}

typedef void (*scalarFnType)(DataStruct*);

void runHostTestFunction(scalarFnType funHost, DataStruct* h_data_reference,
                         int thread_x=1,int thread_y=1,int thread_z=1,
                         int block_x=1, int block_y=1, int block_z=1,
                         int grid=1)
{
  __ptx_sreg_ntid_x = thread_x;
  __ptx_sreg_ntid_y = thread_y;
  __ptx_sreg_ntid_z = thread_z;

  __ptx_sreg_nctaid_x = block_x;
  __ptx_sreg_nctaid_y = block_y;
  __ptx_sreg_nctaid_z = block_z;

//unsigned short __ptx_sreg_clock; TODO

  for(int g=0; g<grid; ++g)
  {
    __ptx_sreg_gridid = g;
    for(int bx=0; bx<block_x; ++bx)
    for(int by=0; by<block_y; ++by)
    for(int bz=0; bz<block_z; ++bz)
    {
      __ptx_sreg_ctaid_x = bx; __ptx_sreg_ctaid_y = by; __ptx_sreg_ctaid_z = bz;
      for(int tx=0; tx<thread_x; ++tx)
      for(int ty=0; ty<thread_y; ++ty)
      for(int tz=0; tz<thread_z; ++tz)
      {
        __ptx_sreg_tid_x = tx; __ptx_sreg_tid_y = ty; __ptx_sreg_tid_z = tz;

        funHost(h_data_reference);
      }
    }
  }
}

bool runRalfFunction(std::string name, scalarFnType fun, CUmodule* hModule, CUdeviceptr d_data,
                     DataStruct *h_data,DataStruct* h_data_reference, unsigned int memSize)
{
  const unsigned inputNr = 10;
  const float scalarInputs[4][inputNr] = {{ 0.f, 3.f, 2.f, 8.f, 10.2f, -1.f, 0.f, 1000.23f, 0.02f, -0.02f },
                                           { 1.f, 2.f, 4.f, 6.f, -14.13f, -13.f, 0.f, 0.02f, 420.001f, -420.001f },
                                           { 2.f, 1.f, 6.f, 4.f, 999.f, -5.f, 0.f, 420.001f, 0.01f, 0.01f },
                                           { 3.f, 0.f, 8.f, 2.f, 0.f, -420.001f, 0.f, 0.01f, 1000.23f, 0.01f }};


  std::cout << "====================== " << name << "===============================\n";
  for (unsigned i=0; i<inputNr; ++i) {
    for (unsigned j=0; j<inputNr; ++j)
    for (unsigned k=0; k<4; ++k)
    {
      setZero(h_data,h_data_reference);
      h_data->fa[0] = h_data_reference->fa[0] = scalarInputs[k][i];
      h_data->fa[1] = h_data_reference->fa[1] = scalarInputs[k][j];

      //run device function
      loadAndRunTestFunction(hModule, name, d_data, h_data, memSize);
      fun(h_data_reference);

      if(!compareData(h_data, h_data_reference))                      //compare Data
      {
        std::cout << "\n Error in Ralf: fa0=" << scalarInputs[k][i]
                  << ", fa1=" << scalarInputs[k][j] << " (" << name << ")\n";
        return false;
      }
    }
  }
  std::cout << " => Test passed!!!\n";
  return true;
}

//dummy functions
float __half2float(short s){return (float)s;}
__m128 __ptx_tex1D(float* ptr, float coordinate){return *(__m128*)ptr;}
__m128 __ptx_tex2D(float* ptr, __m128 coordinates){return *(__m128*)ptr;}
__m128 __ptx_tex3D(float* ptr, __m128 coordinates){return *(__m128*)ptr;}

/*
__m128i __ptx_tex1D(int* ptr, float coordinate){return *(__m128*)ptr;}
__m128i __ptx_tex2D(int* ptr, __m64 coordinates){return *(__m128*)ptr;}
__m128i __ptx_tex3D(int* ptr, __m128 coordinates){return *(__m128*)ptr;}
*/

int main(int argc, char **argv)
{
  //data
  CUdeviceptr  d_data0   = 0;
  CUdeviceptr  d_data1   = 0;
  DataStruct *h_data0  = 0;
  DataStruct *h_data1  = 0;
  DataStruct h_data_reference0;
  DataStruct h_data_reference1;
  unsigned int memSize = sizeof(DataStruct);

  //device references
  CUcontext    hContext = 0;
  CUdevice     hDevice  = 0;
  CUmodule     hModule  = 0;
  CUstream     hStream  = 0;

  // Initialize the device and get a handle to the kernel
  CUresult status = initialize(0, &hContext, &hDevice, &hModule, &hStream);

  // Allocate memory on host and device
  if ((h_data0 = (DataStruct *)malloc(memSize)) == NULL)
    {
      std::cerr << "Could not allocate host memory" << std::endl;
      exit(-1);
    }
  status = cuMemAlloc(&d_data0, memSize);

  if ((h_data1 = (DataStruct *)malloc(memSize)) == NULL)
    {
      std::cerr << "Could not allocate host memory" << std::endl;
      exit(-1);
    }
  status = cuMemAlloc(&d_data1, memSize);
  if (status != CUDA_SUCCESS)
    printf("ERROR: during cuMemAlloc\n");

  ///////////////////////////////////////////////////////////////////////////////
  //======================= test cases ========================================//
  ///////////////////////////////////////////////////////////////////////////////
  std::string name = "";
  unsigned int testnum=0;
  unsigned int passed=0;

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  /////////////////////// Ralf ///////////////////////////////////////////////////
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  if(runRalfFunction("test_phi_scalar", test_phi_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi2_scalar", test_phi2_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi3_scalar", test_phi3_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi4_scalar", test_phi4_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi5_scalar", test_phi5_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi6_scalar", test_phi6_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi7_scalar", test_phi7_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi8_scalar", test_phi8_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_phi9_scalar", test_phi9_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;

  if(runRalfFunction("test_loopbad_scalar", test_loopbad_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_loop23_scalar", test_loop23_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;
  if(runRalfFunction("test_loop13_scalar", test_loop13_scalar, &hModule, d_data0, h_data0, &h_data_reference0, memSize))
    passed++;
  testnum++;

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_GetElementPointer_constant"; /////////////////////
  setZero(h_data0,&h_data_reference0);

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_GetElementPointer_constant(&h_data_reference0);                //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  ///////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_calculate"; /////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 3;
  h_data0->f = h_data_reference0.f = 3.2;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_calculate(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_parquetShader"; /////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->f = h_data_reference0.f = 1;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_parquetShader(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_GetElementPointer_dyn"; /////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 3;
  h_data0->u = h_data_reference0.u = 7;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_GetElementPointer_dyn(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_branch_simple"; // Branch 1 /////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->f = h_data_reference0.f = -4;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_simple(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_branch_simple"; // Branch 2 /////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->f = h_data_reference0.f = 8;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_simple(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_branch_simplePHI"; // Branch 1 /////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->f = h_data_reference0.f = -10;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_simplePHI(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_branch_loop"; //////////////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 100;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_loop(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_math"; //////////////////////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->f = h_data_reference0.f = 1.4;
  h_data0->i = h_data_reference0.i = 3;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_math(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_signedOperands"; //////////////////////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 3;
  h_data0->f = h_data_reference0.f = -7;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_signedOperands(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;


  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_constantOperands"; //////////////////////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 3;
  h_data0->f = h_data_reference0.f = -1.44;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_constantOperands(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_branch_loop_semihard"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 10;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_loop_semihard(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_branch_loop_hard"; // Branch 1 /////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 1;
  h_data0->u = h_data_reference0.u = 3;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_loop_hard(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*////////////*/ name = "test_branch_loop_hard"; // Branch 2 /////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 7;
  h_data0->u = h_data_reference0.u = 10;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_branch_loop_hard(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_binaryInst"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 5;
  h_data0->f = h_data_reference0.f = -121.23;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_binaryInst(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_selp"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = -15;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_selp(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_GetElementPointer_complicated"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 1;
  h_data_reference0.s.s.f = h_data0->s.s.f = 3.11;
  h_data_reference0.s.sa[2].f = h_data0->s.sa[2].f = -4.32;
  h_data_reference0.s.sa[h_data0->i].f = h_data0->s.sa[h_data0->i].f = 111.3;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_GetElementPointer_complicated(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_call"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 10;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_call(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*/////////////*/ name = "test_alloca"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 1;
  h_data0->f = h_data_reference0.f = -3.23;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_alloca(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                     //compare Data
    {passed++; std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_alloca_complicated"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  h_data0->i = h_data_reference0.i = 1;
  h_data0->f = h_data_reference0.f = 23.213;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_alloca_complicated(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;


  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_globalVariables"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_globalVariables(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_specialRegisters_x"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize, 2,3,4, 2,3);   //run device function
  runHostTestFunction(test_specialRegisters_x, &h_data_reference0,   2,3,4, 2,3);   //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;


  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_specialRegisters_y"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize, 2,3,4, 2,3);   //run device function
  runHostTestFunction(test_specialRegisters_x, &h_data_reference0,   2,3,4, 2,3);   //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_dualArgument"; /////////////////////////
  setZero(h_data0,&h_data_reference0);
  setZero(h_data1,&h_data_reference1);

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunDualTestFunction(&hModule, name, d_data0, d_data1, h_data0, h_data1, memSize);   //run device function

  test_dualArgument(&h_data_reference0,&h_data_reference1);   //run host reference
  if(compareData(h_data0,&h_data_reference0))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  if(compareData(h_data1,&h_data_reference1))                      //compare Data
    {passed++;  std::cout << " => Test passed!!!\n";}
  testnum++;  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_vector"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  h_data0->fa[0] = h_data_reference0.fa[0] = 0.43f;
  h_data0->fa[1] = h_data_reference0.fa[1] = 0.234f;
  h_data0->fa[2] = h_data_reference0.fa[2] = 12893.f;
  h_data0->fa[3] = h_data_reference0.fa[3] = 13.33f;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_vector(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                     //compare Data
    {passed++; std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_reg2Const"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  /*
  unsigned int bytes; //size of constant
  CUdeviceptr devptr_const=0;
  status = cuModuleGetGlobal(&devptr_const,
                             &bytes,
                             hModule, "__ptx_constant_data_global");

  cuMemcpyHtoD(devptr_const, h_data0, memSize);
  */

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_reg2Const(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                     //compare Data
    {passed++; std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_constantMemory"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  h_data0->fa[0] = __ptx_constant_data_global.fa[0] = 0.2348f;

  unsigned int bytes; //size of constant
  CUdeviceptr devptr_const=0;
  status = cuModuleGetGlobal(&devptr_const,
                             &bytes,
                             hModule, "__ptx_constant_data_global");

  cuMemcpyHtoD(devptr_const, h_data0, memSize);

  setZero(h_data0,&h_data_reference0);

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  test_constantMemory(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                     //compare Data
    {passed++; std::cout << " => Test passed!!!\n";}
  testnum++;


  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_sharedMemory"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  for(int i = 0; i < ARRAY_N/2; i++)
    h_data0->fa[i*2] = i;

  for(int i = 0; i < ARRAY_N/2; i++)
    h_data0->fa[i*2+1] = -i;

  std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
  loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize, 32,1,1, 1,1);   //run device function

  for(int i = 0; i < ARRAY_N/2; i++)
    h_data_reference0.fa[i] = i;
  for(int i = 0; i < ARRAY_N/2; i++)
    h_data_reference0.fa[i+32] = -i;
  //  runHostTestFunction(test_sharedMemory, &h_data_reference0, 16,1,1, 1,1);                                 //run host reference

  if(compareData(h_data0,&h_data_reference0))                     //compare Data
    {passed++; std::cout << " => Test passed!!!\n";}
  testnum++;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /*///////////////*/ name = "test_lightShader"; /////////////////////////
  setZero(h_data0,&h_data_reference0);

  /*
  unsigned int bytes; //size of constant
  CUdeviceptr devptr_const=0;
  status = cuModuleGetGlobal(&devptr_const,
                             &bytes,
                             hModule, "__ptx_constant_data_global");

  cuMemcpyHtoD(devptr_const, h_data0, memSize);
  */

  if(0) {
    std::cout << "=============== Test " << testnum << ": " << name << " ===================\n";
    loadAndRunTestFunction(&hModule, name, d_data0, h_data0, memSize);   //run device function
  }

  /*
  test_lightShader(&h_data_reference0);                                 //run host reference
  if(compareData(h_data0,&h_data_reference0))                     //compare Data
    {passed++; std::cout << " => Test passed!!!\n";}
  testnum++;
  */

  ///////////////////////////////////////////////////////////////////////////////
  //======================= test cases END ====================================//
  ///////////////////////////////////////////////////////////////////////////////

  // Check the result
  std::cout << "\nPASSED " << passed << " tests" << std::endl;
  std::cout << "FAILED " << (testnum-passed) << " tests" << std::endl;

  // Cleanup
  if (d_data0)
    {
      cuMemFree(d_data0);
      d_data0 = 0;
    }
  if (d_data1)
    {
      cuMemFree(d_data1);
      d_data1 = 0;
    }
  if (h_data0)
    {
      free(h_data0);
      h_data0 = 0;
    }
  if (h_data1)
    {
      free(h_data1);
      h_data1 = 0;
    }
  if (hModule)
    {
      cuModuleUnload(hModule);
      hModule = 0;
    }
  if (hStream)
    {
      cuStreamDestroy(hStream);
      hStream = 0;
    }
  if (hContext)
    {
      cuCtxDestroy(hContext);
      hContext = 0;
    }
  return 0;
}
