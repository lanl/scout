#include <cassert>
#include <map>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cctype>

#include "runtime/opengl/opengl.h"
#include "runtime/scout_gpu.h"

using namespace std;

namespace{

typedef map<const void*, CUmodule> ModuleMap;
typedef map<string, CUdeviceptr> FieldMap;

 struct Mesh{
   FieldMap fieldMap;
 };

 typedef map<string, Mesh*> MeshMap;

 static ModuleMap _moduleMap;
 static MeshMap _meshMap;
 
} // end namespace

bool __sc_gpu = false;

CUdevice __sc_device;
CUcontext __sc_device_context;
CUgraphicsResource __sc_device_resource;
CUstream __sc_device_stream;

void __sc_init_cuda() {
  __sc_gpu = true;

  // Initialize CUDA Driver API.
  assert(cuInit(0) == CUDA_SUCCESS);

  // Acquire a GPU device.
  assert(cuDeviceGet(&__sc_device, 0) == CUDA_SUCCESS);

  // Create a CUDA context for interoperability with OpenGL.
  assert(cuGLCtxCreate(&__sc_device_context, 0, __sc_device) ==
	 CUDA_SUCCESS);
}

void __sc_register_gpu_pbo(GLuint pbo, unsigned int flags){
  assert(cuGraphicsGLRegisterBuffer(&__sc_device_resource, pbo, flags) ==
	 CUDA_SUCCESS);
}


extern "C"
CUresult __sc_get_gpu_module(CUmodule* module, const void* image){
  ModuleMap::iterator itr = _moduleMap.find(image);
  if(itr == _moduleMap.end()){
    CUresult result = cuModuleLoadData(module, image);

    if(result != CUDA_SUCCESS){
      return result;
    }

    _moduleMap[image] = *module;

    return CUDA_SUCCESS;
  }

  *module = itr->second;

  return CUDA_SUCCESS;
}

extern "C"
CUdeviceptr __sc_get_gpu_device_ptr(const char* meshName,
				    const char* fieldName){
  MeshMap::iterator itr = _meshMap.find(meshName);
  Mesh* mesh;
  if(itr == _meshMap.end()){
    mesh = new Mesh;
    _meshMap[meshName] = mesh;
    return 0;
  }
  mesh = itr->second;

  FieldMap::iterator fitr = mesh->fieldMap.find(fieldName);
  if(fitr == mesh->fieldMap.end()){
    return 0;
  }

  return fitr->second;
}

extern "C"
void __sc_put_gpu_device_ptr(const char* meshName,
			     const char* fieldName,
			     CUdeviceptr ptr){
  MeshMap::iterator itr = _meshMap.find(meshName);
  assert(itr != _meshMap.end());
  Mesh* mesh = itr->second;
  mesh->fieldMap[fieldName] = ptr;
}

