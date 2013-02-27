/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
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
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 
#include <cassert>
#include <map>
#include <string>
using namespace std;

#include "scout/Runtime/cuda/CudaUtilities.h"

using namespace scout;

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

bool __sc_cuda = false;
CUgraphicsResource __sc_cuda_device_resource;

// hooks called from lib/Compiler/llvm/Transforms/Driver/CudaDriver.cpp

extern "C"
CUresult __sc_get_cuda_module(CUmodule* module, const void* image){
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
CUdeviceptr __sc_get_cuda_device_ptr(const char* meshName,
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
void __sc_put_cuda_device_ptr(const char* meshName,
                              const char* fieldName,
                              CUdeviceptr ptr){
  MeshMap::iterator itr = _meshMap.find(meshName);
  assert(itr != _meshMap.end());
  Mesh* mesh = itr->second;
  mesh->fieldMap[fieldName] = ptr;
}

