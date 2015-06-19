/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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

#include "scout/Runtime/gpu/GPURuntime.h"

#include <iostream>

#include <map>
#include <vector>
#include <cassert>
#include <cmath>

//#include <sys/time.h>

#include <cuda.h>

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

using namespace std;
using namespace scout;

static const uint8_t FIELD_READ = 0x01;
static const uint8_t FIELD_WRITE = 0x02;

static const uint8_t FIELD_CELL = 0;
static const uint8_t FIELD_VERTEX = 1;
static const uint8_t FIELD_EDGE = 2;
static const uint8_t FIELD_FACE = 3;

static const size_t DEFAULT_THREADS = 128;

/*
namespace{

void sleep(double dt){
  double sec = floor(dt);
  double fsec = dt - sec;
  
  timespec ts;
  ts.tv_sec = sec;
  ts.tv_nsec = fsec*1e9;
  
  nanosleep(&ts, 0);
}

double now(){
  timeval tv;
  gettimeofday(&tv, 0);
  
  return tv.tv_sec + tv.tv_usec/1e6;
}

} // end namespace
*/

class RenderallRuntime : public GPURuntime{
public:
  static void check(CUresult err){
    if(err != CUDA_SUCCESS){
      const char* s;
      cuGetErrorString(err, &s);
      cerr << "RenderallRuntime error: " << s << endl;
      assert(false);
    }
  }

  class MeshField{
  public:
    MeshField(void* hostPtr, CUdeviceptr devPtr, size_t size)
      : hostPtr(hostPtr),
        devPtr(devPtr),
        size(size){

    }

    ~MeshField(){
      CUresult err = cuMemFree(devPtr);
      check(err);
    }

    void* hostPtr;
    CUdeviceptr devPtr;
    size_t size;
  };

  class Mesh{
  public:
    Mesh(uint32_t width, uint32_t height, uint32_t depth)
      : width_(width),
        height_(height),
        depth_(depth){

      if(depth_ > 1){
        rank_ = 3;
      }
      else if(height_ > 1){
        rank_ = 2;
      }
      else{
        rank_ = 1;
      }
    }

    ~Mesh(){
      for(auto& itr : meshFieldMap_){
        delete itr.second;
      }
    }

    MeshField* getField(const char* name){
      auto itr = meshFieldMap_.find(name);
      if(itr != meshFieldMap_.end()){
        return itr->second;
      }

      return 0;
    }

    MeshField* addField(const char* name,
                        void* hostPtr,
                        uint8_t elementType,
                        uint32_t elementSize){
      size_t size;
      
      switch(elementType){
      case FIELD_CELL:
        size = width_ * height_ * depth_;
        break;
      case FIELD_VERTEX:
        switch(rank_){
        case 1:
          size = width_ + 1;
          break;
        case 2:
          size = (width_ + 1) * (height_ + 1);
          break;
        case 3:
          size = (width_ + 1) * (height_ + 1) + (depth_ + 1);
          break;
        }
        break;
      case FIELD_EDGE:
        switch(rank_){
        case 1:
          size = width_;
          break;
        case 2:
          size = (width_ + 1)*height_ + (height_ + 1)*width_;
          break;
        case 3:
          size_t w1 = width_ + 1;
          size_t h1 = height_ + 1;
          size = (w1*height_ + h1*width_)*(depth_ + 1) + w1*h1*depth_;
          break;
        }
        break;
      case FIELD_FACE:
        switch(rank_){
        case 1:
          size = width_;
          break;
        case 2:
          size = (width_ + 1)*height_ + (height_ + 1)*width_;
          break;
        case 3:
          size_t w1 = width_ + 1;
          size_t h1 = height_ + 1;
          size_t d1 = depth_ + 1;
          size = w1*height_*depth_ + h1*width_*depth_ + d1*width_*height_;
          break;
        }
        break;
      default:
        assert(false && "invalid element type");
      }

      size *= elementSize;

      CUdeviceptr devPtr;
      CUresult err = cuMemAlloc(&devPtr, size);
      check(err);

      MeshField* meshField = new MeshField(hostPtr, devPtr, size);
      meshFieldMap_[name] = meshField;
      return meshField;
    }

    uint32_t width(){
      return width_;
    }

    uint32_t height(){
      return height_;
    }

    uint32_t depth(){
      return depth_;
    }

  private:
    typedef map<const char*, MeshField*> MeshFieldMap_;

    MeshFieldMap_ meshFieldMap_;
    uint32_t width_;
    uint32_t height_;
    uint32_t depth_;
    size_t rank_;
  };

  class Kernel;

  class PTXModule{
  public:    
    PTXModule(const char* ptx){
      nlog("t9");
      CUresult err = cuModuleLoadData(&module_, (void*)ptx);
      check(err);
      nlog("t10");
    }

    Kernel* createKernel(Mesh* mesh, const char* kernelName);

  private:
    CUmodule module_;
  };

  class Kernel{
  public:
    class Field{
    public:
      MeshField* meshField;
      uint8_t mode;

      bool isRead(){
        return mode & FIELD_READ;
      }

      bool isWrite(){
        return mode & FIELD_WRITE;
      }
    };

    Kernel(PTXModule* module,
           Mesh* mesh,
           CUfunction function)
      : module_(module),
        mesh_(mesh),
        function_(function),
        ready_(false),
        numThreads_(DEFAULT_THREADS){
      
    }

    ~Kernel(){
      for(auto& itr : fieldMap_){
        delete itr.second;
      }
    }
    
    void setNumThreads(size_t numThreads){
      numThreads_ = numThreads;
    }

    void addField(const char* fieldName,
                  MeshField* meshField,
                  uint8_t mode){

      Field* field = new Field;
      field->meshField = meshField;
      field->mode = mode;

      fieldMap_.insert({fieldName, field});
    }

    void run(){
      if(!ready_){
        CUresult err = cuMemAlloc(&meshPtr_, 8*fieldMap_.size());
        check(err);

        err = cuMemAlloc(&output_, 8);
        check(err);
        
        size_t offset = 0;
        for(auto& itr : fieldMap_){
          Field* field = itr.second;
          MeshField* meshField = field->meshField;
          //err = cuMemcpyHtoD(meshPtr_ + offset, meshField->devPtr, 8);
          check(err);
          offset += 8;
        }
        
        imageW_ = 0;
        imageH_ = 0;

        vSize_[0] = 0;
        vSize_[1] = 0;
        vSize_[2] = 0;

        dataSize_[0] = 0;
        dataSize_[1] = 0;
        dataSize_[2] = 0;

        partitionStart_[0] = 0;
        partitionStart_[1] = 0;
        partitionStart_[2] = 0;

        partitionSize_[0] = 0;
        partitionSize_[1] = 0;
        partitionSize_[2] = 0;

        density_ = 0.0f;
        brightness_ = 0.0f;
        transferOffset_ = 0.0f;
        transferScale_ = 0.0f;

        kernelParams_ = {&output_, &imageW_, &imageH_, &vSize_, &dataSize_,
                         &partitionStart_, &partitionSize_,
                         &density_, &brightness_, &transferOffset_,
                         &transferScale_, &meshPtr_};
        
        for(auto& itr : fieldMap_){
          Field* field = itr.second;
          MeshField* meshField = field->meshField;
          kernelParams_.push_back(&meshField->devPtr);
        }
        ready_ = true;
      }

      CUresult err;

      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        if(field->isRead()){
          MeshField* meshField = field->meshField;
          err = cuMemcpyHtoD(meshField->devPtr, meshField->hostPtr,
                             meshField->size);
          check(err);
        }
      }

      err = cuLaunchKernel(function_, 1, 1, 1,
                           numThreads_, 1, 1, 
                           0, NULL, kernelParams_.data(), NULL);
      check(err);

      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        if(field->isWrite()){
          MeshField* meshField = field->meshField;
          err = cuMemcpyDtoH(meshField->hostPtr, meshField->devPtr, 
                             meshField->size);
          check(err);
          //float* a = (float*)meshField->hostPtr;
          //size_t size = mesh_->width() * mesh_->height() * mesh_->depth();
          //for(size_t i = 0; i < size; ++i){
          //cout << "a[" << i << "] = " << a[i] << endl;
          //}
        }
      }
    }
    
    PTXModule* module(){
      return module_;
    }

    Mesh* mesh(){
      return mesh_;
    }

    bool ready(){
      return ready_;
    }
    
  private:    
    typedef map<string, Field*> FieldMap_;
    typedef vector<void*> KernelParams_;

    CUfunction function_;
    PTXModule* module_;
    Mesh* mesh_;
    bool ready_;
    FieldMap_ fieldMap_;
    KernelParams_ kernelParams_;
    size_t numThreads_;
    CUdeviceptr output_;
    uint32_t imageW_;
    uint32_t imageH_;
    uint32_t vSize_[3];
    int32_t dataSize_[3];
    int32_t partitionStart_[3];
    int32_t partitionSize_[3];
    float density_;
    float brightness_;
    float transferOffset_;
    float transferScale_;
    CUdeviceptr meshPtr_;
  };

  RenderallRuntime(){

  }

  ~RenderallRuntime(){
    CUresult err = cuCtxDestroy(context_);
    check(err);

    for(auto& itr : kernelMap_){
      delete itr.second;
    }

    for(auto& itr : moduleMap_){
      delete itr.second;
    }

    for(auto& itr : meshMap_){
      delete itr.second;
    }
  }

  void init(){
    CUresult err = cuInit(0);
    check(err);
    
    err = cuDeviceGet(&device_, 0);
    check(err);

    err = cuCtxCreate(&context_, 0, device_);
    check(err);

    int threadsPerBlock;
    err = 
      cuDeviceGetAttribute(&threadsPerBlock,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device_);
    check(err);
    numThreads_ = threadsPerBlock;
  }

  void initKernel(const char* meshName,
                  const char* ptx,
                  const char* kernelName,
                  uint32_t width,
                  uint32_t height,
                  uint32_t depth){

    auto kitr = kernelMap_.find(kernelName);
    if(kitr != kernelMap_.end()){
      return;
    }

    Mesh* mesh;
    auto itr = meshMap_.find(meshName);
    if(itr != meshMap_.end()){
      mesh = itr->second;
    }
    else{
      mesh = new Mesh(width, height, depth);
      meshMap_[meshName] = mesh;
    }
    
    nlog("t1");

    PTXModule* module;
    auto mitr = moduleMap_.find(ptx);
    if(mitr != moduleMap_.end()){
      module = mitr->second;
    }
    else{
      nlog("t1a");
      module = new PTXModule(ptx);
      nlog("t1b");
      moduleMap_[meshName] = module;
    }

    nlog("t2");

    Kernel* kernel = module->createKernel(mesh, kernelName);
    kernel->setNumThreads(numThreads_);
    kernelMap_.insert({kernelName, kernel});
  }

  void initField(const char* kernelName,
                 const char* fieldName,
                 void* hostPtr,
                 uint32_t elementSize,
                 uint8_t elementType,
                 uint8_t mode){

    auto kitr = kernelMap_.find(kernelName);
    assert(kitr != kernelMap_.end() && "invalid kernel");

    Kernel* kernel = kitr->second;
    if(kernel->ready()){
      return;
    }

    Mesh* mesh = kernel->mesh();
    MeshField* meshField = mesh->getField(fieldName);
    if(!meshField){
      meshField = mesh->addField(fieldName, hostPtr,
                                 elementType, elementSize); 
    }
    
    kernel->addField(fieldName, meshField, mode);
  }

  void runKernel(const char* kernelName){
    auto kitr = kernelMap_.find(kernelName);
    assert(kitr != kernelMap_.end() && "invalid kernel");

    Kernel* kernel = kitr->second;
    kernel->run();
  }

  void setWindow(void* window){
    window_ = window;
  }

private:
  typedef map<const char*, PTXModule*> ModuleMap_;
  typedef map<const char*, Mesh*> MeshMap_;
  typedef map<const char*, Kernel*> KernelMap_;

  CUdevice device_;
  CUcontext context_;
  size_t numThreads_;

  ModuleMap_ moduleMap_;
  MeshMap_ meshMap_;
  KernelMap_ kernelMap_;
  void* window_;
};

RenderallRuntime::Kernel* 
RenderallRuntime::PTXModule::createKernel(Mesh* mesh, const char* kernelName){
  CUfunction function;
  CUresult err = cuModuleGetFunction(&function, module_, kernelName);
  check(err);
  
  Kernel* kernel = new Kernel(this, mesh, function);
  
  return kernel;
}

namespace{

  RenderallRuntime* _runtime = nullptr;

  RenderallRuntime* _getRuntime(){
    if(_runtime){
      return _runtime;
    }
    
    _runtime = new RenderallRuntime;
    _runtime->init();
    return _runtime;
  }

} // namespace

extern "C"
void __scrt_volren_init_kernel(const char* meshName,
                               const char* data,
                               const char* kernelName,
                               uint32_t width,
                               uint32_t height,
                               uint32_t depth){
  RenderallRuntime* runtime = _getRuntime();
  runtime->initKernel(meshName, data, kernelName, width, height, depth);
}

extern "C"
void __scrt_volren_init_field(const char* kernelName,
                              const char* fieldName,
                              void* hostPtr,
                              uint32_t elementSize,
                              uint8_t elementType){
  ndump(kernelName);
  ndump(fieldName);
  ndump(hostPtr);
  ndump(elementSize);
  ndump(elementType);

  RenderallRuntime* runtime = _getRuntime();
  runtime->initField(kernelName, fieldName, hostPtr,
                     elementSize, elementType, FIELD_READ);
}

extern "C"
void __scrt_volren_run(const char* kernelName, void* window){
  RenderallRuntime* runtime = _getRuntime();
  runtime->setWindow(window);
  runtime->runKernel(kernelName);
}
