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
#include <fstream>

#include <map>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>

#include "scout/Runtime/volren/VolumeRendererWindow.h"
#include "scout/Runtime/opengl/qt/ScoutWindow.h"
#include "scout/Runtime/opengl/qt/QtWindow.h"

#include "CudaVolumeTracer.h"

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

class RenderallRuntime{
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
    Mesh(uint64_t width, uint64_t height, uint64_t depth)
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

    uint64_t width(){
      return width_;
    }

    uint64_t height(){
      return height_;
    }

    uint64_t depth(){
      return depth_;
    }

  private:
    typedef map<const char*, MeshField*> MeshFieldMap_;

    MeshFieldMap_ meshFieldMap_;
    uint64_t width_;
    uint64_t height_;
    uint64_t depth_;
    size_t rank_;
  };

  class Kernel;

  class PTXModule{
  public:    
    PTXModule(const char* scoutPtxDir, const char* ptx){
      CUlinkState linkState;
 
      CUjit_option options[6];
      void* values[6];
      float walltime;
      char error_log[8192], info_log[8192];
      options[0] = CU_JIT_WALL_TIME;
      values[0] = (void*)&walltime;
      options[1] = CU_JIT_INFO_LOG_BUFFER;
      values[1] = (void*)info_log;
      options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
      values[2] = (void*)8192;
      options[3] = CU_JIT_ERROR_LOG_BUFFER;
      values[3] = (void*)error_log;
      options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
      values[4] = (void*)8192;
      options[5] = CU_JIT_LOG_VERBOSE;
      values[5] = (void*)1;

      CUresult err = cuLinkCreate(6, options, values, &linkState);
      check(err);

      string path = scoutPtxDir;
      path += "/cuda_compile_ptx_generated_volumeRender_kernel.cu.ptx";

      err = 
        cuLinkAddFile(linkState, CU_JIT_INPUT_PTX,
                      path.c_str(),
                      0, 0, 0);
      check(err);

      err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx,
                          strlen(ptx) + 1,
                          0, 0, 0, 0);

      void* cubin;
      size_t size;
      err = cuLinkComplete(linkState, &cubin, &size);  
      check(err);

      err = cuModuleLoadData(&module_, cubin);
      check(err);

      err = cuLinkDestroy(linkState);
      check(err);
    }

    Kernel* createKernel(Mesh* mesh,
                         const char* kernelName,
                         void* window,
                         uint64_t width,
                         uint64_t height,
                         uint64_t depth);

  private:
    CUmodule module_;
  };

  class Kernel : public RenderCallback{
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
           void* window,
           CUfunction function,
           uint64_t width,
           uint64_t height,
           uint64_t depth)
      : module_(module),
        mesh_(mesh),
        function_(function),
        ready_(false),
        width_(width),
        height_(height),
        depth_(depth),
        varsSize_(0),
        renderer_(nullptr){

      auto scoutWindow = static_cast<ScoutWindow*>(window);
      winWidth_ = scoutWindow->width();
      winHeight_ = scoutWindow->height();
      window_ = scoutWindow->getVolumeRendererWindow();
    }

    ~Kernel(){
      for(auto& itr : fieldMap_){
        delete itr.second;
      }
    }

    void render_kernel(void* output, float* invMat) override{
      CUresult err;

      if(!ready_){
        imageW_ = window_->widgetWidth();
        imageH_ = window_->widgetHeight();

        blockX_ = 16;
        blockY_ = 16;
        blockZ_ = 1;

        gridX_ = iDivUp(imageW_, blockX_);
        gridY_ = iDivUp(imageH_, blockZ_);
        gridZ_ = 1;

        startX_ = 0;
        startY_ = 0;
        startZ_ = 0;
        
        density_ = 0.05f;
        brightness_ = 1.0f;
        transferOffset_ = 0.0f;
        transferScale_ = 1.0f;

        vector<CUdeviceptr> fields;
        
        for(Field* field : fieldVec_){
          MeshField* meshField = field->meshField;
          fields.push_back(meshField->devPtr);
        }

        err = cuMemAlloc(&invMatPtr_, 16 * 4);
        check(err);

        err = cuMemAlloc(&meshPtr_, 8 * fields.size() + varsSize_);
        check(err);

        err = cuMemcpyHtoD(meshPtr_, fields.data(), 8 * fields.size());
        check(err);
        
        kernelParams_ = {nullptr, &invMatPtr_, 
                         &imageW_, &imageH_,
                         &startX_, &startY_, &startZ_,
                         &width_, &height_, &depth_,
                         &density_, &brightness_, &transferOffset_,
                         &transferScale_, &meshPtr_};
        
        ready_ = true;
      }

      err = cuMemcpyHtoD(invMatPtr_, invMat, 16 * 4);
      check(err); 

      kernelParams_[0] = &output;
    
      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        MeshField* meshField = field->meshField;
        err = cuMemcpyHtoD(meshField->devPtr, meshField->hostPtr,
                           meshField->size);
        check(err);
      }

      err = cuLaunchKernel(function_, gridX_, gridY_, gridZ_,
                           blockX_, blockY_, blockZ_, 
                           0, nullptr, kernelParams_.data(), nullptr);
      check(err);
    }

    void addVar(uint32_t size){
      varsSize_ += size;
    }

    void setVar(uint32_t offset, void* data, uint32_t size){
      CUresult err = 
        cuMemcpyHtoD(meshPtr_ + fieldVec_.size() * 8 + offset, data, size);
      check(err);
    }

    void addField(const char* fieldName,
                  MeshField* meshField,
                  uint8_t mode){

      Field* field = new Field;
      field->meshField = meshField;
      field->mode = mode;

      fieldMap_.insert({fieldName, field});
      fieldVec_.push_back(field);
    }

    int iDivUp(int a, int b){
      return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    void run(){
      if(!renderer_){
        renderer_ = new CudaVolumeTracer(width_, height_, depth_);
        renderer_->setRenderCallback(this);

        window_->setRenderable(renderer_);

        window_->resize(winWidth_, winHeight_);
        window_->show();
      }

      window_->update();
      QtWindow::pollEvents();
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
    typedef vector<Field*> FieldVec_;

    CUfunction function_;
    PTXModule* module_;
    Mesh* mesh_;
    VolumeRendererWindow* window_;
    size_t winWidth_;
    size_t winHeight_;
    bool ready_;
    FieldMap_ fieldMap_;
    FieldVec_ fieldVec_;
    KernelParams_ kernelParams_;
    size_t varsSize_;
    size_t blockX_;
    size_t blockY_;
    size_t blockZ_;
    size_t gridX_;
    size_t gridY_;
    size_t gridZ_;
    CUdeviceptr invMatPtr_;
    uint32_t imageW_;
    uint32_t imageH_;
    uint32_t startX_;
    uint32_t startY_;
    uint32_t startZ_;
    uint64_t width_;
    uint64_t height_;
    uint64_t depth_;
    float density_;
    float brightness_;
    float transferOffset_;
    float transferScale_;
    CUdeviceptr meshPtr_;
    CudaVolumeTracer* renderer_;
  };

  RenderallRuntime(){}

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
  }

  void initKernel(const char* scoutPtxDir,
                  const char* meshName,
                  const char* ptx,
                  const char* kernelName,
                  void* window,
                  uint64_t width,
                  uint64_t height,
                  uint64_t depth){

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

    PTXModule* module;
    auto mitr = moduleMap_.find(ptx);
    if(mitr != moduleMap_.end()){
      module = mitr->second;
    }
    else{
      fstream fstr;

      module = new PTXModule(scoutPtxDir, ptx);
      moduleMap_[meshName] = module;
    }

    Kernel* kernel = 
      module->createKernel(mesh, kernelName, window, width, height, depth);

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

  void setVar(const char* kernelName,
              uint32_t offset,
              void* data,
              uint32_t size){
    auto kitr = kernelMap_.find(kernelName);
    assert(kitr != kernelMap_.end() && "invalid kernel");
    Kernel* kernel = kitr->second;

    if(kernel->ready()){
      kernel->setVar(offset, data, size);
      return;
    }

    kernel->addVar(size);
  }

  void runKernel(const char* kernelName){
    auto kitr = kernelMap_.find(kernelName);
    assert(kitr != kernelMap_.end() && "invalid kernel");

    Kernel* kernel = kitr->second;
    kernel->run();
  }

private:
  typedef map<const char*, PTXModule*> ModuleMap_;
  typedef map<const char*, Mesh*> MeshMap_;
  typedef map<const char*, Kernel*> KernelMap_;

  CUdevice device_;
  CUcontext context_;

  ModuleMap_ moduleMap_;
  MeshMap_ meshMap_;
  KernelMap_ kernelMap_;
  CUmodule kernelModule_;
};

RenderallRuntime::Kernel* 
RenderallRuntime::PTXModule::createKernel(Mesh* mesh,
                                          const char* kernelName,
                                          void* window,
                                          uint64_t width,
                                          uint64_t height,
                                          uint64_t depth){
  CUfunction function;
  //CUresult err = cuModuleGetFunction(&function, module_, kernelName);
  CUresult err = 
    cuModuleGetFunction(&function, module_, "volume_render_wrapper");
  check(err);
  
  Kernel* kernel = 
    new Kernel(this, mesh, window, function, width, height, depth);
  
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
void __scrt_volren_init_kernel(const char* scoutPtxDir,
                               const char* meshName,
                               const char* data,
                               const char* kernelName,
                               void* window,
                               uint64_t width,
                               uint64_t height,
                               uint64_t depth){
  RenderallRuntime* runtime = _getRuntime();
  runtime->initKernel(scoutPtxDir, meshName, data, kernelName,
                      window, width, height, depth);
}

extern "C"
void __scrt_volren_init_field(const char* kernelName,
                              const char* fieldName,
                              void* hostPtr,
                              uint32_t elementSize,
                              uint8_t elementType){
  RenderallRuntime* runtime = _getRuntime();
  runtime->initField(kernelName, fieldName, hostPtr,
                     elementSize, elementType, FIELD_READ);
}

extern "C"
void __scrt_volren_set_var(const char* kernelName,
                           uint32_t offset,
                           void* data,
                           uint32_t size){
  RenderallRuntime* runtime = _getRuntime();
  runtime->setVar(kernelName, offset, data, size);
}

extern "C"
void __scrt_volren_run(const char* kernelName){
  RenderallRuntime* runtime = _getRuntime();
  runtime->runKernel(kernelName);
}
