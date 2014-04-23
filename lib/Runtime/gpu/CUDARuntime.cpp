#include <iostream>

#include <map>
#include <vector>
#include <cassert>
#include <cmath>

//#include <sys/time.h>

#include <cuda.h>

using namespace std;

static const uint8_t FIELD_READ = 0x01;
static const uint8_t FIELD_WRITE = 0x02;
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

class CUDARuntime{
public:
  static void check(CUresult err){
    if(err != CUDA_SUCCESS){
      const char* s;
      cuGetErrorString(err, &s);
      cerr << "CUDARuntime error: " << s << endl;
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

      CUresult err = cuMemAlloc(&widthDev_, 4);
      check(err);     

      err = cuMemcpyHtoD(widthDev_, &width_, 4);
      check(err);
 
      err = cuMemAlloc(&heightDev_, 4);
      check(err);  

      err = cuMemcpyHtoD(heightDev_, &height_, 4);
      check(err);

      err = cuMemAlloc(&depthDev_, 4);
      check(err);

      err = cuMemcpyHtoD(depthDev_, &depth_, 4);
      check(err);
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
                        uint32_t elementSize){

      size_t size = width_ * height_ * depth_ * elementSize;
      
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

    CUdeviceptr& widthDev(){
      return widthDev_;
    }

    uint32_t height(){
      return height_;
    }

    CUdeviceptr& heightDev(){
      return heightDev_;
    }

    uint32_t depth(){
      return depth_;
    }

    CUdeviceptr& depthDev(){
      return depthDev_;
    }

  private:
    typedef map<const char*, MeshField*> MeshFieldMap_;

    MeshFieldMap_ meshFieldMap_;
    uint32_t width_;
    uint32_t height_;
    uint32_t depth_;
    CUdeviceptr widthDev_;
    CUdeviceptr heightDev_;
    CUdeviceptr depthDev_;
  };

  class Kernel;

  class PTXModule{
  public:    
    PTXModule(const char* ptx){
      CUresult err = cuModuleLoadData(&module_, (void*)ptx);
      check(err);
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
        kernelParams_.push_back(&mesh_->widthDev());
        kernelParams_.push_back(&mesh_->heightDev());
        kernelParams_.push_back(&mesh_->depthDev());

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
  };

  static CUDARuntime* get(){
    assert(instance_ && "CUDA runtime has not been initialized");
    return instance_;
  }

  static void init(){
    assert(!instance_ && "CUDA runtime has already been initialized");
    
    instance_ = new CUDARuntime;
    instance_->init_();
  }

  static void finish(){
    if(instance_){
      delete instance_;
      instance_ = 0;
    }
  }

  ~CUDARuntime(){
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

  void init_(){
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

    PTXModule* module;
    auto mitr = moduleMap_.find(ptx);
    if(mitr != moduleMap_.end()){
      module = mitr->second;
    }
    else{
      module = new PTXModule(ptx);
      moduleMap_[meshName] = module;
    }

    Kernel* kernel = module->createKernel(mesh, kernelName);
    kernel->setNumThreads(numThreads_);
    kernelMap_.insert({kernelName, kernel});
  }

  void initField(const char* kernelName,
                 const char* fieldName,
                 void* hostPtr,
                 uint32_t elementSize,
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
      meshField = mesh->addField(fieldName, hostPtr, elementSize); 
    }
    
    kernel->addField(fieldName, meshField, mode);
  }

  void runKernel(const char* kernelName){
    auto kitr = kernelMap_.find(kernelName);
    assert(kitr != kernelMap_.end() && "invalid kernel");

    Kernel* kernel = kitr->second;
    kernel->run();
  }

private:
  CUDARuntime(){}

  typedef map<const char*, PTXModule*> ModuleMap_;
  typedef map<const char*, Mesh*> MeshMap_;
  typedef map<const char*, Kernel*> KernelMap_;

  static CUDARuntime* instance_;
  CUdevice device_;
  CUcontext context_;
  size_t numThreads_;

  ModuleMap_ moduleMap_;
  MeshMap_ meshMap_;
  KernelMap_ kernelMap_;
};

CUDARuntime::Kernel* 
CUDARuntime::PTXModule::createKernel(Mesh* mesh, const char* kernelName){
  CUfunction function;
  CUresult err = cuModuleGetFunction(&function, module_, kernelName);
  check(err);
  
  Kernel* kernel = new Kernel(this, mesh, function);
  
  return kernel;
}

CUDARuntime* CUDARuntime::instance_ = 0;

extern "C"
void __scrt_cuda_init(){
  CUDARuntime::init();
}

extern "C"
void __scrt_cuda_finish(){
  CUDARuntime::finish();
}

extern "C"
void __scrt_cuda_init_kernel(const char* meshName,
                             const char* ptx,
                             const char* kernelName,
                             uint32_t width,
                             uint32_t height,
                             uint32_t depth){
  CUDARuntime* runtime = CUDARuntime::get();
  runtime->initKernel(meshName, ptx, kernelName, width, height, depth);
}

extern "C"
void __scrt_cuda_init_field(const char* kernelName,
                            const char* fieldName,
                            void* hostPtr,
                            uint32_t elementSize,
                            uint8_t mode){
  CUDARuntime* runtime = CUDARuntime::get();
  runtime->initField(kernelName, fieldName, hostPtr, elementSize, mode);
}

extern "C"
void __scrt_cuda_run_kernel(const char* kernelName){
  CUDARuntime* runtime = CUDARuntime::get();
  runtime->runKernel(kernelName);
}
