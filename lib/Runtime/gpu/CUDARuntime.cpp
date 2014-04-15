#include <cuda.h>

using namespace std;

static const uint8_t FIELD_READ = 0x01;
static const uint8_t FIELD_WRITE = 0x02;
static const uint8_t FIELD_READ_WRITE = 0x03;

class CUDARuntime{
public:
  class MeshField{
  public:
    MeshField(void* hostPtr, CUdeviceptr devPtr, size_t size)
      : hostPtr(hostPtr),
        devPtr(devPtr),
        size(size){

    }

    ~MeshField(){
      cuMemFree(devPtr);
    }

    void* hostPtr;
    CUdeviceptr devPtr;
    size_t size;
  };

  class Mesh{
  public:
    Mesh(size_t width, size_t height, size_t depth)
      : width_(width),
        height_(height),
        depth_(depth){
      
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
      assert(err == CUDA_SUCCESS);

      MeshField* meshField = new MeshField(hostPtr, devPtr, size);
      meshFieldMap_[name] = meshField;
      return meshField;
    }

    size_t width(){
      return width_;
    }

    size_t height(){
      return height_;
    }

    size_t depth(){
      return depth_;
    }

  private:
    typedef map<const char*, MeshField*> MeshFieldMap_;

    MeshFieldMap_ meshFieldMap_;
    size_t width_;
    size_t height_;
    size_t depth_;
  };

  class Kernel;

  class PTXModule{
  public:    
    PTXModule(const char* ptx){
      CUresult err = cuModuleLoadData(&module_, (void*)ptx);
      assert(err == CUDA_SUCCESS);
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

    Kernel(PTXModule* module, Mesh* mesh, CUfunction function)
      : module_(module),
        mesh_(mesh),
        function_(function),
        ready_(false){
      
    }

    ~Kernel(){
      for(auto& itr : fieldMap_){
        delete itr.second;
      }
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
        for(auto& itr : fieldMap_){
          Field* field = itr.second;
          MeshField* meshField = field->meshField;
          kernelParams_.push_back(&meshField->devPtr);
        }
        ready_ = true;
      }
      
      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        if(field->isRead()){
          MeshField* meshField = field->meshField;
          cuMemcpyHtoD(meshField->devPtr, meshField->hostPtr,
                       meshField->size); 
        }
      }

      CUresult err = cuLaunchKernel(function_, 1, 1, 1,
                                    mesh_->width(),
                                    mesh_->height(),
                                    mesh_->depth(),
                                    0, NULL, kernelParams_.data(), NULL);
      assert(err == CUDA_SUCCESS);

      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        if(field->isWrite()){
          MeshField* meshField = field->meshField;
          cuMemcpyDtoH(meshField->hostPtr, meshField->devPtr, 
                       meshField->size); 
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
    assert(err == CUDA_SUCCESS);

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
    assert(err == CUDA_SUCCESS);
    
    err = cuDeviceGet(&device_, 0);
    assert(err == CUDA_SUCCESS);

    err = cuCtxCreate(&context_, 0, device_);
    assert(err == CUDA_SUCCESS);
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
      moduleMap_[meshName];
    }

    Kernel* kernel = module->createKernel(mesh, kernelName);
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
    
    if(!kernel->ready()){
      Mesh* mesh = kernel->mesh();
      MeshField* meshField = mesh->getField(fieldName);
      if(!meshField){
        meshField = mesh->addField(fieldName, hostPtr, elementSize); 
      }

      kernel->addField(fieldName, meshField, mode);
    }
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

  ModuleMap_ moduleMap_;
  MeshMap_ meshMap_;
  KernelMap_ kernelMap_;
};

CUDARuntime::Kernel* 
CUDARuntime::PTXModule::createKernel(Mesh* mesh, const char* kernelName){
  CUfunction function;
  CUresult err = cuModuleGetFunction(&function, module_, kernelName);
  assert(err == CUDA_SUCCESS);
  
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
