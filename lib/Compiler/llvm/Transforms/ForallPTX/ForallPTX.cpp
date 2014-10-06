#include "llvm/Transforms/Scout/ForallPTX/ForallPTX.h"

#include <iostream>
#include <memory>

#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/PassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace std;
using namespace llvm;

namespace{

typedef vector<Type*> TypeVec;
typedef vector<Value*> ValueVec;
typedef vector<string> StringVec;

static const uint8_t FIELD_READ = 0x01;
static const uint8_t FIELD_WRITE = 0x02;

/*
static const uint8_t FIELD_CELL = 0;
static const uint8_t FIELD_VERTEX = 1;
static const uint8_t FIELD_EDGE = 2;
static const uint8_t FIELD_FACE = 3;
*/

class CUDAModule{
public:
  
  class Kernel{
  public:
    class Field{
    public:
      Field(const string& name,
            size_t position,
            uint8_t elementType,
            PointerType* type)
        : name_(name),
          position_(position),
          type_(type),
          elementType_(elementType),
          mode_(0),
          value_(0){

      }
      
      void setRead(){
        mode_ |= FIELD_READ;
      }

      void setWrite(){
        mode_ |= FIELD_WRITE;
      }

      uint8_t mode(){
        return mode_;
      }

      uint8_t elementType(){
        return elementType_;
      }
      
      PointerType* type(){
        return type_;
      }

      void setValue(Value* value){
        value_ = value;
      }

      Value* value(){
        return value_;
      }

      size_t position(){
        return position_;
      }

      const string& name(){
        return name_;
      }

    private:      
      string name_;
      size_t position_;
      PointerType* type_;
      uint8_t elementType_;
      uint8_t mode_;
      Value* value_;
    };

    Kernel(CUDAModule& m, MDNode* kernelMD)
      : m_(m),
        context_(m_.context()){

      f_ = cast<Function>(kernelMD->getOperand(0));

      meshName_ = cast<MDString>(kernelMD->getOperand(1))->getString().str();

      MDNode* fieldsNode = cast<MDNode>(kernelMD->getOperand(2));

      size_t pos = 0;
      size_t numFields = fieldsNode->getNumOperands();
      for(size_t i = 0; i < numFields; ++i){
        MDNode* fieldNode = cast<MDNode>(fieldsNode->getOperand(i));

        string fieldName =
          cast<MDString>(fieldNode->getOperand(0))->getString().str();
        
        uint8_t elementType =
        cast<ConstantInt>(fieldNode->getOperand(1))->getZExtValue();

        fieldInfoMap_.insert({fieldName, {pos, elementType}});
        ++pos;
      }
    }
    
    void init(){
      createFields();
      setFieldModes();
      createFunction();
    }

    void clearFunction(Function* f){
      vector<BasicBlock*> bv;

      for(Function::iterator itr = f_->begin(), itrEnd = f_->end();
          itr != itrEnd; ++itr){
        BasicBlock* b = itr;
        bv.push_back(b);
      }

      for(BasicBlock* b : bv){
        b->removeFromParent();
      }
    }

    static void split(const string& in,
                      const string& delimiter,
                      StringVec& out){
      size_t i = 0;
      for(;;){
        size_t pos = in.find(delimiter, i);
        if(pos == string::npos){
          out.push_back(in.substr(i, in.length() - i));
          break;
        }
        out.push_back(in.substr(i, pos - i));
        i = pos + delimiter.length();
      }
    }

    string getMeshFieldName(StringRef name){
      if(!name.startswith("TheMesh.")){
        return "";
      }

      string s = name.str();
      StringVec sv;
      split(s, ".", sv);

      return sv[1];
    }
 
    void createFields(){
      for(Function::iterator itr = f_->begin(), itrEnd = f_->end();
          itr != itrEnd; ++itr){

        BasicBlock* b = itr;
        for(BasicBlock::iterator bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){

          Instruction* i = bitr;
          if(AllocaInst* ai = dyn_cast<AllocaInst>(i)){
            string n = getMeshFieldName(ai->getName());
            if(n.empty()){
              continue;
            }

            Type* et = ai->getType()->getElementType();

            auto fitr = fieldInfoMap_.find(n);
            assert(fitr != fieldInfoMap_.end());
            size_t pos = fitr->second.first;
            uint8_t elementType = fitr->second.second;
            
            createField(n, pos, elementType, PointerType::get(et, 1));
          }
        }
      }
    }  

    void setFieldModes(){
      for(Function::iterator itr = f_->begin(), itrEnd = f_->end();
          itr != itrEnd; ++itr){

        BasicBlock* b = itr;
        for(BasicBlock::iterator bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){

          Instruction* i = bitr;
          if(StoreInst* si = dyn_cast<StoreInst>(i)){
            Value* po = si->getPointerOperand();
            string n = getMeshFieldName(po->getName());
            if(n.empty()){
              continue;
            }

            Field* field = getField(n);
            field->setWrite();
          }
          else if(LoadInst* si = dyn_cast<LoadInst>(i)){
            Value* po = si->getPointerOperand();
            string n = getMeshFieldName(po->getName());
            if(n.empty()){
              continue;
            }

            Field* field = getField(n);
            field->setRead();
          }
        }
      }
    }  

    void createFunction(){
      Module& kernelModule = m_.kernelModule();
      
      //f_->dump();

      TypeVec tv;
      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        tv.push_back(field->type());
      }

      ValueToValueMapTy vm;

      FunctionType* ft = FunctionType::get(m_.voidTy, tv, false);

      kf_ = Function::Create(ft, Function::ExternalLinkage,
                             f_->getName(), &kernelModule);

      Function::arg_iterator aitr = kf_->arg_begin();
      
      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        field->setValue(aitr);
        aitr->setName(itr.first);
        ++aitr;
      }

      for(Function::iterator itr = f_->begin(), itrEnd = f_->end();
          itr != itrEnd; ++itr){

        BasicBlock* b = itr;
        BasicBlock* bk = BasicBlock::Create(context_, b->getName(), kf_);
        vm[b] = bk;

        for(BasicBlock::iterator bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){
          Instruction* i = bitr;

          if(AllocaInst* ai = dyn_cast<AllocaInst>(i)){
            if(ai->getName().startswith("tid.") ||
               ai->getName().startswith("ntid.") ||
               ai->getName().startswith("ctaid.") ||
               ai->getName().startswith("nctaid.") ||
               ai->getName().startswith("TheMesh_addr")){
              continue;
            }
            else{
              string n = getMeshFieldName(ai->getName());
              if(!n.empty()){
                Field* field = getField(n);
                vm[i] = field->value();
                continue;
              }
            }
          }
          else if(LoadInst* li = dyn_cast<LoadInst>(i)){
            if(li->getName().startswith("tid.") ||
               li->getName().startswith("ntid.") ||
               li->getName().startswith("ctaid.") ||
               li->getName().startswith("nctaid.")){
              string n = li->getName().str();
              Function* sf = m_.getSREGFunc(n);
              Instruction* ik = CallInst::Create(sf, li->getName(), bk);
              vm[i] = ik;
              continue;
            }
            else if(li->getName().startswith("TheMesh.width") ||
                    li->getName().startswith("TheMesh.height") ||
                    li->getName().startswith("TheMesh.depth")){
              continue;
            }
          }
          else if(StoreInst* si = dyn_cast<StoreInst>(i)){
            if(si->getPointerOperand()->getName().startswith("TheMesh_addr")){
              continue;
            }
          }

          Instruction* ik = i->clone();
          vm[i] = ik;
          ik->setName(i->getName());
          BasicBlock::InstListType& il = bk->getInstList();
          il.push_back(ik);
        }
      }

      for(Function::iterator itr = kf_->begin(), itrEnd = kf_->end();
          itr != itrEnd; ++itr){

        BasicBlock* b = itr;
        for(BasicBlock::iterator bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){
          Instruction* i = bitr;
          RemapInstruction(i, vm, RF_IgnoreMissingEntries, 0);
        }
      }

      NamedMDNode* annotations = 
        kernelModule.getOrInsertNamedMetadata("nvvm.annotations");
      
      SmallVector<Value*, 3> av;
      av.push_back(kf_);
      av.push_back(MDString::get(context_, "kernel"));
      av.push_back(m_.getInt32(1));

      annotations->addOperand(MDNode::get(context_, av));
    }

    void createRuntimeCalls(){
      Module* module = m_.module();
      clearFunction(f_);

      IRBuilder<> builder(module->getContext());

      BasicBlock* b = BasicBlock::Create(context_, "forall.entry", f_);
      builder.SetInsertPoint(b);
      
      Function::arg_iterator aitr = f_->arg_begin();
      Value* meshPtr = aitr++;
      Value* widthPtr = aitr++;
      Value* heightPtr = aitr++;
      Value* depthPtr = aitr++;

      Value* width = builder.CreateLoad(widthPtr, "width");
      Value* height = builder.CreateLoad(heightPtr, "height");
      Value* depth = builder.CreateLoad(depthPtr, "depth");

      Value* ptx = builder.CreateBitCast(m_.ptxGlobal(), m_.stringTy);
      Value* meshName = builder.CreateGlobalStringPtr(meshName_);
      meshName = builder.CreateBitCast(meshName, m_.stringTy);
      Value* kernelName = builder.CreateGlobalStringPtr(f_->getName());
      kernelName = builder.CreateBitCast(kernelName, m_.stringTy);

      Function* f = module->getFunction("__scrt_gpu_init_kernel");
      
      ValueVec args = {meshName, ptx, kernelName, width, height, depth};

      builder.CreateCall(f, args);

      f = module->getFunction("__scrt_gpu_init_field");

      for(auto& itr : fieldMap_){
        Field* field = itr.second;

        Value* fieldName = builder.CreateGlobalStringPtr(field->name());
        fieldName = builder.CreateBitCast(fieldName, m_.stringTy);

        Value* hostPtr = builder.CreateStructGEP(meshPtr, field->position());
        hostPtr = builder.CreateLoad(hostPtr);
        hostPtr = builder.CreateBitCast(hostPtr, m_.voidPtrTy);

        PointerType* pointerType = field->type();
        Type* type = pointerType->getElementType();
        Value* elementSize = 
          m_.getInt32(type->getPrimitiveSizeInBits()/8);
        
        Value* elementType = m_.getInt8(field->elementType());
        Value* mode = m_.getInt8(field->mode());

        args = {kernelName, fieldName, hostPtr, elementSize, elementType, mode};
        
        builder.CreateCall(f, args);
      }

      f = module->getFunction("__scrt_gpu_run_kernel");
      args = {kernelName};
      builder.CreateCall(f, args);

      builder.CreateRetVoid();
    }

    Field* createField(const string& name,
                       size_t pos,
                       uint8_t elementType,
                       PointerType* type){

      llvm::errs() << "name " << name << "\n";
      auto itr = fieldMap_.find(name);
      assert(itr == fieldMap_.end());

      Field* field = new Field(name, pos, elementType, type);
      fieldMap_.insert({name, field});

      return field;
    }
    
    Field* getField(const string& name){
      auto itr = fieldMap_.find(name);
      assert(itr != fieldMap_.end());
      return itr->second;
    }

  private:
    typedef map<string, Field*> FieldMap_;
    typedef map<string, pair<size_t, uint8_t>> FieldInfoMap_;

    CUDAModule& m_;
    LLVMContext& context_;
    Function* f_;
    Function* kf_;
    FieldMap_ fieldMap_;
    FieldInfoMap_ fieldInfoMap_;
    string meshName_;
  };

  CUDAModule(Module* module)
  : module_(module),
    context_(module->getContext()),
    kernelModule_("kernel_module", context_),
    builder_(context_){

    int32Ty = Type::getInt32Ty(context_);
    int8Ty = Type::getInt8Ty(context_);
    voidTy = Type::getVoidTy(context_);
    stringTy = PointerType::get(int8Ty, 0);
    voidPtrTy = PointerType::get(int8Ty, 0);

    TypeVec tv;
    createFunction("__scrt_cuda_init",
                   voidTy, tv);

    createFunction("__scrt_gpu_finish",
                   voidTy, tv);

    tv = {stringTy, stringTy, stringTy, int32Ty, int32Ty, int32Ty};
    createFunction("__scrt_gpu_init_kernel",
                   voidTy, tv);

    tv = {stringTy, stringTy, voidPtrTy, int32Ty, int8Ty, int8Ty};
    createFunction("__scrt_gpu_init_field",
                   voidTy, tv);

    tv = {stringTy};
    createFunction("__scrt_gpu_run_kernel",
                   voidTy, tv);
  }

  void init(){
    //cerr << "-- CPU module before: " << endl;
    //module_->dump();
    //cerr << "======================" << endl;
    
    NamedMDNode* kernelsMD = module_->getNamedMetadata("scout.kernels");

    vector<Kernel*> kernels;

    for(size_t i = 0; i < kernelsMD->getNumOperands(); ++i){
      MDNode* kernelMD = kernelsMD->getOperand(i);

      Kernel* kernel = new Kernel(*this, kernelMD);
      kernel->init();
      kernels.push_back(kernel);
    }

    //cerr << "--------- kernel module: " << endl;
    //kernelModule_.dump();
    //cerr << "=========================" << endl;

    string ptx = generatePTX();

    Constant* pc = ConstantDataArray::getString(context_, ptx);
    
    ptxGlobal_ = 
      new GlobalVariable(*module_,
                         pc->getType(),
                         true,
                         GlobalValue::PrivateLinkage,
                         pc,
                         "ptx");

    for(Kernel* kernel : kernels){
      kernel->createRuntimeCalls();
      delete kernel;
    }

    //cerr << "--- CPU module after: " << endl;
    //module_->dump();
    //cerr << "======================" << endl;
  }

  Module* module(){
    return module_;
  }

  Module& kernelModule(){
    return kernelModule_;
  }

  Function* getSREGFunc(const string& suffix){
    string name = "llvm.nvvm.read.ptx.sreg." + suffix;

    Function* f = kernelModule_.getFunction(name);
    if(f){
      return f;
    }

    FunctionType* ft = FunctionType::get(int32Ty, false);
    f = Function::Create(ft, Function::ExternalLinkage, name, &kernelModule_);

    return f;
  }

  ConstantInt* getInt32(int32_t v){
    return ConstantInt::get(context_, APInt(32, v, true));
  }

  ConstantInt* getInt8(int8_t v){
    return ConstantInt::get(context_, APInt(8, v, true));
  }

  LLVMContext& context(){
    return context_;
  }

  const Target* findPTXTarget(){
    for(TargetRegistry::iterator itr = TargetRegistry::begin(),
          itrEnd = TargetRegistry::end(); itr != itrEnd; ++itr) {
      if(strcmp(itr->getName(), "nvptx64") == 0) {
        return &*itr;
      }
    }
    
    return 0;
  }

  TargetMachine* createTargetMachine(const Target* target){
    Triple triple(sys::getDefaultTargetTriple());
    triple.setArch(Triple::nvptx64);
    
    return 
      target->createTargetMachine(triple.getTriple(),
                                  "sm_20",
                                  "",
                                  TargetOptions(),
                                  Reloc::Default,
                                  CodeModel::Default,
                                  CodeGenOpt::Aggressive);
  }

  string generatePTX(){
    const Target* target = findPTXTarget();
    assert(target && "failed to find NVPTX target");

    TargetMachine* targetMachine = createTargetMachine(target);

    PassManager passManager;

    passManager.add(createVerifierPass());

    const DataLayout* dataLayout = targetMachine->getSubtargetImpl()->getDataLayout();

    kernelModule_.setDataLayout(dataLayout);

    assert(dataLayout && "failed to get data layout");

    passManager.add(new DataLayoutPass());

    string ptxCode;
    raw_string_ostream rstr(ptxCode);
    formatted_raw_ostream fstr(rstr);

    targetMachine->addPassesToEmitFile(passManager,
                                       fstr,
                                       TargetMachine::CGFT_AssemblyFile,
                                       true);

    passManager.run(kernelModule_);
    fstr.flush();

    return ptxCode;
  }

  Function* createFunction(const string& name,
                           Type* returnType,
                           const TypeVec& argTypes){
    FunctionType* ft =
      FunctionType::get(returnType, argTypes, false);
    
    Function* f =
      Function::Create(ft,
                       Function::ExternalLinkage,
                       name.c_str(), module_);
    
    return f;
  }

  GlobalVariable* ptxGlobal(){
    return ptxGlobal_;
  }
  
  Type* int32Ty;
  Type* int8Ty;
  Type* voidTy;
  Type* stringTy;
  Type* voidPtrTy;

private:
  Module* module_;
  LLVMContext& context_;
  Module kernelModule_;
  IRBuilder<> builder_;
  GlobalVariable* ptxGlobal_;
};

class ForallPTX : public ModulePass{
public:
  static char ID;

  ForallPTX() : ModulePass(ID){}

  void getAnalysisUsage(AnalysisUsage& AU) const override{}

  const char *getPassName() const {
    return "Forall-to-PTX";
  }

  bool runOnModule(Module& M) override{
    CUDAModule cudaModule(&M);
    cudaModule.init();
    return true;
  }
};

char ForallPTX::ID;

} // end namespace

ModulePass* llvm::createForallPTXPass(){
  return new ForallPTX;
}
