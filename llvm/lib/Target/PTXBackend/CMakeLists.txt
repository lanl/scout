add_llvm_target(PTXBackend
  PTXBackend.cpp
  PTXPasses.cpp
  )

add_llvm_library_dependencies(LLVMPTXBackend
  LLVMAnalysis
  LLVMPTXBackendInfo
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMScalarOpts
  LLVMSupport
  LLVMTarget
  LLVMTransformUtils
  LLVMipo
  )

add_subdirectory(TargetInfo)