//===- llvm-build generated file --------------------------------*- C++ -*-===//
//
// Component Library Depenedency Table
//
// Automatically generated file, do not edit!
//
//===----------------------------------------------------------------------===//

struct AvailableComponent {
  /// The name of the component.
  const char *Name;

  /// The name of the library for this component (or NULL).
  const char *Library;

  /// Whether the component is installed.
  bool IsInstalled;

  /// The list of libraries required when linking this component.
  const char *RequiredLibraries[15];
} AvailableComponents[63] = {
  { "all", 0, 1, { "mcjit", "interpreter", "mcdisassembler", "engine", "option", "debuginfo", "gtest_main", "all-targets", "irreader", "instrumentation", "nativecodegen", "tablegen", "lto" } },
  { "all-targets", 0, 1, { "arm", "x86", "nvptx", "cppbackend" } },
  { "analysis", "libLLVMAnalysis.a", 1, { "core", "support", "target" } },
  { "arm", 0, 1, { "armasmprinter", "arminfo", "armdesc", "armasmparser", "armcodegen", "armdisassembler" } },
  { "armasmparser", "libLLVMARMAsmParser.a", 1, { "armdesc", "arminfo", "mc", "mcparser", "support" } },
  { "armasmprinter", "libLLVMARMAsmPrinter.a", 1, { "mc", "support" } },
  { "armcodegen", "libLLVMARMCodeGen.a", 1, { "armasmprinter", "armdesc", "arminfo", "analysis", "asmprinter", "codegen", "core", "mc", "scalaropts", "selectiondag", "support", "target" } },
  { "armdesc", "libLLVMARMDesc.a", 1, { "armasmprinter", "arminfo", "mc", "support" } },
  { "armdisassembler", "libLLVMARMDisassembler.a", 1, { "armdesc", "arminfo", "mc", "support" } },
  { "arminfo", "libLLVMARMInfo.a", 1, { "support" } },
  { "asmparser", "libLLVMAsmParser.a", 1, { "core", "support" } },
  { "asmprinter", "libLLVMAsmPrinter.a", 1, { "analysis", "codegen", "core", "mc", "mcparser", "support", "target", "transformutils" } },
  { "bitreader", "libLLVMBitReader.a", 1, { "core", "support" } },
  { "bitwriter", "libLLVMBitWriter.a", 1, { "core", "support" } },
  { "codegen", "libLLVMCodeGen.a", 1, { "analysis", "core", "mc", "scalaropts", "support", "target", "transformutils" } },
  { "core", "libLLVMCore.a", 1, { "support" } },
  { "cppbackend", 0, 1, { "cppbackendinfo", "cppbackendcodegen" } },
  { "cppbackendcodegen", "libLLVMCppBackendCodeGen.a", 1, { "core", "cppbackendinfo", "support", "target" } },
  { "cppbackendinfo", "libLLVMCppBackendInfo.a", 1, { "support" } },
  { "debuginfo", "libLLVMDebugInfo.a", 1, { "object", "support" } },
  { "engine", 0, 1, { "jit", "native" } },
  { "executionengine", "libLLVMExecutionEngine.a", 1, { "core", "mc", "support" } },
  { "gtest", "libgtest.a", 0, { "support" } },
  { "gtest_main", "libgtest_main.a", 0, { "gtest" } },
  { "instcombine", "libLLVMInstCombine.a", 1, { "analysis", "core", "support", "target", "transformutils" } },
  { "instrumentation", "libLLVMInstrumentation.a", 1, { "analysis", "core", "support", "target", "transformutils" } },
  { "interpreter", "libLLVMInterpreter.a", 1, { "codegen", "core", "executionengine", "support" } },
  { "ipa", "libLLVMipa.a", 1, { "analysis", "core", "support" } },
  { "ipo", "libLLVMipo.a", 1, { "analysis", "core", "ipa", "instcombine", "scalaropts", "support", "target", "transformutils", "vectorize" } },
  { "irreader", "libLLVMIRReader.a", 1, { "asmparser", "bitreader", "core", "support" } },
  { "jit", "libLLVMJIT.a", 1, { "codegen", "core", "executionengine", "support" } },
  { "linker", "libLLVMLinker.a", 1, { "core", "support", "transformutils" } },
  { "lto", "libLLVMLTO.a", 1, { "bitreader", "bitwriter", "core", "ipa", "ipo", "instcombine", "linker", "mc", "mcparser", "objcarcopts", "scalaropts", "support", "target", "transformutils" } },
  { "mc", "libLLVMMC.a", 1, { "object", "support" } },
  { "mcdisassembler", "libLLVMMCDisassembler.a", 1, { "mc", "support" } },
  { "mcjit", "libLLVMMCJIT.a", 1, { "core", "executionengine", "runtimedyld", "support", "target" } },
  { "mcparser", "libLLVMMCParser.a", 1, { "mc", "support" } },
  { "native", 0, 1, { "x86" } },
  { "nativecodegen", 0, 1, { "x86codegen" } },
  { "nvptx", 0, 1, { "nvptxasmprinter", "nvptxinfo", "nvptxdesc", "nvptxcodegen" } },
  { "nvptxasmprinter", "libLLVMNVPTXAsmPrinter.a", 1, { "mc", "support" } },
  { "nvptxcodegen", "libLLVMNVPTXCodeGen.a", 1, { "analysis", "asmprinter", "codegen", "core", "mc", "nvptxasmprinter", "nvptxdesc", "nvptxinfo", "selectiondag", "support", "target" } },
  { "nvptxdesc", "libLLVMNVPTXDesc.a", 1, { "mc", "nvptxasmprinter", "nvptxinfo", "support" } },
  { "nvptxinfo", "libLLVMNVPTXInfo.a", 1, { "support" } },
  { "objcarcopts", "libLLVMObjCARCOpts.a", 1, { "analysis", "core", "support", "transformutils" } },
  { "object", "libLLVMObject.a", 1, { "support" } },
  { "option", "libLLVMOption.a", 1, { "support" } },
  { "runtimedyld", "libLLVMRuntimeDyld.a", 1, { "object", "support" } },
  { "scalaropts", "libLLVMScalarOpts.a", 1, { "analysis", "core", "ipa", "instcombine", "support", "target", "transformutils" } },
  { "selectiondag", "libLLVMSelectionDAG.a", 1, { "analysis", "codegen", "core", "mc", "support", "target", "transformutils" } },
  { "support", "libLLVMSupport.a", 1, {  } },
  { "tablegen", "libLLVMTableGen.a", 1, { "support" } },
  { "target", "libLLVMTarget.a", 1, { "core", "mc", "support" } },
  { "transformutils", "libLLVMTransformUtils.a", 1, { "analysis", "core", "ipa", "support", "target" } },
  { "vectorize", "libLLVMVectorize.a", 1, { "analysis", "core", "support", "target", "transformutils" } },
  { "x86", 0, 1, { "x86utils", "x86asmprinter", "x86info", "x86desc", "x86codegen", "x86asmparser", "x86disassembler" } },
  { "x86asmparser", "libLLVMX86AsmParser.a", 1, { "mc", "mcparser", "support", "x86desc", "x86info" } },
  { "x86asmprinter", "libLLVMX86AsmPrinter.a", 1, { "mc", "support", "x86utils" } },
  { "x86codegen", "libLLVMX86CodeGen.a", 1, { "analysis", "asmprinter", "codegen", "core", "mc", "selectiondag", "support", "target", "x86asmprinter", "x86desc", "x86info", "x86utils" } },
  { "x86desc", "libLLVMX86Desc.a", 1, { "mc", "support", "x86asmprinter", "x86info" } },
  { "x86disassembler", "libLLVMX86Disassembler.a", 1, { "mc", "support", "x86info" } },
  { "x86info", "libLLVMX86Info.a", 1, { "support" } },
  { "x86utils", "libLLVMX86Utils.a", 1, { "support" } },
};
