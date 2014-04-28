include_directories(../)

add_llvm_library(ScoutAsmPrinter
  DwarfUnit.cpp
  )

add_dependencies(ScoutAsmPrinter intrinsics_gen)
