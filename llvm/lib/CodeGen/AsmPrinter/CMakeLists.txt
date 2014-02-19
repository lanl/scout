add_llvm_library(LLVMAsmPrinter
  ARMException.cpp
  AsmPrinter.cpp
  AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp
  DIE.cpp
  DIEHash.cpp
  DwarfAccelTable.cpp
  DwarfCFIException.cpp
  DwarfDebug.cpp
  DwarfException.cpp
  DwarfUnit.cpp
  ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp
  Win64Exception.cpp
<<<<<<< HEAD
# +===== Scout =====================
  Scout/DwarfUnit.cpp
# +=================================
=======
  WinCodeViewLineTables.cpp
>>>>>>> 8c4223e33dc9675bace7e57e9ddbcb16fbccee40
  )

add_dependencies(LLVMAsmPrinter intrinsics_gen)
