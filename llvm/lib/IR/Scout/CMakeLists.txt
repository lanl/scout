include_directories(../)

add_llvm_library(ScoutCore
  DIBuilder.cpp
  DebugInfo.cpp
  )

add_dependencies(ScoutCore intrinsics_gen)
