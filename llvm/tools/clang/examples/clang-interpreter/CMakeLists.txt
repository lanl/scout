include(sc-llvm-link-libs)


set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  JIT
  Support
  nativecodegen
  )

add_clang_executable(clang-interpreter
  main.cpp
  )

add_dependencies(clang-interpreter
  clang-headers
  )

target_link_libraries(clang-interpreter
  clangBasic
  ${SC_LLVM_LINK_LIBS}
  clangCodeGen
  clangDriver
  clangFrontend
  )
