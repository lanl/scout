set(LLVM_TARGET_DEFINITIONS Intrinsics.td)

tablegen(LLVM Intrinsics.gen -gen-intrinsic)

# +===== Scout =================================================================+
add_custom_target(intrinsics_gen ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Intrinsics.gen)
set_target_properties(intrinsics_gen PROPERTIES FOLDER "Tablegenning")
# +=============================================================================+
