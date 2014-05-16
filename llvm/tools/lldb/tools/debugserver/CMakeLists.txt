# ===== Scout ====================================
# ASM-ATT is problematic on osx, build will hang
# if output redirected to file
#project(C CXX ASM-ATT)
project(C CXX)
# ================================================
add_subdirectory(scripts)
add_subdirectory(source)
