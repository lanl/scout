# ===== Scout ====================================
# ASM-ATT is problematic on osx, build will hang
# if output redirected to file
#project(C CXX ASM-ATT)
project(C CXX)
#Don't build the debug server at all, use the one from xcode
#add_subdirectory(scripts)
#add_subdirectory(source)
# ================================================
