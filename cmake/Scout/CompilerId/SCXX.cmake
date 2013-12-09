include(Compiler/GNU)
__compiler_gnu(CXX)

if(CMAKE_SCXX_BOOTSTRAP)
  set(CMAKE_SCXX_LINK_FLAGS ${CMAKE_SCXX_LINK_FLAGS} 
    ${CMAKE_CXX_LINK_FLAGS} -noscstdlib)
endif()
