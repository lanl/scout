include(Compiler/GNU)
__compiler_gnu(CXX)

if(CMAKE_SCXX_BOOTSTRAP)

  set(CMAKE_SCXX_FLAGS ${CMAKE_SCXX_FLAGS} 
    ${CMAKE_CXX_FLAGS} 
    -disable-sc-stdlib)

  set(CMAKE_SCXX_LINK_FLAGS ${CMAKE_SCXX_LINK_FLAGS} 
    ${CMAKE_CXX_LINK_FLAGS} -disable-sc-stdlib)

endif()
