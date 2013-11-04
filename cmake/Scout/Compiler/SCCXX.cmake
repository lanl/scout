include(Compiler/GNU)
__compiler_gnu(CXX)

if(CMAKE_SCCXX_BOOTSTRAP)

  set(CMAKE_SCCXX_FLAGS ${CMAKE_SCCXX_FLAGS} 
    ${CMAKE_CXX_FLAGS} 
    -disable-sc-stdlib)

  set(CMAKE_SCCXX_LINK_FLAGS ${CMAKE_SCCXX_LINK_FLAGS} 
    ${CMAKE_CXX_LINK_FLAGS} -disable-sc-stdlib)

endif()
