include(Compiler/GNU)
__compiler_gnu(CXX)

if(CMAKE_SCC_BOOTSTRAP)

  set(CMAKE_SCC_FLAGS ${CMAKE_SCC_FLAGS} 
    ${CMAKE_CXX_FLAGS} 
    -disable-sc-stdlib)

  set(CMAKE_SCC_LINK_FLAGS ${CMAKE_SCC_LINK_FLAGS} 
    ${CMAKE_CXX_LINK_FLAGS} -disable-sc-stdlib)

endif()
