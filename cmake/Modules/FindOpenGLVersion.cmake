# assumes gl-info has already be built but that is ok as
# we are only using this in the test directory
# output: OPENGL_VERSION_MAJOR 
#         OPENGL_VERSION_MINOR


 find_program(GL_INFO
    NAMES gl-info
    PATH_SUFFIXES bin 
    PATHS 
     ${CMAKE_CURRENT_BINARY_DIR}/bin
     ${CMAKE_CURRENT_BINARY_DIR}/../bin 
    NO_DEFAULT_PATH
    DOC "Path to gl-info"
  )
set(OPENGL_VERSION_MAJOR "-1")
set(OPENGL_VERSION_MINOR "-1")

if (GL_INFO MATCHES "NOTFOUND")
    set(OPENGL_VERSION_MAJOR "0")
    set(OPENGL_VERSION_MINOR "0")
else()
 execute_process(COMMAND ${GL_INFO} "--version" OUTPUT_VARIABLE _OPENGL_VERSION)
    string(REGEX REPLACE ".*([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\1" OPENGL_VERSION_MAJOR ${_OPENGL_VERSION})
    string(REGEX REPLACE ".*([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\2" OPENGL_VERSION_MINOR ${_OPENGL_VERSION})
    unset(_OPENGL_VERSION)
    message(STATUS "OpenGL version: ${OPENGL_VERSION_MAJOR}.${OPENGL_VERSION_MINOR}")
endif()

