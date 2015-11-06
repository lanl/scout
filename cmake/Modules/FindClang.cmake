# - Try to find CLANG if it can't be found use the one we build as part of Scout
# Once done, this will define:
#  CLANG_COMMAND - the clang location
#  CLANGXX_COMMAND - the clang++ location
#

find_program(_CLANG_COMMAND
    NAMES clang
    HINTS $ENV{CLANG_DIR}
    PATHS
    /usr/bin
    /usr/local/bin
    DOC "Path for clang"
    )

if(_CLANG_COMMAND MATCHES "NOTFOUND")
  set(CLANG_COMMAND "${SCOUT_BUILD_DIR}/bin/clang" CACHE FILEPATH "Clang")
else()
  set(CLANG_COMMAND ${_CLANG_COMMAND} CACHE FILEPATH "Clang")
endif()
unset(_CLANG_COMMAND)

find_program(_CLANGXX_COMMAND
    NAMES clang++
    HINTS $ENV{CLANG_DIR}
    PATHS
    /usr/bin
    /usr/local/bin
    DOC "Path for clang++"
    )

if(_CLANGXX_COMMAND MATCHES "NOTFOUND")
  set(CLANGXX_COMMAND "${SCOUT_BUILD_DIR}/bin/clang++" CACHE FILEPATH "Clang++")
else()
  set(CLANGXX_COMMAND ${_CLANGXX_COMMAND} CACHE FILEPATH "Clang++")
#endif()
unset(_CLANGXX_COMMAND)


#hack always use scout clang rather than system clang 
set(CLANG_COMMAND "${SCOUT_BUILD_DIR}/bin/clang" CACHE FILEPATH "Clang")
set(CLANGXX_COMMAND "${SCOUT_BUILD_DIR}/bin/clang++" CACHE FILEPATH "Clang++")

# handle the QUIETLY and REQUIRED arguments and set xxx_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLANG DEFAULT_MSG CLANG_COMMAND CLANGXX_COMMAND)

