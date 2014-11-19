# - Try to find CLANG
# Once done, this will define:
#  CLANG_COMMAND - the clang location
#  CLANGXX_COMMAND - the clang++ location
#

find_program(CLANG_COMMAND
    NAMES clang
    HINTS $ENV{CLANG_DIR}
    PATHS
    /usr/bin
    /usr/local/bin
    ${SCOUT_BUILD_DIR}/bin
    DOC "Path for clang"
    )


find_program(CLANGXX_COMMAND
    NAMES clang++
    HINTS $ENV{CLANG_DIR}
    PATHS
    /usr/bin
    /usr/local/bin
    ${SCOUT_BUILD_DIR}/bin
    DOC "Path for clang++"
    )

#message(STATUS "clang: ${CLANG_COMMAND} ${CLANGXX_COMMAND}")

# handle the QUIETLY and REQUIRED arguments and set xxx_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLANG DEFAULT_MSG CLANG_COMMAND CLANGXX_COMMAND)

