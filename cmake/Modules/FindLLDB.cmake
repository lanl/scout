
set(LLDB_ROOT_DIR
	"${LLDB_ROOT_DIR}"
        "${CMAKE_BINARY_DIR}/llvm"
        "${CMAKE_BINARY_DIR}/../llvm"
	CACHE
	PATH
	"Directory to start our search in")

find_program(LLDB_COMMAND
	NAMES
	lldb
	HINTS
	"${LLDB_ROOT_DIR}"
	PATH_SUFFIXES
	bin
	libexec)

if(LLDB_COMMAND)
	execute_process(COMMAND lldb --version
		COMMAND head -n 1
		OUTPUT_VARIABLE LLDB_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" LLDB_VERSION "${LLDB_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLDB DEFAULT_MSG LLDB_COMMAND)

mark_as_advanced(LLDB_COMMAND)
