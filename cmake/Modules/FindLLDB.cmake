
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

set(LLDB_VERSION "-1")
if(LLDB_COMMAND MATCHES "NOTFOUND" )
  set(LLDB_VERSION "0")
else()
	execute_process(COMMAND lldb --version
		COMMAND head -n 1
		OUTPUT_VARIABLE _LLDB_VERSION
		RESULT_VARIABLE _LLDB_RESULT
		OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_LLDB_RESULT EQUAL 0)
	  string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" LLDB_VERSION "${_LLDB_VERSION}")
        endif()
        unset(_LLDB_VERSION)
        unset(_LLDB_RESULT)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLDB DEFAULT_MSG LLDB_COMMAND)

mark_as_advanced(LLDB_COMMAND)
