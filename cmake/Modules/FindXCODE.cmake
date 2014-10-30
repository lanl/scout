# - Try to find XCODE version
#


set(XCODE_ROOT_DIR
	"${XCODE_ROOT_DIR}"
	CACHE
	PATH
	"Directory to start our search in")

find_program(XCODE_COMMAND
	NAMES
        xcodebuild 	
	HINTS
	"${XCODE_ROOT_DIR}"
	PATH_SUFFIXES
	bin
	libexec)

set(XCODE_VERSION "-1")
if(XCODE_COMMAND MATCHES "NOTFOUND")
  set(XCODE_VERSION "0")
else()
	execute_process(COMMAND xcodebuild -version
		COMMAND head -n 1
		OUTPUT_VARIABLE _XCODE_VERSION
                RESULT_VARIABLE _XCODE_RESULT
		OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_XCODE_RESULT EQUAL 0)
	  string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" XCODE_VERSION "${_XCODE_VERSION}")
        endif()
        unset(_XCODE_VERSION)
        unset(_XCODE_RESULT)
endif()

# handle the QUIETLY and REQUIRED arguments and set xxx_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XCODE DEFAULT_MSG XCODE_COMMAND XCODE_VERSION)

mark_as_advanced(XCODE_COMMAND)
