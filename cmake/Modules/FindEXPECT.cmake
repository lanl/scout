set(EXPECT_ROOT_DIR
	"${EXPECT_ROOT_DIR}"
        /usr
	CACHE
	PATH
	"Directory to start our search in")

find_program(EXPECT_COMMAND
	NAMES
        expect	
	HINTS
	"${EXPECT_ROOT_DIR}"
	PATH_SUFFIXES
	bin
	libexec)

set(EXPECT_VERSION "-1")
if(EXPECT_COMMAND MATCHES "NOTFOUND")
 set(EXPECT_VERSION "0")
else()
	execute_process(COMMAND expect -version
		COMMAND head -n 1
		OUTPUT_VARIABLE _EXPECT_VERSION
                RESULT_VARIABLE _EXPECT_RESULT
		OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_EXPECT_RESULT EQUAL 0) 
	  string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" EXPECT_VERSION "${_EXPECT_VERSION}")
        endif()
        unset(_EXPECT_VERSION)
        unset(_EXPECT_RESULT)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EXPECT DEFAULT_MSG EXPECT_COMMAND EXPECT_VERSION)

mark_as_advanced(EXPECT_COMMAND)
