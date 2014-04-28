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

if(EXPECT_COMMAND)
	execute_process(COMMAND expect -version
		COMMAND head -n 1
		OUTPUT_VARIABLE EXPECT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" EXPECT_VERSION "${EXPECT_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EXPECT DEFAULT_MSG EXPECT_COMMAND EXPECT_VERSION)

mark_as_advanced(EXPECT_COMMAND)
