# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

if(UNIX)
  set(SCCXX_LINK_OR_COPY create_symlink)
  set(SCCXX_DESTDIR $ENV{DESTDIR})
else()
  set(SCCXX_LINK_OR_COPY copy)
endif()

# CMAKE_EXECUTABLE_SUFFIX is undefined on cmake scripts. See PR9286.
if( WIN32 )
  set(EXECUTABLE_SUFFIX ".exe")
else()
  set(EXECUTABLE_SUFFIX "")
endif()

set(bindir "${SCCXX_DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/")
set(scc "scc${EXECUTABLE_SUFFIX}")
set(sccxx "sc++${EXECUTABLE_SUFFIX}")

message("Creating sc++ executable based on ${scc}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E ${SCCXX_LINK_OR_COPY} "${scc}" "${sccxx}"
  WORKING_DIRECTORY "${bindir}")
