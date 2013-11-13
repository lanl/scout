# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.
if(UNIX)
  set(SCXX_LINK_OR_COPY create_symlink)
  set(SCXX_DESTDIR $ENV{DESTDIR})
else()
  set(SCXX_LINK_OR_COPY copy)
endif()

# CMAKE_EXECUTABLE_SUFFIX is undefined on cmake scripts. See PR9286.
if( WIN32 )
  set(EXECUTABLE_SUFFIX ".exe")
else()
  set(EXECUTABLE_SUFFIX "")
endif()

set(bindir "${SCXX_DESTDIR}${SCOUT_BUILD_DIR}/bin/")
set(scc "scc${EXECUTABLE_SUFFIX}")
set(scxx "sc++${EXECUTABLE_SUFFIX}")

message("Creating sc++ executable based: ${bindir}/${scxx} -> ${scc}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E ${SCXX_LINK_OR_COPY} "${scc}" "${scxx}"
  WORKING_DIRECTORY "${bindir}")
