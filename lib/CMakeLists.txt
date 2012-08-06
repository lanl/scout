#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#

project(ScLibraries)

add_subdirectory(standard)    # Scout's standard library
add_subdirectory(Runtime)     # Scout's runtime library
