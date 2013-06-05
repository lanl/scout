#
###########################################################################
# Copyright (c) 2010, Los Alamos National Security, LLC.
# All rights reserved.
# 
#  Copyright 2010. Los Alamos National Security, LLC. This software was
#  produced under U.S. Government contract DE-AC52-06NA25396 for Los
#  Alamos National Laboratory (LANL), which is operated by Los Alamos
#  National Security, LLC for the U.S. Department of Energy. The
#  U.S. Government has rights to use, reproduce, and distribute this
#  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
#  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
#  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
#  derivative works, such modified software should be clearly marked,
#  so as not to confuse it with the version available from LANL.
#
#  Additionally, redistribution and use in source and binary forms,
#  with or without modification, are permitted provided that the
#  following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided 
#      with the distribution.  
#
#    * Neither the name of Los Alamos National Security, LLC, Los
#      Alamos National Laboratory, LANL, the U.S. Government, nor the
#      names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
#
#  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
#  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGE.
#
###########################################################################
#
# Notes 
#  
#  CMake module to search for the hwloc (hardware locality) library. 
#  The module sets the following values:
# 
#    - HWLOC_FOUND       : did we find hwloc? 
#    - HWLOC_INCLUDE_DIR : where to find the header files. 
#    - HWLOC_LIBRARY_DIR : where to find the hwloc libraries. 
#    - HWLOC_LIBRARIES    : libraries to link with when using hwloc. 
#
# If you have hwloc installed in a non-standard location (e.g. your
# home directory) you can use the HWLOC_DIR environment variable to
# help this module find what it is looking for. 
#
#####

##### INCLUDE DIRECTORY AND FILES 
#
find_path(HWLOC_INCLUDE_DIR
  hwloc.h
  PATH_SUFFIXES include include/hwloc 
  HINTS $ENV{HWLOC_DIR}
  PATHS
  $ENV{HWLOC_PREFIX}/include
  /usr/include
  /usr/local/include
  /usr/local/hwloc/include 
  /opt/include
  DOC "HWLOC (Portable Hardware Locality) include file directory."
)

#
#####
  
##### LIBRARIES
#
if (APPLE) 
  set(DYNLIB_EXT "dylib")
else()
  set(DYNLIB_EXT "so")
endif()

find_path(HWLOC_LIBRARY_DIR
  libhwloc.${DYNLIB_EXT}
  PATH_SUFFIXES lib64 lib hwloc/lib hwloc/lib64
  HINTS $ENV{HWLOC_DIR}
  PATHS
  /usr
  /usr/local
  /opt
  DOC "HWLOC (Portable Hardware Locality) library directory."
  )

if (HWLOC_LIBRARY_DIR AND HWLOC_INCLUDE_DIR)
  set(HWLOC_LIBRARIES "-lhwloc")
endif()

if (NOT APPLE) 
find_library(HWLOC_PCI_LIBRARY
  libpci.${DYNLIB_EXT}
  PATH_SUFFIXES lib lib64
  HINTS $ENV{LIBPCI_DIR}
  PATHS
  /
  /usr
  /usr/local
  /opt
  DOC "PCI library (for use w/ HWLOC)."
  )
endif()

if (HWLOC_PCI_LIBRARY) 
  set(HWLOC_LIBRARIES "${HWLOC_LIBRARIES} -lpci")
else()
  message(STATUS "hwloc: warning, hwloc found but PCI library seems to be missing on your system.")
  set(HWLOC_LIBRARIES ${HWLOC_LIBRARIES})
endif()
#
#####

unset(DYNLIB_EXT)

find_package_handle_standard_args(HWLOC
  REQUIRED_VARS 
  HWLOC_INCLUDE_DIR 
  HWLOC_LIBRARY_DIR
  )

