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
#   This module sets the following variables: 
#
#     GLFW_FOUND       - Set to true if GLFW is found. 
#     GLFW_INCLUDE_DIR - Path for reaching GLFW header files. 
#     GLFW_LIBRARY_DIR - Path to GLFW library file(s). 
#     GLFW_LIBRARIES   - Set of libraries required for linking.
#
#####

  ##### INCLUDE DIRECTORY AND FILES 
  # 
  find_path(GLFW_INCLUDE_DIR
    GLFW/glfw3.h
    PATH_SUFFIXES include glfw/include 
    HINTS $ENV{GLFW_DIR}
    PATHS
    /usr
    /usr/local
    /opt/
    /opt/local
    DOC "Path for GLFW's include files."
    )
  #
  ##### 

  ##### LIBRARY DIRECTORY
  # 
  find_path(GLFW_LIBRARY_DIR
    libglfw3.a 
    PATH_SUFFIXES lib64 lib glfw/lib64 glfw/lib 
    HINTS $ENV{GLFW_DIR}
    PATHS
    /usr
    /usr/local
    /opt
    /opt/local 
    DOC "Path to GLFW libraries."
    )
  #
  ##### 

  if (GLFW_LIBRARY_DIR) 
    set(GLFW_LIBRARIES "-lglfw3")
  endif()

  find_package_handle_standard_args(GLFW
    REQUIRED_VARS 
    GLFW_LIBRARY_DIR 
    GLFW_INCLUDE_DIR
    )
