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
#   This is a simplified find module for the SDL library.  
#
#   TODO: We don't currently handle frameworks under MacOS X.  In 
#         addition we don't distinguish between dynamic and static 
#         libraries for SDL.  Do we need to support versioning checks
#         for SDL?
#
#   This module sets the following variables:
#
#     SDL_FOUND       - Set to TRUE if SDL files are found in search paths. 
#     SDL_INCLUDE_DIR - Path for reaching SDL's header files. 
#     SDL_LIBRARY_DIR - Path to SDL's library file(s).
#     SDL_LIBRARIES   - Set of libraries required for linking with SDL. 
# 
#####

  ##### INCLUDE DIRECTORY AND FILES
  #
    find_path(SDL_INCLUDE_DIR 
      SDL/SDL.h
      PATH_SUFFIXES include include/SDL sdl/include
      HINTS $ENV{SDL_DIR}
      PATHS
      /usr
      /usr/local
      /opt
      /opt/local
      DOC "Path for SDL's include files."
      )
  #
  #####


  #####
  #
    find_path(SDL_LIBRARY_DIR
      NAMES libSDL.a libSDL.so
      PATH_SUFFIXES lib64 sdl/lib64 lib sdl/lib lib/x86_64-linux-gnu 
      HINTS $ENV{SDL_DIR}
      /usr
      /usr/local
      /opt
      /opt/local
      DOC "Path to SDL library."
      )
  #
  #####

  if (SDL_LIBRARY_DIR) 
    set(SDL_LIBRARIES "-lSDL -lSDLmain")
  endif()

  find_package_handle_standard_args(SDL
    REQUIRED_VARS 
    SDL_LIBRARY_DIR 
    SDL_INCLUDE_DIR)
