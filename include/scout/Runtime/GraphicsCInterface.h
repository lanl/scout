/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 */

#ifndef __SCOUT_GRAPHICS_H__ 
#define __SCOUT_GRAPHICS_H__

#if defined(__cplusplus)
extern "C" {
#endif

  // These interface calls were prototyped by Pat McCormick in NewRuntime.

  // This call initializes the underlying graphics system.  It should
  // be called prior to any graphics operations -- in our case we hook
  // it into being run before the user's main() function is executed. 
  // 
  // CMS:  I made this return void.  Should make it return bool?  What would we do if it fails?
  void __scrt_init_graphics();

  // This is our sole opaque type for dealing with render targets
  // within the runtime.  Note that we also maintain a C-compatible
  // calling interface so we can support both Scout C and Scout C++
  // without multiple libraries.  That said, we use C++ under the hood
  // and thus C programs have to link with C++ (which we handle within
  // 'scc' so it should hopefully be transparent to the programmer).
  typedef void* __scrt_target_t;

  /// Create an on screen (window-based) rendering target of the given
  /// width and height in pixels.
  __scrt_target_t __scrt_create_window(unsigned short width, unsigned short height);

  float*
  __scrt_window_quad_renderable_colors(unsigned int width,
                                       unsigned int height, 
                                       unsigned int depth,
                                       void* renderTarget);
  
  float*
  __scrt_window_quad_renderable_vertex_colors(unsigned int width,
                                              unsigned int height, 
                                              unsigned int depth,
                                              void* renderTarget);

  float*
  __scrt_window_quad_renderable_edge_colors(unsigned int width,
                                            unsigned int height, 
                                            unsigned int depth,
                                            void* renderTarget);

  void __scrt_window_paint(void* renderTarget);

#if defined(__cplusplus)
}
#endif

#endif

