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

  // This call initializes the underlying graphics system.  It should
  // be called prior to any graphics operations -- in our case we hook
  // it into being run before the user's main() function is executed. 
  bool __scout_init_graphics();
  
  // This is our sole opaque type for dealing with render targets
  // within the runtime.  Note that we also maintain a C-compatible
  // calling interface so we can support both Scout C and Scout C++
  // without multiple libraries.  That said, we use C++ under the hood
  // and thus C programs have to link with C++ (which we handle within
  // 'scc' so it should hopefully be transparent to the programmer).
  typedef void* __scout_target_t;

  /// Create an on screen (window-based) rendering target of the given
  /// width and height in pixels.
  __scout_target_t __scout_create_window(unsigned width, unsigned height);

  /// Create an off screen (image-based) rendering target of the given
  /// width and height in pixels.
  __scout_target_t __scout_create_image(unsigned width, unsigned height);

  /// Destroy the given rendering target.  This should be called when
  /// a render target goes out of scope, or at program exit.
  void __scout_destroy_target(__scout_target_t target);

  /// Bind the given render target -- this makes it the "active"
  /// target for all rendering operations.
  void    __scout_bind_target(__scout_target_t target);

  /// Relase the given render target -- this removes the target from
  /// "active" status.  If the target is not "active" this call
  /// essentially becomes a no-op.
  void    __scout_release_target(__scout_target_t target);

  /// Clear the render target's color buffer to the pre-assigned
  /// background color. 
  void    __scout_clear_target(__scout_target_t target);

  /// Swap the render target's front and back buffers.  If the target
  /// does not use double buffering this call becomes a no-op.
  void    __scout_swap_buffers(__scout_target_t target);


  typedef float float4 __attribute__((ext_vector_type(4)));
  
  /// Obtain a handle to the render target's color buffer.  Note that
  /// in the case of hardware accelerated targets this call will
  /// return null -- as we don't directly have a handle to the data in
  /// the frame buffer.  If you want the actual pixels call
  /// __scout_read_buffer() to obtain a copy of the pixel data.
  float4 *__scout_get_color_buffer(__scout_target_t target);

  /// Obtain a copy of the render target's color buffer.  Note that in
  /// the case of hardware accelerated targets this call will allocate
  /// a buffer and copy pixel data into it.  This buffer will not be
  /// kept up to date with the state of the buffer -- this call should
  /// be made post-rendering but prior to swapping buffers.
  const float4 *__scout_read_buffer(__scout_target_t target);

  /// Save the given render target's pixel data as a PNG image.  If the
  /// file is written successful true is returned, false otherwise. 
  bool __scout_save_as_png(__scout_target_t target, const char *filename);

  /// Save the given byte formated pixel data as a PNG image.  The
  /// pixel data is expected to contain three channels (red, green,
  /// blue) stored in AOS form and is width x height in pixels.  If
  /// the file is successfully written true is returned, false
  /// otherwise.
  bool __scout_write_png(unsigned char *buf8,
                         unsigned width, unsigned height,
                         const char *filename);
  
#if defined(__cplusplus)
}
#endif

#endif

