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
#include <cassert>

#include "scout/new-runtime/graphics.h"
#include "scout/new-runtime/Image.h"
#include "scout/new-runtime/Window.h"

using namespace scout;

extern "C"
__scout_target_t __scout_create_image(unsigned width, unsigned height) {
  assert(width != 0 && height != 0);
  Image *img = new Image(width, height);
  return (__scout_target_t)img;
}

extern "C"
__scout_target_t __scout_create_window(unsigned width, unsigned height) {
  assert(width != 0 && height != 0);
  Window *win = new Window(width, height);
  return (__scout_target_t)win;
}

extern "C"
void __scout_destroy_target(__scout_target_t target) {
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  RT->release(); // Make sure we release the target if it was active.
  delete RT;
}

extern "C"
void __scout_bind_target(__scout_target_t target) {
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  RT->bind();
}

extern "C"
void __scout_release_target(__scout_target_t target) {
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  RT->release();
}

extern "C"
void __scout_clear_target(__scout_target_t target) {
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  RT->clear();
}

extern "C"
void __scout_swap_buffers(__scout_target_t target) {
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  RT->swapBuffers();
}

extern "C"
float4 *__scout_get_color_buffer(__scout_target_t target) {
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  return RT->getColorBuffer();
}

extern "C"
bool __scout_save_as_png(__scout_target_t target, const char *filename) {
  assert(filename != 0);
  RenderTarget *RT = (RenderTarget*)target;
  assert(RT != 0);
  RT->savePNG(filename);
}

