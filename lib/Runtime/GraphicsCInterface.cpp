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

#include "scout/Runtime/GraphicsCInterface.h"
#include "scout/Runtime/opengl/glfw/glfwDevice.h"
#include "scout/Runtime/opengl/glfw/glfwWindow.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

using namespace scout;

extern "C"
void __scrt_init_graphics() {
  // CMS:  We don't really need to keep track of this as a device
  // but we do so since that was already built -- can change that.
  glDevice* glDevice_ = glfwDevice::Instance();
}

extern "C"
__scrt_target_t __scrt_create_window(unsigned short width, unsigned short height) {
  assert(width != 0 && height != 0);
  glWindow *win = new glfwWindow(width, height);
  return (__scrt_target_t)win;
}

static glQuadRenderableVA*
get_renderable(unsigned int width,
               unsigned int height,
               unsigned int depth,
               void* renderTarget){

  glWindow* window = (glWindow*)renderTarget;
  window->makeContextCurrent();

  // TODO:  Check if there is already a quad renderable associated with this window 
  // and if so, try to reuse it.

  glQuadRenderableVA* renderable = 0; 

  if (window->getCurrentRenderable() != NULL) {
    // check here if right kind and size
    renderable = (glQuadRenderableVA*)(window->getCurrentRenderable());
  } else  {
    renderable = new glQuadRenderableVA( glfloat3(0.0, 0.0, 0.0), glfloat3(width, height, 0.0));

    // add to window's list of renderables

    window->addRenderable(renderable);
    window->makeCurrentRenderable(renderable);

    renderable->initialize(NULL); // also does a clear
  }

  return renderable;
}

extern "C"
float*
__scrt_window_quad_renderable_colors(unsigned int width,
                                     unsigned int height,
                                     unsigned int depth,
                                     void* renderTarget){

  glQuadRenderableVA* renderable = 
    get_renderable(width, height, depth, renderTarget);

  assert(renderable && "failed to get renderable");

  return (float*)renderable->map_colors();
}

extern "C"
float*
__scrt_window_quad_renderable_vertex_colors(unsigned int width,
                                            unsigned int height,
                                            unsigned int depth,
                                            void* renderTarget){

  glQuadRenderableVA* renderable = 
    get_renderable(width, height, depth, renderTarget);

  assert(renderable && "failed to get renderable");

  return (float*)renderable->map_vertex_colors();
}

extern "C"
float*
__scrt_window_quad_renderable_edge_colors(unsigned int width,
                                          unsigned int height,
                                          unsigned int depth,
                                          void* renderTarget){

  glQuadRenderableVA* renderable = 
    get_renderable(width, height, depth, renderTarget);

  assert(renderable && "failed to get renderable");

  return (float*)renderable->map_edge_colors();
}

extern "C"
void __scrt_window_paint(void* renderTarget) {
  glWindow* window = (glWindow*)renderTarget;
  // this is funky -- should be a separate function 
  // (__scrt_window_quad_renderable_unmap_colors)
  ((glQuadRenderableVA*)(window->getCurrentRenderable()))->unmap_colors();
  window->paint();
  window->swapBuffers();
  window->pollEvents();
}

extern "C"
void __scrt_window_paint_vertices(void* renderTarget) {
  glWindow* window = (glWindow*)renderTarget;
  // this is funky -- should be a separate function 
  // (__scrt_window_quad_renderable_unmap_colors)
  ((glQuadRenderableVA*)(window->getCurrentRenderable()))->
    unmap_vertex_colors();
  window->paint();
  window->swapBuffers();
  window->pollEvents();
}

extern "C"
void __scrt_window_paint_edges(void* renderTarget) {
  glWindow* window = (glWindow*)renderTarget;
  // this is funky -- should be a separate function 
  // (__scrt_window_quad_renderable_unmap_colors)
  ((glQuadRenderableVA*)(window->getCurrentRenderable()))->
    unmap_edge_colors();
  window->paint();
  window->swapBuffers();
  window->pollEvents();
}
