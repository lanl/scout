/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 */

#include <cassert>
#include "scout/Runtime/framebuffer.h"

using namespace scout;


// ----- framebuffer_rt
//
framebuffer_rt::framebuffer_rt(dim_t w, dim_t h, float r, float g, float b, float a)
{
  assert(w > 0 && h > 0);

  width  = w;
  height = h;

  pixels = new float4[width * height];

  bg_color.x = r;
  bg_color.y = g;
  bg_color.z = b;
  bg_color.w = a;

  this->clear();
}


// ----- ~framebuffer_rt
//
framebuffer_rt::~framebuffer_rt()
{
  delete []pixels;
}


// ----- clear
//
void framebuffer_rt::clear()
{
  index_t size = width * height;

  #pragma omp for schedule (dynamic,width)
  for(index_t i = 0; i < size; ++i) {
    pixels[i] = bg_color;
  }
}

