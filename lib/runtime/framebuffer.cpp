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
#include "runtime/framebuffer.h"

using namespace scout;


// ----- framebuffer_t
//
framebuffer_t::framebuffer_t(dim_t w, dim_t h, float r, float g, float b, float a)
{
  assert(width > 0 && height > 0);
  
  width  = w;
  height = h;

  pixels = new float4[width * height];
  
  bg_color.components[0] = r;
  bg_color.components[1] = g;
  bg_color.components[2] = b;
  bg_color.components[3] = a;

  this->clear();
}


// ----- ~framebuffer_t
//
framebuffer_t::~framebuffer_t()
{
  delete []pixels;
}


// ----- clear
//
void framebuffer_t::clear()
{
  index_t size = width * height;

#pragma omp for schedule (dynamic,width)
  for(index_t i = 0; i < size; ++i) {
    pixels[i].vec = bg_color.vec;
  }
}

    
