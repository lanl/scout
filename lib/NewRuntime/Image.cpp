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
#include <limits.h>
#include "scout/new-runtime/graphics.h"
#include "scout/new-runtime/Image.h"
using namespace scout;

Image::Image(unsigned width, unsigned height)
    : RenderTarget(RTK_image, width, height) {
  ColorBuffer = new float4[Width * Height];
  // By default we do not initialize the image -- this gives us a
  // somewhat similar behavior pattern to the hardware framebuffer
  // semantics. 
}

Image::~Image() {
  delete []ColorBuffer;
}

void Image::bind() {
  if (ColorBuffer) 
    RenderTarget::setActiveTarget(this);
}

void Image::release() {
  RenderTarget *RT = RenderTarget::getActiveTarget();
  if (RT == this) // Don't accidently trash another active RT. 
    RenderTarget::setActiveTarget(0);
}

void Image::clear() {
  if (ColorBuffer) {
    unsigned NPixels = Width * Height;
    for(unsigned i = 0; i < NPixels; ++i) {
      ColorBuffer[i] = Background;
    }
  }
}

bool Image::savePNG(const char *filename) {
  bool retval = false;
  if (ColorBuffer) {
    unsigned NPixels = Width * Height;
    // Note we drop alpha from the channels or you can end up with an
    // odd transparent png image. 
    unsigned char *buf8 = new unsigned char[NPixels * 3];
    for(unsigned npix = 0, upix = 0; npix < NPixels; ++npix, upix += 3) {
      buf8[upix]   = (unsigned char)(ColorBuffer[npix].x * UCHAR_MAX);
      buf8[upix+1] = (unsigned char)(ColorBuffer[npix].y * UCHAR_MAX);
      buf8[upix+2] = (unsigned char)(ColorBuffer[npix].z * UCHAR_MAX);
    }
    retval = __scout_write_png(buf8, Width, Height, filename);
    delete []buf8;
  } 
    
  return retval;
}


