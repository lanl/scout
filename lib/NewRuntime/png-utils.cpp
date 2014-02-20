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
#include <png.h>


#include "scout/new-runtime/graphics.h"
#include "scout/new-runtime/Image.h"
#include "scout/new-runtime/Window.h"

using namespace scout;

extern "C"
bool __scout_write_png(unsigned char *buf8,
                       unsigned width, unsigned height,
                       const char *filename) {

  FILE *fp = fopen(filename, "wb");
  if (! fp) {
    fprintf(stderr, "scout runtime: could not open file: %s\n", filename);
    return false;
  }

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (! png_ptr) {
    fprintf(stderr, "unable to create png struct pointer!\n");
    fclose(fp);
    return false;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (! info_ptr) {
    fprintf(stderr, "unable to create png info structure!\n");
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

    fclose(fp);
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    fprintf(stderr, "png setjmp() error!\n");
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return false;
  }

  png_init_io(png_ptr, fp);
  int bit_depth = 8;
  png_set_IHDR(png_ptr, info_ptr,
               width, height,
               bit_depth,
               PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);


  png_write_info(png_ptr, info_ptr);

  png_bytep *rows = new png_bytep[height];
  if (! rows) {
    fprintf(stderr, "unable to allocate png row pointers!\n");
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return false;
  }

  png_bytep fb_start = (png_bytep)buf8;
  int row_offset = width * sizeof(unsigned char) * 3;
  for(int i = 0; i < height; ++i) {
    rows[i] = fb_start + i * row_offset;
  }

  png_write_image(png_ptr, rows);
  png_write_end(png_ptr, info_ptr);

  fclose(fp);

  delete []rows;
  png_destroy_write_struct(&png_ptr, &info_ptr);
  return true;
}


