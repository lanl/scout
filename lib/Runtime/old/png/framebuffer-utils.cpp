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
#include <limits.h>

#include "scout/Runtime/framebuffer.h"

using namespace scout;

#include <png.h>

// ----- save_framebuffer_as_png
//
// Todo: We're currently saving 8 bpp PNGs, for visualization results
// it would be nice to use 16-bit instead.  I've been unable to get
// this to work with the library...  --pm
//
// Todo: We currenty toss out the alpha channel from the frame
// buffer to avoid strange transparent PNG images -- thus a
// temporary 3-channel image buffer is created on each invocation.
// In a tight loop within Scout this could be a bottleneck...
bool save_framebuffer_as_png(const framebuffer_rt* fb, const char *filename)
{
  assert(fb != 0);
  assert(filename != 0);

  ucharp buf8 = new uchar[fb->width * fb->height * 3];
  index_t npixels = fb->width * fb->height;

  // See above -- this is not a good thing to do on every invocation
  // (although the png library does as well...).
  #pragma omp for schedule (dynamic,width)
  for(dim_t i = 0, j = 0; i < npixels; ++i, j+=3) {
    buf8[j]   = (uchar)(fb->pixels[i].x * UCHAR_MAX);
    buf8[j+1] = (uchar)(fb->pixels[i].y * UCHAR_MAX);
    buf8[j+2] = (uchar)(fb->pixels[i].z * UCHAR_MAX);
  }

  FILE *fp = fopen(filename, "wb");
  if (! fp) {
    fprintf(stderr, "could not save framebuffer as file: %s\n", filename);
    return false;
  }

  png_structp png_ptr;
  png_infop   info_ptr;
  png_bytep   *rows;

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (! png_ptr) {
    fprintf(stderr, "unable to create png struct pointer!\n");
    fclose(fp);
    return false;
  }

  info_ptr = png_create_info_struct(png_ptr);
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
               fb->width, fb->height,
               bit_depth,
               PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);


  png_write_info(png_ptr, info_ptr);

  rows = new png_bytep[fb->height];
  if (! rows) {
    fprintf(stderr, "unable to allocate png row pointers!\n");
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return false;
  }

  png_bytep fb_start = (png_bytep)buf8;
  int row_offset = fb->width * sizeof(uchar) * 3;
  #pragma omp for schedule (dynamic,width)
  for(int i = 0; i < fb->height; ++i) {
    rows[i] = fb_start + i * row_offset;
  }

  png_write_image(png_ptr, rows);
  png_write_end(png_ptr, info_ptr);

  fclose(fp);

  delete []rows;
  png_destroy_write_struct(&png_ptr, &info_ptr);
  delete []buf8;

  return true;
}

