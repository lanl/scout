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

#ifndef SCOUT_RENDERALL_UTILITIES_H_
#define SCOUT_RENDERALL_UTILITIES_H_

#include "scout/types.h"

namespace scout
{
  class framebuffer_rt;
  class viewport_rt;

  enum MapFilterType {
    FILTER_NEAREST,
    FILTER_LINEAR
  };

  void mapToFrameBuffer(const float4 *colors, int dataw, int datah,
                        framebuffer_rt &fb, const viewport_rt &vp,
                        MapFilterType filter);

}

#endif

