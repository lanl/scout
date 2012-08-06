/*
 *           -----  The Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * $Revision$
 * $Date$
 * $Author$
 *
 *----- 
 * 
 */

#ifndef SCOUT_GL_TYPE_TRAITS_H_
#define SCOUT_GL_TYPE_TRAITS_H_

#include "opengl.h"
#include "scout/Runtime/vec_types.h"

namespace scout
{
  template <typename T>
  struct glTypeTraits {
    enum { type = GL_NONE };
  };

#define DEFINE_GL_TYPE_TRAIT(BASE_TYPE, GL_TYPE)        \
  template <>                                           \
  struct glTypeTraits<BASE_TYPE> {                      \
    enum { type = GL_TYPE };                            \
  }

  DEFINE_GL_TYPE_TRAIT(float,   GL_FLOAT);
  DEFINE_GL_TYPE_TRAIT(float4,  GL_FLOAT);

  // ..... glTextureTraits
  // 
  template <typename T>
  class glTextureTraits
  {
   public:

    enum
    {
      iformat    = GL_NONE,
      format     = GL_NONE,
      type       = glTypeTraits<T>::type,
      min_filter = GL_NONE,
      mag_filter = GL_NONE,
      tex_func   = GL_NONE,
      wrap_s     = GL_CLAMP_TO_EDGE,
      wrap_t     = GL_CLAMP_TO_EDGE,
      wrap_r     = GL_CLAMP_TO_EDGE,
      env_mode   = GL_REPLACE
    };
  };


#define DEF_GL_TEXTURE_TRAIT(T, IFORMAT, FORMAT)        \
  template <>                                           \
  class glTextureTraits<T>                              \
  {                                                     \
    public:                                             \
    enum                                                \
    {                                                   \
      iformat    = IFORMAT,                             \
      format     = FORMAT,                              \
      type       = glTypeTraits<T>::type,               \
    };                                                  \
  }

  DEF_GL_TEXTURE_TRAIT(float, GL_LUMINANCE32F_ARB, GL_LUMINANCE);
}

#endif
