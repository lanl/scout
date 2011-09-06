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

#ifndef SCOUT_GL_TEXTURE_H_
#define SCOUT_GL_TEXTURE_H_

#include "glTypeTraits.h"
#include "glTextureParamter.h"

namespace scout
{
  
  // ..... glTexture
  // 
  class glTexture
  {
   public:
    
    glTexture(GLenum target, GLenum iformat, GLenum format, GLenum type);
    virtual ~glTexture();

    GLuint id() const
    { return _id; }

    GLenum target() const
    { return _target; }

    GLenum internalFormat() const
    { return _iformat; }
    
    GLenum pixelFormat() const
    { return _format; }

    GLenum type() const 
    { return _type; }

    void enable() const
    {
      glEnable(_target);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      glActiveTexture(_tex_unit);
      glBindTexture(_target, _id);
    }

    void disable() const
    {
      glDisable(_target);
      glBindTexture(_target, 0);      
    }

    virtual bool canDownload() = 0;

    bool isResident() const;

    virtual void initialize(void* data_p) = 0;
    
    virtual void read(void* dest_p) const = 0;

    void addParameter(GLenum name, GLint param)
    {
      glTextureParameter p = { name, param };
      _parameters.push_back(p);
    }
    
    void addParameter(glTextureParameter& param)
    { _parameters.push_back(param); }

    void assignTextureUnit(GLenum tex_unit)
    { _tex_unit = tex_unit; }

    GLenum textureUnit() const
    { return _tex_unit; }

  protected:
    void setParameters();
    
    glTexParameterList _parameters;
    GLenum             _target;
    GLenum             _iformat;
    GLenum             _format;
    GLenum             _type;

   private:
    GLuint             _id;
    GLenum             _tex_unit;
  };

  typedef shared_pointer<glTexture> pglTexture;
}

#endif
