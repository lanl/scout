/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
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
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 

#ifndef SCOUT_OPENGL_TEXTURE_H_
#define SCOUT_OPENGL_TEXTURE_H_

#include "scout/Runtime/opengl/glTextureParamter.h"

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

    virtual bool canDownload() const = 0;

    bool isResident() const;

    virtual void initialize(const float* data_p) = 0;
    
    virtual void read(float* dest_p) const = 0;

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

    virtual void update(const float *p_data) = 0;

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

}

#endif
