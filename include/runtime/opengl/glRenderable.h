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

#ifndef SCOUT_GL_RENDERABLE_H_
#define SCOUT_GL_RENDERABLE_H_

#include "runtime/opengl/glProgram.h"
#include "runtime/opengl/glCamera.h"

namespace scout
{
  
  // ..... glRenderable
  //
  class glRenderable {

   public:
    
    glRenderable();
    
    virtual ~glRenderable()
    { if (shader_prog != NULL) delete shader_prog; }

    void hide()
    { _hide = true; }

    void show()
    { _hide = false; }

    bool isHidden() const
    { return _hide; }
    
    glProgram* shaderProgram()
    { return shader_prog; }

    void attachShader(glVertexShader* shader)
    { shader_prog->attachShader(shader); }
      
    void attachShader(glGeometryShader* shader)
    { shader_prog->attachShader(shader); }

    void attachShader(glFragmentShader* shader)
    { shader_prog->attachShader(shader); }

    void render(glCamera* camera);
    
    void baseInitialize();

    virtual void initialize(glCamera* camera)
    {
      /* By default this is a no-op but use it if you want to
       * be called once OpenGL is initialized to configure your
       * own settings -- e.g. load and compile shaders, modify the
       * camera's position, modify the light position, etc.
       */
    }

    virtual void draw(glCamera* camera) = 0;

    virtual void timerEvent()
    { /* no-op by default */ }

    virtual void reloadShaders(const glCamera* camera)
    { /* no-op by default */  }

   protected:
    glProgram*   shader_prog;

   private:
    bool         _hide;
  };
  
}

#endif
