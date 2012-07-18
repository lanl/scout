
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

#ifndef SCOUT_GL_FW_H_
#define SCOUT_GL_FW_H_

#include <GL/glfw.h>

#include "runtime/opengl/glToolkit.h"

namespace scout
{
  
  class glFW : public glToolkit {

   public:
    glFW();
    glFW(size_t width, size_t height, glCamera* camera = NULL);
    ~glFW();

   public:
    void resize(size_t width, size_t height);
    void paintMono();    
    void paintStereo();    

    bool processEvent();
    void eventLoop();

    void keyPress(int a, int b);
    void mousePress(int x, int y);
    void mouseMove(int x, int y);
    void swapBuffers() { glfwSwapBuffers(); }
    void update() {}


   private:
    
  };
  
}

#endif // SCOUT_GL_FW_H_

