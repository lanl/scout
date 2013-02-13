
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
#ifndef SCOUT_GL_SDL_H_
#define SCOUT_GL_SDL_H_
#include <SDL/SDL.h>
#include "scout/Runtime/opengl/glToolkit.h"
namespace scout
{
  class glSDL : public glToolkit {

   public:
    glSDL();
    glSDL(size_t width, size_t height, glCamera* camera = NULL);
    ~glSDL();

   public:
    void resize(size_t width, size_t height);
    void update(); // { SDL_UpdateRect(_surface, 0, 0, 0, 0); }
    void paintMono();    
    void paintStereo();    

    bool processEvent();
    void eventLoop();

    void keyPressEvent();
    void keyReleaseEvent();        
    void mousePressLeft();
    void mousePressMiddle();
    void mousePressRight();
    void mouseReleaseLeft();    
    void mouseReleaseMiddle();    
    void mouseReleaseRight();    
    void mouseMoveEvent();
    void resizeEvent();
    void swapBuffers() { SDL_GL_SwapBuffers();}


   private:
    SDL_Surface*      _surface;
    SDL_Event         _event;
  };
}
#endif
