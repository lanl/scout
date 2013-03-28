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


#ifndef SCOUT_GL_SDL_H_
#define SCOUT_GL_SDL_H_
#include <SDL/SDL.h>
#include "scout/Runtime/opengl/glToolkit.h"

const size_t __sc_initial_width = 768;
const size_t __sc_initial_height = 768;

namespace scout
{
  class glSDL : public glToolkit {

   protected:
    glSDL();
    glSDL(size_t width, size_t height, glCamera* camera = NULL);
    ~glSDL();
    glSDL(const glSDL&);
    glSDL& operator= (const glSDL&);

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
    static glSDL* Instance(size_t width = __sc_initial_width, size_t height = __sc_initial_height);

   private:
    static glSDL*     _instance;
    SDL_Surface*      _surface;
    SDL_Event         _event;
  };
}
#endif
