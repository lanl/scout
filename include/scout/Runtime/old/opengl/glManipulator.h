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

#ifndef SCOUT_GL_MANIPULATOR_H_
#define SCOUT_GL_MANIPULATOR_H_

#include "scout/Runtime/opengl/glCamera.h"

namespace scout
{

  // ---- glManipulator
  //
  class glManipulator {

   public:
    glManipulator(int win_width = 0, int win_height = 0);
    
    virtual ~glManipulator() 
    { /* no-op */ }

    virtual bool isActive() const = 0;
    
    virtual void resize(int win_width, int win_height)
    {
      _win_width  = win_width;
      _win_height = win_height;
    }

    virtual void keyPressEvent()       = 0;
    virtual void keyReleaseEvent()     = 0;    
    virtual void mousePressLeft(int x, int y)   = 0;
    virtual void mousePressMiddle(int x, int y)   = 0;
    virtual void mousePressRight(int x, int y)   = 0;
    virtual void mouseReleaseLeft(int x, int y) = 0;
    virtual void mouseReleaseMiddle(int x, int y) = 0;
    virtual void mouseReleaseRight(int x, int y) = 0;
    virtual void mouseMoveEvent(int x, int y)    = 0;

    virtual void timerEvent() = 0;
    virtual void apply() = 0;

    void setCamera(glCamera* cam)
    { _camera = cam; }

    glCamera* camera() const
    { return _camera; }
    
   protected:
    void activate()
    { _active = true; }

    void deactivate()
    { _active = false; }

    bool      _active;
    int       _win_width, _win_height;
    glCamera* _camera;    
  };

}

#endif
