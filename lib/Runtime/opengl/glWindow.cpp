/*
 * ###########################################################################
 * Copyright(c) 2010, Los Alamos National Security, LLC.
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

#include "scout/Runtime/opengl/glWindow.h"
#include <cassert>
#include <limits.h>

using namespace std;
using namespace scout;

/// Create a window of the given width and height.  The window's
/// position is undetermined and the background color will default
/// to black.
glWindow::glWindow(ScreenCoord width, ScreenCoord height, glCamera* camera)
  : RenderTarget(RTK_window, width, height), _frame(width, height), _colorBuffer(0), 
  _camera(camera), _currentRenderable(NULL)  {
  cout << "init window1" << endl;
}

/// Create a window with the given location and size (as described
/// by the given WindowRect).  The background color will default
/// to black.
glWindow::glWindow(const WindowRect &rect)
  : RenderTarget(RTK_window, rect.size.width, rect.size.height), _frame(rect), _colorBuffer(0), 
  _camera(0), _currentRenderable(NULL) {

  cout << "init window2" << endl;
}

glWindow::~glWindow() {
  unsetActive();
  for (RenderableList::iterator it(_renderables.begin()); it != _renderables.end(); ++it)
  {
    delete (*it);
  }
}

void glWindow::setActive() {
    RenderTarget::setActiveTarget(this);
    makeContextCurrent();
}

void glWindow::unsetActive() {
  RenderTarget *RT = RenderTarget::getActiveTarget();
  if (RT == this) { // Don't accidently trash another active RT. 
    RenderTarget::setActiveTarget(0);
    makeContextNotCurrent();
  }
}

void glWindow::clear() {
    assert(RenderTarget::getActiveTarget() == this && "target must be bound before clear()");
    glClearColor(_bgColor.x, _bgColor.y, _bgColor.z, _bgColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
  
const float4 *glWindow::readColorBuffer() {
  assert(RenderTarget::getActiveTarget() == this && "target must be bound before reading buffer");
  if (_colorBuffer == 0) {
    _colorBuffer = new float4[_width * _height];
  }
  glReadPixels(0, 0, _width, _height, GL_RGBA, GL_FLOAT, (GLvoid*)_colorBuffer);
  return _colorBuffer;
}

bool glWindow::savePNG(const char *filename) {
  assert(RenderTarget::getActiveTarget() == this && "target must be bound before reading buffer");
  const float4* buf = readColorBuffer();
  unsigned NPixels = _width * _height;
  unsigned char *buf8 = new unsigned char[NPixels * 3];
  for(unsigned npix = 0, upix = 0; npix < NPixels; ++npix, upix += 3) {
    buf8[upix]   = (unsigned char)(_colorBuffer[npix].x * UCHAR_MAX);
    buf8[upix+1] = (unsigned char)(_colorBuffer[npix].y * UCHAR_MAX);
    buf8[upix+2] = (unsigned char)(_colorBuffer[npix].z * UCHAR_MAX);
  }
  //bool retval = __scout_write_png(buf8, _width, _height, filename);
  delete []buf8;
  //return retval;
  return false;
}


// ---- paintMono
//
void glWindow::paintMono()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  if (_camera != 0) {
    /*
      
    // the renderable is now responsible for doing this...

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(_camera->fov, _camera->aspect, _camera->near, _camera->far);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    gluLookAt(_camera->position.x, _camera->position.y, _camera->position.z,
              _camera->look_at.x, _camera->look_at.y, _camera->look_at.z,
              _camera->up[0], _camera->up[1], _camera->up[2]);
    */
    
    if (_manipulator)
      _manipulator->apply();
    
    RenderableList::iterator it;
    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(_camera);
    }
  } else {
    RenderableList::iterator it;
    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(NULL);
    }
  }
}

void glWindow::paintStereo()
{
#ifdef DOSTEREO
  const double DTOR = 0.0174532925;
  
  if (_camera != 0) {
    /*

    // the renderable is now responsible for doing this...

    double ratio   = _camera->win_width / (double)_camera->win_height;
    double radians = DTOR * _camera->fov / 2.0;
    double wd2     = _camera->near * tan(radians);
    double ndfl    = _camera->near / _camera->focal_length;
    
    float3 view_dir = _camera->look_at - _camera->position;
    float3 r = cross(view_dir, _camera->up);
    r = normalize(r);
    r.x *= _camera->eye_sep / 2.0;
    r.y *= _camera->eye_sep / 2.0;
    r.z *= _camera->eye_sep / 2.0;
    
    // ****** Left
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double left   = -ratio * wd2 - 0.5 * _camera->eye_sep * ndfl;
    double right  =  ratio * wd2 - 0.5 * _camera->eye_sep * ndfl;
    double top    =  wd2;
    double bottom = -wd2;
    oglErrorCheck();
    glFrustum(left, right, bottom, top, _camera->near, _camera->far);
    oglErrorCheck();
    
    glDrawBuffer(GL_BACK_RIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(_camera->position.x + r.x,
              _camera->position.y + r.y,
              _camera->position.z + r.z,
              
              _camera->position.x + r.x + view_dir.x,
              _camera->position.y + r.y + view_dir.y,
              _camera->position.z + r.z + view_dir.z,
              
              _camera->up[0],
              _camera->up[1],
              _camera->up[2]);
    
    */

    if (_manipulator)
      _manipulator->apply();
    
    RenderableList::iterator it;
    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(_camera);
    }
    
    // ****** Right
    /*

    // the renderable is now responsible for doing this...

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    left   = -ratio * wd2 + 0.5 * _camera->eye_sep * ndfl;
    right  =  ratio * wd2 + 0.5 * _camera->eye_sep * ndfl;
    glFrustum(left, right, bottom, top, _camera->near, _camera->far);
    
    
    glDrawBuffer(GL_BACK_LEFT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    gluLookAt(_camera->position.x - r.x,
              _camera->position.y - r.y,
              _camera->position.z - r.z,
              _camera->position.x - r.x + view_dir.x,
              _camera->position.y - r.y + view_dir.y,
              _camera->position.z - r.z + view_dir.z,
              _camera->up[0],
              _camera->up[1],
              _camera->up[2]);
    */

    if (_manipulator)
      _manipulator->apply();
    
    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(_camera);
    }
  }
#endif
}
