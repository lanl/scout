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

#include "runtime/opengl/glToolkit.h"
#include "runtime/opengl/opengl.h"

using namespace scout;
using namespace std;

glToolkit::glToolkit(glCamera* camera) 
{
  _img_seq_num    = 0;
  _save_frames    = false;
  _stereo_mode    = false;
  _ignore_events  = false;
  _camera  =  camera;
  _manipulator = NULL;
}

// ---- ~glToolkit
//
glToolkit::~glToolkit()
{
}

// ---- saveFrames
//
void glToolkit::saveFrames(bool save_frames, const std::string& basename)
{
  _save_frames = save_frames;
  _frame_basename = basename;
  update();
}


// ---- initialize
//
void glToolkit::initialize()
{
  glClearColor(_bg_color[0], _bg_color[1], _bg_color[2], _bg_color[3]);
  glEnable(GL_DEPTH_TEST);

  RenderableList::iterator it;
  for(it = _renderables.begin(); it != _renderables.end(); ++it) {
    (*it)->baseInitialize();
    (*it)->initialize(_camera);
  }

}

// ---- paint
//
void glToolkit::paint()
{
  //cout << "paint" << endl;

  if (_ignore_events)
    return;

  if (_stereo_mode)
    paintStereo();
  else
    paintMono();
}


// ---- resize
//
void glToolkit::resize(int width, int height)
{
  //uncomment this if you want the sim to follow the size of the window
  //otherwise, it stays the same size regardless of window resize.

//  glViewport(0, 0, width, height);


  if (_camera != 0) {
    _camera->resize(width, height);
  }

  if (_manipulator)
    _manipulator->resize(width, height);
}


void glToolkit::saveState()
 {
     glPushAttrib(GL_ALL_ATTRIB_BITS);
     glMatrixMode(GL_PROJECTION);
     glPushMatrix();
     glMatrixMode(GL_MODELVIEW);
     glPushMatrix();
 }

void glToolkit::restoreState()
 {
     glMatrixMode(GL_PROJECTION);
     glPopMatrix();
     glMatrixMode(GL_MODELVIEW);
     glPopMatrix();
     glPopAttrib();
 }

void glToolkit::setDisplayText(int x, int y, string str)
{
  _text_x = x;
  _text_y = y;
  _display_text = str;
}

