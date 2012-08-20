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
#include <iostream>
#include <cassert>

#include <GL/glfw.h>

#include "runtime/opengl/glFW.h"
#include "runtime/opengl/vectors.h"

#ifdef __APPLE__

#include "runtime/init_mac.h"

#endif

using namespace std;
using namespace scout;

extern glFW* __sc_glfw;

void __handleMouseButton(int x, int y){
  __sc_glfw->mousePress(x, y);
}

void __handleMouseMove(int x, int y){
  __sc_glfw->mouseMove(x, y);
}

void __handleKeyPress(int x, int y){
  __sc_glfw->keyPress(x, y);
}

// ---- glFW
//
glFW::glFW()
{
  glFW(500, 500);
}

glFW::glFW(size_t width, size_t height, glCamera* camera)
  : glToolkit(camera)
{
  assert(glfwInit());

  size_t accumRedBits = 8;
  size_t accumGreenBits = 8;
  size_t accumBlueBits = 8;
  size_t fsaaSamples = 1;

  glfwOpenWindowHint(GLFW_ACCUM_RED_BITS, accumRedBits);
  glfwOpenWindowHint(GLFW_ACCUM_GREEN_BITS, accumGreenBits);
  glfwOpenWindowHint(GLFW_ACCUM_BLUE_BITS, accumBlueBits);
  glfwOpenWindowHint(GLFW_FSAA_SAMPLES, fsaaSamples);

  /*
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
  */

  size_t redBits = 8;
  size_t greenBits = 8;
  size_t blueBits = 8;
  size_t alphaBits = 8;
  size_t depthBits = 16;
  size_t stencilBits = 0;

  // 0 values mean use defaults

  assert(glfwOpenWindow(width, height, redBits, greenBits, blueBits, 
                        alphaBits, depthBits, stencilBits, GLFW_WINDOW));

  glfwEnable(GLFW_STICKY_KEYS);
  
  // do not automatically poll events when swap buffers is called
  glfwDisable(GLFW_AUTO_POLL_EVENTS);

  glfwSetMouseButtonCallback(__handleMouseButton);
  glfwSetMousePosCallback(__handleMouseMove);
  glfwSetKeyCallback(__handleKeyPress);

  resize(width, height);
  initialize();
}

// ---- ~glFW
//
glFW::~glFW()
{
  glfwTerminate();
}

void glFW::resize(size_t width, size_t height)
{ 
  glToolkit::resize(width, height);
}

// ---- paintMono
//
void glFW::paintMono()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (_camera != 0) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(_camera->fov, _camera->aspect, _camera->near, _camera->far);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(_camera->position.x, _camera->position.y, _camera->position.z,
        _camera->look_at.x, _camera->look_at.y, _camera->look_at.z,
        _camera->up[0], _camera->up[1], _camera->up[2]);

    if (_manipulator)
      _manipulator->apply();    

    RenderableList::iterator it;
    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(_camera);
    }
  }

}

void glFW::paintStereo()
{
#ifdef DOSTEREO
  const double DTOR = 0.0174532925;

  if (_camera != 0) {
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
    OpenGLErrorCheck();
    glFrustum(left, right, bottom, top, _camera->near, _camera->far);
    OpenGLErrorCheck();

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


    if (_manipulator)
      _manipulator->apply();

    RenderableList::iterator it;
    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(_camera);
    }

    // ****** Left 
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

    if (_manipulator)
      _manipulator->apply();

    for(it = _renderables.begin(); it != _renderables.end(); ++it) {
      if ((*it)->isHidden() == false)
        (*it)->render(_camera);
    }
  }
#endif
}

void glFW::keyPress(int a, int b)
{
  if (_ignore_events)
    return;
}

void glFW::mousePress(int x, int y)
{
  if (_ignore_events)
    return;

  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseReleaseLeft(x, y);
    update();
  }
}

void glFW::mouseMove(int x, int y)
{
  if (_ignore_events)
    return;

  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseMoveEvent(x, y);
    update();
  }
}

bool glFW::processEvent()
{
  glfwPollEvents();

  return false;
}

