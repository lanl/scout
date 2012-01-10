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
#include "runtime/opengl/glSDL.h"
#include "runtime/opengl/vectors.h"

#ifdef __APPLE__

#include "runtime/init_mac.h"

#endif

using namespace std;
using namespace scout;

extern glSDL* __sc_glsdl;


// ---- glSDL
//
glSDL::glSDL()
{
  glSDL(500, 500);
}

glSDL::glSDL(size_t width, size_t height)
  :_surface(NULL), glToolkit()
{


#ifdef __APPLE__
  scoutInitMac();
#endif

  if(SDL_Init(SDL_INIT_VIDEO) < 0){
    cerr << "Error: failed to initialize SDL." << endl;
    exit(1);
  }

  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
  SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);

  SDL_GL_SetAttribute(SDL_GL_ACCUM_RED_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_GREEN_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_BLUE_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_ALPHA_SIZE, 8);

  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 2);

  resize(width, height);
  initialize();

}

// ---- ~glSDL
//
glSDL::~glSDL()
{
  if (_surface != NULL) SDL_FreeSurface(_surface);
}

void glSDL::resize(size_t width, size_t height)
{ 

  if (_surface) {
    SDL_FreeSurface(_surface);
  }

  _surface = SDL_SetVideoMode(width, height, 32,
      SDL_HWSURFACE |
      SDL_RESIZABLE |
      SDL_GL_DOUBLEBUFFER |
      SDL_OPENGL);

  if (_surface == NULL) {
    cerr << "Error: SDL_SetVideoMode failed: " << SDL_GetError() << endl;
    exit(1);
  }

  glToolkit::resize(width, height);

}

// ---- paintMono
//
void glSDL::paintMono()
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

void glSDL::paintStereo()
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

// ---- keyPressEvent
//
void glSDL::keyPressEvent()
{
  if (_ignore_events)
    return;

  switch(_event.key.keysym.sym) {

    case SDLK_ESCAPE:
      _event.type = SDL_QUIT;
      break;

    case SDLK_t:
      // not sure what to do here
      break;

    case SDLK_r:
      {
        RenderableList::iterator it;
        for(it = _renderables.begin(); it != _renderables.end(); ++it) {
          (*it)->reloadShaders(_camera);
        }
        update();
      }
      break;

    default:
      break;
  }
}


// --- keyReleaseEvent
//
void glSDL::keyReleaseEvent()
{

}


// ---- mousePressLeft
//
void glSDL::mousePressLeft()
{
  if (_ignore_events)
    return;

  if (_manipulator)
    _manipulator->mousePressLeft(_event.motion.x, _event.motion.y);
}

// ---- mousePressMiddle
//
void glSDL::mousePressMiddle()
{
  if (_ignore_events)
    return;

  if (_manipulator)
    _manipulator->mousePressMiddle(_event.motion.x, _event.motion.y);
}

// ---- mousePressRight
//
void glSDL::mousePressRight()
{
  if (_ignore_events)
    return;

  if (_manipulator)
    _manipulator->mousePressRight(_event.motion.x, _event.motion.y);
}

// ---- mouseReleaseLeft
//
void glSDL::mouseReleaseLeft()
{
  if (_ignore_events)
    return;

  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseReleaseLeft(_event.motion.x, _event.motion.y);
    update();
  }
}


// ---- mouseReleaseMiddle
//
void glSDL::mouseReleaseMiddle()
{
  if (_ignore_events)
    return;

  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseReleaseMiddle(_event.motion.x, _event.motion.y);
    update();
  }
}


// ---- mouseReleaseRight
//
void glSDL::mouseReleaseRight()
{
  if (_ignore_events)
    return;

  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseReleaseMiddle(_event.motion.x, _event.motion.y);
    update();
  }
}

// ---- mouseMoveEvent
//
void glSDL::mouseMoveEvent()
{
  if (_ignore_events)
    return;

  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseMoveEvent(_event.motion.x, _event.motion.y);
    update();
  }
}


void glSDL::resizeEvent() 
{
  resize(_event.resize.w, _event.resize.h);
  paint();

}

// pass it a function to execute on each iteration
void glSDL::eventLoop() 
{
  bool done = false;
  while(!done) {
    done = processEvent();
    paint();
    SDL_GL_SwapBuffers(); 
  }
}

bool glSDL::processEvent()
{
  bool done = false;
  if (!SDL_PollEvent(&_event)) return done; 

  switch(_event.type) {
    case SDL_USEREVENT:
      // do something?
      break;

    case SDL_KEYDOWN:
      keyPressEvent();  
      if (_event.type == SDL_QUIT) 
      {
        done = true;
      }
      break;

    case SDL_MOUSEBUTTONDOWN:
      switch(_event.button.button) {
        case SDL_BUTTON_LEFT:
          mousePressLeft();
          break;
        case SDL_BUTTON_MIDDLE:
          mousePressMiddle();
          break;
        case SDL_BUTTON_RIGHT:
          mousePressRight();
          break;
        default:
          break;
      }
      break;

    case SDL_MOUSEBUTTONUP:
      switch(_event.button.button) {
        case SDL_BUTTON_LEFT:
          mouseReleaseLeft();
          break;
        case SDL_BUTTON_MIDDLE:
          mouseReleaseMiddle();
          break;
        case SDL_BUTTON_RIGHT:
          mouseReleaseRight();
          break;
        default:
          break;
      }
      break;

    case SDL_MOUSEMOTION:
      mouseMoveEvent();
      break;

    case SDL_VIDEORESIZE:
      resizeEvent();
      break;

    case SDL_QUIT:
      done = true;
      break;

    default:
      break;
  }   // End switch

  return done;
}

void glSDL::update() {
  paint();
}


