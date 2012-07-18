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

#ifndef SCOUT_GL_TOOLKIT_H_
#define SCOUT_GL_TOOLKIT_H_
#include <deque>
#include "runtime/opengl/glCamera.h"
#include "runtime/opengl/glRenderable.h"
#include "runtime/opengl/glManipulator.h"

namespace scout
{

  // ..... glToolkit
  //
  class glToolkit {

   public:
  
    glToolkit(glCamera* camera = NULL);

    void addRenderable(glRenderable *r) 
    { _renderables.push_back(r); }

    glCamera* camera() { return _camera; }

    void setCamera(glCamera* c)
    {
      _camera = c;
      if (_manipulator)
        _manipulator->setCamera(c);
    }

    glManipulator* manipulator()
    { return _manipulator; }

    void setManipulator(glManipulator* m)
    {
      _manipulator = m;
      if (_camera)
        _manipulator->setCamera(_camera);
    }
    
    void refresh()
    { update(); }

    void saveFrames(bool save_frames, const std::string& basename);

    bool isSavingFrames() const
    { return _save_frames; }

    void enableStereo()
    {
      _stereo_mode = true;
      update();
    }

    void disableStereo()
    {
      _stereo_mode = false;
      update();
    }

    void disableEvents()
    { _ignore_events = true; }

    void enableEvents()
    { _ignore_events = false; }    

    void setBackgroundColor(const glfloat4& c)
    { _bg_color = c; }

    void setBackgroundColor(float r, float g, float b, float a = 1.0)
    {
      _bg_color[0] = r;
      _bg_color[1] = g;
      _bg_color[2] = b;
      _bg_color[3] = a;
    }

		void setDisplayText(int x, int y, std::string text);

   public:
    virtual ~glToolkit();
    void initialize(); 
    virtual void update() = 0; 
    void paint(); 
    virtual void paintMono() = 0;    
    virtual void paintStereo() = 0;    
    void resize(int width, int height);

    virtual bool processEvent() = 0;
    virtual void eventLoop() = 0;

    virtual void keyPressEvent(){};
    virtual void keyReleaseEvent(){};        
    virtual void mousePressLeft(){};
    virtual void mousePressMiddle(){};
    virtual void mousePressRight(){};
    virtual void mouseReleaseLeft(){};    
    virtual void mouseReleaseMiddle(){};    
    virtual void mouseReleaseRight(){};    
    virtual void mouseMoveEvent(){};
    virtual void resizeEvent(){};

    void saveState();
    void restoreState();

   protected:
    
    typedef std::deque<glRenderable*>  RenderableList;
    RenderableList _renderables;

    bool           _ignore_events;
    bool           _stereo_mode;
    bool           _save_frames;
    int            _img_seq_num;
    std::string    _frame_basename;
    std::string    _display_text;
		int						 _text_x, _text_y;
    glCamera       *_camera;
    glManipulator  *_manipulator;
    
    glfloat4         _bg_color;
  };
  
}

#endif

