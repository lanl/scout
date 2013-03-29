/*
 *  
 *###########################################################################
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
 */ 

#ifndef SCOUT_GL_TOOLKIT_H_
#define SCOUT_GL_TOOLKIT_H_
#include <deque>
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/opengl/glRenderable.h"
#include "scout/Runtime/opengl/glManipulator.h"

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
    int            _text_x, _text_y;
    glCamera       *_camera;
    glManipulator  *_manipulator;
    
    glfloat4       _bg_color;
  };
  
}

#endif

