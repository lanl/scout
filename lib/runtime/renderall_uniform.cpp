/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * -----
 * 
 */

#include "runtime/renderall_uniform.h"

#ifdef __APPLE__

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#else

#include <GL/gl.h>
#include <GL/glu.h>

#endif

#include <iostream>

#include <SDL/SDL.h>

#include "runtime/scout_gpu.h"

#include "runtime/base_types.h"
#include "runtime/opengl/glTexture1D.h"
#include "runtime/opengl/glTexture2D.h"
#include "runtime/opengl/glTextureBuffer.h"
#include "runtime/opengl/glTexCoordBuffer.h"
#include "runtime/opengl/glVertexBuffer.h"

using namespace std;
using namespace scout;

// ------  LLVM - globals written to by LLVM

float4* _pixels;
CUdeviceptr __sc_device_renderall_uniform_colors;

// -------------

extern SDL_Surface* __sc_sdl_surface;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;
void __sc_init_sdl(size_t width, size_t height);

namespace{
  
  const size_t WINDOW_WIDTH  = 768;
  const size_t WINDOW_HEIGHT = 768;

} // end namespace

namespace scout{

  class renderall_uniform_rt_{
  public:
    renderall_uniform_rt_(renderall_uniform_rt* o)
    : o_(o){

      if(!__sc_sdl_surface){
	__sc_init_sdl(__sc_initial_width, __sc_initial_height); 
      }

      init();
    }

    ~renderall_uniform_rt_(){
      destroy();
    }

    void init(){
      glMatrixMode(GL_PROJECTION);
    
      glLoadIdentity();

      size_t width = o_->width();
      size_t height = o_->height();
      size_t depth = o_->depth();
      
      static const float pad = 0.05;
    
      if(height == 0){
	float px = pad * width;
	gluOrtho2D(-px, width + px, -px, width + px);

	init(width);
      }
      else{
	if(width >= height){
	  float px = pad * width;
	  float py = (1 - float(height)/width) * width * 0.50;
	  gluOrtho2D(-px, width + px, -py - px, width - py + px);
	}
	else{
	  float py = pad * height;
	  float px = (1 - float(width)/height) * height * 0.50;
	  gluOrtho2D(-px - py, width + px + py, -py, height + py);
	}

	init(width, height);
      }

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glClearColor(0.5, 0.55, 0.65, 0.0);

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      SDL_GL_SwapBuffers();
    }

    void begin(){
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if(__sc_gpu){
	map_gpu_resources();
      }
      else{
	map_colors();
      }
    }

    void end(){
      if(__sc_gpu){
	unmap_gpu_resources();
      }
      else{
	unmap_colors();
      }

      exec();

      SDL_GL_SwapBuffers();

      SDL_Event evt;
      if(!SDL_PollEvent(&evt)){
	return;
      }

      switch(evt.type){
        case SDL_QUIT:
	{
	  exit(0);
	}
        case SDL_KEYDOWN:
	{
	  switch(evt.key.keysym.sym){
	    case SDLK_ESCAPE:
	    {
	      exit(0);
	    }
	    default:
	    {
	      break;
	    }
	  }
	  break;
	}
        case SDL_VIDEORESIZE:
	{
	  __sc_init_sdl(evt.resize.w, evt.resize.h);
	  
	  destroy();
	  init();
	  break;
	}
      }
    }

    void fill_vbo(float x0,
		  float y0,
		  float x1,
		  float y1){
  
      float* verts = (float*)vbo_->mapForWrite();

      verts[0] = x0;
      verts[1] = y0;
      verts[2] = 0.0f;
  
      verts[3] = x1;
      verts[4] = y0;
      verts[5] = 0.f;
  
      verts[6] = x1;
      verts[7] = y1;
      verts[8] = 0.0f;
  
      verts[9] = x0;
      verts[10] = y1;
      verts[11] = 0.0f;
  
      vbo_->unmap();
    }

    void fill_tcbo_2d(float x0,
		      float y0,
		      float x1,
		      float y1){

      float* coords = (float*)tcbo_->mapForWrite();

      coords[0] = x0;
      coords[1] = y0;
  
      coords[2] = x1;
      coords[3] = y0;
  
      coords[4] = x1;
      coords[5] = y1;
  
      coords[6] = x0;
      coords[7] = y1;
  
      tcbo_->unmap();

    }

    void fill_tcbo_1d(float start, float end){
      float* coords = (float*)tcbo_->mapForWrite();
  
      coords[0] = start;
      coords[1] = end;
      coords[2] = end;
      coords[3] = start;

      tcbo_->unmap();
    }

    void map_gpu_resources(){
      assert(cuGraphicsMapResources(1, &__sc_device_resource, 0) == CUDA_SUCCESS);

      size_t bytes;
      assert(cuGraphicsResourceGetMappedPointer(&__sc_device_renderall_uniform_colors, &bytes, __sc_device_resource) == CUDA_SUCCESS);
    }

    void unmap_gpu_resources(){
      assert(cuGraphicsUnmapResources(1, &__sc_device_resource, 0) == CUDA_SUCCESS);

      pbo_->bind();
      tex_->initialize(0);
      pbo_->release();
    }

    void register_gpu_pbo(GLuint pbo, unsigned int flags){
      assert(cuGraphicsGLRegisterBuffer(&__sc_device_resource, pbo, flags) ==
	     CUDA_SUCCESS);
    }

    void init(dim_t xdim){
      ntexcoords_ = 1;
      tex_ = new glTexture1D(xdim);
      tex_->addParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      tex_->addParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      tex_->addParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      pbo_ = new glTextureBuffer;
      pbo_->bind();
      pbo_->alloc(sizeof(float) * 4 * xdim, GL_STREAM_DRAW_ARB);
      pbo_->release();

      vbo_ = new glVertexBuffer;
      vbo_->bind();
      vbo_->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);  // we use a quad for 1D meshes...
      fill_vbo(0.0f, 0.0f, float(xdim), float(xdim));
      vbo_->release();
      nverts_ = 4;

      tcbo_ = new glTexCoordBuffer;
      tcbo_->bind();
      tcbo_->alloc(sizeof(float) * 4, GL_STREAM_DRAW_ARB);  // one-dimensional texture coordinates.
      fill_tcbo_1d(0.0f, 1.0f);
      tcbo_->release();

      OpenGLErrorCheck();

      if(__sc_gpu){
	register_gpu_pbo(pbo_->id(),
			 CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
      }
    }

    void init(dim_t xdim, dim_t ydim){
      ntexcoords_ = 2;
      tex_ = new glTexture2D(xdim, ydim);
      tex_->addParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      tex_->addParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      tex_->addParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      //tex_->initialize(0);

      pbo_ = new glTextureBuffer;
      pbo_->bind();
      pbo_->alloc(sizeof(float) * 4 * xdim * ydim, GL_STREAM_DRAW_ARB);
      pbo_->release();

      vbo_ = new glVertexBuffer;
      vbo_->bind();
      vbo_->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);
      fill_vbo(0.0f, 0.0f, float(xdim), float(ydim));
      vbo_->release();
      nverts_ = 4;

      tcbo_ = new glTexCoordBuffer;
      tcbo_->bind();
      tcbo_->alloc(sizeof(float) * 8, GL_STREAM_DRAW_ARB);  // two-dimensional texture coordinates.
      fill_tcbo_2d(0.0f, 0.0f, 1.0f, 1.0f);
      tcbo_->release();

      OpenGLErrorCheck();

      if(__sc_gpu){
	register_gpu_pbo(pbo_->id(), 
			 CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
      }
    }

    void destroy(){
      delete pbo_;
      delete vbo_;
      delete tcbo_;
      delete tex_;
    }

    void map_colors(){
      _pixels = (float4*)pbo_->mapForWrite();
    }

    void unmap_colors(){
      pbo_->unmap();
      pbo_->bind();
      tex_->initialize(0);
      pbo_->release();
    }

    void exec(){
      pbo_->bind();
      tex_->enable();
      tex_->update(0);
      pbo_->release();
  
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      tcbo_->bind();
      glTexCoordPointer(ntexcoords_, GL_FLOAT, 0, 0);
  
      glEnableClientState(GL_VERTEX_ARRAY);
      vbo_->bind();
      glVertexPointer(3, GL_FLOAT, 0, 0);
  
      glDrawArrays(GL_POLYGON, 0, nverts_);
  
      glDisableClientState(GL_VERTEX_ARRAY);
      vbo_->release();
  
      glDisableClientState(GL_TEXTURE_COORD_ARRAY);
      tcbo_->release();
  
      tex_->disable();
    }

  private:
    renderall_uniform_rt* o_;

    glVertexBuffer* vbo_;
    glTexture* tex_;
    glTextureBuffer* pbo_;
    glTexCoordBuffer* tcbo_;
    unsigned short ntexcoords_;
    unsigned int nverts_;
  };

} // end namespace scout

renderall_uniform_rt::renderall_uniform_rt(size_t width,
					   size_t height,
					   size_t depth)
  : renderall_base_rt(width, height, depth){
  
  x_ = new renderall_uniform_rt_(this);
  
}

renderall_uniform_rt::~renderall_uniform_rt(){
  delete x_;
}

void renderall_uniform_rt::begin(){
  x_->begin();
}

void renderall_uniform_rt::end(){
  x_->end();
}

void __sc_begin_uniform_renderall(size_t width,
				  size_t height,
				  size_t depth){
  if(!__sc_renderall){
      __sc_renderall = new renderall_uniform_rt(width, height, depth);
  }
  
  __sc_renderall->begin();
  
}
