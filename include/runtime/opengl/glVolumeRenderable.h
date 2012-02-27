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

#ifndef GL_VOLUME_RENDERABLE_H_
#define GL_VOLUME_RENDERABLE_H_

#include <cassert>
#include <string>
#include <list>

#include "runtime/opengl/glCamera.h"
#include "runtime/opengl/glTexture.h"
#include "runtime/opengl/glTexture2D.h"
#include "runtime/opengl/glTextureBuffer.h"
#include "runtime/opengl/glRenderable.h"
#include "runtime/opengl/glVertexBuffer.h"
#include "runtime/opengl/glTexCoordBuffer.h"
#include "runtime/volren/hpgv/hpgv.h"


// ----- glVolumeRenderable 
// 

namespace scout
{

  class glVolumeRenderable: public glRenderable {

    public:

      glVolumeRenderable(int npx, int npy, int npz,
          int nx, int ny, int nz, double* x, double* y, double* z,
          int win_width, int win_height, 
          glCamera* camera, trans_func_t* trans_func,
          int id, int root, MPI_Comm gcomm);

      ~glVolumeRenderable();

      void initialize(glCamera* camera);

      float4* map_colors();
      void unmap_colors();

      void setVolumeData(void* dataptr);

      void draw(glCamera* camera);

    void setMinPoint(glfloat3 pt)
    { _min_pt = pt; }

    void setMaxPoint(glfloat3 pt)
    { _max_pt = pt; }

    void setTexture(glTexture *texture)
    { _texture = texture; }

    glTexture* texture() const
    { return _texture; }

    void setTexCoordScale(float s)
    { _tex_coord_scale = s; }

  private:
    void fill_vbo(float x0, float y0, float x1, float y1);
    void fill_tcbo2d(float x0, float y0, float x1, float y1);

   private:
      glVertexBuffer* _vbo;
      glTexture* _texture;
      glTextureBuffer* _pbo;
      glTexCoordBuffer* _tcbo;
      unsigned short _ntexcoords;
      unsigned int _nverts;
      glfloat3 _min_pt;
      glfloat3 _max_pt;
      float _tex_coord_scale;

    private:
      void initializeRenderer(glCamera* camera);
      void initializeOpenGL(glCamera* camera);
      void createBlock();
      void render();
      void writePPM(double time);

    private:
      para_input_t*   _para_input;
      block_t*        _block;
      int             _npx, _npy, _npz, _nx, _ny, _nz, _win_height, 
                      _win_width, _id, _root, _groupsize;
      double          *_x, *_y, *_z;
      trans_func_t*   _trans_func;
      MPI_Comm        _gcomm;

    public:
      void*           _data;
      
  };

}

#endif
