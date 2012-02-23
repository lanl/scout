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
#include "runtime/opengl/glRenderable.h"
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

      void setVolumeData(void* dataptr);

      void draw(glCamera* camera);

      //GLuint get_buffer_object_id() { return _abo->id(); } 

      //glyph_vertex* map_data() 
      //{ return (glyph_vertex*)_abo->mapForWrite(); }

      //void unmap_vertex_data() { _abo->unmap(); }

    private:
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
      
      
      //glAttributeBuffer      *_abo;      // attributes

  };

}

#endif
