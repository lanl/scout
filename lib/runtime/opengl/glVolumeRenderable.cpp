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
#include "runtime/opengl/glVolumeRenderable.h"
#include "runtime/volren/hpgv/hpgv_block.h"
#include "runtime/volren/hpgv/matrix_util.h"
#include <mpi.h>

using namespace std;
using namespace scout;

glVolumeRenderable::glVolumeRenderable(int npx, int npy, int npz,
    int nx, int ny, int nz, double* x, double* y, double* z, 
    int win_width, int win_height, glCamera* camera, trans_func_t* trans_func,
    int id, int root, MPI_Comm gcomm)
    :_npx(npx), _npy(npy), _npz(npz), _nx(nx), _ny(ny), _nz(nz),
    _x(x), _y(y), _z(z), _win_width(win_width), _win_height(win_height),
    _trans_func(trans_func), _id(id), _root(root), _gcomm(gcomm)
    
{
  if (!hpgv_vis_valid()) {
    hpgv_vis_init(MPI_COMM_WORLD, _root);
  }

  size_t xdim = win_width;
  size_t ydim = win_height;

  setMinPoint(glfloat3(0, 0, 0));
  setMaxPoint(glfloat3(xdim, ydim, 0));

  // Set up vis params  
  initialize(camera);
  hpgv_vis_para(_para_input);

  // set up opengl stuff
  _ntexcoords = 2;
  _texture = new glTexture2D(xdim, ydim);
  _texture->addParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  _texture->addParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _texture->addParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  _pbo = new glTextureBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * xdim * ydim, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);
  fill_vbo(_min_pt.x, _min_pt.y, _max_pt.x, _max_pt.y);
  _vbo->release();
  _nverts = 4;

  _tcbo = new glTexCoordBuffer;
  _tcbo->bind();
  _tcbo->alloc(sizeof(float) * 8, GL_STREAM_DRAW_ARB);  // two-dimensional texture coordinates.
  fill_tcbo2d(0.0f, 0.0f, 1.0f, 1.0f);
  _tcbo->release();

  OpenGLErrorCheck();

  //if (id == root) printf("finished hpgv_vis_para()\n");

}

glVolumeRenderable::~glVolumeRenderable() 
{
  if (_para_input) {
    hpgv_para_delete(_para_input);
  }

  if (_block) {
    free(_block);
  }

  if (_texture != 0) delete _texture;
  if (_pbo != 0) delete _pbo;
  if (_vbo != 0) delete _vbo;
  if (_tcbo != 0) delete _tcbo;
  _texture = NULL;
  _pbo = NULL;
  _vbo = NULL;
  _tcbo = NULL;

}

    
void glVolumeRenderable::initialize(glCamera* camera)
{
  initializeRenderer(camera);
  initializeOpenGL(camera);
}



void glVolumeRenderable::initializeRenderer(glCamera* camera)
{
  /* new param struct */
  para_input_t *pi = (para_input_t *)calloc(1, sizeof(para_input_t));
  HPGV_ASSERT_P(_id, pi, "Out of memory.", HPGV_ERR_MEM);

  // no need to set colormap for that data structure anymore
  // but it wants this info for compositing 

  pi->image_format = HPGV_RGBA;  
  pi->image_type = HPGV_FLOAT;

  // compute view and projection matrices

  int width = _win_width;
  int height = _win_height;

  double mym[16];
  getViewMatrix((double)camera->position[0], (double)camera->position[1], 
      (double) camera->position[2], (double)camera->look_at[0], 
      (double)camera->look_at[1], (double)camera->look_at[2],
      (double)camera->up[0], (double)camera->up[1], (double)camera->up[2], 
      (double (*)[4])&mym[0]);

  /*
     int i,j;
     fprintf(stderr, "My modelview matrix:\n");
     for (i = 0; i < 4; i++ ) {
     for (j = 0; j < 4; j++ ) {
     fprintf(stderr, "%lf ", mym[j*4 + i]);
     }
     fprintf(stderr, "\n");
     }
     fprintf(stderr, "\n");
     */

  // transpose mine and use it -- Hongfeng takes a transpose of opengl matrix
  transposed(&pi->para_view.view_matrix[0], &mym[0]);

  getProjectionMatrix((double)camera->fov, (double)_win_width/_win_height, 
      (double)camera->near, (double)camera->far, (double (*)[4])&mym[0]);

  /*
     fprintf(stderr, "My projection matrix:\n");
     for (i = 0; i < 4; i++ ) {
     for (j = 0; j < 4; j++ ) {
     fprintf(stderr, "%lf ", mym[j*4 + i]);
     }
     fprintf(stderr, "\n");
     }
     fprintf(stderr, "\n");
     */

  // transpose mine and use it -- Hongfeng takes a transpose of opengl matrix
  transposed(&pi->para_view.proj_matrix[0], &mym[0]);

  // probably want to use framebuffer_rt and viewport_rt for this?
  // for now, hardwire
  pi->para_view.view_port[0] = 0;
  pi->para_view.view_port[1] = 0;
  pi->para_view.view_port[2] = _win_width;
  pi->para_view.view_port[3] = _win_height;
  pi->para_view.frame_width = _win_width;
  pi->para_view.frame_height = _win_height;

  // hardwire to 1 image, 1 volume, and no particles for now
  // To change this we increase num_vol to the number of different
  // vars we want to vis, then set the corresponding id for each
  // variable H2, H, O, O2, etc. (which enables the value lookup in the data).
  pi->num_image = 1;
  pi->para_image = (para_image_t *)calloc(1, sizeof(para_image_t));
  pi->para_image->num_particle = 0;
  pi->para_image->num_vol = 1;
  pi->para_image->id_vol = (int*)calloc (1, sizeof (int));
  pi->para_image->id_vol[0] = 0;
  pi->para_image->sampling_spacing = 1.297920;
  pi->para_image->tf_vol = (para_tf_t *)calloc(1, sizeof(para_tf_t));


  pi->para_image->light_vol = (para_light_t *)calloc(1, sizeof(para_light_t));
  pi->para_image->light_vol[0].withlighting = 1;
  pi->para_image->light_vol[0].lightpar[0] =  0.2690;//ambient reflection coef
  pi->para_image->light_vol[0].lightpar[1] =  0.6230;//diffuse reflection coef
  pi->para_image->light_vol[0].lightpar[2] =  0.8890;//specular reflection coef
  pi->para_image->light_vol[0].lightpar[3] =  128.0000; //spec. refl. exponent

  _para_input =  pi;

}

void glVolumeRenderable::initializeOpenGL(glCamera* camera)
{
 glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  size_t width = _max_pt.x - _min_pt.x;
  size_t height = _max_pt.y - _min_pt.y;

  static const float pad = 0.05;

  if(height == 0){
    float px = pad * width;
    gluOrtho2D(-px, width + px, -px, width + px);

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

  }

  glMatrixMode(GL_MODELVIEW);

  glLoadIdentity();

  glClearColor(0.5, 0.55, 0.65, 0.0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void glVolumeRenderable::setVolumeData(void* dataptr) 
{
  _data = dataptr;
}
 
// assume _data has been set before this is called
void glVolumeRenderable::createBlock()
{
  /* new block */
  _block = (block_t *)calloc(1, sizeof(block_t));
  HPGV_ASSERT_P(_id, _block, "Out of memory.", HPGV_ERR_MEM);

  /* mpi */
  _block->mpiid = _id;
  _block->mpicomm = _gcomm;
  MPI_Comm_size(_gcomm, &_groupsize);

  /* new header */
  blk_header_t header;
  header_new(_id, _gcomm, _groupsize, _x, _y, _z, _nx, _ny, _nz, 
      _npx, _npy, _npz, &header);
  memcpy(&(_block->blk_header), &header, sizeof(blk_header_t));

  block_add_volume(_block, HPGV_DOUBLE, _data, 0);

  //hpgv_block_print(id, _root, block);

  /*  initalize timing module   */
  if (!HPGV_TIMING_VALID()) {
    HPGV_TIMING_INIT(_root, *_gcomm);
  }

  static int init_timing = HPGV_FALSE;

  if (init_timing == HPGV_FALSE) {
    init_timing = HPGV_TRUE;

    HPGV_TIMING_NAME(MY_STEP_SLOVE_TIME,      "T_slove");
    HPGV_TIMING_NAME(MY_STEP_VIS_TIME,        "T_vis");
    HPGV_TIMING_NAME(MY_STEP_SAVE_IMAGE_TIME, "T_saveimg");
    HPGV_TIMING_NAME(MY_STEP_GHOST_VOL_TIME,  "T_ghostvol");
    HPGV_TIMING_NAME(MY_ALL_SLOVE_TIME,       "T_tslove");
    HPGV_TIMING_NAME(MY_ALL_VIS_TIME,         "T_tvis");
    HPGV_TIMING_NAME(MY_ENDTOEND_TIME,        "T_end2end");

    HPGV_TIMING_BEGIN(MY_ENDTOEND_TIME);
    HPGV_TIMING_BEGIN(MY_STEP_SLOVE_TIME);
    HPGV_TIMING_BEGIN(MY_ALL_SLOVE_TIME);
  }

}

void glVolumeRenderable::render()
{

  MPI_Comm_size(_gcomm, &_groupsize);

  HPGV_TIMING_END(MY_STEP_SLOVE_TIME);
  HPGV_TIMING_END(MY_ALL_SLOVE_TIME);

  HPGV_TIMING_BEGIN(MY_STEP_VIS_TIME);
  HPGV_TIMING_COUNT(MY_STEP_VIS_TIME);

  HPGV_TIMING_BEGIN(MY_ALL_VIS_TIME);


  if ( !_id && _id == _root) {
    //fprintf(stderr, "render call at %.3e.\n", 0);
  }

  int i, j;


  MPI_Barrier(_gcomm);

  if (_id == _root) {
    struct tm *start_time;
    time_t start_timer;
    time(&start_timer);
    start_time = localtime(&start_timer);

    //fprintf(stderr, "tstep render call starts at %02d:%02d:%02d\n",
    //    start_time->tm_hour, start_time->tm_min, start_time->tm_sec);
  }


  HPGV_TIMING_BEGIN(MY_STEP_GHOST_VOL_TIME);
  HPGV_TIMING_COUNT(MY_STEP_GHOST_VOL_TIME);

  /* exchange ghost area */
  for (i = 0; i < _para_input->num_image; i++) {
    para_image_t *image = &(_para_input->para_image[i]);

    for (j = 0; j < image->num_vol; j++) {
      block_exchange_boundary(_block, image->id_vol[j]);
    }
  }
  if (_id == _root) printf("finished ghost area exchange\n");

  HPGV_TIMING_END(MY_STEP_GHOST_VOL_TIME);

  hpgv_vis_render(_block, _root, _gcomm, 0, _trans_func);

  if (_id == _root) printf("finished hpgv_vis_render()\n");

  HPGV_TIMING_END(MY_STEP_VIS_TIME);

  HPGV_TIMING_END(MY_ALL_VIS_TIME);

}

void glVolumeRenderable::writePPM(double time) 
{
  HPGV_TIMING_BEGIN(MY_STEP_SAVE_IMAGE_TIME);
  HPGV_TIMING_COUNT(MY_STEP_SAVE_IMAGE_TIME);

  if (_id == _root) {

    char filename[MAXLINE];
    char varname[MAXLINE];

    char *imgptr   = (char *) hpgv_vis_get_imageptr();
    int  imgformat = hpgv_vis_get_imageformat();
    int  imgtype   = hpgv_vis_get_imagetype();
    long imgsize   = _para_input->para_view.frame_width *
      _para_input->para_view.frame_height *
      hpgv_formatsize(imgformat) *
      hpgv_typesize(imgtype);

    int i;
    if (1) {
      for (i = 0; i < _para_input->num_image; i++) {

        para_image_t *image = &(_para_input->para_image[i]);

        sprintf(varname, "test");
/*
        for (j = 0; j < image->num_vol; j++) {
          sprintf(varname, "%s_v%s", varname,
              theDataName[image->id_vol[j]]);
        }

        if (image->num_particle == 1) {
          sprintf(varname, "%s_v%s", varname,
              theDataName[image->vol_particle]);
        }
*/
        snprintf(filename, MAXLINE, "%s/image_p%04d_s%04d%s_t%.3e.ppm",
            ".", _groupsize, 0, varname, time);

        hpgv_vis_saveppm(_para_input->para_view.frame_width,
            _para_input->para_view.frame_height,
            imgformat,
            imgtype,
            imgptr + imgsize * i,
            filename);
        printf("finished hpgv_vis_saveppm()\n");
      }
    }
  }
}


void glVolumeRenderable::draw(glCamera* camera)
{
  createBlock();
  render();

  if (_id == _root) {

    // now put it up in the window
    float4* _colors = map_colors();

    hpgv_vis_copyraw( _para_input->para_view.frame_width,
        _para_input->para_view.frame_height,
        _para_input->image_format,
        _para_input->image_type,
        _para_input->num_image,
        0,
        hpgv_vis_get_imageptr(),
        (void*)_colors);
/*
    for (int i = 0; i < _para_input->para_view.frame_height; i++) {
      for (int j = 0; j < _para_input->para_view.frame_width; j++) {
          float4 color = _colors[(i*_para_input->para_view.frame_width) + j];
        printf(" %f:%f:%f", color.components[0], color.components[1], 
          color.components[2]); 
      }
      printf("\n");
    } 
*/
    unmap_colors();

    _pbo->bind();
    _texture->enable();
    _texture->update(0);
    _pbo->release();

    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    _tcbo->bind();
    glTexCoordPointer(_ntexcoords, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    _vbo->bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);

    OpenGLErrorCheck();

    glDrawArrays(GL_POLYGON, 0, _nverts);

    glDisableClientState(GL_VERTEX_ARRAY);
    _vbo->release();

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    _tcbo->release();

    _texture->disable();

  }
}

void glVolumeRenderable::fill_vbo(float x0,
    float y0,
    float x1,
    float y1)
{

  float* verts = (float*)_vbo->mapForWrite();

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

  _vbo->unmap();
}


void glVolumeRenderable::fill_tcbo2d(float x0,
    float y0,
    float x1,
    float y1)
{

  float* coords = (float*)_tcbo->mapForWrite();

  coords[0] = x0;
  coords[1] = y0;

  coords[2] = x1;
  coords[3] = y0;

  coords[4] = x1;
  coords[5] = y1;

  coords[6] = x0;
  coords[7] = y1;

  _tcbo->unmap();

}

float4* glVolumeRenderable::map_colors()
{
  return (float4*)_pbo->mapForWrite();
}


void glVolumeRenderable::unmap_colors()
{
  _pbo->unmap();
  _pbo->bind();
  _texture->initialize(0);
  _pbo->release();
}


