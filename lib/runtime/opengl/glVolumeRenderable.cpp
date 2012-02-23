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
  //  setMinPoint(min_pt);
  //  setMaxPoint(max_pt);

  if (!hpgv_vis_valid()) {
    hpgv_vis_init(MPI_COMM_WORLD, _root);
  }

  // Set up vis params  
  initialize(camera);
  hpgv_vis_para(_para_input);

  if (id == root) printf("finished hpgv_vis_para()\n");


}

glVolumeRenderable::~glVolumeRenderable() 
{
  if (_para_input) {
    hpgv_para_delete(_para_input);
  }

  if (_block) {
    free(_block);
  }
}

    
void glVolumeRenderable::initialize(glCamera* camera)
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
  writePPM(0);
}
