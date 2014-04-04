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
#include "scout/Runtime/opengl/glSurfaceRenderable.h"

using namespace std;
using namespace scout;

glSurfaceRenderable::glSurfaceRenderable(int width, int height, int depth, float* vertices, 
    float* normals, float* colors, int num_vertices, glCamera* camera)
:_nx(width), _ny(height), _nz(depth), _vertices(vertices), _normals(normals), _colors(colors), 
  _num_vertices(num_vertices)
{
  // do we need width, height and depth?
  initialize(camera);
}

// init opengl and compute view and projection matrices in SDL
void glSurfaceRenderable::initialize(glCamera *camera) 
{

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_SMOOTH);

  // good old-fashioned fixed function lighting
  float white[] = { 0.8, 0.8, 0.8, 1.0 };
  float black[] = { 0.0, 0.0, 0.0, 1.0 };
  float lightPos[] = { camera->position[0], camera->position[1], camera->position[2], 1.0 };

  glLightfv(GL_LIGHT0, GL_AMBIENT, white);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
  glLightfv(GL_LIGHT0, GL_SPECULAR, black);
  glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);

  // Setup the view of the cube.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective( camera->fov, camera->aspect, camera->near, camera->far);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(camera->position[0], camera->position[1], camera->position[2],
      camera->look_at[0], camera->look_at[1], camera->look_at[2],
      camera->up[0], camera->up[1], camera->up[2]);

}


void glSurfaceRenderable::draw(glCamera* camera)
{
  // not sure what to do with camera in draw
  // it is really not needed
  for (int i = 0; i < _num_vertices; i++) {
    //printf("drawcolor[%d]: %f, %f, %f, %f \n", i, *(_colors+4*i), *(_colors+4*i+1), *(_colors+4*i+2), *(_colors+4*i+3));
  }
  glColorPointer(4, GL_FLOAT, 0, _colors);
  glNormalPointer(GL_FLOAT, 0, _normals);
  glVertexPointer(4, GL_FLOAT, 0, _vertices);
  glDrawArrays(GL_TRIANGLES, 0, _num_vertices);
}

