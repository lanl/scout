//
// This code builds a vertex list (array of coordinates that make up
// the mesh vertices).  In a Scout language sense these are our
// 'vertices' too.  From there it builds an index set that points into
// those vertices to represent the edges (vertex-to-vertex lines) and
// then the cells -- the quads (squares in this case) that represent 
// the mesh cells.  So, there's a reasonbly straightforward one-to-one 
// mapping between Scout terminology and this code.  
//
// The key piece from a renderall perspective will be to build the
// color array that each element above (vertex, edge, cell) is colored
// by.  So, the renderall should fill the color array and then call
// the renderable.
//
// From the command line you can compile this with:
//
// c++ mesh.cpp -L/usr/local/lib -framework OpengL -framework Cocoa -framework IOKit -framework CoreVideo -lglfw3 
// 
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
using namespace std;

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

unsigned mesh_width  = 0;
unsigned mesh_height = 0;
unsigned mesh_vert_width = 0;
unsigned mesh_vert_height = 0;


unsigned *vertices = 0;
unsigned *edges    = 0;
unsigned *cells    = 0;


void init_mesh(unsigned width, unsigned height, unsigned cell_size) {
  if (vertices == 0) {
    mesh_width       = width;
    mesh_height      = height;
    mesh_vert_width  = mesh_width  + 1;
    mesh_vert_height = mesh_height + 1;
    cerr << "create mesh: " << width << "x" << height << " cells.\n";
    cerr << "             " << mesh_vert_width << "x" << mesh_vert_height << " vertices.\n";
    cerr << "             " << mesh_vert_width * mesh_vert_height << " total vertices.\n";
    cerr << "             " << mesh_width*mesh_vert_height*2 + mesh_height*mesh_vert_width*2
         << " total edge vertices.\n";              
    vertices = new unsigned[mesh_vert_width * mesh_vert_height * 2];
    edges    = new unsigned[mesh_width * mesh_vert_height * 2  + mesh_height * mesh_vert_width * 2];
    cells    = new unsigned[mesh_width * mesh_height * 4];
  }

  unsigned i = 0;
  for(unsigned y = 0; y < mesh_vert_height; y++) {  
    for(unsigned x = 0; x < mesh_vert_width; x++) {
      vertices[i] = x * cell_size;
      i++;
      vertices[i] = y * cell_size;
      i++;
      cout << "(" << vertices[i-2] << ", " << vertices[i-1] << ")\n";
    }
  }

  i = 0;
  for(unsigned y = 0; y <= height; y++) {
    for(unsigned x = 0; x < width; x++) {
      edges[i] = y * (width+1) + x;
      i++;
      edges[i] = y * (width+1) + x + 1;
      i++;
    }
  }
  for(unsigned x = 0; x <= width; x++) {
    for(unsigned y = 0; y < height; y++) {
      edges[i] = y * (width+1) + x;
      i++;
      edges[i] = (y + 1) * (width+1) + x;
      i++;
    }
  }

  i = 0;
  for(unsigned y = 0; y < height; y++) {
    for(unsigned x = 0; x < width; x++) {
      unsigned i0, i1;
      i0 = y * (width+1) + x;
      i1 = (y + 1) * (width+1) + x;
      cells[i] = i1;
      i++;
      cells[i] = i0;
      i++;
      cells[i] = i0+1;
      i++;
      cells[i] = i1+1;
      i++;
      printf("[%d,%d,%d,%d]\n", cells[i-4], cells[i-3], cells[i-2], cells[i-1]);

    }
  }
}

void init_opengl() {
  glShadeModel(GL_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_POINT_SIZE);
  glPointSize(3.0);  
  glClearColor(0.3, 0.3, 0.35, 0.0);
}

void keypress(GLFWwindow* window, int key, int scancode, int action, int mode) {
  if (action != GLFW_PRESS)
    return;

  switch(key) {

    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GL_TRUE);
      break;
  }
}

void resize(GLFWwindow *window, int width, int height) {
  float ratio = 1.0f;
  if (height > 0) 
    ratio = (float)width / float(height);

  glViewport(0, 0, width, height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-5, 5, -5, 5, -1.0, 1.0);
}

void render_mesh(GLFWwindow *window) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();


  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(2, GL_INT, 0, vertices);
  glColor4f(1.0, 1.0, 1.0, 0.0);    
  glDrawArrays(GL_POINTS, 0, mesh_vert_width * mesh_vert_height);
  
  glColor4f(0.7, 0.7, .7, 0.0);  
  glDrawElements(GL_LINES, mesh_width * mesh_vert_height * 2 +
                 mesh_height * mesh_vert_width * 2, GL_UNSIGNED_INT, edges);
  
  glColor4f(0.5, 0.5, 0.5, 0.0);
  glDrawElements(GL_QUADS, mesh_width * mesh_height * 4, GL_UNSIGNED_INT, cells);  
  glDisableClientState(GL_VERTEX_ARRAY);  
  glfwSwapBuffers(window);
}

int main(int argc, char *argv[]) {
  
  init_mesh(4, 2, 1);
  
  if (!glfwInit())
    return(1);

  GLFWwindow *window = glfwCreateWindow(640, 640, "uniform mesh", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return(1);
  }
  

  glfwSetKeyCallback(window, keypress);
  glfwSetFramebufferSizeCallback(window, resize);

  glfwMakeContextCurrent(window);
  init_opengl();  
  int win_width, win_height;
  glfwGetFramebufferSize(window, &win_width, &win_height);
  resize(window, win_width, win_height);
  
  while(! glfwWindowShouldClose(window)) {
    render_mesh(window);
    glfwPollEvents();
  }
  
  return 0;
}
