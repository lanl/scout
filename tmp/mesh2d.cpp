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

unsigned mesh_width       = 0;
unsigned mesh_height      = 0;
unsigned mesh_vert_width  = 0;
unsigned mesh_vert_height = 0;
unsigned *vertices        = 0;
unsigned *edges           = 0;
unsigned *cells           = 0;
float    *cell_colors     = 0;

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
    vertices      = new unsigned[mesh_vert_width * mesh_vert_height * 2];
    cell_colors   = new float[mesh_width * mesh_height * 3];
  }

  unsigned i = 0;
  unsigned j = 0;
  for(unsigned y = 0; y < mesh_vert_height; y++) {  
    for(unsigned x = 0; x < mesh_vert_width; x++) {
      vertices[i] = x * cell_size;
      i++;
      vertices[i] = y * cell_size;
      i++;
      cout << j << ". [" << vertices[i-2] << ", " << vertices[i-1] << "] (" << i-2 << ")\n";
      j+=1;
    }
  }
  
  i = 0;
  j = 0;
  int ncells = mesh_width * mesh_height;
  for(unsigned cc = 0; cc < ncells; cc++) {
    cell_colors[i]   = float(cc) / ncells;
    cell_colors[i+1] = 0.0;
    cell_colors[i+2] = 0.0;
    i+=3;
  }
}

void init_opengl() {
  glShadeModel(GL_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_POINT_SIZE);
  glPointSize(3.0);  
  glClearColor(0.0, 0.0, 0.35, 0.0);
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
  glOrtho(-100, 100, -100, 100, -1.0, 1.0);
}

void render_mesh(GLFWwindow *window) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  unsigned ci = 0;
  for(unsigned y = 0; y < mesh_height; y++) {
    for(unsigned x = 0; x < mesh_width; x++) {
      
      glColor3f(cell_colors[ci], cell_colors[ci+1], cell_colors[ci+2]);
      ci+=3;

      glBegin(GL_QUADS);
      int i = (y * (mesh_width+1) * 2) + (x*2);
        glVertex2i(vertices[i], vertices[i+1]);
        glVertex2i(vertices[i+2], vertices[i+3]);
        printf("cell(%d): [%d,%d], [%d,%d], ",
               i, vertices[i], vertices[i+1],
               vertices[i+2], vertices[i+3]);
               i = ((y+1) * (mesh_width+1) * 2) + (x*2);
        glVertex2i(vertices[i+2], vertices[i+3]);  
        glVertex2i(vertices[i], vertices[i+1]);        
        printf("(%d) [%d,%d], [%d,%d]\n", i,
               vertices[i+2], vertices[i+3],
               vertices[i], vertices[i+1]);

      glEnd();
    }
    printf("---\n");
  }
  printf("\n\n");  
  glfwSwapBuffers(window);
}

int main(int argc, char *argv[]) {
  
  init_mesh(33,37, 2);
  
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
