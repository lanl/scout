
// Compile on the mac with:
//   c++ mesh3d.cpp -I/usr/local/include -L/usr/local/lib -lglfw3
//   -framework OpenGL -framework CoreFoundation -framework IOKit
//   -framework CoreVideo -framework Cocoa
//
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
using namespace std;

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

unsigned mesh_width        = 0;
unsigned mesh_height       = 0;
unsigned mesh_depth        = 0;
unsigned mesh_vert_width   = 0;
unsigned mesh_vert_height  = 0;
unsigned mesh_vert_depth   = 0;
unsigned *vertices         = 0;
unsigned *edges            = 0;
unsigned *faces            = 0;
unsigned *face_color_index = 0;
float    *face_colors      = 0;

void init_mesh(unsigned width, unsigned height, unsigned depth, unsigned cell_size) {
  
  if (vertices == 0) {
    mesh_width       = width;
    mesh_height      = height;
    mesh_depth       = depth;
    mesh_vert_width  = mesh_width  + 1;
    mesh_vert_height = mesh_height + 1;
    mesh_vert_depth  = mesh_depth  + 1;
    
    cerr << "create mesh: " << width << "x" << height << "x" << depth << " cells.\n";
    vertices = new unsigned[mesh_vert_width * mesh_vert_height * mesh_vert_depth * 3];
    
    edges    = new unsigned[mesh_width  * mesh_vert_height * mesh_vert_depth * 2 +
                            mesh_height * mesh_vert_width  * mesh_vert_depth * 2 +
                            mesh_depth  * mesh_vert_height * mesh_vert_width * 2];
    
    faces    = new unsigned[mesh_width * mesh_height * mesh_vert_depth * 4 +
                            mesh_width * mesh_vert_height * mesh_depth * 4 + 
                            mesh_vert_width * mesh_height * mesh_depth * 4];

    face_colors      = new float[mesh_width * mesh_height * mesh_depth * 6 * 3];
    face_color_index = new unsigned[mesh_width * mesh_height * mesh_vert_depth * 4 +
                                    mesh_width * mesh_vert_height * mesh_depth * 4 + 
                                    mesh_vert_width * mesh_height * mesh_depth * 4];    
  }

  int i = 0;
  for(unsigned z = 0; z <= mesh_depth; z++) {
    for(unsigned y = 0; y <= mesh_height; y++) {  
      for(unsigned x = 0; x <= mesh_width; x++) {
        vertices[i]   = x * cell_size;
        vertices[i+1] = y * cell_size;
        vertices[i+2] = z * cell_size;
        printf("%03d: (%d, %d, %d)\n", i, vertices[i], vertices[i+1], vertices[i+1]);
        i+=3;
      }
    }
  }

  i = 0;
  float face_color[3];
  for(unsigned f = 0; f < 6; f++) {
    for(unsigned z = 0; z < mesh_depth; z++) {
      for(unsigned y = 0; y < mesh_height; y++) {
        for(unsigned x = 0; x < mesh_width; x++) {
          face_colors[i]   = float(x+1) / mesh_width;
          face_colors[i+1] = float(y+1) / mesh_height;
          face_colors[i+2] = float(z+1) / mesh_depth;
          i+=3;          
        }
      }
    }
  }
  
        


  i = 0;
  for(unsigned z = 0; z <= mesh_depth; z++) {
    for(unsigned y = 0; y <= mesh_height; y++) {
      for(unsigned x = 0; x < mesh_width; x++) {
        edges[i]   = (z * (mesh_height+1) * (mesh_width+1)) + y * (mesh_width+1) + x;
        edges[i+1] = edges[i] + 1; 
        i+=2;
      }
    }
  }
  
  for(unsigned z = 0; z <= mesh_depth; z++) {
    for(unsigned x = 0; x <= mesh_width; x++) {
      for(unsigned y = 0; y < mesh_height; y++) {
        edges[i]   = (z * (mesh_height+1) * (mesh_width+1)) + y * (mesh_width+1) + x;
        edges[i+1] = (z * (mesh_height+1) * (mesh_width+1)) + (y+1) * (mesh_width+1) + x;
        i+=2;
      }
    }
  }

  for(unsigned x = 0; x <= mesh_width; x++) {
    for(unsigned y = 0; y <= mesh_height; y++) {
      for(unsigned z = 0; z < mesh_depth; z++) {
        edges[i]   = (z * (mesh_height+1) * (mesh_width+1)) + y * (mesh_width+1) + x;
        edges[i+1] = ((z+1) * (mesh_height+1) * (mesh_width+1)) + y * (mesh_width+1) + x;
        i+=2;
      }
    }
  }

  i = 0;
  unsigned face = 0;
  /* z == 0 */
  for(unsigned y = 0; y < mesh_height; y++) {
    for(unsigned x = 0; x < mesh_width; x++) {
      unsigned v0, v1, v2, v3;
      v0 = y * (mesh_width+1) + x;
      v1 = y * (mesh_width+1) + x + 1;
      v2 = (y+1) * (mesh_width+1) + x + 1;
      v3 = (y+1) * (mesh_width+1) + x;
      faces[i]   = v0;
      faces[i+1] = v1;
      faces[i+2] = v2;
      faces[i+3] = v3;
      face_color_index[i]   = face;
      face_color_index[i+1] = face;
      face_color_index[i+2] = face;
      face_color_index[i+3] = face;
      i += 4;
      face++;
    }
  }
  
  for(unsigned z = 1; z <= mesh_depth; z++) {
    for(unsigned y = 0; y < mesh_height; y++) {
      for(unsigned x = 0; x < mesh_width; x++) {
        unsigned v0, v1, v2, v3;
        v0 = z * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x;
        v1 = z * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x + 1;
        v2 = z * (mesh_width+1) * (mesh_height+1) + (y+1) * (mesh_width+1) + x + 1;
        v3 = z * (mesh_width+1) * (mesh_height+1) + (y+1) * (mesh_width+1) + x;
        faces[i]   = v0;
        faces[i+1] = v1;
        faces[i+2] = v2;
        faces[i+3] = v3;
        i += 4;
        face_color_index[i]   = face;
        face_color_index[i+1] = face;
        face_color_index[i+2] = face;
        face_color_index[i+3] = face;
        i += 4;
        face++;
      }
    }
  }

  
  printf("1st face count = %d\n", face);

  for(unsigned y = 0; y <= mesh_height; y++) {  
    for(unsigned z = 0; z < mesh_depth; z++) {
      for(unsigned x = 0; x < mesh_width; x++) {
        faces[i]   = z * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x;
        faces[i+1] = z * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x + 1;
        faces[i+2] = (z+1) * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x + 1;
        faces[i+3] = (z+1) * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x;

        face_color_index[i]   = face;
        face_color_index[i+1] = face;
        face_color_index[i+2] = face;
        face_color_index[i+3] = face;
        
        i += 4;
        face++;
      }
    }
  }
  
  printf("2nd face count = %d\n", face);  

  for(unsigned x = 0; x <= mesh_width; x++) {
    for(unsigned y = 0; y < mesh_height; y++) {    
      for(unsigned z = 0; z < mesh_depth; z++) {    
        faces[i]   = z * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x;
        faces[i+1] = (z+1) * (mesh_width+1) * (mesh_height+1) + y * (mesh_width+1) + x;
        faces[i+2] = (z+1) * (mesh_width+1) * (mesh_height+1) + (y+1) * (mesh_width+1) + x; 
        faces[i+3] = z * (mesh_width+1) * (mesh_height+1) + (y+1) * (mesh_width+1) + x;

        face_color_index[i]   = face;
        face_color_index[i+1] = face;
        face_color_index[i+2] = face;
        face_color_index[i+3] = face;
        
        i +=4;
        face++;        
      }
    }
  }

  printf("faces = %d\n", face);
}


void init_opengl() {
  glShadeModel(GL_FLAT);
  glEnable(GL_DEPTH_TEST);
  glPointSize(5.0);
  glLineWidth(3.0);
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
  //glOrtho(-10, 10, -10, 10, -20.0, 20.0);
  glFrustum(-2.0, 2.0, -2.0, 2.0, 6.0, 55);
  glMatrixMode(GL_MODELVIEW);  
}


void render_mesh(GLFWwindow *window, float degrees) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  gluLookAt(1.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  glRotatef(degrees, 1.0, 1.0, 0.0);
  glTranslatef(-1.0f, -1.5f, -2.0f);  

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_INT, 0, vertices);
  
  glColor4f(1.0, 1.0, 1.0, 0.0);      
  glDrawArrays(GL_POINTS, 0, mesh_vert_width * mesh_vert_height * mesh_vert_depth);
  
  glColor4f(0.7, 0.7, .7, 0.0);    
  glDrawElements(GL_LINES,
                 mesh_vert_depth * mesh_width * mesh_vert_height * 2 +
                 mesh_vert_depth * mesh_height * mesh_vert_width * 2 +
                 mesh_depth * mesh_vert_width * mesh_vert_height * 2, 
                 GL_UNSIGNED_INT, edges);  

  /*
  glEnableClientState(GL_COLOR_ARRAY);
  glColorPointer(3, GL_FLOAT, 0, face_colors);
  glColor4f(0.4, 0.4, .4, 0.0);      
  glDrawElements(GL_QUADS,
                 mesh_width * mesh_height * mesh_vert_depth * 4 +
                 mesh_vert_width * mesh_height * mesh_depth * 4 +
                 mesh_vert_width * mesh_vert_height * mesh_depth * 4,
                 GL_UNSIGNED_INT, faces);
  */
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glfwSwapBuffers(window);
}


int main(int argc, char *argv[]) {
  
  init_mesh(2, 3, 4, 1);
  
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
  

  float degrees = 0.0f;
  while(! glfwWindowShouldClose(window)) {
    render_mesh(window, degrees);
    degrees += 1.0f;
    if (degrees >= 360.0f)
      degrees = 0.0f;
    glfwPollEvents();
  }
  
  return 0;
}
