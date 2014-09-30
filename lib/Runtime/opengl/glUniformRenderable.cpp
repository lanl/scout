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

#include <unistd.h>
#include <cstdlib>
#include <iostream>

#include <OpenGL/gl3.h>

#include "scout/Runtime/opengl/glUniformRenderable.h"

#define GLSL(shader) "#version 410 core\n" #shader

using namespace std;
using namespace scout;

namespace{

const char* vertexVS = 
GLSL(
  layout(location = 0) in vec3 position;
  layout(location = 1) in vec4 color;

  out VS_OUT{
    vec4 color;
  } vs_out;

  void main () {
    gl_Position = vec4(position, 1.0f);
    vs_out.color = color;
  }
);

const char* vertexGS =
GLSL(
  layout(points) in;
  layout(points, max_vertices = 1) out;

  in VS_OUT{
    vec4 color;
  } gs_in[1];

  out GS_OUT{
    vec4 color;
  } gs_out;

  uniform mat4 mvp;
  uniform float pointSize;

  void main(){
    gl_Position = mvp * gl_in[0].gl_Position;
    gs_out.color = gs_in[0].color;
    gl_PointSize = pointSize;
    EmitVertex();

    EndPrimitive();
  }
);

const char* vertexFS =
GLSL(
  in GS_OUT{
    vec4 color;
  } fs_in;

  out vec4 color;

  void main(void){
    color = fs_in.color;
  }
);

const char* edgeVS = 
GLSL(
  layout(location = 0) in vec3 position;
  layout(location = 1) in vec4 color;

  out VS_OUT{
    vec4 color;
  } vs_out;

  void main () {
    gl_Position = vec4(position, 1.0f);
    vs_out.color = color;
  }
);

const char* edgeGS =
GLSL(
  layout(lines_adjacency) in;
  layout(triangle_strip, max_vertices = 16) out;

  in VS_OUT{
    vec4 color;
  } gs_in[4];

  out GS_OUT{
    vec4 color;
  } gs_out;

  uniform mat4 mvp;
  uniform float edgeWidth;

  void main(){

    vec4 v1 = gl_in[0].gl_Position;
    vec4 v2 = gl_in[1].gl_Position;
    vec4 v3 = gl_in[2].gl_Position;
    vec4 v4 = gl_in[3].gl_Position;
  
    float w = edgeWidth;

    gl_Position = mvp * (v1 + vec4(0.0, -w, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v1 + vec4(0.0, w, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v2 + vec4(0.0, -w, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v2 + vec4(0.0, w, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();
  
    EndPrimitive();

    gl_Position = mvp * (v2 + vec4(-w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[1].color;
    EmitVertex();

    gl_Position = mvp * (v3 + vec4(-w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[1].color;
    EmitVertex();

    gl_Position = mvp * (v2 + vec4(w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[1].color;
    EmitVertex();

    gl_Position = mvp * (v3 + vec4(w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[1].color;
    EmitVertex();
  
    EndPrimitive();

    gl_Position = mvp * (v4 + vec4(0.0, -w, 0.0, 0.0));
    gs_out.color = gs_in[2].color;
    EmitVertex();

    gl_Position = mvp * (v4 + vec4(0.0, w, 0.0, 0.0));
    gs_out.color = gs_in[2].color;
    EmitVertex();

    gl_Position = mvp * (v3 + vec4(0.0, -w, 0.0, 0.0));
    gs_out.color = gs_in[2].color;
    EmitVertex();

    gl_Position = mvp * (v3 + vec4(0.0, w, 0.0, 0.0));
    gs_out.color = gs_in[2].color;
    EmitVertex();
  
    EndPrimitive();

    gl_Position = mvp * (v1 + vec4(-w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[3].color;
    EmitVertex();

    gl_Position = mvp * (v4 + vec4(-w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[3].color;
    EmitVertex();

    gl_Position = mvp * (v1 + vec4(w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[3].color;
    EmitVertex();

    gl_Position = mvp * (v4 + vec4(w, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[3].color;
    EmitVertex();
  
    EndPrimitive();
  }
);

const char* edgeFS =
GLSL(
  in GS_OUT{
    vec4 color;
  } fs_in;

  out vec4 color;

  void main(void){
    color = fs_in.color;
  }
);

const char* cellVS = 
GLSL(
  layout(location = 0) in vec3 position;
  layout(location = 1) in vec4 color;

  out VS_OUT{
    vec4 color;
  } vs_out;

  void main () {
    gl_Position = vec4(position, 1.0f);
    vs_out.color = color;
  }
);

const char* cellGS =
GLSL(
  layout(points) in;
  layout(triangle_strip, max_vertices = 4) out;

  in VS_OUT{
    vec4 color;
  } gs_in[1];

  out GS_OUT{
    vec4 color;
  } gs_out;

  uniform mat4 mvp;

  void main(){

    vec4 v = gl_in[0].gl_Position;

    gl_Position = mvp * v;
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v + vec4(0.0, 1.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v + vec4(1.0, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v + vec4(1.0, 1.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    EndPrimitive();
  }
);

const char* cellFS =
GLSL(
  in GS_OUT{
    vec4 color;
  } fs_in;

  out vec4 color;

  void main(void){
    color = fs_in.color;
  }
);

GLuint compileShader(const char* code, GLenum type){
  GLuint res = glCreateShader(type);

  if(!res){
    return 0;
  }
  
  glShaderSource(res, 1, &code, NULL);
  
  glCompileShader(res);
  
  GLint status = 0;
  glGetShaderiv(res, GL_COMPILE_STATUS, &status);
  
  if(!status){
    char buf[4096];    
    glGetShaderInfoLog(res, 4096, NULL, buf);
    cerr << buf << endl;

    glDeleteShader(res);
    return 0;
  }
  
  return res;
}

GLuint loadShader(const char* path, GLenum type){
  FILE* fp = fopen(path, "rb");
  if(!fp){
    return 0;
  }
  
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  
  char* buf = new char[size + 1];
  
  if(!buf){
    fclose(fp);
    return 0;
  }
  
  fread(buf, 1, size, fp);
  buf[size] = 0;
  fclose(fp);
  
  GLuint shader = compileShader(buf, type);

  delete[] buf;

  return shader;
}

} // end namespace

glUniformRenderable::glUniformRenderable(size_t width, size_t height)
  : width_(width),
    height_(height),
    drawCells_(false),
    drawEdges_(false),
    drawVertices_(false){
  
}


glUniformRenderable::~glUniformRenderable(){
  
}


void glUniformRenderable::initialize(glCamera* camera){
  GLuint vs = compileShader(cellVS, GL_VERTEX_SHADER);
  assert(vs);

  GLuint gs = compileShader(cellGS, GL_GEOMETRY_SHADER);
  assert(gs);

  GLuint fs = compileShader(cellFS, GL_FRAGMENT_SHADER);
  assert(fs);

  cellProgram_ = glCreateProgram();

  glAttachShader(cellProgram_, vs);
  glAttachShader(cellProgram_, gs);
  glAttachShader(cellProgram_, fs);

  glDeleteShader(vs);  
  glDeleteShader(gs);
  glDeleteShader(fs);
  
  glLinkProgram(cellProgram_);

  /*
  vs = compileShader(edgeVS, GL_VERTEX_SHADER);
  assert(vs);

  gs = compileShader(edgeGS, GL_GEOMETRY_SHADER);
  assert(gs);

  fs = compileShader(edgeFS, GL_FRAGMENT_SHADER);
  assert(fs);

  edgeProgram_ = glCreateProgram();

  glAttachShader(edgeProgram_, vs);
  glAttachShader(edgeProgram_, gs);
  glAttachShader(edgeProgram_, fs);

  glDeleteShader(vs);  
  glDeleteShader(gs);
  glDeleteShader(fs);
  
  glLinkProgram(edgeProgram_);

  vs = compileShader(vertexVS, GL_VERTEX_SHADER);
  assert(vs);

  gs = compileShader(vertexGS, GL_GEOMETRY_SHADER);
  assert(gs);

  fs = compileShader(vertexFS, GL_FRAGMENT_SHADER);
  assert(fs);

  vertexProgram_ = glCreateProgram();

  glAttachShader(vertexProgram_, vs);
  glAttachShader(vertexProgram_, gs);
  glAttachShader(vertexProgram_, fs);

  glDeleteShader(vs);  
  glDeleteShader(gs);
  glDeleteShader(fs);
  
  glLinkProgram(vertexProgram_);
  */

  mvpCellLoc_ = glGetUniformLocation(cellProgram_, "mvp");
  //mvpEdgeLoc_ = glGetUniformLocation(edgeProgram_, "mvp");
  //mvpVertexLoc_ = glGetUniformLocation(vertexProgram_, "mvp");
  //edgeWidthEdgeLoc_ = glGetUniformLocation(edgeProgram_, "edgeWidth");
  //pointSizeVertexLoc_ = glGetUniformLocation(vertexProgram_, "pointSize");

  float pad = 0.05;
  float near = 0.00;
  float far = 1000.0;

  float left;
  float right;
  float bottom;
  float top;

  if(width_ >= height_){
    float px = pad * width_;
    float py = (1 - float(height_)/width_) * width_ * 0.50;

    left = -px;
    right = width_ + px;
    bottom = -py - px;
    top = width_ - py + px;
  }
  else{
    float py = pad * height_;
    float px = (1 - float(width_)/height_) * height_ * 0.50;

    left = -px - py;
    right = width_ + px + py;
    bottom = -py;
    top = height_ + py;
  }

  mvp_ = vmath::ortho(left, right, bottom, top, near, far);

  size_t width1 = width_ + 1;
  size_t height1 = height_ + 1;
  numCells_ = width_ * height_;
  numVertices_ = width1 * height_;
  numEdges_ = width_ * height1 + width1 * height_;

  cellColors_ = new glColorBuffer;
  cellColors_->bind();
  cellColors_->alloc(sizeof(float) * 4 * numCells_, GL_STREAM_DRAW_ARB);
  cellColors_->release();

  cellPoints_ = new glVertexBuffer;
  cellPoints_->bind();
  cellPoints_->alloc(sizeof(float) * 3 * numCells_, GL_STREAM_DRAW_ARB);
  cellPoints_->release();

  float* points = (float*)cellPoints_->mapForWrite();
  size_t i = 0;
  for(size_t y = 0; y < height_; ++y) {
    for(size_t x = 0; x < width_; ++x) {
      points[i++] = x;
      points[i++] = y;
      points[i++] = 0.0f; 
    }
  }
  cellPoints_->unmap();

  /*
  edgeColors_ = new glColorBuffer;
  edgeColors_->bind();
  edgeColors_->alloc(sizeof(float) * 4 * numEdges_, GL_STREAM_DRAW_ARB);
  edgeColors_->release();

  vertexColors_ = new glColorBuffer;
  vertexColors_->bind();
  vertexColors_->alloc(sizeof(float) * 4 * numVertices_, GL_STREAM_DRAW_ARB);
  vertexColors_->release();
  */

  cellVAO_ = 0;
  glGenVertexArrays(1, &cellVAO_);
  glBindVertexArray(cellVAO_);
  cellPoints_->bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  cellColors_->bind();
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindAttribLocation(cellProgram_, 0, "position");
  glBindAttribLocation(cellProgram_, 1, "color");
  
  cellPoints_->release();
  cellColors_->release();

  /*
  vertexVAO_ = 0;
  glGenVertexArrays(1, &vertexVAO_);
  glBindVertexArray(vertexVAO_);
  vertexPoints_->bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  vertexColors_->bind();
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindAttribLocation(vertexProgram_, 0, "position");
  glBindAttribLocation(vertexProgram_, 1, "color");

  edgeVAO_ = 0;
  glGenVertexArrays(1, &edgeVAO_);
  glBindVertexArray(edgeVAO_);
  edgeLinesAdj_->bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  edgeColors_->bind();
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindAttribLocation(edgeProgram_, 0, "position");
  glBindAttribLocation(edgeProgram_, 1, "color");

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  */

  glClearColor(0.5, 0.55, 0.65, 1.0);
}

float4* glUniformRenderable::map_colors(){
  drawCells_ = true;
  return (float4*)cellColors_->mapForWrite();
}

float4* glUniformRenderable::map_vertex_colors(){
  drawVertices_ = true;
  return (float4*)vertexColors_->mapForWrite();
}

float4* glUniformRenderable::map_edge_colors(){
  drawEdges_ = true;
  return (float4*)edgeColors_->mapForWrite();
}

void glUniformRenderable::unmap_colors(){
  if(!drawCells_){
    return;
  }
  
  cellColors_->unmap();
}

void glUniformRenderable::unmap_vertex_colors(){
  if(!drawVertices_){
    return;
  }
  
  vertexColors_->unmap();
}

void glUniformRenderable::unmap_edge_colors(){
  if(!drawEdges_){
    return;
  }
  
  edgeColors_->unmap();
}

void glUniformRenderable::draw(glCamera* camera){
  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(cellProgram_);
  glUniformMatrix4fv(mvpCellLoc_, 1, GL_FALSE, mvp_);
  glBindVertexArray(cellVAO_);
  glDrawArrays(GL_POINTS, 0, numCells_);

  /*
  glUseProgram(edgeProgram);
  glUniform1f(widthEdgeLoc, 0.05);
  glUniformMatrix4fv(mvpEdgeLoc, 1, GL_FALSE, mvp);
  glBindVertexArray(edgeVAO);
  glDrawArrays(GL_LINES_ADJACENCY, 0, n * 4);

  glUseProgram(vertexProgram);
  glUniform1f(pointSizeVertexLoc, 5.0);
  glUniformMatrix4fv(mvpVertexLoc, 1, GL_FALSE, mvp);
  glBindVertexArray(vertexVAO);
  glDrawArrays(GL_POINTS, 0, n);
  */
}
