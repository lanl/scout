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
  uniform float size;

  void main(){
    gl_Position = mvp * gl_in[0].gl_Position;
    gs_out.color = gs_in[0].color;
    gl_PointSize = size;
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

const char* horizEdgeGS =
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
  uniform float size;

  void main(){

    vec4 v = gl_in[0].gl_Position;
  
    gl_Position = mvp * (v + vec4(0.0, -size, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v + vec4(0.0, size, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v + vec4(1.0, -size, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (v + vec4(1.0, size, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    EmitVertex();
  
    EndPrimitive();
  }
);

const char* vertEdgeGS =
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
  uniform float size;
  uniform int start;

  void main(){

    vec4 v = gl_in[0].gl_Position;
  
    gl_Position = mvp * (v + vec4(-size, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    gl_PrimitiveID = start + gl_PrimitiveIDIn;
    EmitVertex();

    gl_Position = mvp * (v + vec4(-size, 1.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    gl_PrimitiveID = start + gl_PrimitiveIDIn;
    EmitVertex();

    gl_Position = mvp * (v + vec4(size, 0.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    gl_PrimitiveID = start + gl_PrimitiveIDIn;
    EmitVertex();

    gl_Position = mvp * (v + vec4(size, 1.0, 0.0, 0.0));
    gs_out.color = gs_in[0].color;
    gl_PrimitiveID = start + gl_PrimitiveIDIn;
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
  delete cellPoints_;
  delete vertexPoints_;
  delete horizEdgePoints_;
  delete vertEdgePoints_;
    
  delete cellColors_;
  delete vertexColors_;
  delete edgeColors_;
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

  vs = compileShader(edgeVS, GL_VERTEX_SHADER);
  assert(vs);

  gs = compileShader(horizEdgeGS, GL_GEOMETRY_SHADER);
  assert(gs);

  fs = compileShader(edgeFS, GL_FRAGMENT_SHADER);
  assert(fs);

  horizEdgeProgram_ = glCreateProgram();

  glAttachShader(horizEdgeProgram_, vs);
  glAttachShader(horizEdgeProgram_, gs);
  glAttachShader(horizEdgeProgram_, fs);

  glDeleteShader(vs);  
  glDeleteShader(gs);
  glDeleteShader(fs);
  
  glLinkProgram(horizEdgeProgram_);

  vs = compileShader(edgeVS, GL_VERTEX_SHADER);
  assert(vs);

  gs = compileShader(vertEdgeGS, GL_GEOMETRY_SHADER);
  assert(gs);

  fs = compileShader(edgeFS, GL_FRAGMENT_SHADER);
  assert(fs);

  vertEdgeProgram_ = glCreateProgram();

  glAttachShader(vertEdgeProgram_, vs);
  glAttachShader(vertEdgeProgram_, gs);
  glAttachShader(vertEdgeProgram_, fs);

  glDeleteShader(vs);  
  glDeleteShader(gs);
  glDeleteShader(fs);
  
  glLinkProgram(vertEdgeProgram_);

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

  mvpCellLoc_ = glGetUniformLocation(cellProgram_, "mvp");

  mvpHorizEdgeLoc_ = glGetUniformLocation(horizEdgeProgram_, "mvp");
  mvpVertEdgeLoc_ = glGetUniformLocation(vertEdgeProgram_, "mvp");

  sizeHorizEdgeLoc_ = 
    glGetUniformLocation(horizEdgeProgram_, "size");

  sizeVertEdgeLoc_ = 
    glGetUniformLocation(vertEdgeProgram_, "size");
  
  startVertEdgeLoc_ = glGetUniformLocation(vertEdgeProgram_, "start");

  sizeVertexLoc_ = glGetUniformLocation(vertexProgram_, "size");
  mvpVertexLoc_ = glGetUniformLocation(vertexProgram_, "mvp");

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
  numVertices_ = width1 * height1;
  numHorizEdges_ = width_ * height1;
  numVertEdges_ = width1 * height_;
  size_t numEdges = numHorizEdges_ + numVertEdges_;

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

  edgeColors_ = new glColorBuffer;
  edgeColors_->bind();
  edgeColors_->alloc(sizeof(float) * 4 * numEdges, GL_STREAM_DRAW_ARB);
  edgeColors_->release();

  horizEdgePoints_ = new glVertexBuffer;
  horizEdgePoints_->bind();
  horizEdgePoints_->alloc(sizeof(float) * 3 * numHorizEdges_,
                          GL_STREAM_DRAW_ARB);
  horizEdgePoints_->release();

  i = 0;
  float* edges = (float*)horizEdgePoints_->mapForWrite();
  for(size_t y = 0; y < height1; ++y){
    for(size_t x = 0; x < width_; ++x){
      edges[i++] = x;
      edges[i++] = y;      
      edges[i++] = 0.0f; 
    }
  }
  horizEdgePoints_->unmap();

  vertEdgePoints_ = new glVertexBuffer;
  vertEdgePoints_->bind();
  vertEdgePoints_->alloc(sizeof(float) * 3 * numVertEdges_,
                          GL_STREAM_DRAW_ARB);
  vertEdgePoints_->release();

  i = 0;
  edges = (float*)vertEdgePoints_->mapForWrite();
  for(size_t x = 0; x < width1; ++x){
    for(size_t y = 0; y < height_; ++y){
      edges[i++] = x;
      edges[i++] = y;      
      edges[i++] = 0.0f; 
    }
  }
  vertEdgePoints_->unmap();

  vertexColors_ = new glColorBuffer;
  vertexColors_->bind();
  vertexColors_->alloc(sizeof(float) * 4 * numVertices_, GL_STREAM_DRAW_ARB);
  vertexColors_->release();

  vertexPoints_ = new glVertexBuffer;
  vertexPoints_->bind();
  vertexPoints_->alloc(sizeof(float) * 3 * numVertices_, GL_STREAM_DRAW_ARB);
  vertexPoints_->release();

  points = (float*)vertexPoints_->mapForWrite();
  i = 0;
  for(size_t y = 0; y < height1; ++y) {
    for(size_t x = 0; x < width1; ++x) {
      points[i++] = x;
      points[i++] = y;
      points[i++] = 0.0f; 
    }
  }
  vertexPoints_->unmap();

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
  
  horizEdgeVAO_ = 0;
  glGenVertexArrays(1, &horizEdgeVAO_);
  glBindVertexArray(horizEdgeVAO_);
  horizEdgePoints_->bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  edgeColors_->bind();
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindAttribLocation(horizEdgeProgram_, 0, "position");
  glBindAttribLocation(horizEdgeProgram_, 1, "color");

  vertEdgeVAO_ = 0;
  glGenVertexArrays(1, &vertEdgeVAO_);
  glBindVertexArray(vertEdgeVAO_);
  vertEdgePoints_->bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  edgeColors_->bind();
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindAttribLocation(vertEdgeProgram_, 0, "position");
  glBindAttribLocation(vertEdgeProgram_, 1, "color");

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

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  vertexSize_ = 100.0f/max(width_, height_);
  edgeSize_ = vertexSize_/50.f;

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
  if(drawCells_){
    glUseProgram(cellProgram_);
    glUniformMatrix4fv(mvpCellLoc_, 1, GL_FALSE, mvp_);
    glBindVertexArray(cellVAO_);
    glDrawArrays(GL_POINTS, 0, numCells_);
  }

  if(drawEdges_){
    glUseProgram(horizEdgeProgram_);
    glUniform1f(sizeHorizEdgeLoc_, edgeSize_);
    glUniformMatrix4fv(mvpHorizEdgeLoc_, 1, GL_FALSE, mvp_);
    glBindVertexArray(horizEdgeVAO_);
    glDrawArrays(GL_POINTS, 0, numHorizEdges_);

    glUseProgram(vertEdgeProgram_);
    glUniform1i(startVertEdgeLoc_, numHorizEdges_);
    glUniform1f(sizeVertEdgeLoc_, edgeSize_);
    glUniformMatrix4fv(mvpVertEdgeLoc_, 1, GL_FALSE, mvp_);
    glBindVertexArray(vertEdgeVAO_);
    glDrawArrays(GL_POINTS, 0, numVertEdges_);
  }

  if(drawVertices_){
    glUseProgram(vertexProgram_);
    glUniformMatrix4fv(mvpVertexLoc_, 1, GL_FALSE, mvp_);
    glUniform1f(sizeVertexLoc_, vertexSize_);
    glBindVertexArray(vertexVAO_);
    glDrawArrays(GL_POINTS, 0, numVertices_);
  }
}
