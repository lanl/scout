/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#version 120

uniform float windowWidth; 
uniform float near, far;

attribute float radius;
attribute vec4 color;

varying vec3 WSc;
varying vec4 center;
varying float WSr; 
varying float pointSize;
varying float cameraDepth;
varying vec4 varycolor;

void main()
{
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
  if (radius > 0.0) {
    vec4 P = gl_ModelViewMatrix * gl_Vertex;
    vec4 C = gl_Position;
    vec4 S = gl_ProjectionMatrix * vec4(P.x + radius, P.yzw);

    WSr = (S - C).x;
    
    C /= C.w;
    S /= S.w;
    WSc = C.xyz;
  
    center = P; 
    pointSize = windowWidth * (S - C).x;
    gl_PointSize = pointSize;
  } else {
    gl_PointSize = 0.0;
  }
  gl_FrontColor = color;
  varycolor = color;
}
